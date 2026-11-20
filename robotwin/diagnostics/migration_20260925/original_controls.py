"""Restore the original inference-only LaWM chain for matched offline controls."""
import gc,json,time,hashlib
from pathlib import Path
import torch
from torch.utils.data import default_collate
from audit import ROOT,OUT,RUN,qwen_reference,sample,difference
from robotwin.model import RobotwinPolicy
from robotwin.data import FutureDataset
from starVLA.model.framework.vlas.lawam import VLMToLAMQFormer
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2

@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.16);torch.manual_seed(2026)
    cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text())
    stats=json.loads(Path(cfg['statistics_path']).read_text())
    ds=FutureDataset(cfg['dataset_root'],stats,'val')
    indices=json.loads((RUN/'validation_samples.json').read_text())['indices'][:8]
    print('Loading retained model and original LaWM decoder',flush=True)
    model=RobotwinPolicy.from_baseline(cfg).eval()
    decoder=LAMDecoder_v2(context_dim=1024,input_dim=768,num_queries=1,num_layers=12,
        num_heads=16,dropout=0.,grid_hw=(16,16),train_in_latent=True,
        ffn_expansion_factor=4,num_embodiments=32,code_dim=32,last_ln=True)
    mapper=VLMToLAMQFormer(vlm_hidden_dim=2048,lam_code_dim=32,num_layers=1,num_heads=8,ffn_expansion_factor=4.,dropout=0.)
    state=torch.load(cfg['baseline_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    def take(prefix):return {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    decoder.load_state_dict(take('policy_backend.lam.decoder.'),strict=True,assign=True)
    mapper.load_state_dict(take('policy_backend.vlm_to_lam.'),strict=True,assign=True)
    del state;gc.collect()
    model=model.cuda();decoder=decoder.cuda().eval();mapper=mapper.cuda().eval()
    report=dict(started_at=time.time(),scope='Offline paired inputs/noise; reconstructed original inference chain, no training or online rollout',
        source=cfg['baseline_checkpoint'],decoder_strict_load=True,mapper_strict_load=True,rows=[])
    def save():
        p=OUT/'original_controls.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(OUT/'original_controls.json')
    save()
    for i,index in enumerate(indices):
        b=default_collate([ds[int(index)]])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            parity,enc,allcur,rawcur=qwen_reference(model,b)
            dino=model.current_dino(b['current_main'])
            latent=mapper(allcur[rawcur[2]].reshape(1,8,2048))
            future=decoder(dino,latent)[:,0]
            def condition(keep_lawm,keep_queries):
                current,valid=(allcur,rawcur[1]) if keep_queries else (enc['current'],enc['current_valid'])
                vis=torch.cat((dino,future),1) if keep_lawm else dino
                base=torch.cat((vis,model.flow.enc_vlm(current)),1)
                im=torch.cat((torch.ones(1,vis.shape[1],device='cuda',dtype=torch.bool),torch.zeros_like(valid)),1)
                vm=torch.cat((torch.zeros(1,vis.shape[1],device='cuda',dtype=torch.bool),valid),1)
                return base,im,vm,None,None,None
            torch.manual_seed(800+i);noise=torch.randn(1,50,32,device='cuda')
            pred={}
            for name,lawm,queries,h in [('original36',True,True,36),('original8',True,True,8),
                ('no_lawm36',False,True,36),('no_queries36',True,False,36),('aba2_36',False,False,36),('aba2_8',False,False,8)]:
                pred[name]=sample(model.flow,condition(lawm,queries),noise,h,'native')[:,:8]
            target=b['actions'][:,:8,:16].cuda();mask=b['actions_mask'][:,:8,:16].cuda()
            mse=lambda y:float(((y-target).square()*mask).sum()/mask.sum())
            row=dict(sample=i,index=int(index),episode=int(b['episode_id'][0]),t=int(b['t'][0]),qwen_parity=parity,
                normalized_action_mse={k:mse(v) for k,v in pred.items()},
                original36_effects={k:difference(v,pred['original36']) for k,v in pred.items() if k!='original36'},
                latent_shape=list(latent.shape),future_shape=list(future.shape))
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
    report['completed']=True;report['finished_at']=time.time();report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30;save()

if __name__=='__main__':main()
