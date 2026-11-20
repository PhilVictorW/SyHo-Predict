"""Two deployment policies; original arm imports the original LaWAM checkout."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'robotwin/round10'
SERVER_OUT=Path(os.environ.get('ROUND10_SERVER_OUTPUT',str(R/'online')))
SERVER_LABEL=os.environ.get('ROUND10_SERVER_LABEL','')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

class Policy:
    def __init__(self,arm,ckpt):
        self.arm=arm;self.ckpt=str(Path(ckpt).resolve());self.calls=0
        if arm=='original':
            original=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM');sys.path.insert(0,str(original))
            from deployment.model_server.server_policy import load_policy_from_checkpoint
            import starVLA.model.framework.lawam_framework as module
            assert Path(module.__file__).resolve().is_relative_to(original)
            self.model=load_policy_from_checkpoint(self.ckpt,use_bf16=True,device='cuda').eval().requires_grad_(False)
            self.code_source=module.__file__;self.teacher=None
            assert self.model.policy_backend.flow._compute_dtype()==torch.bfloat16
            assert self.model.policy_backend.flow.config.horizon_sec==1.2
        else:
            from robotwin.round6.model import RetainedTeacher
            from robotwin.round7.model import CurrentQueryStudent
            cfg=json.loads((ROOT/'robotwin/configs/stage_I.json').read_text());assert cfg['baseline_checkpoint']==self.ckpt
            recipe=json.loads((ROOT/'robotwin/round7/candidate_recipe.json').read_text());teacher=RetainedTeacher(cfg)
            for path in [recipe['source_checkpoint'],recipe['teacher_checkpoint']]:
                payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False);assert payload['manifest']['base_config']==cfg
                named=dict(teacher.named_parameters())
                with torch.no_grad():
                    for n,v in payload['adapter'].items():named[n].copy_(v)
                del named,payload
            self.teacher=teacher.cuda().eval().requires_grad_(False);teacher.policy.checkpoint_blocks=False;teacher.flow.checkpoint_blocks=False
            self.model=CurrentQueryStudent(teacher).cuda().eval().requires_grad_(False)
            init=torch.load(recipe['initializers'],map_location='cpu',weights_only=False);self.model.set_queries(*init['queries']['fixed_queries']);self.model.gate.load_state_dict(init['gate']);del init
            assert not self.model.gate.scales.any()
            def reject(*args,**kw):raise AssertionError('Real future input is forbidden in deployed student')
            teacher.policy.encode_future_images=reject;teacher.policy.future.forward=reject
            self.code_source=str(ROOT/'robotwin/round7/model.py')
    @torch.inference_mode()
    def candidate(self,example,noise):
        def image(x):
            a=np.asarray(x);assert a.dtype==np.uint8 and a.shape==(256,256,3)
            return torch.from_numpy(a.copy()).permute(2,0,1)
        obs=dict(current_main=image(example['primary_image'][0])[None],current_wrist=torch.stack([image(x) for x in example['wrist_image']])[None],instruction=[example['lang']])
        with torch.autocast('cuda',dtype=torch.bfloat16):return self.model(obs,noise)
    @torch.inference_mode()
    def predict_action(self,examples,**kwargs):
        assert len(examples)==1,'Paired pilot requires exactly one active slot'
        e=dict(examples[0]);seed=int(e.pop('_noise_seed'))
        assert set(e)=={'primary_image','wrist_image','lang','embodiment_id','action_hz'}
        assert len(e['primary_image'])==1 and len(e['wrist_image'])==2
        assert int(e['embodiment_id'])==1 and float(e['action_hz'])==30.
        torch.manual_seed(seed)
        if self.arm=='original':
            result=self.model.predict_action([e]);actions=np.asarray(result['normalized_actions'],dtype=np.float32)
        else:
            # Same BF16 Gaussian draws as the original deployed flow; keep repaired Euler state FP32.
            noise=torch.randn(1,50,32,device='cuda',dtype=torch.bfloat16).float()
            actions=self.candidate(e,noise).float().cpu().numpy()
        expected=(1,36,32) if self.arm=='original' else (1,36,16)
        assert actions.shape==expected and np.isfinite(actions).all(),actions.shape
        self.calls+=1
        if self.calls==1 or self.calls%50==0:print(json.dumps(dict(arm=self.arm,calls=self.calls,noise_seed=seed,shape=list(actions.shape))),flush=True)
        return dict(normalized_actions=actions)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','compatible'],required=True);p.add_argument('--ckpt_path',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--ready',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES');torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25);torch.use_deterministic_algorithms(True)
    started=time.time();policy=Policy(a.arm,a.ckpt_path)
    fixture=np.load(R/'fixture.npz');info=json.loads((R/'fixture.json').read_text())
    example=dict(primary_image=[fixture['main']],wrist_image=[fixture['left'],fixture['right']],lang=info['instruction'],embodiment_id=1,action_hz=30.,_noise_seed=290001)
    x=policy.predict_action([example])['normalized_actions'];y=policy.predict_action([example])['normalized_actions'];repeat_delta=float(np.max(np.abs(x-y)));assert np.allclose(x,y,atol=1e-2,rtol=1e-2),repeat_delta
    check=dict(passed=True,repeat_max_abs=float(np.max(np.abs(x-y))),shape=list(x.shape),effective_action_shape=[1,36,16],noise_dtype='bfloat16',code_source=policy.code_source)
    if a.arm=='compatible':
        from robotwin.round6.train import physical_metrics
        # Replay the exact old fixed validation frame and FP32 noise before deploying the wrapper.
        b=torch.load(R/'fixture_batch.pt',map_location='cpu',weights_only=False)
        torch.manual_seed(800);noise=torch.randn(1,50,32,device='cuda')
        prediction=policy.candidate(example,noise);measured=physical_metrics(prediction,b,policy.teacher.policy.statistics)
        error=max(abs(measured[k]-info['initial_metrics'][k]) for k in measured);default_limit=4e-2 if os.environ.get('CUDA_VISIBLE_DEVICES')=='5' else 1e-6;limit=float(os.environ.get('ROUND10_PREFLIGHT_LIMIT',default_limit));assert error<limit,error
        check['offline_wrapper_reproduction_max_abs']=error
    write(SERVER_OUT/f'{a.arm}{SERVER_LABEL}_preflight.json',check);np.save(SERVER_OUT/f'{a.arm}{SERVER_LABEL}_fixture_actions.npy',x)
    meta=dict(ckpt_path=policy.ckpt,arm=a.arm,server_type='round10_paired_deployment',horizon=36,action_hz=30,uses_future=False,uses_state=False,code_source=policy.code_source)
    write(a.ready,dict(pid=os.getpid(),ready=True,load_seconds=time.time()-started,preflight=check,**meta))
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    print(json.dumps(dict(ready=True,**meta)),flush=True)
    WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=meta).serve_forever()
if __name__=='__main__':main()
