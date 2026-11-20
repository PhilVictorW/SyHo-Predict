"""Read-only localization of reference/live divergence; no optimizer."""
import json, os, contextlib, torch
from pathlib import Path
from torch.func import functional_call
from torch.utils.data import default_collate
from robotwin.round6.model import RetainedTeacher, FutureActionDataset
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid
R=Path(__file__).resolve().parents[2]
class Probe(RetainedTeacher):
    def forward(self,batch,route='train',prepared=None,**kw):
        if route=='conditions':return self.conditions(batch,prepared)
        return super().forward(batch,route,prepared,**kw)
def diff(a,b):
    return [dict(shape=list(x.shape),dtype=str(x.dtype),max=float((x.float()-y.float()).abs().max()),mse=float((x.float()-y.float()).square().mean())) for x,y in zip(a,b)]
def emit(**kw):print(json.dumps(kw),flush=True)
def ac(cache=True):return torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=cache)
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(260929)
    cfg=json.loads((R/'robotwin/configs/stage_I.json').read_text());m=Probe(cfg)
    state=torch.load(R/'robotwin/round4/runs/future_aux/step_000200/adapter.pt',map_location='cpu',mmap=True,weights_only=False)
    named=dict(m.named_parameters())
    with torch.no_grad():
        for k,v in state['adapter'].items():named[k].copy_(v)
    del state,named
    m=m.cuda();m.initialize_reference();m.train();refs=m.__dict__['_reference_parameters']
    emit(loaded=True,dit_training=m.flow.DiT.training,dropout_modules=sum(isinstance(x,torch.nn.Dropout) and x.p>0 for x in m.flow.DiT.modules()),ref_parameters=len(refs),reference_initial_equal=all(torch.equal(dict(m.named_parameters())[k],v) for k,v in refs.items()))
    ds=FutureActionDataset(cfg['dataset_root'],m.policy.statistics,'val')
    idx=json.loads((R/'robotwin/round6/runs/control/samples.json').read_text())['validation_indices'][0]
    b=default_collate([ds[idx]])
    with ac():
        prep=m.prepare(b)
        c=m.conditions(b,prep)
        with torch.no_grad():cn=m.conditions(b,prep)
        with torch.no_grad(),ac(False):cnn=m.conditions(b,prep)
        with torch.no_grad(),ac(False):cr=functional_call(m,refs,(b,),dict(route='conditions',prepared=prep),strict=False)
        emit(conditions_grad_vs_nograd=diff(c,cn),conditions_grad_vs_nocache=diff(c,cnn),conditions_grad_vs_ref=diff(c,cr),ref_vs_nocache=diff(cr,cnn))
        torch.manual_seed(600000);noise=torch.randn(1,50,32,device='cuda');index=int(torch.randint(10,()).item())
        with torch.no_grad(),ac(False):x,target,t,hz,emb,valid=functional_call(m,refs,(b,),dict(route='reference_trace',prepared=prep,noise=noise,time_index=index),strict=False)
        pred=m.flow.velocity(x,t,hz,emb,valid,c)
        with torch.no_grad():pn=m.flow.velocity(x,t,hz,emb,valid,cn)
        with torch.no_grad(),ac(False):pnn=m.flow.velocity(x,t,hz,emb,valid,cnn);pr=m.flow.velocity(x,t,hz,emb,valid,cr)
        emit(index=index,velocity_live_vs_target=diff([pred],[target]),velocity_nograd_vs_target=diff([pn],[target]),velocity_nocache_vs_target=diff([pnn],[target]),velocity_refcond_vs_target=diff([pr],[target]))
        # Isolate whether conditions or velocity grad/no_grad differs.
        with torch.no_grad(),ac(False):samecond=m.flow.velocity(x,t,hz,emb,valid,c)
        emit(velocity_samecond_grad_vs_nograd=diff([pred],[samecond]))
        # Run all math in training graph mode with immutable snapshot parameters.
        with ac(False):cg=functional_call(m,refs,(b,),dict(route='conditions',prepared=prep),strict=False)
        emit(conditions_reference_grad_mode=diff(c,cg))
    emit(done=True)
if __name__=='__main__':main()
