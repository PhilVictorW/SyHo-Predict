"""Disposable W0-based Stage-II smoke fixture; no optimizer updates or checkpoints."""
import json, sys, time, gc, random
from pathlib import Path
WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parents[3]
sys.path[:0]=[str(WORK),str(ROOT)]
import numpy as np
import torch
from torch.utils.data import DataLoader
from a2_goal.config import load_plan, atomic_json
from a2_goal.offload import stream_frozen_teacher
from ablation.aba1.libero_common.data import FutureDataset
from ablation.aba1.libero_common.model import GoalLocalPolicy,make_stage_two
import ablation.aba1.libero_common.model as module
from transformers.initialization import no_init_weights

def main():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device=torch.device('cuda:0')
    plan,cfg=load_plan()
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    original=module.Qwen3VLModel
    def loaded_qwen(*args,**kwargs):
        with no_init_weights(): return original(*args,**kwargs)
    module.Qwen3VLModel=loaded_qwen
    tick=time.time()
    teacher=GoalLocalPolicy.from_baseline(cfg)
    initialization=teacher.initialization_report
    # The positive step is a test fixture only; never exported or used for training.
    teacher.source_step=1
    system=make_stage_two(teacher,cfg)
    system.student.to(device)
    dataset=FutureDataset(cfg['dataset_root'],system.student.statistics,split='train',
        seed=cfg['split_seed'],val_ratio=cfg['val_ratio'],local=False,goal=True)
    batch=next(iter(DataLoader(dataset,batch_size=1,num_workers=0)))
    assert not any(k.startswith('local_') for k in batch)
    assert tuple(system.student.q_goal.shape)==(16,2048)
    # Actual-model streamed and resident Teacher outputs must agree bit for bit.
    teacher.to(device);teacher.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        resident=teacher.encode(batch)
    reference={layer:roles[0].detach().cpu() for layer,roles in resident['layers'].items()}
    del resident
    stream_frozen_teacher(teacher,device)
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        streamed=teacher.encode(batch)
    for layer,target in reference.items():
        torch.testing.assert_close(streamed['layers'][layer][0].cpu(),target,rtol=0,atol=0)
    del streamed,reference
    frozen={n:(p,p._version) for n,p in teacher.named_parameters()}
    losses=[]
    for _ in range(2):
        with torch.autocast('cuda',dtype=torch.bfloat16): values=system(batch)
        values['loss'].backward()
        assert torch.isfinite(system.student.q_goal.grad).all() and system.student.q_goal.grad.norm()>0
        losses.append({k:float(v.detach()) for k,v in values.items()})
    assert all(p._version==version and p.grad is None for p,version in frozen.values())
    report=dict(test_fixture_only=True,formal_training_updates=0,checkpoint_written=False,
        teacher_source='W0-based disposable fixture',initialization=initialization,
        goal_query_shape=list(system.student.q_goal.shape),local_present=False,
        future_offsets=[24,40,56,72,88],teacher_streaming_bitwise_equivalent=True,
        distill_layers=cfg['distill_layers'],losses=losses,teacher_frozen_unchanged=True,
        stage_II_peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3,
        seconds=time.time()-tick)
    atomic_json(WORK/'checks/cuda_validation.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
