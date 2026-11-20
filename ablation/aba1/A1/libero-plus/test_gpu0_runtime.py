"""Runtime placement, batch invariance, and conservative memory admission checks."""
import copy,json,os,sys,tempfile,unittest
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a1_local.config_runtime import load_config,runtime_placement,stage_spec
from run_stage_ii_gpu0 import select_capacity

def distributed_worker(rank,world,init_file,output):
 import torch
 import torch.distributed as dist
 from a1_local.distributed import DistributedMaster
 from goal_local.optim import MasterAdamW
 torch.set_num_threads(1);torch.manual_seed(2026)
 dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=world)
 model=torch.nn.Linear(4,2);reference=copy.deepcopy(model)
 torch.manual_seed(17);x=torch.randn(32,4);y=torch.randn(32,2)
 opt=DistributedMaster(model.parameters(),1e-3,1e-8);opt.broadcast_weights();opt.zero_grad()
 for i in range(rank,32,world):
  ((model(x[i:i+1])-y[i:i+1]).square().mean()*world/32).backward()
 opt.step(1e-3,1.0)
 if rank==0:
  ref_opt=MasterAdamW(reference.parameters(),1e-3,1e-8,offload=True)
  ((reference(x)-y).square().mean()).backward()
  for p,m in zip(reference.parameters(),ref_opt.masters):m.grad=p.grad.detach().float().clone()
  torch.nn.utils.clip_grad_norm_(ref_opt.masters,1.0,foreach=False)
  ref_opt.optimizer.step()
  difference=max(float((p.detach()-m.detach()).abs().max()) for p,m in zip(model.parameters(),ref_opt.masters))
  Path(output).write_text(json.dumps(dict(max_difference=difference)))
 dist.destroy_process_group()

class RuntimePlacement(unittest.TestCase):
 def test_world_counts_keep_batch_and_teacher_config(self):
  cfg=load_config(WORK/'config.json','A1');old=copy.deepcopy(cfg)
  for n in range(1,33):
   spec=stage_spec(cfg,'II',runtime_placement(cfg,'II',[0],n))
   self.assertEqual(sum(spec['accumulation_per_rank']),32)
   self.assertEqual(spec['rank_to_gpu'],[0]*n)
   self.assertEqual(spec['stop_after_updates'],2500)
  self.assertEqual(cfg,old)
 def test_stage_i_placement_cannot_be_overridden(self):
  cfg=load_config(WORK/'config.json','A1')
  with self.assertRaises(ValueError):runtime_placement(cfg,'I',[0],1)
  with self.assertRaises(ValueError):runtime_placement(cfg,'II',[0],33)
 def test_capacity_reserves_memory_and_declines_unsafe_launch(self):
  c=select_capacity(60000,8000,150000,30000)
  self.assertEqual(c['processes'],5)
  self.assertLessEqual(c['processes']*c['per_rank_budget_mib']+c['reserve_mib'],60000)
  self.assertEqual(select_capacity(10000,8000,150000,30000)['processes'],0)
  self.assertEqual(select_capacity(60000,8000,20000,30000)['processes'],0)
 def test_one_and_five_rank_updates_match_global_batch_32(self):
  import torch.multiprocessing as mp
  with tempfile.TemporaryDirectory(prefix='a1_gpu0_gloo_') as temp:
   for world in (1,5):
    init=str(Path(temp)/f'group{world}');out=str(Path(temp)/f'result{world}.json')
    mp.spawn(distributed_worker,args=(world,init,out),nprocs=world,join=True)
    self.assertLess(json.loads(Path(out).read_text())['max_difference'],2e-6)
if __name__=='__main__':unittest.main()
