import copy,json,sys,tempfile,unittest
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a1_local.config_runtime import load_config,runtime_placement,stage_spec
from a1_local.elastic import GlobalBatches,validate_resume_contract
from run_stage_ii_auto_gpu import choose_mapping,BUDGET,RESERVE,FIRST_HOST,HOST_RESERVE

def resume_worker(rank,world,init_file,checkpoint,resuming,result):
 import torch
 import torch.distributed as dist
 from a1_local.distributed import DistributedMaster
 from a1_local.checkpointing import make_recovery
 from goal_local.optim import MasterAdamW
 torch.set_num_threads(1);torch.manual_seed(5)
 dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=world)
 model=torch.nn.Linear(4,2);reference=copy.deepcopy(model)
 opt=DistributedMaster(model.parameters(),1e-3,1e-8)
 contract=dict(world=world,physical_gpus=list(range(world)),rank_to_gpu=list(range(world)),sampler='global_batch32_v1',schedule=2500)
 if resuming:
  state=torch.load(checkpoint,weights_only=False)
  assert validate_resume_contract(state['contract'],contract,True)
  opt.load_state_dict(state['optimizer'] if rank==0 else None)
 else:opt.broadcast_weights()
 torch.manual_seed(17);x=torch.randn(64,4);y=torch.randn(64,2)
 offset=32 if resuming else 0;opt.zero_grad()
 for i in range(rank,32,world):
  ((model(x[offset+i:offset+i+1])-y[offset+i:offset+i+1]).square().mean()*world/32).backward()
 opt.step(1e-3,1)
 if rank==0:
  if not resuming:
   states=[dict(rank=r,epoch=0,consumed=1,rng={'torch':torch.get_rng_state()}) for r in range(world)]
   torch.save(make_recovery(1,contract,opt.state_dict(),None,states),checkpoint)
  else:
   ref=MasterAdamW(reference.parameters(),1e-3,1e-8,offload=True)
   for start in (0,32):
    reference.zero_grad();((reference(x[start:start+32])-y[start:start+32]).square().mean()).backward()
    for p,m in zip(reference.parameters(),ref.masters):m.grad=p.grad.float().clone()
    torch.nn.utils.clip_grad_norm_(ref.masters,1,foreach=False);ref.optimizer.step()
    with torch.no_grad():
     for p,m in zip(reference.parameters(),ref.masters):p.copy_(m)
   diff=max(float((p.detach()-q.detach()).abs().max()) for p,q in zip(model.parameters(),reference.parameters()))
   Path(result).write_text(json.dumps({'max_difference':diff}))
 dist.barrier();dist.destroy_process_group()

class AutomaticPlacement(unittest.TestCase):
 def test_uneven_mapping_and_batch(self):
  cfg=load_config(WORK/'config.json','A1');before=copy.deepcopy(cfg)
  spec=stage_spec(cfg,'II',runtime_placement(cfg,'II',[5,0,4],None,[5,0,5,4,5]))
  self.assertEqual(spec['rank_to_gpu'],[5,0,5,4,5]);self.assertEqual(sum(spec['accumulation_per_rank']),32)
  self.assertIsNone(spec['processes_per_gpu']);self.assertEqual(cfg,before)
 def test_sampler_exact_global_order_across_resize_and_epoch(self):
  for world in (1,3,6,32):
   streams=[]
   for rank in range(world):
    sampler=GlobalBatches(range(103),world,rank,2026);sampler.resume_step(1)
    streams.append(iter(sampler))
   actual=[]
   for u in (1,2):
    actual.extend(next(streams[slot%world])[0] for slot in range(32))
   reference=GlobalBatches(range(103),1,0,2026);reference.resume_step(1)
   self.assertEqual(actual,[b[0] for b in reference])
  sampler=GlobalBatches(range(103),3,0,2026)
  self.assertEqual(sampler.resume_step(3),(1,0))
 def test_only_placement_can_change_on_resume(self):
  old=dict(world=1,rank_to_gpu=[0],sampler='global_batch32_v1',schedule=2500,code_sha256='fixed')
  new=dict(old,world=3,rank_to_gpu=[0,0,5])
  self.assertTrue(validate_resume_contract(old,new,True))
  for key,value in (('schedule',100),('code_sha256','changed'),('sampler','old')):
   with self.assertRaises(ValueError):validate_resume_contract(old,dict(new,**{key:value}),True)
  with self.assertRaises(ValueError):validate_resume_contract(old,new,False)
 def test_capacity_needs_safety_margin_and_limits_host(self):
  capacities={0:8705,1:2*BUDGET+RESERVE,2:BUDGET+RESERVE-1,3:BUDGET+RESERVE}
  m,detail=choose_mapping(capacities,200000)
  self.assertEqual(len(m),3);self.assertNotIn(0,m);self.assertNotIn(2,m)
  self.assertEqual(m.count(1),2)
  self.assertEqual(choose_mapping(capacities,HOST_RESERVE+FIRST_HOST)[1]['host_limit'],1)
  self.assertEqual(choose_mapping(capacities,200000,blocked=[1])[0],[3])
 def test_real_optimizer_resume_from_two_to_three_ranks(self):
  import torch.multiprocessing as mp
  with tempfile.TemporaryDirectory(prefix='a1_elastic_test_') as d:
   p=Path(d);checkpoint=str(p/'trainer.pt');result=str(p/'result.json')
   for world,resume in ((2,False),(3,True)):
    mp.spawn(resume_worker,args=(world,str(p/f'init{world}'),checkpoint,resume,result),nprocs=world,join=True)
   self.assertLess(json.loads(Path(result).read_text())['max_difference'],2e-6)
if __name__=='__main__':unittest.main()
