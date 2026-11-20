"""CPU checks for the 4-to-8 sample cursor and guarded resume transition."""
import copy,itertools,json,unittest
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
from a2_stage2_fixed8.sampling import GlobalBatches
from a2_stage2_fixed8.runtime import WORK,POLICY,CHECKPOINT_POLICY,selected_spec,load_config,code_identity,digest_json
from a2_stage2_fixed8.resume import validate_resume_contract

class ResumeChecks(unittest.TestCase):
 def test_same_global_samples_after_3200(self):
  data=range(2125407);old=[]
  for rank in range(4):
   sampler=DistributedSampler(data,num_replicas=4,rank=rank,seed=2026,shuffle=True,drop_last=True)
   old.append(list(itertools.islice(iter(sampler),800,808)))
  expected=[old[r][j] for j in range(8) for r in range(4)]
  actual={}
  for rank in range(8):
   batches=GlobalBatches(data,8,rank,2026);batches.resume_step(100)
   values=[v[0] for v in itertools.islice(iter(batches),batches.local_count)]
   actual.update(zip(range(rank,32,8),values))
  self.assertEqual(expected,[actual[i] for i in range(32)])
  self.assertEqual(len(set(expected)),32)
 def test_fixed_layout_and_batch(self):
  cfg=load_config(WORK/'config.json','A2');s=selected_spec(cfg,8)
  self.assertEqual(s['rank_to_gpu'],[2,5,6,7,2,5,6,7])
  self.assertEqual({g:s['rank_to_gpu'].count(g) for g in [2,5,6,7]},{2:2,5:2,6:2,7:2})
  self.assertEqual(sum(s['accumulation_per_rank']),32)
  self.assertEqual(s['save_every'],100)
  self.assertEqual(s['stop_after_updates'],2500)
  with self.assertRaises(ValueError):selected_spec(cfg,9)
 def test_resume_guards(self):
  meta=json.loads((WORK/'logs/resume100_fixed10_20261030/source_recovery_metadata.json').read_text())
  old=meta['contract'];new=copy.deepcopy(old)
  new.update(world=8,physical_gpus=[2,5,6,7],rank_to_gpu=POLICY['base_rank_to_gpu'],accumulation=4,
             accumulation_per_rank=[4,4,4,4,4,4,4,4],rank_policy=POLICY['rank_policy'],
             code_sha256=digest_json(code_identity()),sampler='global_batch32_v1')
  recovery=dict(step=100,contract=old,rank_states=meta['rank_states'])
  self.assertTrue(validate_resume_contract(recovery,new))
  bad=copy.deepcopy(new);bad['schedule_total_steps']+=1
  with self.assertRaises(ValueError):validate_resume_contract(recovery,bad)
  bad=copy.deepcopy(recovery);bad['rank_states'][0]['consumed']-=1
  with self.assertRaises(ValueError):validate_resume_contract(bad,new)
  self.assertFalse(validate_resume_contract(dict(step=200,contract=new),new))

if __name__=='__main__':unittest.main()
