"""Actual tiny model exports verify 50-step retention and failed-save ordering."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
W=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(W),str(W.parents[3])]
import torch
from a1_local.checkpoints import save_weights
from a1_local.config import load_plan, file_sha256
class Processor:
 def save_pretrained(self,p):
  p.mkdir();(p/'processor.json').write_text('{}')
class Checkpoint50(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=W/'checks');self.root=Path(self.temp.name);self.output=self.root/'stage_II';self.output.mkdir()
  self.teacher=self.root/'stage_I/checkpoints/step_001000/model.pt';self.teacher.parent.mkdir(parents=True);self.teacher.write_bytes(b'preserved teacher')
  self.policy=torch.nn.Linear(2,2);self.policy.processor=Processor();self.policy.mode='student';self.policy.cfg={'fixture':True};self.policy.statistics={}
  config=SimpleNamespace(to_dict=lambda:{})
  self.policy.qwen=SimpleNamespace(config=config);self.policy.dino=SimpleNamespace(config=config)
  self.spec=load_plan()[0]['stage_II']
  self.assertEqual((self.spec['save_every'],self.spec['keep']),(50,1))
 def tearDown(self):self.temp.cleanup()
 def save(self,step):return save_weights(self.policy,self.output,'II',step,self.spec,{})
 def test_new_checkpoint_commits_before_previous_is_removed(self):
  original=torch.save
  for step in [50,100,150]:
   previous=self.output/'checkpoints'/f'step_{step-50:06d}'
   def during_save(*args,**kwargs):
    if step>50:self.assertTrue((previous/'checkpoint_complete.json').exists())
    self.assertFalse((self.output/'checkpoints'/f'step_{step:06d}').exists())
    return original(*args,**kwargs)
   with patch('torch.save',side_effect=during_save):entry=self.save(step)
   self.assertEqual([p.name for p in (self.output/'checkpoints').iterdir()],[f'step_{step:06d}'])
   self.assertEqual(file_sha256(Path(entry['path'])/'model.pt'),entry['model_sha256'])
   self.assertEqual(json.loads((self.output/'latest.json').read_text())['step'],step)
   self.assertEqual(json.loads((self.output/'retention.json').read_text())['steps'],[step])
   self.assertEqual(self.teacher.read_bytes(),b'preserved teacher')
 def test_failed_new_save_preserves_previous(self):
  previous=self.save(50)
  def fail(state,path):Path(path).write_bytes(b'partial');raise OSError('simulated interrupted write')
  with patch('torch.save',side_effect=fail):
   with self.assertRaisesRegex(OSError,'interrupted write'):self.save(100)
  self.assertTrue((Path(previous['path'])/'model.pt').exists())
  self.assertEqual(json.loads((self.output/'latest.json').read_text())['step'],50)
  self.assertFalse((self.output/'checkpoints/step_000100').exists())
 def test_final_step_is_kept_and_off_schedule_is_rejected(self):
  with self.assertRaises(ValueError):self.save(49)
  self.save(1950);self.save(2000)
  self.assertEqual([p.name for p in (self.output/'checkpoints').iterdir()],['step_002000'])
  self.assertEqual(json.loads((self.output/'final.json').read_text())['step'],2000)
if __name__=='__main__':unittest.main(verbosity=2)
