"""Reject incomplete or stale training state before a Stage-II-only launch."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
W=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(W),str(W.parents[3])]
from a1_local.launcher import validate_start
from a1_local.config import OWNER
class StageIIStart(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=W/'checks');self.root=Path(self.temp.name)
  self.folder=self.root/'stage_I';self.ckpt=self.folder/'checkpoints/step_001000';self.ckpt.mkdir(parents=True)
  self.marker=dict(owner=OWNER,stage='I',step=1000,mode='posterior',model_sha256='expected')
  self.put(self.ckpt/'checkpoint_complete.json',self.marker)
  self.put(self.folder/'final.json',dict(self.marker,path=str(self.ckpt)))
  self.put(self.folder/'status.json',dict(phase='completed',step=1000))
 def tearDown(self):self.temp.cleanup()
 def put(self,p,d):p.write_text(json.dumps(d))
 def test_valid_teacher_is_verified(self):
  with patch('a1_local.launcher.verify_teacher',return_value={'verified':True}) as verify:
   self.assertEqual(validate_start(self.root,{},'II'),{'verified':True});verify.assert_called_once_with(self.ckpt,{})
 def test_incomplete_teacher_and_foreign_final(self):
  self.put(self.folder/'status.json',dict(phase='saving',step=1000))
  with self.assertRaises(ValueError):validate_start(self.root,{},'II')
  self.put(self.folder/'status.json',dict(phase='completed',step=1000))
  self.put(self.folder/'final.json',dict(self.marker,path='/tmp/foreign_teacher'))
  with self.assertRaises(ValueError):validate_start(self.root,{},'II')
 def test_teacher_corruption_is_not_ignored(self):
  with patch('a1_local.launcher.verify_teacher',side_effect=ValueError('checksum mismatch')):
   with self.assertRaisesRegex(ValueError,'checksum mismatch'):validate_start(self.root,{},'II')
 def test_existing_student_or_pipeline_is_never_overwritten(self):
  (self.root/'stage_II').mkdir()
  with self.assertRaises(FileExistsError):validate_start(self.root,{},'II')
  (self.root/'stage_II').rmdir();self.put(self.root/'pipeline.json',{'phase':'training'})
  with self.assertRaises(FileExistsError):validate_start(self.root,{},'II')
 def test_default_stage_I_preserves_existing_teacher(self):
  with self.assertRaises(FileExistsError):validate_start(self.root,{},'I')
if __name__=='__main__':unittest.main(verbosity=2)
