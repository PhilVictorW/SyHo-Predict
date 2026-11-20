"""CPU-only tests for the external evaluation trigger and result completeness."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK))
import watch_full_eval as watch


class AutoEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=WORK/'checks')
        self.run=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def put(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    def completed_run(self):
        for stage,step,mode in [('I',1000,'posterior'),('II',2000,'student')]:
            folder=self.run/('stage_'+stage)
            ckpt=folder/'checkpoints'/f'step_{step:06d}'
            ckpt.mkdir(parents=True)
            (ckpt/'model.pt').write_bytes(('fake_cpu_fixture_'+stage).encode())
            marker=dict(owner=watch.OWNER,stage=stage,step=step,mode=mode,model_sha256=watch.file_sha256(ckpt/'model.pt'))
            self.put(ckpt/'checkpoint_complete.json',marker)
            self.put(folder/'final.json',dict(marker,path=str(ckpt)))
            self.put(folder/'status.json',dict(phase='completed',step=step))
        self.student=self.run/'stage_II/checkpoints/step_002000'
        self.put(self.run/'pipeline.json',dict(phase='completed',student_checkpoint=str(self.student)))
    def test_training_must_fully_exit_before_inference(self):
        self.assertIsNone(watch.ready_checkpoint(self.run))
        self.completed_run()
        self.put(self.run/'pipeline.json',dict(phase='training',student_checkpoint=str(self.student)))
        self.assertIsNone(watch.ready_checkpoint(self.run))
    def test_only_complete_step2000_can_trigger(self):
        self.completed_run()
        self.assertEqual(watch.ready_checkpoint(self.run),self.student)
        self.put(self.run/'stage_II/status.json',dict(phase='completed',step=1500))
        with self.assertRaises(ValueError):watch.ready_checkpoint(self.run)
        self.put(self.run/'stage_II/status.json',dict(phase='completed',step=2000))
        (self.student/'checkpoint_complete.json').unlink()
        with self.assertRaises(ValueError):watch.ready_checkpoint(self.run)
    def test_metadata_provenance_and_corruption(self):
        self.completed_run()
        cfg=watch.read(WORK/'config.json')
        teacher=self.run/'stage_I/checkpoints/step_001000'
        meta=dict(format=watch.FORMAT,mode='student',step=2000,config=cfg,
                  extra=dict(teacher_source=dict(path=str(teacher),step=1000,
                      model_sha256=watch.read(teacher/'checkpoint_complete.json')['model_sha256'])))
        with patch.object(watch,'model_metadata',return_value=meta):
            expected=watch.file_sha256(self.student/'model.pt')
            self.assertEqual(watch.verify_checkpoint(self.run,self.student,cfg),expected)
            meta['mode']='posterior'
            with self.assertRaises(ValueError):watch.verify_checkpoint(self.run,self.student,cfg)
            meta['mode']='student'
            (self.student/'model.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):watch.verify_checkpoint(self.run,self.student,cfg)
    def test_all_40_tasks_require_50_episodes(self):
        output=self.run/'evaluation'
        tasks=[dict(suite=s,task_id=t,complete=True,episodes=50) for s in watch.SUITES for t in range(10)]
        summary=dict(status='complete',episodes_completed=2000,expected_episodes=2000,checkpoint_sha256='sha',tasks=tasks)
        self.put(output/'summary.json',summary)
        self.assertTrue(watch.complete(output,'sha'))
        self.assertFalse(watch.complete(output,'wrong_sha'))
        tasks[0]['episodes']=49
        self.put(output/'summary.json',summary)
        self.assertFalse(watch.complete(output,'sha'))
        tasks[0]=dict(tasks[1])
        self.put(output/'summary.json',summary)
        self.assertFalse(watch.complete(output,'sha'))


if __name__=='__main__':unittest.main(verbosity=2)
