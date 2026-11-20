"""Exercise actual student restore, sample offsets, lineage checks and imported retention."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
W=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(W),str(W.parents[3])]
import torch
from a1_local.config import load_plan, learning_rate
from a1_local.sampling import RankSampler
from a1_local.checkpoints import save_weights, verify_student, seed_student_checkpoint, load_student_weights

class Processor:
    def save_pretrained(self,p):
        p.mkdir(); (p/'processor.json').write_text('{}')

class Resume(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=W/'checks');self.root=Path(self.temp.name)
        self.output=self.root/'old/stage_II';self.output.mkdir(parents=True)
        self.plan,self.cfg=load_plan();self.spec=self.plan['stage_II']
        self.policy=torch.nn.Linear(2,2)
        self.policy.cfg=self.cfg;self.policy.mode='student';self.policy.statistics={}
        self.policy.processor=Processor()
        config=SimpleNamespace(to_dict=lambda:{})
        self.policy.qwen=config;self.policy.dino=config
        self.policy.qwen=SimpleNamespace(config=config);self.policy.dino=SimpleNamespace(config=config)
        self.teacher=dict(step=1000,model_sha256='teacher_digest',path='/old/teacher')
        self.entry=save_weights(self.policy,self.output,'II',450,self.spec,dict(teacher_source=self.teacher))
        self.ckpt=Path(self.entry['path'])
    def tearDown(self): self.temp.cleanup()
    def test_actual_model_load_overwrites_parameters_without_aliasing(self):
        source=verify_student(self.ckpt,self.cfg,dict(self.teacher,path='/new/teacher'),self.spec)
        restored=torch.nn.Linear(2,2);restored.cfg=self.cfg
        self.assertEqual(load_student_weights(restored,source),450)
        for a,b in zip(self.policy.parameters(),restored.parameters()):
            self.assertTrue(torch.equal(a,b));self.assertNotEqual(a.data_ptr(),b.data_ptr());self.assertTrue(b.requires_grad)
        self.assertFalse(source['optimizer_restored'])
    def test_corrupt_student_is_rejected(self):
        with (self.ckpt/'model.pt').open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):verify_student(self.ckpt,self.cfg,self.teacher,self.spec)
    def test_wrong_teacher_or_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'provenance'):
            verify_student(self.ckpt,self.cfg,dict(self.teacher,model_sha256='wrong'),self.spec)
        with self.assertRaisesRegex(ValueError,'configuration'):
            verify_student(self.ckpt,dict(self.cfg,seed=-1),self.teacher,self.spec)
    def test_resume_checkpoint_is_pruned_only_after_step500(self):
        source=verify_student(self.ckpt,self.cfg,self.teacher,self.spec)
        out=self.root/'new/stage_II'
        imported=seed_student_checkpoint(source,out)
        self.assertEqual((Path(imported['path'])/'model.pt').stat().st_ino,(self.ckpt/'model.pt').stat().st_ino)
        self.assertNotEqual((Path(imported['path'])/'checkpoint_complete.json').stat().st_ino,(self.ckpt/'checkpoint_complete.json').stat().st_ino)
        save_weights(self.policy,out,'II',500,self.spec,dict(teacher_source=self.teacher,resume_source=imported))
        self.assertEqual([p.name for p in (out/'checkpoints').iterdir()],['step_000500'])
        self.assertTrue((self.ckpt/'model.pt').exists())
    def test_sampler_and_schedule_continue_at451(self):
        for rank,share in enumerate([11,11,10]):
            full=list(RankSampler(65000,2000,rank,2026,world=3))
            resumed=RankSampler(65000,2000,rank,2026,world=3,start_step=450)
            self.assertEqual(list(resumed),full[450*share:])
            self.assertEqual(len(resumed),1550*share)
        self.assertGreater(learning_rate(450,self.spec,1),.9)
        with self.assertRaises(ValueError):RankSampler(65000,2000,0,2026,world=3,start_step=2001)
if __name__=='__main__':unittest.main(verbosity=2)
