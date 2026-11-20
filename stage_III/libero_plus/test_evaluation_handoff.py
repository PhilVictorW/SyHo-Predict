"""Meaningful completion-gate, queue recovery and metric denominator checks."""
import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from .eval_common import (initialize_queue, connect, claim, finish, requeue_errors,
                          result_summary, training_complete)


class HandoffTests(unittest.TestCase):
    def test_requires_committed_final_pass(self):
        done=dict(phase='completed_with_skips',groups_processed=8380,batch=262,
                  groups_completed=8362,groups_skipped=18)
        self.assertTrue(training_complete(done))
        for key,val in [('phase','collecting'),('groups_processed',8379),('batch',261),('groups_completed',8361)]:
            self.assertFalse(training_complete(done|{key:val}))

    def test_concurrent_claim_retry_and_result_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            tasks=[dict(suite='libero_spatial',category='Camera',task_id=i,name=f'task{i}') for i in range(32)]
            initialize_queue(folder,tasks)
            def worker(slot):
                db=connect(folder);out=[]
                while (job:=claim(db,slot)) is not None:
                    state='timeout' if job['id']==0 else 'done'
                    finish(db,job,slot,state,dict(success=(job['id']%2==0) if state=='done' else None))
                    out.append(job['id'])
                db.close();return out
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                ids=sum(list(pool.map(worker,range(8))),[])
            self.assertEqual(sorted(ids),list(range(32)))
            protocol=dict(tasks=tasks,base_student='frozen')
            result=result_summary(folder,protocol)
            self.assertEqual(result['processed'],32);self.assertEqual(result['done'],31)
            self.assertEqual(result['timeout'],1);self.assertEqual(result['successes'],15)
            self.assertAlmostEqual(result['success_rate'],15/31)
            self.assertAlmostEqual(result['success_rate_full_denominator'],15/32)
            self.assertFalse(result['complete'])
            self.assertEqual(requeue_errors(folder),1)
            db=connect(folder);job=claim(db,0)
            self.assertEqual(job['attempts'],2)
            finish(db,job,0,'done',dict(success=True));db.close()
            result=result_summary(folder,protocol)
            self.assertTrue(result['complete']);self.assertEqual(result['successes'],16)
            self.assertEqual(len(result['cells']),4)
            self.assertTrue(all(len(v)==7 for v in result['cells'].values()))
            initialize_queue(folder,tasks)
            db=connect(folder);self.assertIsNone(claim(db,0));db.close()
            with self.assertRaises(ValueError):initialize_queue(folder,[dict(t, name='wrong') for t in tasks])

    def test_retry_limit_and_stale_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);tasks=[dict(suite='libero_goal',category='Noise',task_id=1,name='task')]
            initialize_queue(folder,tasks);db=connect(folder)
            for attempt in range(1,4):
                job=claim(db,0);self.assertEqual(job['attempts'],attempt)
                finish(db,job,0,'timeout',dict(success=None))
                with self.assertRaises(RuntimeError):finish(db,job,0,'done',dict(success=True))
                self.assertEqual(requeue_errors(folder),int(attempt<3))
            self.assertIsNone(claim(db,0));db.close()

    def test_mean_gate_is_input_dependent_and_does_not_sample(self):
        import torch
        from .eval_server import MeanGateActor
        actor=MeanGateActor()
        with torch.no_grad():actor.head.weight.normal_(0,.01)
        x=torch.randn(2,6144)
        before=torch.get_rng_state().clone();a=actor.collect(x,seed=1);b=actor.collect(x,seed=999)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(a['eta'],b['eta']))
        self.assertTrue(torch.allclose(a['eta'],actor.deterministic_eta(x)))
        self.assertFalse(torch.allclose(a['eta'][0],a['eta'][1]))
        self.assertIsNone(a['old_logp'])


if __name__=='__main__':unittest.main()
