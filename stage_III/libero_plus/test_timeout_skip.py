"""Regression tests for skip/continue, policy exclusion and resumable counts."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from stage_III.libero_plus import train, full_train
from stage_III.libero_plus.gate_actor import GateActor
from stage_III.libero_plus.group_policy import load_skip, record_skip
from stage_III.libero_plus.runtime_safety import file_sha256, wait_with_progress


class TimeoutSkipTests(unittest.TestCase):
    def test_partial_timeout_skips_then_next_group_completes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'trajectories').mkdir()
            actor = root/'actor.pt'; actor.write_bytes(b'actor-v16')
            task = dict(suite='libero_10', task_id=3, name='fixture')
            task_path=root/'task.json'; task_path.write_text(json.dumps(task))
            def row(g, i):
                return dict(group_id=g, trajectory_idx=i, suite=task['suite'], task_id=3,
                            task_name=task['name'], initial_state_idx=0, decisions=[{}],
                            base_student_checkpoint=str(train.STUDENT.resolve()),
                            actor_checkpoint=str(actor.resolve()), actor_sha256=file_sha256(actor),
                            success=True, reward=1.05)
            (root/'trajectories/group_0000_traj_00.json').write_text(json.dumps(row(0,0)))
            with patch.object(train.subprocess, 'Popen'), patch.object(train, 'wait_with_progress', side_effect=TimeoutError('forced stall')) as wait:
                skipped=train.collect_group(root,task_path,0,4,actor,'http://unused',0,timeout_policy='skip')
            self.assertEqual(wait.call_count,1)
            self.assertEqual(skipped['status'],'skipped_timeout')
            self.assertEqual(skipped['trajectories_used'],0)
            self.assertEqual(skipped['partial_trajectories'],1)
            self.assertTrue((root/'skipped_trajectories/group_0000/group_0000_traj_00.json').exists())
            self.assertFalse((root/'trajectories/group_0000_traj_00.json').exists())
            with patch.object(train.subprocess, 'Popen') as popen:
                self.assertEqual(train.collect_group(root,task_path,0,4,actor,'http://unused',0,timeout_policy='skip'),skipped)
                popen.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
                load_skip(root,0,task,'other-policy',4)
            def finish(*a, **kw):
                for i in range(4):
                    (root/f'trajectories/group_0001_traj_{i:02d}.json').write_text(json.dumps(row(1,i)))
                return 0
            with patch.object(train.subprocess,'Popen'), patch.object(train,'wait_with_progress',side_effect=finish):
                valid=train.collect_group(root,task_path,1,4,actor,'http://unused',0,timeout_policy='skip')
            with patch.object(full_train,'train_iteration', return_value={'groups':1,'trajectories':4}) as update:
                metrics=full_train.update_collected_groups(None,None,[skipped,valid],{})
                self.assertEqual(update.call_args.args[2],[valid])
                self.assertEqual(metrics['groups_skipped'],1)
            with patch.object(full_train,'train_iteration') as update:
                metrics=full_train.update_collected_groups(None,None,[skipped],{})
                update.assert_not_called()
                self.assertFalse(metrics['optimizer_step'])
                self.assertEqual(metrics['trajectories'],0)

    def test_import_budget_and_hard_limit(self):
        with tempfile.TemporaryDirectory() as d:
            progress=Path(d)/'progress.json'
            progress.write_text(json.dumps({'phase':'导入仿真环境'}))
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.3)'])
            self.assertEqual(wait_with_progress(proc, progress, stall_timeout=.05,
                             import_timeout=1.0, hard_timeout=3, poll_seconds=.02),0)
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'])
            with self.assertRaises(TimeoutError):
                wait_with_progress(proc, progress, stall_timeout=.05,
                                   import_timeout=10, hard_timeout=.1, poll_seconds=.02)
            self.assertIsNotNone(proc.poll())

    def test_full_coordinator_continues_and_resumes_with_skip_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); run=root/'run'; initial=root/'initial.pt'
            actor=GateActor(); optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
            train.save_actor(initial,actor,optimizer,0,{'learning_rate':1e-4},{})
            tasks=[dict(suite='fixture',task_id=i,name=f'task{i}') for i in range(2)]
            def collect(run_dir, task_path, group, n, actor_path, *a, **kw):
                self.assertEqual(kw['timeout_policy'],'skip')
                if group==0:
                    return record_skip(run_dir,group,tasks[group],file_sha256(actor_path),n,'forced integration timeout')
                return [dict(suite='fixture',task_id=group,success=True,reward=1.05,decisions=[{}]) for _ in range(n)]
            argv=['full_train','--run',str(run),'--limit-groups','2','--groups-per-update','1',
                  '--gpus','0','--ports','29700','--initial-actor',str(initial)]
            with patch.object(full_train,'STAGE3',root), patch.object(full_train,'ensure_manifest',return_value=tasks), \
                 patch.object(full_train,'verify_baseline'), patch.object(full_train,'prepare_student_cache',return_value=root/'cache'), patch.object(full_train,'start_server',return_value=('fake','http://unused',None)), \
                 patch.object(full_train,'stop_server'), patch.object(full_train,'hot_reload'), \
                 patch.object(full_train,'collect_group',side_effect=collect) as collector, \
                 patch.object(full_train,'train_iteration',return_value={'groups':1,'trajectories':4,'updated':True}) as update:
                with patch.object(sys,'argv',argv): full_train.main()
                state=json.loads((run/'status.json').read_text())
                self.assertEqual(state['phase'],'completed_with_skips')
                self.assertEqual((state['groups_processed'],state['groups_completed'],state['groups_skipped'],state['trajectories_completed']),(2,1,1,4))
                self.assertEqual(state['optimizer_updates'],1)
                self.assertEqual(update.call_count,1)
                self.assertEqual(collector.call_count,2)
                checkpoint=torch.load(run/'checkpoints/actor_current.pt',weights_only=False)
                self.assertEqual(checkpoint['iteration'],2)
                migrated=list(argv)
                migrated[migrated.index('--gpus')+1]='2'
                migrated[migrated.index('--ports')+1]='29701'
                with patch.object(sys,'argv',migrated+['--resume']): full_train.main()
                migrated_config=json.loads((run/'config.json').read_text())
                self.assertEqual(migrated_config['gpus'],[2])
                history=json.loads((run/'resource_history.jsonl').read_text().splitlines()[-1])
                self.assertEqual(history['old_gpus'],[0])
                self.assertEqual(history['new_gpus'],[2])
                self.assertEqual(collector.call_count,2)
                self.assertEqual(update.call_count,1)
                self.assertEqual(json.loads((run/'status.json').read_text())['groups_completed'],1)


if __name__=='__main__': unittest.main()
