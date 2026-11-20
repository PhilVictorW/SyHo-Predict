"""Regression checks for simulator stalls and resume cache safety."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from stage_III.libero_plus.runtime_safety import (
    cached_trajectory, file_sha256, patch_reset, wait_with_progress,
)


class RandomizationError(Exception):
    pass


class FakeWrapper:
    def __init__(self, error=None):
        self.error, self.calls = error, 0
        self.env = self
    def reset(self):
        success = False
        while not success:
            try:
                obs = self.inner()
                success = True
            except RandomizationError:
                pass
            finally:
                continue
        return obs
    def inner(self):
        self.calls += 1
        if self.error:
            raise self.error
        return {'image': 'unchanged'}


class SafetyTests(unittest.TestCase):
    def test_reset_success_and_exception_propagation(self):
        patch_reset(FakeWrapper, max_attempts=3)
        self.assertEqual(FakeWrapper().reset(), {'image': 'unchanged'})
        with self.assertRaisesRegex(ValueError, 'bad state'):
            FakeWrapper(ValueError('bad state')).reset()
        obj = FakeWrapper(RandomizationError())
        with self.assertRaisesRegex(RuntimeError, 'retries exhausted'):
            obj.reset()
        self.assertEqual(obj.calls, 3)

    def test_hung_child_is_terminated(self):
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
            begin = time.monotonic()
            with self.assertRaises(TimeoutError):
                wait_with_progress(p, Path(d)/'missing', stall_timeout=0.15,
                                   hard_timeout=2, poll_seconds=0.05)
            self.assertIsNotNone(p.poll())
            self.assertLess(time.monotonic()-begin, 3)

    def test_active_child_not_timed_out(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'heartbeat'
            code = "import time; from pathlib import Path; p=Path("+repr(str(path))+");\nfor _ in range(10): p.write_text('alive'); time.sleep(0.08)"
            p = subprocess.Popen([sys.executable, '-c', code])
            self.assertEqual(wait_with_progress(p, path, stall_timeout=0.4,
                                               hard_timeout=3, poll_seconds=0.05), 0)

    def test_cache_rejects_other_policy_and_task(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            actor, student, output = root/'actor', root/'student', root/'traj.json'
            actor.write_text('actor_v7')
            task = dict(suite='libero_10',task_id=1273,name='task')
            row = dict(group_id=234,trajectory_idx=0,suite=task['suite'],task_id=1273,
                       task_name='task',initial_state_idx=0,decisions=[{}],
                       base_student_checkpoint=str(student.resolve()),
                       actor_checkpoint=str(actor.resolve()),actor_sha256=file_sha256(actor))
            kw=dict(group=234,trajectory=0,task=task,student=student,actor=actor,actor_hash=file_sha256(actor))
            output.write_text(json.dumps(row))
            self.assertEqual(cached_trajectory(output,**kw),row)
            row['actor_sha256']='wrong'
            output.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError,'different actor'):
                cached_trajectory(output,**kw)
            row['actor_sha256']=file_sha256(actor)
            row['task_id']=999
            output.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
                cached_trajectory(output,**kw)


if __name__ == '__main__':
    unittest.main()
