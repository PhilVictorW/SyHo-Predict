"""Offline checks for identity, timeout and incomplete-result accounting."""
import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import eval_common as common
import bench_bridge as bridge

class Checks(unittest.TestCase):
    def test_exact_40_minute_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(common.timed_out(d,100,2499))
            self.assertTrue(common.timed_out(d,100,2500))
    def test_log_and_heartbeat_cannot_hide_stall(self):
        with tempfile.TemporaryDirectory() as d:
            common.write(Path(d)/'activity.json',dict(last_progress_at=100,updated_at=99999))
            (Path(d)/'bridge.log').write_text('heartbeat')
            self.assertTrue(common.timed_out(d,100,2501))
    def test_executed_progress_renews_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            common.write(Path(d)/'activity.json',dict(last_progress_at=2400,updated_at=2400,env_steps=72))
            self.assertFalse(common.timed_out(d,100,2501))
    def test_partial_results_and_missing_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            key='task__demo_clean';job=Path(d)/'jobs'/key
            common.write(job/'a/b/tasks/task/episodes.partial.json',
                         dict(episodes=[dict(episode_id=0,success=True),dict(episode_id=1,success=False)]))
            common.write(job/'result.json',dict(status='skipped_timeout'))
            result=common.aggregate(d,[dict(key=key,task='task',mode='demo_clean')])
            self.assertEqual(result['episodes'],2)
            self.assertEqual(result['successes'],1)
            self.assertEqual(result['skipped_cells'],1)
            self.assertEqual(result['completed_cells'],0)
            self.assertEqual(result['success_rate'],0.5)
    def test_repeated_action_and_observation_messages_do_not_renew(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',{'ROBOTWIN_EVAL_ROOT':d}):
            with patch.object(bridge.time,'time',return_value=100):
                conn=bridge.ActivityConnection(Mock())
                conn.send(dict(type='action_request',episode_id=0,seed=1,step=0))
            with patch.object(bridge.time,'time',return_value=200):
                conn.send(dict(type='observation_ready',episode_id=0,seed=1,step=0))
                conn.send(dict(type='action_request',episode_id=0,seed=1,step=0))
            self.assertEqual(conn.progress['last_progress_at'],100)
            with patch.object(bridge.time,'time',return_value=300):
                conn.send(dict(type='action_request',episode_id=0,seed=1,step=1))
            self.assertEqual(conn.progress['last_progress_at'],300)
    def test_identity_mismatch_rejected(self):
        transport=bridge.ActorTransport.__new__(bridge.ActorTransport)
        transport.expected=dict(student_sha256='s',actor_sha256='a')
        with self.assertRaises(ValueError):
            transport.check(dict(student_sha256='s',actor_sha256='wrong'))
    def test_mean_actor_rpc_preserves_actions_and_noise(self):
        transport=bridge.ActorTransport.__new__(bridge.ActorTransport)
        transport.url='http://127.0.0.1:1'
        transport.expected=dict(student_sha256='s',actor_sha256='a')
        actions=np.arange(36*16,dtype=np.float32).reshape(36,16)/1000
        frame=np.zeros((256,256,3),dtype=np.uint8)
        def fake_call(url,path,req,timeout):
            self.assertEqual(req['policy'],'actor_mean')
            self.assertEqual(req['noise_seed'],123)
            self.assertEqual(timeout,2400)
            self.assertEqual(bridge.array_decode(req['main']).shape,(256,256,3))
            return dict(student_sha256='s',actor_sha256='a',
                        actions=bridge.array_encode(actions),gate={'eta':1.01})
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',{'ROBOTWIN_EVAL_ROOT':d}), patch.object(bridge,'call',fake_call):
            got=transport.predict_action({'examples':[dict(primary_image=[frame],wrist_image=[frame,frame],
                                                        lang='test',_noise_seed=123)]})
        np.testing.assert_array_equal(got['data']['normalized_actions'][0],actions)

if __name__=='__main__':unittest.main(verbosity=2)
