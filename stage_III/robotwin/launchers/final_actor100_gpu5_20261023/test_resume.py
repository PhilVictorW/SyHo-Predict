import json
import tempfile
import unittest
from pathlib import Path
from resume_state import load_resume, durable_write

class ResumeChecks(unittest.TestCase):
    def test_keep_outcomes_skip_invalid_and_retry_unfinished_seed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            records = [dict(episode_id=0, seed=100002, success=False),
                       dict(episode_id=1, seed=100003, success=True)]
            durable_write(p/'episodes.partial.json', dict(episodes=records))
            (p/'environment_events.jsonl').write_text(json.dumps(dict(type='candidate_invalid', seed=100001, error=None))+'\n')
            before = (p/'episodes.partial.json').read_bytes()
            state = load_resume(p, [100002,100001,100003,100000,100004], 100000, 50)
            self.assertEqual(state['records'], records)
            self.assertEqual((state['completed'],state['successes']), (2,1))
            self.assertEqual(state['fixed_seed_index'], 3)
            self.assertEqual(state['next_episode_id'], 2)
            self.assertEqual((p/'episodes.partial.json').read_bytes(), before)
            state['records'].append(dict(episode_id=2,seed=100000,success=True))
            durable_write(p/'episodes.partial.json', dict(episodes=state['records']))
            again = load_resume(p, [100002,100001,100003,100000,100004], 100000, 50)
            self.assertEqual((again['completed'],again['fixed_seed_index']), (3,4))
            self.assertEqual(again['records'][:2], records)
    def test_transient_error_seed_is_retried(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'environment_events.jsonl').write_text(json.dumps(dict(type='candidate_invalid',seed=100000,error='temporary setup error'))+'\n')
            self.assertEqual(load_resume(p,[100000,100001],100000,50)['fixed_seed_index'],0)
    def test_nonfixed_cursor_and_torn_last_event(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            durable_write(p/'episodes.partial.json',dict(episodes=[dict(episode_id=0,seed=100001,success=True)]))
            (p/'environment_events.jsonl').write_text(json.dumps(dict(type='candidate_invalid',seed=100002,error=None))+'\n{"time":')
            self.assertEqual(load_resume(p,[],100000,50)['next_seed'],100003)
    def test_corrupt_outcomes_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episodes.partial.json';p.write_text('{"episodes":')
            with self.assertRaises(json.JSONDecodeError):load_resume(p.parent,[100000],100000,50)
            self.assertEqual(p.read_text(),'{"episodes":')
    def test_duplicate_or_foreign_outcomes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);row=dict(episode_id=0,seed=100001,success=True)
            durable_write(p/'episodes.partial.json',dict(episodes=[row,row]))
            with self.assertRaises(ValueError):load_resume(p,[100000,100001],100000,50)
            durable_write(p/'episodes.partial.json',dict(episodes=[row]))
            with self.assertRaises(ValueError):load_resume(p,[100000],100000,50)


class ResumeIntegration(unittest.TestCase):
    def test_actual_evaluation_loop_keeps_old_rows_and_runs_only_remainder(self):
        import os
        from contextlib import ExitStack
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        import resumable_round10_bridge as bridge
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            p=Path(d)
            rows=[dict(slot_id=0,episode_id=0,seed=100001,success=True,steps=3),
                  dict(slot_id=0,episode_id=1,seed=100002,success=False,steps=4)]
            durable_write(p/'episodes.partial.json',dict(episodes=rows))
            conn=Mock();conn.recv.side_effect=[dict(type='candidate_valid',slot_id=0,seed=100003),
                dict(type='episode_done',slot_id=0,episode_id=2,seed=100003,success=True,steps=5)]
            child=Mock();proc=Mock();proc.is_alive.return_value=False
            ctx=Mock();ctx.Pipe.return_value=(conn,child);ctx.Process.return_value=proc
            model=SimpleNamespace(env_action_type='ee',pair_instructions={},reset=Mock(),close=Mock())
            stack.enter_context(patch.dict(os.environ,ROBOTWIN_FIXED_SEEDS='100001,100002,100003',ROBOTWIN_NUM_SLOTS='1'))
            stack.enter_context(patch.object(bridge,'build_save_dir',return_value=p))
            stack.enter_context(patch.object(bridge,'prepare_task_args',return_value=({},'256x256')))
            stack.enter_context(patch.object(bridge,'get_model',return_value=model))
            stack.enter_context(patch.object(bridge,'_print_task_config'))
            stack.enter_context(patch.object(bridge.mp,'get_context',return_value=ctx))
            stack.enter_context(patch.object(bridge,'mp_wait',return_value=[conn]))
            args=dict(task_name='test',task_config='demo_clean',policy_name='test',ckpt_setting='test',test_num=3,seed=0)
            self.assertEqual(bridge.run_batched_eval(args),0)
            summary=json.loads((p/'summary.json').read_text())
            self.assertEqual(summary['episodes'][:2],rows)
            self.assertEqual((len(summary['episodes']),summary['successes']),(3,2))
            requests=[c.args[0] for c in conn.send.call_args_list if c.args[0].get('cmd')=='try_seed']
            self.assertEqual(requests,[dict(cmd='try_seed',seed=100003)])
            accepted=[c.args[0] for c in conn.send.call_args_list if c.args[0].get('cmd')=='accept_candidate']
            self.assertEqual(accepted,[dict(cmd='accept_candidate',episode_id=2)])

if __name__=='__main__':unittest.main(verbosity=2)
