"""Regression tests for physical action semantics and checkpoint isolation."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from goal_local import eval_client
import student_action_protocol as adapter
import student_eval_common as common


class ActionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.checkpoint=self.root/'step_000500';self.checkpoint.mkdir()
        (self.checkpoint/'checkpoint_complete.json').write_text(json.dumps(dict(step=500,stage='II',mode='student')))
        self.root_patch=patch.object(adapter,'TRAIN',self.root);self.root_patch.start()
        adapter.ensure_checkpoint_protocol(self.checkpoint)
        self.actions=np.arange(56,dtype=np.float32).reshape(8,7)/100
        self.actions[:,6]=[0,1,0,1,1,0,1,0]
        self.example=dict(primary_image=[np.zeros((256,256,3),dtype=np.uint8)],
            wrist_image=[np.zeros((256,256,3),dtype=np.uint8)],lang='pick up the bowl')

    def tearDown(self):
        self.root_patch.stop();self.temp.cleanup()

    def test_physical_gripper_and_motion_through_real_core(self):
        path=Path(__file__).resolve().parents[2]/'examples/LIBERO/eval_files/libero_eval_core.py'
        nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_binarize_gripper_open','invert_gripper_action')]
        ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        client=adapter.PlusStudentClient(self.checkpoint,'unused',action_protocol=common.ACTION_PROTOCOL)
        with patch.object(eval_client,'rpc',return_value=dict(normalized=False,actions=self.actions.tolist())) as rpc:
            for i in range(16):
                raw=client.step(self.example)['raw_action'];expected=self.actions[i%8]
                np.testing.assert_array_equal(raw['world_vector'],expected[:3])
                np.testing.assert_array_equal(raw['rotation_delta'],expected[3:6])
                physical=ns['invert_gripper_action'](ns['_binarize_gripper_open'](raw['open_gripper']))[0]
                self.assertEqual(physical,-1 if expected[6]==1 else 1)
                np.testing.assert_array_equal(client.actions,self.actions)
            self.assertEqual(rpc.call_count,2)

    def test_legacy_client_is_unchanged(self):
        legacy=eval_client.StudentClient(self.checkpoint,'unused')
        with patch.object(eval_client,'rpc',return_value=dict(normalized=False,actions=self.actions.tolist())):
            for row in self.actions:
                raw=legacy.step(self.example)['raw_action']
                self.assertEqual(raw['open_gripper'][0],row[6])

    def test_rejects_non_plus_checkpoint(self):
        with self.assertRaises(ValueError):
            adapter.PlusStudentClient(self.root.parent/'original_libero','unused',action_protocol=common.ACTION_PROTOCOL)

    def test_rejects_missing_or_conflicting_semantics(self):
        with self.assertRaises(ValueError):
            adapter.PlusStudentClient(self.checkpoint,'unused',action_protocol={})
        (self.checkpoint/'action_semantics.json').write_text('{}')
        with self.assertRaises(ValueError):adapter.ensure_checkpoint_protocol(self.checkpoint)

    def test_collector_rejects_old_episode_semantics(self):
        folder=self.root/'evaluation';folder.mkdir();(folder/'episodes').mkdir()
        task=dict(suite='libero_spatial',task_id=0,category='Camera')
        protocol=dict(step=500,smoke=False,tasks=[task],checkpoint=str(self.checkpoint),action_protocol=common.ACTION_PROTOCOL)
        (folder/'protocol.json').write_text(json.dumps(protocol))
        row=dict(task,episode_idx=0,checkpoint=str(self.checkpoint),success=True)
        (folder/'episodes/old.json').write_text(json.dumps(row))
        with self.assertRaises(ValueError):common.collect(folder)


if __name__=='__main__':unittest.main(verbosity=2)
