"""Cache integrity and nonfatal service-startup timeout regressions."""
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from stage_III.libero_plus import full_train
from stage_III.libero_plus.checkpoint_cache import prepare_student_cache, validate_cache, digest


class CacheStartupTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source'; source.mkdir(); (source/'model.pt').write_bytes(b'exact immutable student bytes')
        (source/'processor').mkdir(); (source/'processor/config.json').write_text('{"size":256}')
        return source, digest(source/'model.pt')

    def test_cache_reuses_verified_copy_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source, expected=self.fixture(root)
            original=(source/'model.pt').read_bytes()
            cache=prepare_student_cache(source,expected,root/'cache')
            self.assertEqual((cache/'model.pt').read_bytes(),original)
            self.assertEqual(prepare_student_cache(source,expected,root/'cache'),cache)
            (cache/'model.pt').write_bytes(b'bad cache')
            with self.assertRaises(RuntimeError):validate_cache(cache,source,expected)
            rebuilt=prepare_student_cache(source,expected,root/'cache')
            self.assertNotEqual(rebuilt,cache)
            self.assertEqual((rebuilt/'model.pt').read_bytes(),original)
            self.assertEqual((source/'model.pt').read_bytes(),original)
            (source/'processor/config.json').write_text('{"size":128}')
            with self.assertRaises(RuntimeError):validate_cache(rebuilt,source,expected)

    def test_wrong_source_hash_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source,_=self.fixture(root)
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                prepare_student_cache(source,'0'*64,root/'cache')
            self.assertFalse((root/'cache/latest.json').exists())

    def test_startup_request_timeout_and_long_wait_do_not_terminate_server(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); actor=root/'actor.pt';actor.write_text('fixture')
            proc=Mock();proc.poll.return_value=None
            ready={'checkpoint':str(full_train.STUDENT),'actor_checkpoint':str(actor.resolve()),'load_state':'ready'}
            replies=[urllib.error.URLError('temporary timeout'),{'load_state':'idle'},
                     {'load_state':'loading','load_phase':'slow disk'},ready]
            with patch.object(full_train.subprocess,'Popen',return_value=proc), \
                 patch.object(full_train,'rpc',side_effect=replies) as rpc, \
                 patch.object(full_train.time,'sleep'), \
                 patch.object(full_train.time,'monotonic',side_effect=[0,4000,4001]):
                handle=full_train.start_server(root,actor,1,29700,0)
            self.assertIs(handle[0],proc)
            proc.terminate.assert_not_called()
            self.assertEqual([call.args[1] for call in rpc.call_args_list],['/health','/health','/start_load','/health'])
            handle[2].close()

    def test_async_load_returns_and_does_not_duplicate_requests(self):
        from stage_III.libero_plus.server import Engine
        with tempfile.TemporaryDirectory() as d:
            engine=Engine('cpu',str(Path(d)/'ready.json'))
            entered=threading.Event(); release=threading.Event(); done=threading.Event()
            def load(checkpoint,actor):
                entered.set(); release.wait(3)
                engine.checkpoint=checkpoint;engine.actor_checkpoint=actor
                engine.load_state='ready';done.set()
            with patch.object(engine,'load',side_effect=load) as loader:
                first=engine.start_load('student','actor')
                self.assertEqual(first['load_state'],'loading')
                self.assertTrue(entered.wait(1))
                self.assertEqual(engine.start_load('student','actor')['load_state'],'loading')
                self.assertEqual(loader.call_count,1)
                release.set(); self.assertTrue(done.wait(2))
                self.assertEqual(engine.info()['load_state'],'ready')


if __name__=='__main__':unittest.main()
