"""Verify rolling publication and collective low-space waiting without GPU use."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch.distributed as dist
import torch.multiprocessing as mp
from a2_stage2 import checkpointing as cp


def disk_worker(rank, directory):
    root = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(root / 'gloo'), rank=rank, world_size=2)
    states = []
    free = iter([10, 70, 100])
    try:
        with patch.object(cp.shutil, 'disk_usage', side_effect=lambda path: SimpleNamespace(free=next(free))):
            cp.wait_for_space(root, 100 if rank == 0 else None, rank, dist,
                              lambda **row: states.append(row), 100, poll_seconds=.05)
        (root / f'rank{rank}.json').write_text(json.dumps(states))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class Save100(unittest.TestCase):
    def make_temp(self, root, step):
        path = root / f'.step_{step:06d}.incomplete'
        path.mkdir()
        (path / 'model.pt').write_bytes(f'model {step}'.encode())
        (path / 'trainer.pt').write_bytes(f'optimizer {step}'.encode())
        return path

    def metadata(self, step):
        return dict(format='test', complete=True, resumable=True, step=step,
                    stage='II', config_sha256='same-experiment')

    def validate(self, path, **options):
        self.assertEqual(options, dict(require_resume=True, verify_hash=True))
        marker = json.loads((path / 'checkpoint_complete.json').read_text())
        self.assertEqual(marker['save_every'], 100)
        for name, info in marker['files'].items():
            self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), info['sha256'])

    def test_100_200_300_publish_before_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in [100, 200, 300]:
                temp = self.make_temp(root, step)
                final = root / f'step_{step:06d}'
                previous = root / f'step_{step-100:06d}'
                def validate(path, **options):
                    self.assertFalse(final.exists())
                    if step > 100:
                        self.assertTrue(previous.exists())
                    self.validate(path, **options)
                cp.commit_checkpoint(temp, final, self.metadata(step), ['model.pt', 'trainer.pt'], validate)
                self.assertEqual([p.name for p in root.glob('step_*')], [final.name])
                self.assertEqual(json.loads((root / 'latest.json').read_text())['step'], step)
                retention = json.loads((root / 'recovery_retention.json').read_text())
                self.assertEqual(retention['save_every'], 100)
                self.assertEqual(retention['full_resume'], str(final))

    def test_failed_new_validation_retains_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / 'step_000100'
            cp.commit_checkpoint(self.make_temp(root, 100), first, self.metadata(100),
                                 ['model.pt', 'trainer.pt'], self.validate)
            def fail(*args, **kwargs):
                raise ValueError('checksum mismatch')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                cp.commit_checkpoint(self.make_temp(root, 200), root / 'step_000200',
                                     self.metadata(200), ['model.pt', 'trainer.pt'], fail)
            self.assertTrue((first / 'trainer.pt').exists())
            self.assertEqual(json.loads((root / 'latest.json').read_text())['step'], 100)

    def test_two_ranks_wait_then_resume_when_disk_space_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(disk_worker, args=(directory,), nprocs=2, join=True)
            rows = [json.loads((Path(directory) / f'rank{rank}.json').read_text()) for rank in range(2)]
            self.assertEqual(rows[0], rows[1])
            self.assertEqual([row['phase'] for row in rows[0]],
                             ['waiting_checkpoint_space', 'waiting_checkpoint_space', 'checkpoint'])
            self.assertEqual([row['checkpoint_free_bytes'] for row in rows[0]], [10, 70, 100])

    def test_disk_read_error_is_reported_to_peers(self):
        class Broadcast:
            def broadcast_object_list(self, message, src):
                self.sent = message[0]
        fake = Broadcast()
        with patch.object(cp.shutil, 'disk_usage', side_effect=OSError('read error')):
            with self.assertRaisesRegex(OSError, 'read error'):
                cp.wait_for_space(Path('.'), 100, 0, fake, lambda **row: None, 100)
        self.assertIn('error', fake.sent)


if __name__ == '__main__':
    unittest.main(verbosity=2)
