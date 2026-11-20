"""Small CPU checks for partitioning, synchronized AdamW and checkpoint retention."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from .config import WORK, OWNER, load_plan, learning_rate
from .sampling import RankSampler, backward_weight
from .checkpoints import retain_latest
from .launcher import commands, evaluation_command, evaluation_complete
from ablation.aba1.plus_common.distributed import DistributedMaster


def distributed_worker(rank, rendezvous, world=3):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=world)
    try:
        value = torch.nn.Parameter(torch.tensor([.2, -.1]))
        opt = DistributedMaster([value], .01, .0001, bucket_elements=1)
        opt.broadcast_weights()
        reference = value.detach().clone().requires_grad_(True)
        ref_opt = torch.optim.AdamW([reference], lr=.01, weight_decay=.0001, betas=(.9,.95), eps=1e-8, foreach=False)
        x = torch.arange(64, dtype=torch.float32).reshape(32,2)/64
        y = torch.linspace(-.3, .4, 32)
        for step in range(3):
            opt.zero_grad(); ref_opt.zero_grad()
            for i in range(rank,32,world):
                ((value@x[i]-y[i]).square()*backward_weight(world)).backward()
            opt.step(.01, 1.)
            ((x@reference-y).square().mean()).backward()
            torch.nn.utils.clip_grad_norm_([reference],1.)
            ref_opt.step()
            torch.testing.assert_close(value,reference,atol=1e-7,rtol=1e-6)
    finally:
        dist.destroy_process_group()


class Runtime(unittest.TestCase):
    def test_automatic_pipeline_and_full_evaluation(self):
        plan, _ = load_plan()
        output = WORK/'runs'/'test_only'
        stages = commands(plan, output)
        self.assertEqual(list(stages), ['I', 'II'])
        for command in stages.values():
            self.assertIn(f"--nproc-per-node={plan['world_size']}", command)
            self.assertIn('a2_goal.training', command)
        command = evaluation_command(plan, output)
        self.assertEqual(command[command.index('--protocol')+1], 'full')
        self.assertEqual(command[command.index('--checkpoint')+1], str(output/'stage_II/checkpoints/step_002000'))
        from goal_local.eval_common import SUITES
        with tempfile.TemporaryDirectory(dir=WORK/'checks') as folder:
            path = Path(folder)
            tasks = [dict(suite=suite, task_id=task, complete=True, episodes=50) for suite in SUITES for task in range(10)]
            summary = dict(status='complete', episodes_completed=2000, expected_episodes=2000,
                           checkpoint_sha256='test', tasks=tasks)
            (path/'summary.json').write_text(json.dumps(summary))
            evaluation_complete(path, 'test')
            summary['episodes_completed'] = 1999
            (path/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError): evaluation_complete(path, 'test')

    def test_partition(self):
        ranks = [list(RankSampler(260726,2000,r,2026)) for r in range(3)]
        self.assertEqual([len(x) for x in ranks], [22000,22000,20000])
        self.assertEqual(len(set(sum(ranks,[]))),64000)
        order = torch.randperm(260726,generator=torch.Generator().manual_seed(2026)).tolist()
        for step in [0,1,1999]:
            actual = [ranks[i % 3][step*len(range(i % 3,32,3))+i//3] for i in range(32)]
            self.assertEqual(actual,order[step*32:(step+1)*32])
    def test_retention_and_schedule(self):
        plan,_ = load_plan()
        with tempfile.TemporaryDirectory(dir=WORK/'checks') as folder:
            folder=Path(folder)
            for step in [500,1000,1500,2000]:
                path=folder/f'step_{step:06d}';path.mkdir()
                (path/'checkpoint_complete.json').write_text(json.dumps(dict(owner=OWNER,stage='II',step=step)))
                kept=retain_latest(folder,'II',2)
                self.assertLessEqual(len(kept),2)
            self.assertEqual(kept,[1500,2000])
            self.assertEqual(sorted(p.name for p in folder.iterdir()),['step_001500','step_002000'])
        for stage in ['I','II']:
            spec=plan['stage_'+stage]
            self.assertAlmostEqual(learning_rate(spec['warmup_steps']-1,spec,1.),1.)
            self.assertAlmostEqual(learning_rate(spec['steps']-1,spec,1.),.05)
    def test_four_process_gradient_equivalence(self):
        with tempfile.TemporaryDirectory(dir=WORK/'checks') as folder:
            mp.spawn(distributed_worker,args=(str(Path(folder)/'gloo'),4),nprocs=4,join=True)

    def test_three_process_gradient_equivalence(self):
        with tempfile.TemporaryDirectory(dir=WORK/'checks') as folder:
            mp.spawn(distributed_worker,args=(str(Path(folder)/'gloo'),),nprocs=3,join=True)


if __name__ == '__main__':
    (WORK/'checks').mkdir(exist_ok=True)
    unittest.main()
