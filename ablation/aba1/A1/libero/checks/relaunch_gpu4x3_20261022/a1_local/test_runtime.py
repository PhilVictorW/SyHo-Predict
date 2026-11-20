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
from ablation.aba1.plus_common.distributed import DistributedMaster


def distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
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
            for i in range(rank,32,4):
                ((value@x[i]-y[i]).square()*backward_weight()).backward()
            opt.step(.01, 1.)
            ((x@reference-y).square().mean()).backward()
            torch.nn.utils.clip_grad_norm_([reference],1.)
            ref_opt.step()
            torch.testing.assert_close(value,reference,atol=1e-7,rtol=1e-6)
    finally:
        dist.destroy_process_group()


class Runtime(unittest.TestCase):
    def test_partition(self):
        ranks = [list(RankSampler(260726,2000,r,2026)) for r in range(4)]
        self.assertEqual([len(x) for x in ranks], [16000]*4)
        self.assertEqual(len(set(sum(ranks,[]))),64000)
        order = torch.randperm(260726,generator=torch.Generator().manual_seed(2026)).tolist()
        for step in [0,1,1999]:
            actual = [ranks[r][step*8+i] for i in range(8) for r in range(4)]
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
            mp.spawn(distributed_worker,args=(str(Path(folder)/'gloo'),),nprocs=4,join=True)


if __name__ == '__main__':
    (WORK/'checks').mkdir(exist_ok=True)
    unittest.main()
