import unittest,tempfile
from pathlib import Path
import torch
from goal_local.optim import MasterAdamW
from goal_local.train import atomic_json,validation_indices

class RuntimeTests(unittest.TestCase):
    def test_master_updates_survive_bf16_rounding_and_resume(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.bfloat16));unused=torch.nn.Parameter(torch.tensor([3.]))
        bridge=MasterAdamW([p,unused],lr=1e-4,weight_decay=0.)
        for _ in range(100):
            p.grad=torch.ones_like(p);bridge.step()
        self.assertLess(float(p),.995);self.assertAlmostEqual(float(bridge.masters[0]),.99,places=4)
        self.assertEqual(float(unused),3.)
        q=torch.nn.Parameter(torch.tensor([0.],dtype=torch.bfloat16));r=torch.nn.Parameter(torch.tensor([0.]))
        restored=MasterAdamW([q,r],lr=1e-4,weight_decay=0.)
        restored.load_state_dict(bridge.state_dict())
        torch.testing.assert_close(q,p);torch.testing.assert_close(r,unused)
        p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);bridge.step();restored.step()
        torch.testing.assert_close(restored.masters[0],bridge.masters[0])
    def test_validation_worker_seed_does_not_change_flow_rng(self):
        from torch.utils.data import DataLoader,TensorDataset
        loader=DataLoader(TensorDataset(torch.arange(8)),batch_size=2,
            generator=torch.Generator().manual_seed(2028))
        torch.manual_seed(2026);before=torch.get_rng_state().clone()
        iterator=iter(loader);next(iterator)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_atomic_status(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status.json';atomic_json(p,{'step':1});self.assertEqual(json.loads(p.read_text())['step'],1)
            atomic_json(p,{'step':2});self.assertEqual(json.loads(p.read_text())['step'],2)
if __name__=='__main__':unittest.main()
