"""Miniature CPU tests, never use real training frames or a production model."""
import json,tempfile,unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from unittest.mock import patch
from a1_local.config_runtime import load_config,stage_spec,source_paths
from ablation.aba1.plus_common.test_contracts import (
    Contracts,tiny_policy,attach_inputs,current,batch_for)
from ablation.aba1.plus_common.test_runtime import Runtime
from a1_local.model import make_stage_two
from a1_local.distributed import DistributedMaster
from goal_local.optim import MasterAdamW

def four_rank_fixture(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,world_size=6,rank=rank)
    try:
        p=nn.Parameter(torch.tensor([.5,-.25],dtype=torch.bfloat16))
        q=nn.Parameter(torch.tensor([.1]))
        unused=nn.Parameter(torch.tensor([3.]))
        opt=DistributedMaster([p,q,unused],1e-5,1e-8,bucket_elements=2)
        opt.zero_grad()
        for _ in range(len(range(rank,32,6))):
            loss=p.float().sum()*(rank+1)*(6/32)
            if rank==5:loss=loss+q.sum()*6*(6/32)
            loss.backward()
        opt.step(1e-5,1.)
        if rank==0:
            ref=MasterAdamW([nn.Parameter(torch.tensor([.5,-.25],dtype=torch.bfloat16)),
                            nn.Parameter(torch.tensor([.1])),nn.Parameter(torch.tensor([3.]))],1e-5,1e-8)
            for master,grad in zip(ref.masters,([108/32,108/32],[30/32],None)):
                master.grad=torch.tensor(grad) if grad is not None else None
            torch.nn.utils.clip_grad_norm_(ref.masters,1.,error_if_nonfinite=True,foreach=False)
            ref.optimizer.step()
            for actual,wanted in zip(opt.master.masters,ref.masters):
                torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
        for value in (p,q,unused):
            low=value.detach().float().clone();high=low.clone()
            dist.all_reduce(low,op=dist.ReduceOp.MIN);dist.all_reduce(high,op=dist.ReduceOp.MAX)
            torch.testing.assert_close(low,high,rtol=0,atol=0)
    finally:dist.destroy_process_group()

class A1(unittest.TestCase):
    def test_runtime_budgets_and_sources(self):
        cfg=load_config(Path(__file__).with_name("config.json"),"A1")
        for stage,steps,save in (("I",3000,3000),("II",2500,500)):
            spec=stage_spec(cfg,stage)
            self.assertEqual(sum(spec["accumulation_per_rank"])*spec["batch_size"],32)
            self.assertEqual(spec["stop_after_updates"],steps)
            self.assertEqual(spec["save_every"],save)
        self.assertEqual(source_paths(cfg,"II")["teacher"].name,"step_003000")
        self.assertEqual(cfg["a1_training"]["physical_gpus"],[0,1,2,3,5,6])
        from a1_local.config_runtime import physical_gpus,rank_device
        self.assertEqual([physical_gpus(cfg)[rank_device(cfg,r)] for r in range(6)],[0,1,2,3,5,6])
        self.assertEqual(cfg["a1_training"]["processes_per_gpu"],1)
        self.assertEqual(stage_spec(cfg,"II")["cumulative_offset"]+2500,5500)
        self.assertFalse(cfg["interaction"])
    def test_teacher_action_only_and_student_initialization(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)
        teacher=tiny_policy("A1","posterior")
        attach_inputs(teacher,current())
        losses=teacher(batch_for("A1"))
        self.assertEqual(set(losses),{"loss","flow"})
        losses["loss"].backward()
        self.assertGreater(float(teacher.future.local.query.grad.abs().sum()),0)
        before={k:v.detach().clone() for k,v in teacher.state_dict().items() if not k.startswith("future.")}
        teacher.zero_grad(set_to_none=True);teacher.source_step=3000
        teacher.cfg["a1_training"]=json.loads(Path(__file__).with_name("config.json").read_text())["a1_training"]
        with patch("a1_local.model.validate_config"),patch("ablation.aba1.plus_common.model.validate_config"):
            teacher.source_step=5000
            with self.assertRaises(ValueError):make_stage_two(teacher)
            teacher.source_step=3000
            system=make_stage_two(teacher)
        student=system.student
        self.assertEqual(student.initialization_report["teacher_step"],3000)
        self.assertEqual(tuple(student.q_local.shape),(16,32))
        self.assertFalse(hasattr(student,"gate"));self.assertFalse(hasattr(student,"q_goal"))
        for key,value in before.items():torch.testing.assert_close(student.state_dict()[key],value,rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
    def test_six_rank_offloaded_gradient_mean(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(four_rank_fixture,args=(str(Path(d)/"rendezvous"),),nprocs=6,join=True)

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(A1)
    for name in ("test_role_mask_isolation_and_padding","test_resampler_slots_missing_and_gradients",
                 "test_single_scale_teacher_student_distillation_and_frozen_copy",
                 "test_distillation_mean_and_no_valid"):
        suite.addTest(Contracts(name))
    for name in ("test_selective_rlds_decode_and_episode_tail","test_selective_frozen_future_encoding",
                 "test_action_injection_locations_single_role_and_plan",
                 "test_absent_future_disables_teacher_injection_without_nan"):
        suite.addTest(Runtime(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
