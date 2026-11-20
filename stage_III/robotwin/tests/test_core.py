"""CPU-only numerical/contracts tests; no real model or simulator is loaded."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import numpy as np
from goal_local.roles import RelevanceGate
from stage_III.robotwin.common import load_config,FORMAT,WORK
from stage_III.robotwin.gate_actor import GateActor,goal_local_update,pool_features
from stage_III.robotwin.grpo import reward,advantages,make_batch,update_actor
from stage_III.robotwin.make_manifests import build,assert_disjoint,validate_manifest
from stage_III.robotwin.acceptance import paired_statistics

torch.set_num_threads(2)

def cache_group(actor, rewards=(True,False,True,False), lengths=(1,2,3,1)):
    result=[]
    for i,(success,k) in enumerate(zip(rewards,lengths)):
        features=torch.randn(k,6144)
        rec=actor.decide(features,seed=700+i)
        decisions=[];position=0
        for j in range(k):
            d={n:(v[j].tolist() if n=="features" else float(v[j])) for n,v in rec.items()}
            used=1 if success else 6//k+int(j<6%k)
            d.update(executed_steps=used,chunk_start_step=position)
            position+=used
            decisions.append(d)
        result.append(dict(actor_sha256="actor1",student_sha256="student1",scene_id="scene",
            initial_state_hash="state",instruction="task",task="t",mode="demo_clean",
            group_id=0,trajectory_idx=i,decisions=decisions,env_steps=position,max_steps=6,
            env_config_hash="cfg",policy="actor_sample",
            termination="success" if success else "step_limit",success=success,
            reward=reward(success,position,6)))
    return result

class GateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.gate=RelevanceGate(dim=8,relevance_dim=4,heads=2,deep_layers=2,scale=.1)
        self.g=torch.randn(2,3,8);self.l=torch.randn(2,4,8)

    def test_eta_one_preserves_original_arithmetic(self):
        old,r=self.gate(self.g,self.l,0)
        new,r2=goal_local_update(self.gate,self.g,self.l,0,torch.ones(2))
        self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(r,r2))

    def test_eta_zero_closes_only_message(self):
        x,_=goal_local_update(self.gate,self.g,self.l,0,0)
        self.assertTrue(torch.equal(x,self.l))

    def test_zero_scale_cannot_be_fixed_by_actor(self):
        with torch.no_grad():self.gate.scales.zero_()
        a,_=goal_local_update(self.gate,self.g,self.l,0,.4)
        b,_=goal_local_update(self.gate,self.g,self.l,0,1.8)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a,self.l))

    def test_scale_first_then_message_gradients(self):
        with torch.no_grad():self.gate.scales.zero_()
        out,_=goal_local_update(self.gate,self.g,self.l,0)
        out.square().mean().backward()
        self.assertGreater(float(self.gate.scales.grad.abs().sum()),0)
        self.assertEqual(float(self.gate.messages[0].in_proj_weight.grad.abs().sum()),0)
        self.gate.zero_grad()
        with torch.no_grad():self.gate.scales.fill_(.01)
        out,_=goal_local_update(self.gate,self.g,self.l,0)
        out.square().mean().backward()
        self.assertGreater(float(self.gate.messages[0].in_proj_weight.grad.abs().sum()),0)

    def test_per_environment_eta_broadcast(self):
        out,_=goal_local_update(self.gate,self.g,self.l,0,torch.tensor([0.,1.]))
        old,_=self.gate(self.g,self.l,0)
        self.assertTrue(torch.equal(out[0],self.l[0]))
        self.assertTrue(torch.equal(out[1],old[1]))

    def test_invalid_eta_rejected(self):
        for v in ([-1,1],[1,3],[1,1,1],[float("nan")]):
            with self.assertRaises(ValueError):goal_local_update(self.gate,self.g,self.l,0,v)

    def test_pool_dimensions_padding_and_old_query_inclusion(self):
        h=torch.randn(2,36,2048);valid=torch.ones(2,20,dtype=torch.bool);valid[0,0]=False
        a=pool_features(h,valid,20)
        self.assertEqual(a.shape,(2,6144))
        h[0,0]=1000
        self.assertTrue(torch.equal(a,pool_features(h,valid,20)))
        h[0,19]+=torch.arange(2048)*.01
        self.assertFalse(torch.equal(a[0,4096:],pool_features(h,valid,20)[0,4096:]))

class ActorTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(41);self.actor=GateActor();self.cfg=load_config()["rl"]

    def test_parameter_count_and_initial_mean(self):
        self.assertEqual(sum(p.numel() for p in self.actor.parameters()),1573634)
        rec=self.actor.decide(torch.randn(2,6144),deterministic=True)
        self.assertTrue(torch.equal(rec["eta"],torch.ones(2)))

    def test_actor_remains_fp32_under_autocast(self):
        with torch.autocast("cpu",dtype=torch.bfloat16):
            r=self.actor.decide(torch.randn(3,6144),seed=8)
        for v in r.values():self.assertEqual(v.dtype,torch.float32)

    def test_rng_isolation_and_reproducibility(self):
        x=torch.randn(2,6144);before=torch.get_rng_state().clone()
        a=self.actor.decide(x,seed=90);b=self.actor.decide(x,seed=90)
        self.assertTrue(torch.equal(a["z"],b["z"]))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_group_advantage_and_failure_reward(self):
        self.assertEqual(reward(False,1,100),0)
        a,std=advantages([[0,0,0,0],[1,1,0,0]])
        self.assertTrue(torch.equal(a[0],torch.zeros(4)))
        self.assertTrue(torch.allclose(a[1],torch.tensor([1.,1.,-1.,-1.])))
        self.assertEqual(float(std[1]),.5)

    def test_length_weighted_trajectory_objective(self):
        b,stats=make_batch([cache_group(self.actor)],self.cfg)
        self.assertAlmostEqual(float(b["weight"].sum()),1,places=6)
        self.assertAlmostEqual(float(b["weight"][:1].sum()),.25)
        self.assertAlmostEqual(float(b["weight"][1:3].sum()),.25)
        self.assertAlmostEqual(float(b["weight"][3:6].sum()),.25,places=6)

    def test_logprob_mismatch_blocks_update(self):
        g=cache_group(self.actor);g[0]["decisions"][0]["old_logp"]+=.1
        opt=torch.optim.Adam(self.actor.parameters(),lr=3e-5)
        with self.assertRaises(RuntimeError):update_actor(self.actor,opt,[g],self.cfg)

    def test_mixed_versions_and_incomplete_groups_rejected(self):
        g=cache_group(self.actor);g[1]["actor_sha256"]="different"
        with self.assertRaises(ValueError):make_batch([g],self.cfg)
        with self.assertRaises(ValueError):make_batch([g[:3]],self.cfg)

    def test_eta_instead_of_z_is_rejected(self):
        g=cache_group(self.actor);g[0]["decisions"][0]["z"]=1.2
        with self.assertRaises(ValueError):make_batch([g],self.cfg)

    def test_zero_advantage_preserves_adam_momentum_state(self):
        opt=torch.optim.Adam(self.actor.parameters(),lr=3e-5)
        update_actor(self.actor,opt,[cache_group(self.actor)],self.cfg)
        before={k:v.clone() for k,v in self.actor.state_dict().items()}
        steps=[float(v["step"]) for v in opt.state.values()]
        g=cache_group(self.actor,rewards=(False,)*4)
        result=update_actor(self.actor,opt,[g],self.cfg)
        self.assertFalse(result["updated"])
        self.assertEqual(steps,[float(v["step"]) for v in opt.state.values()])
        for k,v in before.items():self.assertTrue(torch.equal(v,self.actor.state_dict()[k]))

    def test_same_actual_state_and_action_accounting_required(self):
        g=cache_group(self.actor);g[0]["initial_state_hash"]="other"
        with self.assertRaises(ValueError):make_batch([g],self.cfg)
        g=cache_group(self.actor);g[0]["env_steps"]+=1
        with self.assertRaises(ValueError):make_batch([g],self.cfg)

    def test_partial_failure_is_not_a_complete_rl_trajectory(self):
        g=cache_group(self.actor)
        g[1]["max_steps"]=100
        for r in g:r["max_steps"]=100
        with self.assertRaises(ValueError):make_batch([g],self.cfg)

    def test_chunk_gaps_are_rejected(self):
        g=cache_group(self.actor);g[0]["decisions"][0]["chunk_start_step"]=1
        with self.assertRaises(ValueError):make_batch([g],self.cfg)

class ManifestAndAdmissionTests(unittest.TestCase):
    def test_candidate_manifest_is_not_prepared_and_seeds_disjoint(self):
        m=build(load_config())
        self.assertEqual(len({(r["task"],r["mode"]) for r in m["rows"]}),100)
        self.assertEqual(len(m["rows"]),len({r["reset_seed"] for r in m["rows"]}))
        with self.assertRaises(ValueError):validate_manifest(m)

    def test_physical_split_leak_rejected_even_with_different_ids(self):
        row=dict(scene_id="one",task="t",mode="demo_clean",split="train",instruction="do",
                 initial_state_hash="same",prepared=True)
        m=dict(format=FORMAT,kind="prepared_scenes",prepared=True,complete=True,split="train",rows=[row])
        n=copy.deepcopy(m);n["split"]="val";n["rows"][0].update(split="val",scene_id="two")
        with self.assertRaises(ValueError):assert_disjoint(m,n)

    def paired(self):
        rows=[];arms={"baseline":[],"interaction":[]}
        for task in range(8):
            for mode in ("demo_clean","demo_randomized"):
                for i in range(3):
                    r=dict(scene_id=f"{task}_{mode}_{i}",task=str(task),mode=mode,split="val",
                           initial_state_hash=f"{mode}_{i}",env_config_hash="cfg",instruction="do",prepared=True)
                    rows.append(r)
                    base=dict(r,success=True,termination="success",noise_repeat=0)
                    arms["baseline"].append(dict(base,policy="baseline"))
                    arms["interaction"].append(dict(base,policy="fixed",eta=1))
        m=dict(format=FORMAT,kind="prepared_scenes",prepared=True,complete=True,split="val",rows=rows)
        report=dict(complete=True,skips=[],split="val",arms=arms)
        cfg=load_config();cfg["acceptance"]["bootstrap_samples"]=40
        return m,report,cfg

    def test_incomplete_manifest_and_quota_shortfall_rejected(self):
        m,e,c=self.paired();m["complete"]=False
        with self.assertRaises(ValueError):validate_manifest(m)
        m["complete"]=True;m["requested_cells"]=[["0","demo_clean"]];m["quota_per_cell"]=3
        with self.assertRaises(ValueError):validate_manifest(m)

    def test_paired_identical_success_passes_noninferiority(self):
        m,e,c=self.paired()
        x=paired_statistics(e,m,c)
        self.assertTrue(x["passed"]);self.assertEqual(x["confidence_interval"],[0.,0.])

    def test_paired_degradation_fails_admission(self):
        m,e,c=self.paired()
        for r in e["arms"]["interaction"]:r.update(success=False,termination="step_limit")
        self.assertFalse(paired_statistics(e,m,c)["passed"])

    def test_missing_or_mismatched_scenes_fail_admission(self):
        m,e,c=self.paired();e["arms"]["interaction"].pop()
        with self.assertRaises(ValueError):paired_statistics(e,m,c)
        m,e,c=self.paired();e["arms"]["interaction"][0]["instruction"]="different"
        with self.assertRaises(ValueError):paired_statistics(e,m,c)

class SafetyTests(unittest.TestCase):
    def test_workflow_default_does_not_launch(self):
        from stage_III.robotwin.workflow import main
        with patch("sys.argv",["workflow"]),patch("subprocess.run") as run,patch("builtins.print"):
            main()
        run.assert_not_called()

    def test_explicit_execution_and_gpu_are_required(self):
        from argparse import Namespace
        from stage_III.robotwin.common import require_execute
        for a in (Namespace(execute=False,gpu="0"),Namespace(execute=True,gpu=None)):
            with self.assertRaises(SystemExit):require_execute(a)

    def test_heartbeat_does_not_extend_real_progress_timeout(self):
        from stage_III.robotwin.runtime import wait_owned
        import subprocess
        from unittest.mock import Mock
        proc=Mock();proc.poll.return_value=None;proc.wait.side_effect=subprocess.TimeoutExpired("x",2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"progress.json";p.write_text(json.dumps(dict(updated_at=2000,last_progress_at=0)))
            with patch("stage_III.robotwin.runtime.time.time",side_effect=[0,2000]),patch("stage_III.robotwin.runtime.stop_owned") as stop:
                with self.assertRaises(TimeoutError):wait_owned(proc,p,1800,99999)
            stop.assert_called_once_with(proc)

    def test_normalized_action_wire_roundtrip(self):
        from stage_III.robotwin.rpc import array_encode,array_decode
        x=np.arange(36*16,dtype=np.float32).reshape(36,16)
        self.assertTrue(np.array_equal(array_decode(array_encode(x)),x))

    def test_wrong_routing_config_rejected(self):
        from stage_III.robotwin.common import load_config
        c=load_config();c["control_site"]="residual_strength"
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config.json";p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):load_config(p)

if __name__=="__main__":unittest.main()
