"""Behavioral runtime/provenance tests; all updates are tiny CPU unit fixtures."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .test_contracts import tiny_policy,current,attach_inputs,batch_for,STATISTICS,config
from .model import make_stage_two,initialize_new_modules
from .data import IndexedFutureDataset,ActionNormalizer
from .config import *
from .training import DistributedBatches
from .evaluation import summarize
from .deployment import PlusStudentClient
from .preflight import plan,check_complete
from goal_local.optim import MasterAdamW

def distributed_fixture(rank,path,result):
    import torch.distributed as dist
    from .distributed import DistributedMaster
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+path,world_size=2,rank=rank)
    try:
        p=nn.Parameter(torch.tensor([.5,-.25],dtype=torch.bfloat16))
        q=nn.Parameter(torch.tensor([.1]));unused=nn.Parameter(torch.tensor([3.]))
        opt=DistributedMaster([p,q,unused],.001,1e-8,bucket_elements=2)
        opt.zero_grad();p.grad=torch.full_like(p,float(rank+1))
        if rank==1:q.grad=torch.tensor([2.])
        opt.step(.001,1.)
        if rank==0:
            torch.save(opt.state_dict(),result)
            refp=nn.Parameter(torch.tensor([.5,-.25],dtype=torch.bfloat16))
            refq=nn.Parameter(torch.tensor([.1]));refunused=nn.Parameter(torch.tensor([3.]))
            ref=MasterAdamW([refp,refq,refunused],.001,1e-8)
            # Same global average + FP32 master clipping as the production Stage II.
            for master,grad in zip(ref.masters,([1.5,1.5],[1.],None)):
                master.grad=torch.tensor(grad) if grad is not None else None
            torch.nn.utils.clip_grad_norm_(ref.masters,1.,foreach=False,error_if_nonfinite=True)
            ref.optimizer.step()
            for actual,expected in zip(opt.master.masters,ref.masters):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        dist.barrier()
        first=[x.detach().clone() for x in (p,q,unused)]
        opt.zero_grad();p.grad=torch.full_like(p,2.);q.grad=torch.ones_like(q)
        opt.step(.001,1.);expected=[x.detach().clone() for x in (p,q,unused)]
        saved=torch.load(result,weights_only=False) if rank==0 else None
        opt.load_state_dict(saved)
        for actual,wanted in zip((p,q,unused),first):torch.testing.assert_close(actual,wanted,atol=0,rtol=0)
        opt.zero_grad();p.grad=torch.full_like(p,2.);q.grad=torch.ones_like(q);opt.step(.001,1.)
        for actual,wanted in zip((p,q,unused),expected):torch.testing.assert_close(actual,wanted,atol=0,rtol=0)
    finally:dist.destroy_process_group()

class Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_selective_rlds_decode_and_episode_tail(self):
        for local,goal,count in [(False,False,2),(True,False,4),(False,True,7),(True,True,9)]:
            ds=object.__new__(IndexedFutureDataset);ds.use_local=local;ds.use_goal=goal
            ds.actions=np.zeros((100,7),dtype=np.float32);ds.normalizer=ActionNormalizer(STATISTICS)
            episode=dict(length=100,dataset_from_index=0,tasks=["test"],episode_index=0)
            calls=[]
            def image(e,t,view,valid=True):
                if valid:
                    self.assertTrue(0<=t<100);calls.append((t,view))
                return torch.zeros(3,256,256,dtype=torch.uint8)
            ds._image=image
            row=ds.sample(episode,0);self.assertEqual(len(calls),count)
            self.assertEqual("goal_main" in row,goal);self.assertEqual("local_main" in row,local)
            if goal:self.assertEqual([x[0] for x in calls if x[0]>8],[24,40,56,72,88])
            calls.clear();row=ds.sample(episode,94)
            self.assertEqual(calls,[(94,0),(94,1)])
            self.assertEqual(int(row["actions_mask"].sum()),42)
            if local:self.assertFalse(row["local_valid"])
            if goal:self.assertFalse(row["goal_frame_valid"].any())
            with self.assertRaises(IndexError):ds.sample(episode,100)

    def test_selective_frozen_future_encoding(self):
        for variant in ("A1","A2"):
            policy=tiny_policy(variant,"posterior");calls=[]
            class Batch(dict):
                def to(self,device):return self
            def process(images,return_tensors):
                calls.append(len(images));return Batch(pixel_values=torch.zeros(len(images),4),
                    image_grid_thw=torch.zeros(len(images),3))
            policy.processor=types.SimpleNamespace(image_processor=process)
            policy.qwen.get_image_features=lambda pixels,grid,return_dict:types.SimpleNamespace(
                pooler_output=[torch.ones(4,32) for _ in pixels])
            embedding=nn.Embedding(2,32)
            policy.qwen.get_input_embeddings=lambda:embedding
            batch=batch_for(variant)
            if variant=="A1":
                batch.update(local_main=torch.zeros(2,3,256,256,dtype=torch.uint8),
                    local_wrist=torch.zeros(2,3,256,256,dtype=torch.uint8))
            else:batch["goal_main"]=torch.zeros(2,5,3,256,256,dtype=torch.uint8)
            local,goal=policy.encode_future_images(batch)
            self.assertEqual(calls,[2]);self.assertEqual(local is None,variant=="A2")
            self.assertEqual(goal is None,variant=="A1")

    def test_student_never_uses_future_validity_as_action_gate(self):
        for variant in ("A1","A2","A3","A0"):
            policy=tiny_policy(variant);attach_inputs(policy,current());batch=batch_for(variant)
            encoded=policy.encode(batch);before=policy.action_conditions(batch,encoded)
            if "local_valid" in batch:batch["local_valid"].zero_()
            if "goal_frame_valid" in batch:batch["goal_frame_valid"].zero_()
            after=policy.action_conditions(batch,policy.encode(batch))
            for a,b in zip(before,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(after[-1].all())

    def test_action_injection_locations_single_role_and_plan(self):
        for variant in ("A1","A2"):
            policy=tiny_policy(variant);attach_inputs(policy,current());calls=[]
            for key,module in policy.flow.goal_attn.items():
                module.register_forward_hook(lambda m,a,o,key=key:calls.append(("g",int(key)+1,a[0].shape[1],a[1].shape[1])))
            for key,module in enumerate(policy.flow.local_attn):
                module.register_forward_hook(lambda m,a,o,key=key:calls.append(("l",key+1,a[0].shape[1],a[1].shape[1])))
            loss=policy(batch_for(variant))["loss"];forward_calls=list(calls);loss.backward();calls=forward_calls
            self.assertEqual(policy.flow.plan.shape[0],4);self.assertEqual(len(policy.flow.DiT.transformer_blocks),16)
            self.assertEqual([i for role,i,q,k in calls if role=="g"],[1,5,9,13] if variant=="A2" else [])
            self.assertEqual([i for role,i,q,k in calls if role=="l"],list(range(1,17)) if variant=="A1" else [])
            self.assertTrue(all(k==16 for role,i,q,k in calls))
            self.assertGreater(float(policy.flow.plan.grad.abs().sum()),0.)

    def test_absent_future_disables_teacher_injection_without_nan(self):
        for variant in ("A1","A2"):
            teacher=tiny_policy(variant,"posterior");attach_inputs(teacher,current())
            batch=batch_for(variant)
            if teacher.kl:batch["local_valid"].zero_()
            if teacher.kg:batch["goal_frame_valid"].zero_()
            encoded=teacher.encode(batch);conditions=teacher.action_conditions(batch,encoded)
            self.assertFalse(conditions[-1].any())
            encoded2={**encoded,"local":encoded["local"]+123,"goal":encoded["goal"]-321}
            other=teacher.action_conditions(batch,encoded2)
            torch.manual_seed(2);a=teacher.flow.loss(batch["actions"],batch["actions_mask"],
                batch["action_hz"],batch["embodiment_id"],batch["remaining_actions"],conditions)
            torch.manual_seed(2);b=teacher.flow.loss(batch["actions"],batch["actions_mask"],
                batch["action_hz"],batch["embodiment_id"],batch["remaining_actions"],other)
            self.assertTrue(torch.isfinite(a));torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_module_seeded_common_initialization(self):
        a=tiny_policy("A0");b=tiny_policy("A1","posterior");c=tiny_policy("A2","posterior")
        before=torch.get_rng_state().clone()
        for model in (a,b,c):initialize_new_modules(model)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.testing.assert_close(a.flow.plan,b.flow.plan,rtol=0,atol=0)
        torch.testing.assert_close(a.flow.plan,c.flow.plan,rtol=0,atol=0)
        torch.testing.assert_close(a.flow.local_proj.weight,b.flow.local_proj.weight,rtol=0,atol=0)
        torch.testing.assert_close(a.flow.goal_attn["0"].in_proj_weight,c.flow.goal_attn["0"].in_proj_weight,rtol=0,atol=0)

    def test_a3_paired_step0_shared_weights_and_reject_trained_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            teacher=tiny_policy("A3","posterior");teacher.source_step=5000
            teacher.cfg["paired_student_init"]=str(path)
            state={k:v.clone() for k,v in teacher.state_dict().items() if not k.startswith("future.")}
            full=tiny_policy("A4")
            state.update({k:v.clone() for k,v in full.state_dict().items() if k.startswith("gate.") or k in ("q_goal","q_local")})
            payload=dict(format="goal_local_v1",mode="student",step=0,config=teacher.cfg,
                extra={"teacher_step":5000},state_dict=state)
            torch.save(payload,path/"model.pt")
            atomic_json(path/"checkpoint_complete.json",dict(step=0,teacher_step=5000))
            original=copy.deepcopy(teacher)
            with patch("ablation.aba1.plus_common.model.validate_config"):
                system=make_stage_two(teacher,initial_path=path)
                for k,v in system.student.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
                self.assertFalse(hasattr(system.student,"gate"))
                payload["step"]=1;torch.save(payload,path/"model.pt")
                with self.assertRaises(ValueError):make_stage_two(original,initial_path=path)

    def test_fixed_schedule_matches_original_lambda_lr_and_resume(self):
        import math
        for total,stop in ((16000,5000),(10000,4000)):
            parameter=nn.Parameter(torch.tensor([1.]))
            opt=torch.optim.AdamW([parameter],lr=1e-5)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:lr_factor(n,total))
            for n in range(stop):
                expected=min(1,(n+1)/500)*(.05+.95*.5*(1+math.cos(math.pi*max(0,n-500)/(total-500))))
                self.assertAlmostEqual(opt.param_groups[0]["lr"],1e-5*expected,places=16)
                opt.step();sched.step()
        self.assertNotEqual(lr_factor(999,16000),lr_factor(999,1000))

    def test_distributed_anchor_cursor_exact_recovery(self):
        ds=list(range(103))
        for world,batch in ((2,2),(4,1)):
            ranks=[]
            for rank in range(world):
                sampler=DistributedBatches(ds,batch,world,rank,2026);sampler.reset(3)
                full=list(sampler);ranks.append({i for row in full for i in row})
                sampler.reset(3,7);self.assertEqual(list(sampler),full[7:])
            for a in range(world):
                for b in range(a):self.assertFalse(ranks[a]&ranks[b])
        self.assertEqual(stage_spec(config("A0"),"action1")["schedule_total_steps"],16000)
        self.assertEqual(stage_spec(config("A0"),"action2")["cumulative_offset"],5000)

    def test_fp32_master_distributed_average_unused_and_resume(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(distributed_fixture,args=(str(Path(temp)/"rendezvous"),str(Path(temp)/"state.pt")),nprocs=2,join=True)

    def test_plus_gripper_flips_once_without_mutating_chunk(self):
        from goal_local.eval_client import StudentClient
        client=object.__new__(PlusStudentClient)
        client.actions=np.array([[.1,.2,.3,.4,.5,.6,1.]],dtype=np.float32)
        client.cursor=0;client.chunk=0;original=client.actions.copy()
        result=client.step({},0)
        self.assertEqual(float(result["raw_action"]["open_gripper"][0]),0.)
        np.testing.assert_array_equal(client.actions,original)
        np.testing.assert_array_equal(result["raw_action"]["world_vector"],original[0,:3])

    def test_incomplete_results_do_not_claim_full_accuracy(self):
        tasks=[dict(suite="libero_spatial",task_id=i,category="Camera") for i in (0,1)]
        protocol=dict(tasks=tasks,scope="full",checkpoint="fake",model_sha256="m",protocol_sha256="p")
        row=dict(suite="libero_spatial",task_id=0,category="Camera",success=True,num_actions=13,
            episode_idx=0,checkpoint="fake",model_sha256="m",protocol_sha256="p",action_protocol_id=ACTION_PROTOCOL["id"])
        result=summarize(protocol,[row],{"failed":1})
        self.assertIsNone(result["success_rate"]);self.assertEqual(result["expected"],2)
        self.assertEqual(result["completed_success_rate"],1.);self.assertEqual(result["infrastructure_errors"],1)
        with self.assertRaises(ValueError):summarize(protocol,[row,row])
        second=dict(row,task_id=1,success=False)
        result=summarize(protocol,[row,second])
        self.assertTrue(result["complete"]);self.assertEqual(result["success_rate"],.5)

    def test_real_qwen_rotary_role_spans_and_final_norm(self):
        from transformers import Qwen3VLConfig,Qwen3VLModel
        from .model import GoalLocalPolicy
        qc=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,
            num_hidden_layers=16,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
            rope_parameters={"rope_type":"default","rope_theta":10000,"mrope_section":[2,3,3]}),
            vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,
                out_hidden_size=64,deepstack_visual_indexes=[0,1,2]))
        qc._attn_implementation="sdpa";qc.text_config._attn_implementation="sdpa"
        for variant in ("A1","A2","A3","A4"):
            cfg=config(variant);cfg["checkpoint_blocks"]=False
            cfg["flow"]=dict(cfg["flow"],vlm_dim=64,hidden_dim=32,vision_dim=24,attention_heads=4)
            with patch("ablation.aba1.plus_common.model.validate_config"):
                policy=GoalLocalPolicy(cfg,Qwen3VLModel(qc),nn.Linear(1,1),None,STATISTICS,"student")
            b,n=2,24;hidden=torch.randn(b,n,64);valid=torch.ones(b,n,dtype=torch.bool);valid[1,0]=False
            am=torch.zeros_like(valid);fm=am.clone();am[:,2:10]=True;fm[:,12:20]=True
            c=(hidden,valid,am,fm,torch.arange(n)[None,None].expand(3,b,-1),torch.zeros_like(valid),[])
            g=policy.q_goal[None].expand(b,-1,-1) if policy.kg else hidden.new_empty(b,0,64)
            l=policy.q_local[None].expand(b,-1,-1) if policy.kl else hidden.new_empty(b,0,64)
            normed=[]
            handle=policy.qwen.language_model.norm.register_forward_hook(lambda m,a,o:normed.append(o.detach()))
            result=policy.language(c,g,l);handle.remove()
            self.assertEqual(result["current"].shape,(b,n-16,64))
            for role,index,count in (("goal",0,policy.kg),("local",1,policy.kl)):
                self.assertEqual(result[role].shape,(b,count,64))
                torch.testing.assert_close(result["layers"][16][index],result[role],rtol=0,atol=0)
            torch.testing.assert_close(result["goal"],normed[0][:,n:n+policy.kg],rtol=0,atol=0)
            if variant=="A3":
                other=policy.language(c,g+torch.randn_like(g),l)
                torch.testing.assert_close(result["local"],other["local"],rtol=0,atol=0)
            (result["goal"].sum()+result["local"].sum()).backward()
            if policy.kg:self.assertIsNotNone(policy.q_goal.grad)
            if policy.kl:self.assertIsNotNone(policy.q_local.grad)

    def test_eval_rejects_untrained_or_wrong_root_checkpoint(self):
        from .deployment import validate_checkpoint
        cfg=config("A0")
        meta=dict(mode="student",step=0,config=cfg,extra={"stage":"action1","action_protocol":ACTION_PROTOCOL})
        with patch("ablation.aba1.plus_common.deployment.check_complete",return_value=(meta,{})):
            with self.assertRaises(ValueError):validate_checkpoint("/tmp/not_an_ablation",cfg)
        meta["step"]=1000
        with patch("ablation.aba1.plus_common.deployment.check_complete",return_value=(meta,{})):
            with self.assertRaises(ValueError):validate_checkpoint("/tmp/not_an_ablation",cfg)

    def test_checkpoint_commit_requires_processor_and_full_resume(self):
        from .preflight import check_complete
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);cfg=config("A0")
            torch.save(dict(format=FORMAT,mode="student",step=1,config=cfg,extra={"stage":"action1"}),path/"model.pt")
            torch.save({"optimizer":"tiny"},path/"trainer.pt")
            (path/"processor").mkdir();(path/"processor/config.json").write_text("{}")
            names=["model.pt","trainer.pt","processor/config.json"]
            marker=dict(format=FORMAT,complete=True,resumable=True,step=1,config_sha256=digest_json(cfg),
                files={name:dict(sha256=file_sha256(path/name)) for name in names})
            atomic_json(path/"checkpoint_complete.json",marker)
            check_complete(path,require_resume=True,verify_hash=True)
            (path/"processor/config.json").write_text('{"changed":true}')
            with self.assertRaises(ValueError):check_complete(path,verify_hash=True)
            (path/"processor/config.json").write_text("{}")
            marker["resumable"]=False;atomic_json(path/"checkpoint_complete.json",marker)
            (path/"trainer.pt").unlink()
            check_complete(path,verify_hash=True)
            with self.assertRaises(ValueError):check_complete(path,require_resume=True)

if __name__=="__main__":unittest.main()
