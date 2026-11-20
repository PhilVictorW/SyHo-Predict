import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from transformers import Qwen3VLConfig,Qwen3VLModel
from goal_local.roles import FutureResamplers,RelevanceGate,role_attention_mask,role_distillation,remove_old_queries
from goal_local.action import HierarchicalFlow
from goal_local.data import indices
from goal_local.model import GoalLocalPolicy,make_stage_two
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig


def tiny_policy():
    cfg=json.loads(Path('configs/goal_local/default.json').read_text());cfg['checkpoint_blocks']=False
    cfg['flow'].update(vlm_dim=64,hidden_dim=32,vision_dim=24,attention_heads=4,num_vision_tokens=4,num_embodiments=1)
    qcfg=Qwen3VLConfig(text_config=dict(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=16,num_attention_heads=4,num_key_value_heads=2,head_dim=16,rope_parameters={'rope_type':'default','rope_theta':10000,'mrope_section':[2,3,3]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=64,deepstack_visual_indexes=[0,1,2]))
    qcfg._attn_implementation='sdpa';qcfg.text_config._attn_implementation='sdpa'
    stats={'franka':{'action':{'min':[-1]*7,'max':[1]*7}}}
    return GoalLocalPolicy(cfg,Qwen3VLModel(qcfg),nn.Linear(1,1),SimpleNamespace(),stats)


def current(b=2):
    h=torch.randn(b,24,64,requires_grad=True);valid=torch.ones(b,24,dtype=torch.bool)
    valid[1,0]=False
    am=torch.zeros_like(valid);am[:,2:10]=True
    fm=torch.zeros_like(valid);fm[:,12:20]=True
    pos=torch.arange(24)[None,None].expand(3,b,-1)
    return h,valid,am,fm,pos,torch.zeros_like(valid),[]


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2);torch.manual_seed(2)
    def test_future_boundaries(self):
        li,gi,lv,gv=indices(91,100)
        self.assertTrue(lv);self.assertFalse(gv.any())
        self.assertFalse(indices(92,100)[2])
    def test_mask_and_query_removal(self):
        c=current();mask=role_attention_mask(c[1]);allowed=torch.isfinite(mask[:,0])
        self.assertFalse(allowed[:,:24,24:].any());self.assertFalse(allowed[:,24:32,32:].any());self.assertFalse(allowed[:,32:,24:32].any())
        self.assertTrue(allowed[:,24:32,24:32].all())
        h,m=remove_old_queries(c[0],c[1],c[2],c[3]);self.assertEqual(h.shape,(2,8,64));self.assertFalse(m[1,0])
    def test_resampler_missing_future(self):
        r=FutureResamplers(64);local=torch.randn(2,2,4,64);goal=torch.randn(2,5,4,64)
        lv=torch.tensor([True,False]);gv=torch.tensor([[1,0,1,0,0],[0,0,0,0,0]],dtype=torch.bool)
        g,l=r(local,goal,lv,gv)
        goal2=goal.clone();goal2[~gv]=10000
        g2,l2=r(local,goal2,lv,gv)
        torch.testing.assert_close(g,g2);self.assertTrue(torch.equal(g[1],torch.zeros_like(g[1])))
        (g.square().mean()+l.square().mean()).backward();self.assertIsNotNone(r.goal.query.grad)
    def test_masked_distillation(self):
        s=torch.randn(2,8,16,requires_grad=True);t=s.detach().clone();t[1]=float('nan')
        loss=role_distillation(s,t,torch.tensor([True,False]));self.assertTrue(torch.isfinite(loss));loss.backward()
        zero=role_distillation(s,t,torch.tensor([False,False]));self.assertEqual(float(zero),0.)
    def test_teacher_role_isolation_and_gradient(self):
        model=tiny_policy().eval();c=current();g=torch.randn(2,8,64,requires_grad=True);l=torch.randn(2,8,64,requires_grad=True)
        out=model.language(c,g,l);other=model.language(c,g,l+torch.randn_like(l)*10)
        torch.testing.assert_close(out['current'],other['current']);torch.testing.assert_close(out['goal'],other['goal'])
        other=model.language(c,g+torch.randn_like(g)*10,l)
        torch.testing.assert_close(out['local'],other['local'])
        (out['goal'].square().mean()+out['local'].square().mean()).backward();self.assertGreater(float(g.grad.abs().sum()),0)
    def test_student_freeze_and_gate(self):
        system=make_stage_two(tiny_policy());system.train()
        self.assertFalse(system.teacher.training);self.assertTrue(all(not p.requires_grad for p in system.teacher.parameters()))
        self.assertNotEqual(system.teacher.act_query.data_ptr(),system.student.act_query.data_ptr())
        c=current();s=system.student;out=s.language(c,s.q_goal[None].expand(2,-1,-1),s.q_local[None].expand(2,-1,-1))
        loss=out['local'][...,:5].sum();loss.backward()
        self.assertGreater(float(s.gate.proj_l.weight.grad.abs().sum()),0)
        self.assertGreater(float(s.q_goal.grad.abs().sum()),0)
        frozen=system.teacher.act_query.clone();torch.optim.SGD([p for p in s.parameters() if p.requires_grad],lr=.01).step()
        torch.testing.assert_close(frozen,system.teacher.act_query)
        self.assertTrue(all(p.grad is None for p in system.teacher.parameters()))
    def test_flow_topology_baseline_bypass_and_tail(self):
        model=tiny_policy();flow=model.flow.eval();b=2
        cur=torch.randn(b,8,64);valid=torch.ones(b,8,dtype=torch.bool);dino=torch.randn(b,4,24);g=torch.randn(b,8,64);l=torch.randn(b,8,64)
        cond=flow.conditions(cur,valid,dino,g,l)
        x=torch.randn(b,50,32);hz=torch.full((b,),20.);eid=torch.zeros(b,dtype=torch.long);tv=torch.arange(50)[None].expand(b,-1)<8;t=torch.tensor([5,7])
        predicted=flow.velocity(x,t,hz,eid,tv,cond,bypass_roles=True)
        base,im,vm,*_=cond
        time=flow.time_encoder(torch.arange(50)[None].expand(b,-1)/20)*tv[...,None]
        h=flow.action_encoder(x,eid)*tv[...,None]
        expected=flow.action_decoder(flow.DiT(hidden_states=h,encoder_hidden_states=base,timestep=t,hidden_attention_mask=tv,image_mask=im,vlm_mask=vm,encoder_attention_mask=im|vm,hidden_positional_embeddings=time),eid)*tv[...,None]
        torch.testing.assert_close(predicted,expected)
        calls=[];handles=[]
        for i,a in flow.goal_attn.items():handles.append(a.register_forward_hook(lambda m,args,out,i=i:calls.append(('g',int(i)+1,args[0].shape[1]))))
        for i,a in enumerate(flow.local_attn):handles.append(a.register_forward_hook(lambda m,args,out,i=i:calls.append(('l',i+1,args[0].shape[1]))))
        masks=torch.zeros(b,50,32,dtype=torch.bool);masks[0,:8,:7]=True;masks[1,:1,:7]=True
        loss=flow.loss(torch.randn(b,50,32),masks,hz,eid,torch.tensor([20,1]),cond,repeats=1)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual([i for role,i,n in calls if role=='g'],[1,5,9,13]);self.assertEqual(len([i for role,i,n in calls if role=='l']),16)
        self.assertGreater(float(flow.goal_proj.weight.grad.abs().sum()),0);self.assertGreater(float(flow.plan.grad.abs().sum()),0)
        for handle in handles:handle.remove()
        with self.assertRaises(ValueError):flow.loss(torch.randn(b,50,32),masks,hz,eid,torch.tensor([20,20]),cond)
        disabled=flow.conditions(cur,valid,dino,g,l,torch.zeros(b,2,dtype=torch.bool))
        altered=flow.conditions(cur,valid,dino,g*100,l*100,torch.zeros(b,2,dtype=torch.bool))
        torch.testing.assert_close(flow.velocity(x,t,hz,eid,tv,disabled),flow.velocity(x,t,hz,eid,tv,altered))

if __name__=='__main__':unittest.main()
