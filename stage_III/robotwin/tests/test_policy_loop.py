"""Exercise the actual wrapper loop using small CPU stand-ins, not checkpoints."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from stage_III.robotwin.policy import InteractionPolicy

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.gain=nn.Parameter(torch.full((2048,),1.001),requires_grad=False)
    def forward(self,h,**kwargs):return h*self.gain

class Language(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.ModuleList([Block() for _ in range(16)]);self.norm=nn.Identity()
    def rotary_emb(self,*args):return None

class Message(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(.2))
    def forward(self,q,k,v,need_weights=False):return self.weight*(q+v.mean(1,keepdim=True)),None

def model_and_current():
    m=InteractionPolicy.__new__(InteractionPolicy);nn.Module.__init__(m)
    lm=Language()
    m.__dict__["_teacher"]=SimpleNamespace(policy=SimpleNamespace(qwen=SimpleNamespace(language_model=lm)))
    m.cfg={"distill_layers":[4,8,12,16]};m.checkpoint_blocks=False;m.mode="student"
    m.q_goal=nn.Parameter(torch.randn(8,2048)*.1);m.q_local=nn.Parameter(torch.randn(8,2048)*.1)
    m.register_buffer("baseline_goal",m.q_goal.detach().clone(),persistent=False)
    m.register_buffer("baseline_local",m.q_local.detach().clone(),persistent=False)
    m.gate=nn.Module()
    m.gate.norm_g=nn.LayerNorm(2048);m.gate.norm_l=nn.LayerNorm(2048)
    m.gate.proj_g=nn.Linear(2048,4);m.gate.proj_l=nn.Linear(2048,4)
    m.gate.null_key=nn.Parameter(torch.zeros(1,4))
    m.gate.messages=nn.ModuleList([Message() for _ in range(8)])
    m.gate.scales=nn.Parameter(torch.full((8,),.01))
    b,length=2,21
    h=torch.randn(b,length,2048);valid=torch.ones(b,length,dtype=torch.bool)
    positions=torch.arange(length)[None,None].expand(3,b,-1)
    current=(h,valid,torch.zeros_like(valid),torch.zeros_like(valid),positions,torch.zeros_like(valid),[])
    return m,current,lm

class PolicyLoopTests(unittest.TestCase):
    def test_shapes_and_unidirectional_goal_invariance(self):
        m,c,_=model_and_current()
        a=m.language_roles(c,eta=0);b=m.language_roles(c,eta=1)
        self.assertEqual(b["goal"].shape,(2,8,2048))
        self.assertEqual(b["gate"]["features"].shape,(2,6144))
        self.assertTrue(torch.equal(a["goal"],b["goal"]))
        self.assertGreater(float((a["local"]-b["local"]).abs().max()),0)
        self.assertEqual(set(b["layers"]),{4,8,12,16})

    def test_gradients_cross_frozen_backbone(self):
        m,c,lm=model_and_current()
        m.language_roles(c)["local"].square().mean().backward()
        self.assertGreater(float(m.q_local.grad.abs().sum()),0)
        self.assertGreater(float(m.q_goal.grad.abs().sum()),0)
        self.assertGreater(float(m.gate.scales.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in lm.parameters()))

    def test_checkpoint_recomputation_matches_gradients(self):
        torch.manual_seed(2);m,c,_=model_and_current()
        m.train()
        m.language_roles(c)["local"].square().mean().backward()
        grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
        m.zero_grad();m.checkpoint_blocks=True
        m.language_roles(c)["local"].square().mean().backward()
        for n,p in m.named_parameters():
            if n in grads:self.assertTrue(torch.allclose(p.grad,grads[n],atol=1e-6,rtol=1e-5),n)

    def test_actor_sample_once_and_broadcast_to_deep_layers(self):
        m,c,_=model_and_current();m.eval()
        class Actor:
            calls=0
            def decide(self,features,seed,deterministic=False):
                self.calls+=1
                return dict(features=features,eta=torch.full((len(features),),1.4))
        actor=Actor()
        a=m.language_roles(c,actor=actor,gate_seed=7)
        b=m.language_roles(c,eta=1.4)
        self.assertEqual(actor.calls,1)
        self.assertTrue(torch.equal(a["local"],b["local"]))

if __name__=="__main__":unittest.main()
