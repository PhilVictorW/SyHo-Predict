import copy,json,types,unittest
from collections import Counter
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from .config import WORK,load_config
from .model import A0Policy
from .distributed import ExactRankBatchSampler
from .data import CurrentDataset

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(32,96);self.proj=nn.Linear(32,32)
    def forward(self,x,attention_mask=None,**kw):
        q,k,v=self.qkv(x).chunk(3,-1)
        return x+self.proj(F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],attn_mask=attention_mask)[:,0])

class Qwen(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Linear(32,32)
        self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=32))
        self.language_model=nn.Module();self.language_model.layers=nn.ModuleList([Block() for _ in range(16)])
        self.language_model.norm=nn.LayerNorm(32);self.language_model.rotary_emb=lambda *a:None

def tiny():
    c=load_config(WORK/'config.json');c['flow'].update(vlm_dim=32,vision_dim=32,hidden_dim=32,attention_heads=4,num_vision_tokens=2)
    return A0Policy(c,Qwen(),nn.Linear(32,32),None,{'franka':{'action':{'min':[-1.]*7,'max':[1.]*7}}})

class Tests(unittest.TestCase):
    def test_exact_epoch_and_resume(self):
        for world,micro,acc in ((1,1,8),(2,8,2),(4,4,2),(5,1,8)):
            batch=world*micro*acc
            for n in (1,3,7,31,32,33,40,83):
                def sampler(r,cursor=0):
                    return ExactRankBatchSampler(n,r,2026,0,cursor=cursor,global_batch=batch,world=world,micro_batch=micro)
                ranks=[list(sampler(r)) for r in range(world)]
                self.assertEqual(len({len(x) for x in ranks}),1)
                self.assertEqual(len(sampler(0)),len(ranks[0]))
                samples=[entry[0] for seq in ranks for row in seq for entry in row if entry[1]>0]
                self.assertEqual(Counter(samples),Counter(range(n)))
                for start in range(0,len(ranks[0]),acc):
                    entries=[item for seq in ranks for row in seq[start:start+acc] for item in row]
                    weighted=sum(x[0]*x[1]/world for x in entries)
                    selected=[x[0] for x in entries if x[1]>0]
                    self.assertAlmostEqual(weighted,sum(selected)/len(selected))
                if n>batch:
                    for r in range(world):self.assertEqual(ranks[r][acc:],list(sampler(r,batch)))

    def test_batch_tail_loss_normalization(self):
        m=tiny();b=3
        actions=torch.zeros(b,50,32);actions[0]=1.;actions[1]=2.;actions[2]=3.
        remaining=torch.tensor([8,2,1]);mask=torch.zeros_like(actions,dtype=torch.bool)
        for i,k in enumerate(remaining):mask[i,:k,:7]=True
        conditions=m.flow.conditions(torch.randn(b,16,32),torch.randn(b,2,32))
        with patch.object(m.flow,'sample_noise',side_effect=lambda shape,device,dtype:torch.zeros(shape,device=device,dtype=dtype)), patch.object(m.flow,'sample_time',side_effect=lambda n,device,dtype:torch.full((n,),.5,device=device,dtype=dtype)), patch.object(m.flow,'velocity',side_effect=lambda x,*args:torch.zeros_like(x)):
            values=m.flow.loss(actions,mask,torch.full((b,),20.),torch.full((b,),25),remaining,conditions,repeats=2,reduction='none')
        torch.testing.assert_close(values,torch.tensor([1.,4.,9.]))
        # A padded sample contributes exactly zero even if its loss is nonzero.
        self.assertAlmostEqual(float((values*torch.tensor([.5,.5,0.])).sum()),2.5)

    def test_single_latent_gradients_and_memory(self):
        torch.manual_seed(1);m=tiny()
        c=torch.cat((m.act_query,m.flow_action_query,torch.randn(4,32)),0)[None]
        v=torch.ones(1,20,dtype=torch.bool);a=torch.zeros_like(v);f=a.clone();a[:,:8]=True;f[:,8:16]=True
        pos=torch.arange(20)[None,None].expand(3,1,-1)
        latent=m.latent_features((c,v,a,f,pos,torch.zeros_like(v),[]))
        self.assertEqual(tuple(latent.shape),(1,16,32))
        dino=torch.randn(1,2,32);conditions=m.flow.conditions(latent,dino)
        self.assertEqual(tuple(conditions[0].shape),(1,18,32))
        self.assertTrue(torch.equal(conditions[0][:,:2],dino))
        self.assertEqual(int(conditions[1].sum()),2);self.assertEqual(int(conditions[2].sum()),16)
        actions=torch.randn(1,50,32);mask=torch.zeros_like(actions,dtype=torch.bool);mask[:,:8,:7]=True
        hz=torch.tensor([20.]);emb=torch.tensor([25]);remaining=torch.tensor([8])
        loss=m.flow.loss(actions,mask,hz,emb,remaining,conditions,repeats=1);loss.backward()
        for param in (m.latent_query,m.act_query,m.flow_action_query,m.flow.enc_vlm.net[0].weight):
            self.assertTrue(torch.isfinite(param.grad).all());self.assertGreater(float(param.grad.abs().sum()),0)
        self.assertFalse(any(p.requires_grad for p in m.dino.parameters()))
        for name in ('future','gate','q_goal','q_local'):self.assertFalse(hasattr(m,name))
        self.assertFalse(hasattr(m.flow,'plan'))

    def test_dit_routes_and_dino_effect(self):
        torch.manual_seed(7);m=tiny();m.eval()
        latent=torch.randn(1,16,32);dino=torch.randn(1,2,32);cond=m.flow.conditions(latent,dino)
        used=[];handles=[]
        def hook(i):
            return lambda mod,args,kw:used.append((i,kw.get('encoder_attention_mask')))
        for i,b in enumerate(m.flow.DiT.transformer_blocks):handles.append(b.register_forward_pre_hook(hook(i),with_kwargs=True))
        x=torch.randn(1,50,32);valid=torch.arange(50)[None]<8
        args=(x,torch.tensor([1]),torch.tensor([20.]),torch.tensor([25]),valid)
        out=m.flow.velocity(*args,cond)
        for h in handles:h.remove()
        for i,mask in used:
            if i%2:self.assertIsNone(mask)
            else:self.assertTrue(torch.equal(mask,cond[2] if i%4==0 else cond[1]))
        other=m.flow.velocity(*args,m.flow.conditions(latent,dino+1))
        self.assertGreater(float((out-other).abs().sum()),0)
        pred=m.flow.sample(cond,torch.tensor([20.]),torch.tensor([25]),noise=torch.zeros(1,50,32),steps=2)
        self.assertEqual(tuple(pred.shape),(1,8,7));self.assertTrue(torch.isfinite(pred).all())

    def test_actual_data_current_only(self):
        c=load_config(WORK/'config.json');stats=json.loads(__import__('pathlib').Path(c['statistics_path']).read_text())
        d=CurrentDataset(c['dataset_root'],stats,decode=False)
        self.assertEqual(len(d),2125407)
        calls=[];original=d._image
        d._image=lambda e,t,v,valid=True:(calls.append((e['episode_index'],t,v)) or original(e,t,v,valid))
        e=d.selected[0];batch=d.sample(e,e['length']-1)
        self.assertEqual(len(calls),2);self.assertTrue(all(t==e['length']-1 for _,t,_ in calls))
        self.assertFalse(any(k.startswith(('local','goal','future')) for k in batch))
        self.assertEqual(int(batch['actions_mask'].sum()),7);d.close()

    def test_timestep_embedding_matches_original(self):
        m=tiny();t=torch.tensor([0,500,999])
        torch.testing.assert_close(m.flow.timestep_embedding(t),m.flow.DiT.timestep_encoder(t),atol=0,rtol=0)

    def test_budget(self):
        import math
        c=load_config(WORK/'config.json')
        from .preflight import inspect
        from .evaluation import validate_final_budget
        plan=inspect(c)
        self.assertEqual(plan['total_updates'],1000);self.assertEqual(plan['total_training_anchors'],32000)
        self.assertNotIn('epochs',c)
        self.assertTrue(c['uses_dino']);self.assertEqual(c['physical_gpus'],[0,7])
        meta=dict(step=1000,extra=dict(smoke=False,budget_complete=True,completed_updates=1000,stop_after_updates=1000,seen_anchors=32000,completed_epochs=0,cursor=32000))
        validate_final_budget(meta,c)
        for changes in ({'smoke':True},{'budget_complete':False},{'completed_updates':999},{'seen_anchors':31968}):
            bad=copy.deepcopy(meta);bad['extra'].update(changes)
            with self.assertRaises(ValueError):validate_final_budget(bad,c)

if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
