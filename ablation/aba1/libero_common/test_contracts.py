"""CPU contract tests with miniature tensors; no W0, real training or LIBERO rollout."""
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
from torch.nn import functional as F
from .config import ROOT, load_config, validate_config, LAYOUTS
from .roles import role_attention_mask, FutureResamplers, role_distillation
from .model import GoalLocalPolicy, make_stage_two
from .data import FutureDataset, ActionNormalizer, MAIN, WRIST
from .training import (EpochBatchSampler,epoch_updates,accumulation_groups,
                       capture_rng,restore_rng,learning_rate_factor)
from .evaluation import episode_rows,summarize
from goal_local.optim import MasterAdamW


class TinyBlock(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.qkv=nn.Linear(dim,dim*3);self.out=nn.Linear(dim,dim)
    def forward(self,h,attention_mask=None,**kwargs):
        q,k,v=self.qkv(h).chunk(3,-1)
        attended=F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],attn_mask=attention_mask)
        return h+self.out(attended[:,0])


class TinyLanguage(nn.Module):
    def __init__(self,dim):
        super().__init__();self.layers=nn.ModuleList([TinyBlock(dim) for _ in range(16)]);self.norm=nn.LayerNorm(dim)
    def rotary_emb(self,*args): return None


class TinyQwen(nn.Module):
    def __init__(self,dim=32):
        super().__init__();self.visual=nn.Linear(dim,dim);self.language_model=TinyLanguage(dim)
        self.config=types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=dim))


STATISTICS={'franka':{'action':{'min':[-1.]*7,'max':[1.]*7}}}


def config(variant):
    source=variant if variant!='A4' else 'A3'
    cfg=load_config(ROOT/'ablation/aba1'/source/'libero/config.json')
    if variant=='A4':cfg.update(ablation='A4',interaction=True)
    return cfg


def tiny_policy(variant,mode='student'):
    cfg=config(variant)
    cfg['flow']=dict(cfg['flow'],vlm_dim=32,vision_dim=32,hidden_dim=32,attention_heads=4,
                     num_vision_tokens=2)
    with patch('ablation.aba1.libero_common.model.validate_config'):
        return GoalLocalPolicy(cfg,TinyQwen(),nn.Linear(32,32),None,STATISTICS,mode)


def current(batch=2):
    length=20;hidden=torch.randn(batch,length,32);valid=torch.ones(batch,length,dtype=torch.bool)
    am=torch.zeros_like(valid);fm=am.clone();am[:,:8]=True;fm[:,8:16]=True
    pos=torch.arange(length)[None,None].expand(3,batch,-1)
    return hidden,valid,am,fm,pos,torch.zeros_like(valid),[]


def attach_inputs(policy,c):
    policy.encode_current_inputs=types.MethodType(lambda self,*args:None,policy)
    policy.current_embeddings=types.MethodType(lambda self,inputs:c,policy)
    policy.current_dino=types.MethodType(lambda self,images:torch.zeros(len(images),2,32),policy)
    if policy.mode=='posterior':
        local=torch.randn(c[0].shape[0],2,4,32) if policy.kl else None
        goal=torch.randn(c[0].shape[0],5,4,32) if policy.kg else None
        policy.encode_future_images=types.MethodType(lambda self,batch:(local,goal),policy)


def batch_for(variant,b=2):
    actions=torch.randn(b,50,32);mask=torch.zeros_like(actions,dtype=torch.bool);mask[:,:8,:7]=True
    batch=dict(current_main=torch.zeros(b,3,256,256,dtype=torch.uint8),current_wrist=torch.zeros(b,3,256,256,dtype=torch.uint8),
               instruction=['test']*b,actions=actions,actions_mask=mask,action_hz=torch.full((b,),20.),
               embodiment_id=torch.full((b,),25,dtype=torch.long),remaining_actions=torch.full((b,),8))
    if variant!='A0':
        kl,kg,_=LAYOUTS[variant]
        if kl:batch['local_valid']=torch.tensor([True,False][:b])
        if kg:batch['goal_frame_valid']=torch.tensor([[True,False,True,False,False],[False]*5][:b])
    return batch


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_variant_validation_and_paths(self):
        for variant in LAYOUTS:
            cfg=config(variant);validate_config(cfg)
            cfg['local_slots']+=1
            with self.assertRaises(ValueError):validate_config(cfg)

    def test_role_mask_isolation_and_padding(self):
        valid=torch.tensor([[True,True,False],[False,True,True]])
        for kl,kg,_ in LAYOUTS.values():
            mask=role_attention_mask(valid,kg,kl)[:,0]
            self.assertEqual(mask.shape,(2,19,19))
            self.assertTrue(torch.isneginf(mask[:,:3,3:]).all())
            if kl and kg:
                self.assertTrue(torch.isneginf(mask[:,3:3+kg,3+kg:]).all())
                self.assertTrue(torch.isneginf(mask[:,3+kg:,3:3+kg]).all())
            self.assertTrue(torch.isfinite(mask).any(-1).all())

    def test_resampler_slots_missing_and_gradients(self):
        for kl,kg in [(16,0),(0,16),(8,8)]:
            resampler=FutureResamplers(32,kl,kg)
            local=torch.randn(2,2,4,32) if kl else None
            goal=torch.randn(2,5,4,32) if kg else None
            lv=torch.tensor([True,False]);gv=torch.tensor([[True,False,False,False,False],[False]*5])
            g,l=resampler(local,goal,lv,gv)
            self.assertEqual(g.shape,(2,kg,32));self.assertEqual(l.shape,(2,kl,32))
            self.assertTrue(torch.isfinite(g).all() and torch.isfinite(l).all())
            (g.square().mean() if kg else l.square().mean()).backward()
            module=resampler.goal if kg else resampler.local
            self.assertIsNotNone(module.query.grad)
            self.assertTrue(torch.equal((g if kg else l)[1],torch.zeros_like((g if kg else l)[1])))

    def test_current_only_a0_flow_reaches_queries_and_gate(self):
        torch.manual_seed(5);policy=tiny_policy('A0');attach_inputs(policy,current())
        loss=policy(batch_for('A0'))['loss'];self.assertTrue(torch.isfinite(loss));loss.backward()
        for tensor in [policy.q_local,policy.q_goal,policy.gate.scales]:
            self.assertIsNotNone(tensor.grad);self.assertGreater(float(tensor.grad.abs().sum()),0.)
        self.assertFalse(hasattr(policy,'future'))

    def test_a3_isolation_and_a4_message_path(self):
        for variant in ['A3','A4']:
            torch.manual_seed(1);policy=tiny_policy(variant);policy.eval();c=current()
            g=policy.q_goal[None].expand(2,-1,-1);l=policy.q_local[None].expand(2,-1,-1)
            first=policy.language(c,g,l);second=policy.language(c,g+torch.randn_like(g),l)
            self.assertTrue(torch.equal(first['current'],second['current']))
            if variant=='A3':
                self.assertFalse(hasattr(policy,'gate'))
                self.assertTrue(torch.equal(first['local'],second['local']))
            else:self.assertFalse(torch.equal(first['local'],second['local']))

    def test_single_scale_teacher_student_distillation_and_frozen_copy(self):
        for variant in ['A1','A2','A3']:
            torch.manual_seed(2);teacher=tiny_policy(variant,'posterior');teacher.source_step=1
            c=current()
            with patch('ablation.aba1.libero_common.model.validate_config'):
                system=make_stage_two(teacher)
            attach_inputs(teacher,c);attach_inputs(system.student,c)
            before={k:v.clone() for k,v in teacher.state_dict().items()}
            losses=system(batch_for(variant));losses['loss'].backward()
            wanted={'loss','flow'}|({'local_distill'} if system.student.kl else set())|({'goal_distill'} if system.student.kg else set())
            self.assertEqual(set(losses),wanted)
            if variant=='A1':
                self.assertFalse(hasattr(system.student,'q_goal'))
                self.assertFalse(hasattr(system.student.flow,'goal_proj'))
            if variant=='A2':
                self.assertFalse(hasattr(system.student,'q_local'))
                self.assertFalse(hasattr(system.student.flow,'local_proj'))
            for name,value in teacher.state_dict().items():self.assertTrue(torch.equal(value,before[name]))
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            teacher_storage={p.data_ptr() for p in teacher.parameters()}
            self.assertTrue(all(p.data_ptr() not in teacher_storage for p in system.student.parameters()))
            for role in ['local','goal']:
                query=getattr(system.student,'q_'+role,None)
                if query is not None:self.assertGreater(float(query.grad.abs().sum()),0.)

    def test_full_topology_matches_upstream(self):
        from goal_local.model import GoalLocalPolicy as Original
        torch.manual_seed(7);new=tiny_policy('A4')
        old_cfg=copy.deepcopy(new.cfg);old_cfg['slots']=8
        torch.manual_seed(7)
        with patch('goal_local.model.validate_config'):
            old=Original(old_cfg,TinyQwen(),nn.Linear(32,32),None,STATISTICS,'student')
        self.assertEqual(set(new.state_dict()),set(old.state_dict()))
        for key,value in new.state_dict().items():self.assertTrue(torch.equal(value,old.state_dict()[key]),key)
        c=current();attach_inputs(new,c);attach_inputs(old,c)
        batch=batch_for('A0')
        torch.manual_seed(99);a=new(batch)['loss']
        torch.manual_seed(99);b=old(batch)['loss']
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_paired_initialization_and_reject_trained_initial(self):
        teacher=tiny_policy('A3','posterior');teacher.source_step=12
        original=copy.deepcopy(teacher)
        with patch('ablation.aba1.libero_common.model.validate_config'):
            first=make_stage_two(teacher).student
            first.source_step=0;first.source_extra={'initialization_only':True}
            cfg=copy.deepcopy(first.cfg);cfg.update(ablation='A4',interaction=True)
            second=make_stage_two(original,cfg,first).student
            for key,value in first.state_dict().items():self.assertTrue(torch.equal(value,second.state_dict()[key]),key)
            original=tiny_policy('A3','posterior');original.source_step=12
            first.source_step=1
            with self.assertRaises(ValueError):make_stage_two(original,initial=first)

    def test_checkpoint_roundtrip_all_variants_and_shape_rejection(self):
        class Processor:
            def save_pretrained(self,path):Path(path).mkdir(parents=True,exist_ok=True)
        for variant in ['A0','A1','A2','A3']:
            policy=tiny_policy(variant)
            policy.processor=Processor()
            policy.qwen.config.to_dict=lambda: {'tiny':True}
            policy.dino.config=types.SimpleNamespace(to_dict=lambda: {'model_type':'tiny'})
            qc=types.SimpleNamespace(text_config=types.SimpleNamespace(),vision_config=types.SimpleNamespace())
            with tempfile.TemporaryDirectory() as temp:
                policy.save(temp,17,extra={'completed_epochs':1})
                with patch('ablation.aba1.libero_common.model.validate_config'), \
                     patch('ablation.aba1.libero_common.model.Qwen3VLConfig.from_dict',return_value=qc), \
                     patch('ablation.aba1.libero_common.model.Qwen3VLModel',side_effect=lambda cfg:TinyQwen()), \
                     patch('ablation.aba1.libero_common.model.AutoConfig.for_model',return_value=None), \
                     patch('ablation.aba1.libero_common.model.AutoModel.from_config',side_effect=lambda cfg:nn.Linear(32,32)), \
                     patch('ablation.aba1.libero_common.model.AutoProcessor.from_pretrained',return_value=Processor()):
                    loaded=GoalLocalPolicy.load(temp)
                    self.assertEqual(loaded.source_step,17);self.assertEqual(loaded.cfg,policy.cfg)
                    for key,value in policy.state_dict().items():self.assertTrue(torch.equal(value,loaded.state_dict()[key]),key)
                    file=Path(temp)/'model.pt';payload=torch.load(file,weights_only=False)
                    # Detach mmapped tensors before rewriting their backing file.
                    payload['state_dict']={k:v.clone() for k,v in payload['state_dict'].items()}
                    name='q_local' if policy.kl else 'q_goal'
                    payload['state_dict'][name]=payload['state_dict'][name][:-1]
                    corrupt=Path(temp)/'bad';corrupt.mkdir();torch.save(payload,corrupt/'model.pt')
                    with self.assertRaises(RuntimeError):GoalLocalPolicy.load(corrupt)

    def test_distillation_mean_and_no_valid(self):
        student=torch.randn(2,16,32,requires_grad=True);teacher=torch.randn_like(student)
        zero=role_distillation(student,teacher,torch.zeros(2,dtype=torch.bool));zero.backward()
        self.assertEqual(float(zero),0.);self.assertTrue(torch.isfinite(student.grad).all())
        identical=role_distillation(student,student.detach(),torch.ones(2,dtype=torch.bool))
        self.assertLess(abs(float(identical)),1e-6)

    def fake_dataset(self,local,goal):
        dataset=object.__new__(FutureDataset);dataset.use_local=local;dataset.use_goal=goal
        dataset.root=Path('/unused');dataset.info={'video_path':'{video_key}_{chunk_index}_{file_index}.mp4'}
        dataset.timestamps=np.arange(100)*.05;dataset.actions=np.zeros((100,7),dtype=np.float32)
        dataset.normalizer=ActionNormalizer(STATISTICS);dataset.decode=True
        episode={'length':100,'dataset_from_index':0,'tasks':['test'],'episode_index':0}
        for camera in [MAIN,WRIST]:
            for field,value in [('chunk_index',0),('file_index',0),('from_timestamp',12.)]:
                episode[f'videos/{camera}/{field}']=value
        return dataset,episode

    def test_data_only_decodes_required_future_and_keeps_tails(self):
        for local,goal,main_count,wrist_count in [(False,False,1,1),(True,False,2,2),(False,True,6,1),(True,True,7,2)]:
            dataset,episode=self.fake_dataset(local,goal);calls=[]
            def read(path,timestamps):
                calls.append(timestamps);return [np.zeros((256,256,3),dtype=np.uint8) for _ in timestamps]
            with patch('ablation.aba1.libero_common.data.read_frames',side_effect=read):
                row=dataset.sample(episode,0)
                self.assertEqual([len(x) for x in calls],[main_count,wrist_count])
                self.assertEqual(calls[0][0],12.)
                self.assertEqual('local_valid' in row,local);self.assertEqual('goal_frame_valid' in row,goal)
                calls.clear();tail=dataset.sample(episode,94)
                self.assertEqual(int(tail['actions_mask'].sum()),6*7)
                self.assertEqual([len(x) for x in calls],[1,1])

    def test_exact_epoch_and_resume_cursor(self):
        batches=list(EpochBatchSampler(260726,2,2026,0))
        flat=[i for batch in batches for i in batch]
        self.assertEqual(len(flat),260726);self.assertEqual(len(set(flat)),260726)
        groups=list(accumulation_groups(batches,16))
        self.assertEqual(len(groups),8148);self.assertEqual(sum(map(len,groups[-1])),22)
        cursor=32*7
        self.assertEqual(list(EpochBatchSampler(260726,2,2026,0,cursor)),batches[cursor//2:])
        self.assertEqual(500*epoch_updates(260726,2,16),4074000)
        self.assertEqual(len(list(EpochBatchSampler(23,2,3,0))[-1]),1)
        self.assertAlmostEqual(learning_rate_factor(4073999,4074000,500),.05)

    def test_tail_weighting_and_optimizer_rng_resume(self):
        self.assertNotIn('cuda',capture_rng())
        torch.manual_seed(32);weight=nn.Parameter(torch.randn(3));optim=MasterAdamW([weight],.001,1e-8)
        x=torch.randn(22,3);y=torch.randn(22)
        for xb,yb in zip(x.split(2),y.split(2)):
            ((xb@weight-yb).square().mean()*len(xb)/22).backward()
        actual=weight.grad.clone();weight.grad=None
        (x@weight-y).square().mean().backward()
        torch.testing.assert_close(weight.grad,actual)
        optim.step()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'resume.pt';torch.save(dict(optimizer=optim.state_dict(),rng=capture_rng()),path)
            def step_once(param,opt):
                opt.zero_grad();loss=(param*torch.randn(3)).sum();loss.backward();opt.step()
            step_once(weight,optim);expected=weight.detach().clone()
            resumed=nn.Parameter(torch.zeros(3));other=MasterAdamW([resumed],.001,1e-8)
            saved=torch.load(path,weights_only=False);other.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            step_once(resumed,other);torch.testing.assert_close(resumed,expected,rtol=0,atol=0)

    def test_evaluation_denominators_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);task=out/'tasks/libero_spatial__task00';task.mkdir(parents=True)
            row=dict(task_id=0,episode_idx=0,success=False,num_actions=250)
            file=task/'episodes.checkpoint.jsonl';file.write_text(json.dumps(row)+'\n')
            summary=summarize(out,10,'fake')
            self.assertEqual(summary['expected_episodes'],400);self.assertEqual(summary['episodes_completed'],1)
            self.assertEqual(summary['successes'],0);self.assertEqual(summary['status'],'incomplete')
            with file.open('a') as f:f.write(json.dumps(dict(row,success=True))+'\n')
            with self.assertRaises(ValueError):episode_rows(task,0,10)


if __name__=='__main__': unittest.main()
