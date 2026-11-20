"""Reuse Goal/Local role attention and action blocks with RoboTwin I/O adapters."""
import copy
import torch
from torch import nn
from PIL import Image

from goal_local.model import GoalLocalPolicy, validate_config
from goal_local.roles import FutureResamplers, RelevanceGate
from goal_local.action import HierarchicalFlow
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig, build_time_grid
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages
from .data import ActionNormalizer


class ThreeViewFuture(FutureResamplers):
    def __init__(self, dim=2048):
        super().__init__(dim)
        self.view = nn.Parameter(torch.randn(3,dim)*.02)

    def forward(self, local, goal, local_valid, goal_valid):
        b, views, n, d = local.shape
        assert views == 3
        l = (local + self.view[None,:,None]).reshape(b,views*n,d)
        g = (goal + self.horizon[None,:,None]).flatten(1,2)
        lm = local_valid[:,None].expand(-1,views*n)
        gm = goal_valid[:,:,None].expand(-1,-1,goal.shape[2]).flatten(1)
        return self.goal(g,gm), self.local(l,lm)


class RobotwinFlow(HierarchicalFlow):
    def loss(self, actions, mask, hz, embodiment, remaining, conditions, repeats=2):
        _, _, count = build_time_grid(self.config.horizon_sec,hz,actions.shape[1])
        expected = torch.arange(actions.shape[1],device=actions.device)[None] < torch.minimum(count,remaining)[:,None]
        if not torch.all(count == 8) or not torch.all(mask[:,:,:16] == expected[:,:,None]) or mask[:,:,16:].any():
            raise ValueError('RoboTwin requires the episode-tail mask over 8 steps and 16 EEF dimensions')
        def repeat(x): return x.repeat_interleave(repeats,0)
        actions,mask,hz,embodiment = map(repeat,(actions,mask,hz,embodiment))
        conditions = tuple(repeat(x) for x in conditions)
        noise = self.sample_noise(actions.shape,actions.device,actions.dtype)
        tau = self.sample_time(actions.shape[0],actions.device,actions.dtype)[:,None,None]
        noisy = ((1-tau)*noise+tau*actions)*mask.any(-1)[...,None]
        bucket = (tau[:,0,0]*self.config.num_timestep_buckets).long().clamp(0,self.config.num_timestep_buckets-1)
        pred = self.velocity(noisy,bucket,hz,embodiment,mask.any(-1),conditions)
        return (((pred.float()-(actions-noise).float())**2)*mask).sum()/mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, conditions, hz, embodiment, noise=None, steps=10):
        b = hz.shape[0]
        x = conditions[0].new_empty(b,50,self.config.action_dim).normal_() if noise is None else noise.clone()
        _,valid,count = build_time_grid(self.config.horizon_sec,hz,50)
        assert torch.all(count == 8)
        x *= valid[...,None]
        for i in range(steps):
            bucket = torch.full((b,),min(int(i/steps*self.config.num_timestep_buckets),self.config.num_timestep_buckets-1),device=x.device,dtype=torch.long)
            x = (x+self.velocity(x,bucket,hz,embodiment,valid,conditions)/steps)*valid[...,None]
        return x[:,:8,:16]


class RobotwinPolicy(GoalLocalPolicy):
    # from_baseline/load inherited: cls ensures the same strict retained-weight audit.
    def __init__(self,cfg,qwen,dino,processor,statistics,mode='posterior',with_action=True):
        nn.Module.__init__(self)
        validate_config(cfg)
        if cfg.get('dataset_type') != 'robotwin_eef' or mode not in ('posterior','student'): raise ValueError('Wrong adapter/config')
        self.cfg,self.mode = copy.deepcopy(cfg),mode
        self.qwen,self.dino,self.processor = qwen,dino,processor
        self.statistics,self.normalizer = statistics,ActionNormalizer(statistics)
        dim = qwen.config.text_config.hidden_size
        self.act_query = nn.Parameter(torch.randn(8,dim)*.02)
        self.flow_action_query = nn.Parameter(torch.randn(8,dim)*.02)
        self.flow = RobotwinFlow(ConditionalFlowMatchingConfig(**cfg['flow']),cfg['role_scale']) if with_action else None
        if mode == 'posterior': self.future = ThreeViewFuture(dim)
        else:
            self.q_goal = nn.Parameter(torch.randn(8,dim)*.02)
            self.q_local = nn.Parameter(torch.randn(8,dim)*.02)
            self.gate = RelevanceGate(dim,scale=cfg['gate_scale'])
        self.qwen.visual.requires_grad_(False).eval()
        self.dino.requires_grad_(False).eval()
        self.checkpoint_blocks = bool(cfg['checkpoint_blocks'])
        if self.flow is not None: self.flow.checkpoint_blocks = self.checkpoint_blocks
        self.frozen_teacher = False

    def encode_current_inputs(self,main,wrist,instructions):
        def pil(x): return Image.fromarray(x.permute(1,2,0).cpu().numpy())
        if wrist.shape[1:] != (2,3,256,256): raise ValueError('Expected both wrist cameras')
        messages = build_qwenvl_messages(images=[[pil(x)] for x in main],
            wrist_images=[[pil(v) for v in x] for x in wrist],instructions=instructions,
            placeholder_token='<ACT_PH>',act_queries=8,flow_queries=8)
        result = self.processor.apply_chat_template(messages,tokenize=True,padding=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
        return {k:v.to(self.device) for k,v in result.items() if isinstance(v,torch.Tensor)}

    @torch.no_grad()
    def encode_future_images(self,batch):
        lv,gv = batch['local_valid'].bool(),batch['goal_frame_valid'].bool()
        images,locations = [],[]
        for i in range(len(lv)):
            if lv[i]:
                views = [batch['local_main'][i],*batch['local_wrist'][i]]
                for v,x in enumerate(views):
                    images.append(Image.fromarray(x.permute(1,2,0).cpu().numpy()));locations.append(('l',i,v))
            for j in range(5):
                if gv[i,j]:
                    images.append(Image.fromarray(batch['goal_main'][i,j].permute(1,2,0).cpu().numpy()));locations.append(('g',i,j))
        chunks,n = [],64
        if images:
            processed = self.processor.image_processor(images=images,return_tensors='pt').to(self.device)
            chunks = self.qwen.get_image_features(processed['pixel_values'],processed['image_grid_thw'],return_dict=True).pooler_output
            sizes = {x.shape[0] for x in chunks}
            if len(sizes)!=1: raise ValueError('Future grids must match')
            n = sizes.pop()
        dtype = self.qwen.get_input_embeddings().weight.dtype
        local = torch.zeros(len(lv),3,n,self.act_query.shape[-1],device=self.device,dtype=dtype)
        goal = torch.zeros(len(lv),5,n,self.act_query.shape[-1],device=self.device,dtype=dtype)
        for (kind,i,j),features in zip(locations,chunks): (local if kind=='l' else goal)[i,j] = features
        return local,goal

    @torch.no_grad()
    def predict_action(self,main,wrist,instructions,noise=None):
        if self.mode!='student': raise ValueError('Posterior teacher is not an online policy')
        self.eval()
        batch=dict(current_main=main,current_wrist=wrist,instruction=instructions)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            cond=self.action_conditions(batch,self.encode(batch))
            actions=self.flow.sample(cond,torch.full((len(instructions),),30.,device=self.device),
                torch.ones(len(instructions),device=self.device,dtype=torch.long),noise=noise)
        return self.normalizer.denormalize(actions)
