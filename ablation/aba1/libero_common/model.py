"""Explicit role-masked Qwen execution, independent posterior and student policies."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel, AutoProcessor, Qwen3VLConfig, Qwen3VLModel
from .roles import FutureResamplers, RelevanceGate, role_attention_mask, remove_old_queries, role_distillation
from .action import HierarchicalFlow
from .data import ActionNormalizer
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages


from .config import validate_config, FORMAT


class GoalLocalPolicy(nn.Module):
    def __init__(self, cfg, qwen, dino, processor, statistics, mode='posterior', with_action=True):
        super().__init__()
        validate_config(cfg)
        if cfg['ablation'] == 'A0' and mode != 'student':
            raise ValueError('A0 must initialize directly as a student')
        if mode not in ('posterior','student'): raise ValueError(mode)
        self.kl, self.kg = cfg['local_slots'], cfg['goal_slots']
        self.cfg, self.mode = copy.deepcopy(cfg), mode
        self.qwen, self.dino, self.processor = qwen, dino, processor
        self.statistics = statistics
        self.normalizer = ActionNormalizer(statistics)
        dim = qwen.config.text_config.hidden_size
        self.act_query = nn.Parameter(torch.randn(8,dim)*.02)
        self.flow_action_query = nn.Parameter(torch.randn(8,dim)*.02)
        self.flow = HierarchicalFlow(ConditionalFlowMatchingConfig(**cfg['flow']), cfg['role_scale'], self.kg, self.kl) if with_action else None
        if mode == 'posterior':
            self.future = FutureResamplers(dim, self.kl, self.kg)
        else:
            if self.kg: self.q_goal = nn.Parameter(torch.randn(self.kg,dim)*.02)
            if self.kl: self.q_local = nn.Parameter(torch.randn(self.kl,dim)*.02)
            if cfg['interaction']: self.gate = RelevanceGate(dim, scale=cfg['gate_scale'])
        self.qwen.visual.requires_grad_(False).eval()
        self.dino.requires_grad_(False).eval()
        self.checkpoint_blocks = bool(cfg['checkpoint_blocks'])
        if self.flow is not None: self.flow.checkpoint_blocks = self.checkpoint_blocks
        self.frozen_teacher = False

    def train(self, mode=True):
        super().train(False if self.frozen_teacher else mode)
        self.qwen.visual.eval(); self.dino.eval()
        return self

    @property
    def device(self): return self.act_query.device

    @classmethod
    def from_baseline(cls, cfg):
        validate_config(cfg)
        state = torch.load(cfg['baseline_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        prefix = 'policy_backend.'
        def take(p): return {k[len(p):]:v for k,v in state.items() if k.startswith(p)}
        qs = take(prefix+'vlm.model.')
        qc = AutoConfig.from_pretrained(cfg['qwen_path'], local_files_only=True)
        if qc.model_type != 'qwen3_vl': raise ValueError('Validated implementation requires Qwen3-VL')
        qc.text_config.num_hidden_layers = 16
        qc.text_config.vocab_size = qs['language_model.embed_tokens.weight'].shape[0]
        qc._attn_implementation = 'sdpa'; qc.text_config._attn_implementation = 'sdpa'; qc.vision_config._attn_implementation = 'sdpa'
        # Create real rotary buffers; meta allocation would leave nonpersistent buffers uninitialized.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            qwen = Qwen3VLModel(qc)
        finally: torch.set_default_dtype(previous_dtype)
        qwen.load_state_dict(qs, strict=True, assign=True)
        dino = AutoModel.from_config(AutoConfig.from_pretrained(cfg['dino_path'], local_files_only=True))
        ds = take(prefix+'lam.vision_encoder.model.')
        dino.load_state_dict(ds, strict=True, assign=True)
        processor = AutoProcessor.from_pretrained(cfg['qwen_path'], local_files_only=True)
        processor.tokenizer.add_special_tokens({'additional_special_tokens':['<ACT_PH>']})
        if processor.chat_template is None: processor.chat_template = processor.tokenizer.chat_template
        statistics = json.loads(Path(cfg['statistics_path']).read_text())
        model = cls(cfg, qwen, dino, processor, statistics, mode='student' if cfg['ablation']=='A0' else 'posterior')
        with torch.no_grad():
            model.act_query.copy_(state[prefix+'act_query']); model.flow_action_query.copy_(state[prefix+'flow_action_query'])
        fs = take(prefix+'flow.')
        missing, unexpected = model.flow.load_state_dict(fs, strict=False)
        allowed = ('plan','goal_proj.','local_proj.','goal_attn.','local_attn.','goal_scale','local_scale')
        if unexpected or any(not k.startswith(allowed) for k in missing):
            raise ValueError(f'Unexpected flow migration: missing={missing}, unexpected={unexpected}')
        model.initialization_report = dict(source=cfg['baseline_checkpoint'], qwen_tensors=len(qs), dino_tensors=len(ds), flow_tensors=len(fs),
            new_flow_keys=missing, discarded_prefixes=['lam.encoder','lam.decoder','vlm_to_lam'], strict_retained_load=True)
        return model

    def encode_current_inputs(self, main, wrist, instructions):
        from PIL import Image
        def pil(x): return Image.fromarray(x.permute(1,2,0).cpu().numpy())
        messages = build_qwenvl_messages(images=[[pil(x)] for x in main], wrist_images=[[pil(x)] for x in wrist],
            instructions=instructions, placeholder_token='<ACT_PH>', act_queries=8, flow_queries=8)
        encoded = self.processor.apply_chat_template(messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        return {k:v.to(self.device) for k,v in encoded.items() if isinstance(v,torch.Tensor)}

    def current_embeddings(self, inputs):
        ids, valid = inputs['input_ids'], inputs['attention_mask'].bool()
        hidden = self.qwen.get_input_embeddings()(ids)
        mask = ids == self.processor.tokenizer.convert_tokens_to_ids('<ACT_PH>')
        if not torch.all(mask.sum(-1)==16): raise ValueError('Current template must contain exactly 16 old queries')
        ranks = mask.cumsum(-1)
        am, fm = mask & (ranks<=8), mask & (ranks>8)
        hidden = hidden.clone()
        hidden[am] = self.act_query[None].expand(ids.shape[0],-1,-1).reshape(-1,hidden.shape[-1]).to(hidden.dtype)
        hidden[fm] = self.flow_action_query[None].expand(ids.shape[0],-1,-1).reshape(-1,hidden.shape[-1]).to(hidden.dtype)
        with torch.no_grad():
            visual = self.qwen.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'], return_dict=True)
        features = torch.cat(visual.pooler_output,0).to(hidden.dtype)
        image_mask, _ = self.qwen.get_placeholder_mask(ids, inputs_embeds=hidden, image_features=features)
        hidden = hidden.masked_scatter(image_mask, features)
        # aba2 calls Qwen with inputs_embeds only and no cached rope_deltas.
        # Its actual current positions are sequential (including padding), not image-grid MRoPE.
        # Preserve that behavior and use the same appended positions for both branches.
        positions = torch.arange(ids.shape[1], device=ids.device)[None,None].expand(3,ids.shape[0],-1)
        return hidden, valid, am, fm, positions, image_mask[...,0], visual.deepstack_features

    @torch.no_grad()
    def encode_future_images(self, batch):
        """Only valid images are processed. Preserve horizon/view identity after scatter."""
        from PIL import Image
        b = batch['current_main'].shape[0]
        lv = batch['local_valid'].bool() if self.kl else torch.zeros(b,dtype=torch.bool)
        gv = batch['goal_frame_valid'].bool() if self.kg else torch.zeros(b,5,dtype=torch.bool)
        images, locations = [], []
        for i in range(b):
            if lv[i]:
                for v,key in enumerate(('local_main','local_wrist')):
                    images.append(Image.fromarray(batch[key][i].permute(1,2,0).cpu().numpy())); locations.append(('l',i,v))
            for j in range(5):
                if gv[i,j]:
                    images.append(Image.fromarray(batch['goal_main'][i,j].permute(1,2,0).cpu().numpy())); locations.append(('g',i,j))
        dim = self.act_query.shape[-1]
        if images:
            processed = self.processor.image_processor(images=images, return_tensors='pt').to(self.device)
            visual = self.qwen.get_image_features(processed['pixel_values'], processed['image_grid_thw'], return_dict=True)
            chunks = visual.pooler_output
            sizes = {x.shape[0] for x in chunks}
            if len(sizes)!=1: raise ValueError('LIBERO fixed-resolution future views must share grid size')
            n = sizes.pop()
        else:
            n, chunks = 64, []
        dtype = self.qwen.get_input_embeddings().weight.dtype
        local = torch.zeros(b,2,n,dim,device=self.device,dtype=dtype) if self.kl else None
        goal = torch.zeros(b,5,n,dim,device=self.device,dtype=dtype) if self.kg else None
        for (kind,i,j), features in zip(locations,chunks):
            (local if kind=='l' else goal)[i,j] = features
        return local, goal

    def language(self, current, goal, local):
        hidden, valid, am, fm, positions, visual_mask, deepstack = current
        b, length, _ = hidden.shape
        hidden = torch.cat((hidden,goal.to(hidden.dtype),local.to(hidden.dtype)),1)
        # Identical role positions in posterior/student; current multimodal RoPE remains native.
        start = positions.amax((0,2))+1
        appended = start[:,None]+torch.arange(16,device=hidden.device)[None]
        positions = torch.cat((positions,appended[None].expand(positions.shape[0],-1,-1)), -1)
        visual_mask = F.pad(visual_mask,(0,16),value=False)
        attn = role_attention_mask(valid,self.kg,self.kl,dtype=hidden.dtype)
        lm = self.qwen.language_model
        rotary = lm.rotary_emb(hidden, positions[-3:])
        layer_roles, gates = {}, {}
        cache_position = torch.arange(hidden.shape[1],device=hidden.device)
        for i, block in enumerate(lm.layers):
            def run(h, i=i, block=block):
                h = block(h,attention_mask=attn,position_ids=positions[0],cache_position=cache_position,
                    position_embeddings=rotary, use_cache=False)
                if i < len(deepstack): h = lm._deepstack_process(h,visual_mask,deepstack[i])
                if self.mode=='student' and self.cfg['interaction'] and i>=8:
                    g,l = h[:,length:length+self.kg],h[:,length+self.kg:]
                    updated, _ = self.gate(g,l,i-8)
                    h = torch.cat((h[:,:length+self.kg],updated),1)
                return h
            hidden = checkpoint(run,hidden,use_reentrant=False) if self.checkpoint_blocks and self.training else run(hidden)
            if i+1 in self.cfg['distill_layers']:
                layer_roles[i+1] = (hidden[:,length:length+self.kg], hidden[:,length+self.kg:])
        hidden = lm.norm(hidden)
        # Both teacher/student use final norm at layer 16, raw block outputs at 4/8/12.
        layer_roles[16] = (hidden[:,length:length+self.kg],hidden[:,length+self.kg:])
        base, base_valid = remove_old_queries(hidden[:,:length],valid,am,fm)
        return dict(current=base,current_valid=base_valid,goal=hidden[:,length:length+self.kg],local=hidden[:,length+self.kg:],layers=layer_roles)

    @torch.no_grad()
    def current_dino(self, images):
        x = images.to(self.device,dtype=torch.float32)/255
        mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
        std = x.new_tensor([.229,.224,.225])[None,:,None,None]
        output = self.dino(pixel_values=(x-mean)/std,output_hidden_states=True)
        features = output.hidden_states[-2][:,5:]
        return F.layer_norm(features,(features.shape[-1],),eps=1e-5)

    def encode(self, batch, inputs=None):
        if inputs is None: inputs=self.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
        current = self.current_embeddings(inputs)
        b = current[0].shape[0]
        if self.mode=='posterior':
            l,g = self.encode_future_images(batch)
            goal,local = self.future(l,g,batch['local_valid'].to(self.device) if self.kl else None,batch['goal_frame_valid'].to(self.device) if self.kg else None)
        else:
            # No access to future images or future validity anywhere in the student encoding.
            empty = current[0].new_empty(b,0,current[0].shape[-1])
            goal = self.q_goal[None].expand(b,-1,-1) if self.kg else empty
            local = self.q_local[None].expand(b,-1,-1) if self.kl else empty
        return self.language(current,goal,local)

    def action_conditions(self,batch,encoded):
        dino=self.current_dino(batch['current_main'])
        validity=None
        if self.mode=='posterior':
            b = batch['current_main'].shape[0]
            gv = batch['goal_frame_valid'].any(-1).to(self.device) if self.kg else torch.zeros(b,device=self.device,dtype=torch.bool)
            lv = batch['local_valid'].to(self.device) if self.kl else torch.zeros(b,device=self.device,dtype=torch.bool)
            validity=torch.stack((gv,lv),-1)
        return self.flow.conditions(encoded['current'],encoded['current_valid'],dino,encoded['goal'],encoded['local'],validity)

    def action_loss(self,batch,encoded):
        return self.flow.loss(batch['actions'].to(self.device),batch['actions_mask'].to(self.device),batch['action_hz'].to(self.device),
            batch['embodiment_id'].to(self.device),batch['remaining_actions'].to(self.device),self.action_conditions(batch,encoded),
            repeats=self.cfg['repeated_diffusion_steps'])

    def forward(self,batch):
        encoded=self.encode(batch)
        loss=self.action_loss(batch,encoded)
        return dict(loss=loss,flow=loss)

    @torch.no_grad()
    def predict_action(self,main,wrist,instructions,noise=None):
        if self.mode!='student': raise ValueError('Deployment only supports a student checkpoint')
        self.eval()
        if main.shape[1:] != (3,256,256) or wrist.shape != main.shape:
            raise ValueError('predict_action requires uint8 RGB [B,3,256,256] for both views')
        if main.dtype != torch.uint8 or wrist.dtype != torch.uint8:
            raise ValueError('predict_action expects uint8 images, before normalization')
        batch=dict(current_main=main,current_wrist=wrist,instruction=instructions)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            encoded=self.encode(batch)
            conditions=self.action_conditions(batch,encoded)
            b=len(instructions)
            actions=self.flow.sample(conditions,torch.full((b,),20.,device=self.device),torch.full((b,),25,device=self.device,dtype=torch.long),noise=noise)
        return self.normalizer.denormalize(actions)

    def save(self,path,step=0,extra=None):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.processor.save_pretrained(path/'processor')
        torch.save(dict(format=FORMAT,mode=self.mode,config=self.cfg,step=step,
            qwen_config=self.qwen.config.to_dict(),dino_config=self.dino.config.to_dict(),statistics=self.statistics,
            state_dict=self.state_dict(),extra=extra or {}),path/'model.pt')

    @classmethod
    def load(cls,path,with_action=True,legacy_config=None):
        path=Path(path)
        payload=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=False)
        if payload.get('format') != FORMAT:
            if not (payload.get('format')=='goal_local_v1' and payload['mode']=='posterior' and legacy_config is not None and legacy_config['ablation'] in ('A3','A4')):
                raise ValueError('Only an explicit full Stage-I legacy teacher may be imported')
            old = payload['config']
            if old['seed']!=legacy_config['split_seed'] or old['val_ratio']!=legacy_config['val_ratio']:
                raise ValueError('Legacy teacher split mismatch')
            for key in ('baseline_checkpoint','dataset_root','statistics_path','flow','goal_horizons','horizon','llm_layers','distill_layers'):
                if old[key] != legacy_config[key]: raise ValueError(f'Legacy teacher mismatch: {key}')
            payload['extra'] = dict(payload.get('extra',{}), legacy_import=True, original_teacher_config=copy.deepcopy(old))
            payload['config'] = copy.deepcopy(legacy_config)
        qc=Qwen3VLConfig.from_dict(payload['qwen_config'])
        qc._attn_implementation='sdpa';qc.text_config._attn_implementation='sdpa';qc.vision_config._attn_implementation='sdpa'
        previous_dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16);qwen=Qwen3VLModel(qc)
        finally:torch.set_default_dtype(previous_dtype)
        dc=dict(payload['dino_config']);kind=dc.pop('model_type')
        dino=AutoModel.from_config(AutoConfig.for_model(kind,**dc))
        model=cls(payload['config'],qwen,dino,AutoProcessor.from_pretrained(path/'processor',local_files_only=True),payload['statistics'],payload['mode'],with_action)
        state={k:v for k,v in payload['state_dict'].items() if with_action or not k.startswith('flow.')}
        model.load_state_dict(state,strict=True,assign=True)
        model.source_step=payload['step']
        model.source_extra=payload.get('extra',{})
        return model


def make_stage_two(posterior, cfg=None, initial=None):
    if posterior.mode != 'posterior' or getattr(posterior, 'source_step', 0) < 1:
        raise ValueError('Stage II requires a trained posterior Teacher')
    cfg = copy.deepcopy(cfg or posterior.cfg)
    validate_config(cfg)
    if cfg['ablation']=='A0': raise ValueError('A0 cannot inherit a posterior teacher')
    permitted = {'ablation', 'interaction', 'seed', 'learning_rate', 'warmup_steps'}
    for key in set(cfg) | set(posterior.cfg):
        if key not in permitted and cfg.get(key) != posterior.cfg.get(key):
            raise ValueError(f'Teacher/student contract mismatch: {key}')
    if cfg['ablation'] in ('A1','A2') and cfg['ablation'] != posterior.cfg['ablation']:
        raise ValueError('Single-scale student requires its own single-scale teacher')
    student = type(posterior)(cfg,copy.deepcopy(posterior.qwen),copy.deepcopy(posterior.dino),
                             posterior.processor,posterior.statistics,'student')
    shared = {k:v for k,v in posterior.state_dict().items() if not k.startswith('future.')}
    missing, unexpected = student.load_state_dict(shared,strict=False)
    if unexpected or any(not k.startswith(('q_goal','q_local','gate.')) for k in missing):
        raise ValueError((missing,unexpected))
    if initial is not None:
        if initial.mode != 'student' or initial.source_step != 0 or not initial.source_extra.get('initialization_only'):
            raise ValueError('Paired initialization must have zero Stage-II updates')
        for key in set(initial.cfg) | set(cfg):
            if key not in ('ablation','interaction') and initial.cfg.get(key) != cfg.get(key):
                raise ValueError(f'Paired initialization mismatch: {key}')
        common = {k:v for k,v in initial.state_dict().items() if not k.startswith('gate.')}
        # Only newly initialized queries may differ from the common teacher-derived weights.
        for key, value in shared.items():
            if not torch.equal(common[key], value):
                raise ValueError(f'Initialization is not from this teacher: {key}')
        missing, unexpected = student.load_state_dict(common,strict=False)
        if unexpected or any(not k.startswith('gate.') for k in missing):
            raise ValueError((missing,unexpected))
    posterior.flow = None
    posterior.requires_grad_(False); posterior.frozen_teacher=True; posterior.eval()
    return DistillationSystem(student,posterior)


class DistillationSystem(nn.Module):
    def __init__(self,student,teacher):
        super().__init__(); self.student=student; self.teacher=teacher

    def train(self,mode=True):
        super().train(mode); self.teacher.eval(); return self

    def forward(self,batch):
        with torch.no_grad(): target=self.teacher.encode(batch)
        encoded=self.student.encode(batch)
        flow=self.student.action_loss(batch,encoded)
        cfg=self.student.cfg
        losses=dict(flow=flow)
        total=flow
        for name,slot,index,mask in (
            ('local',self.student.kl,1,'local_valid'),
            ('goal',self.student.kg,0,'goal_frame_valid')):
            if not slot: continue
            valid=batch[mask].to(flow.device)
            if name=='goal': valid=valid.any(-1)
            terms=[role_distillation(encoded['layers'][layer][index],target['layers'][layer][index],
                                     valid,cfg['gram_alpha']) for layer in cfg['distill_layers']]
            value=torch.stack(terms).mean()
            losses[name+'_distill']=value
            total=total+cfg['lambda_'+name]*value
        return dict(loss=total,**losses)
