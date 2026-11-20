"""A0: single latent16, current Qwen context, VLM-only action conditioning."""
import copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig,AutoProcessor,Qwen3VLConfig,Qwen3VLModel
from .attention import latent_attention_mask,remove_old_queries
from .action import LatentFlow
from .data import ActionNormalizer
from .config import FORMAT
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages

class A0Policy(nn.Module):
    def __init__(self,cfg,qwen,processor,statistics):
        super().__init__()
        self.cfg,self.mode=copy.deepcopy(cfg),'student'
        self.qwen,self.processor,self.statistics=qwen,processor,statistics
        self.normalizer=ActionNormalizer(statistics)
        d=qwen.config.text_config.hidden_size
        self.act_query=nn.Parameter(torch.randn(8,d)*.02)
        self.flow_action_query=nn.Parameter(torch.randn(8,d)*.02)
        self.latent_query=nn.Parameter(torch.randn(16,d)*.02)
        self.flow=LatentFlow(ConditionalFlowMatchingConfig(**cfg['flow']))
        self.qwen.visual.requires_grad_(False).eval()

    @property
    def device(self):return self.act_query.device

    def train(self,mode=True):
        super().train(mode);self.qwen.visual.eval();return self

    @classmethod
    def from_baseline(cls, cfg):
        assert cfg["architecture"] == FORMAT
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
        processor = AutoProcessor.from_pretrained(cfg['qwen_path'], local_files_only=True)
        processor.tokenizer.add_special_tokens({'additional_special_tokens':['<ACT_PH>']})
        if processor.chat_template is None: processor.chat_template = processor.tokenizer.chat_template
        statistics = json.loads(Path(cfg['statistics_path']).read_text())
        model = cls(cfg, qwen, processor, statistics)
        with torch.no_grad():
            model.act_query.copy_(state[prefix+'act_query']); model.flow_action_query.copy_(state[prefix+'flow_action_query'])
        fs = take(prefix+'flow.')
        discarded = fs.pop('cfg_embeddings')
        model.flow.load_state_dict(fs, strict=True, assign=True)
        model.initialization_report = dict(source=cfg['baseline_checkpoint'], qwen_tensors=len(qs), flow_tensors=len(fs),
            new_parameters=['latent_query'], discarded_prefixes=['lam.', 'vlm_to_lam.', 'flow.cfg_embeddings'],
            strict_retained_load=True, latent_shape=[16,2048], uses_dino=False)
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

    def encode(self,batch):
        inputs=self.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
        hidden,valid,am,fm,positions,visual_mask,deepstack=self.current_embeddings(inputs)
        b,length,_=hidden.shape
        hidden=torch.cat((hidden,self.latent_query[None].expand(b,-1,-1).to(hidden.dtype)),1)
        appended=positions.amax((0,2))[:,None]+1+torch.arange(16,device=hidden.device)[None]
        positions=torch.cat((positions,appended[None].expand(3,-1,-1)),-1)
        visual_mask=F.pad(visual_mask,(0,16),value=False)
        attn=latent_attention_mask(valid,dtype=hidden.dtype)
        lm=self.qwen.language_model
        rotary=lm.rotary_emb(hidden,positions[-3:])
        cache_position=torch.arange(hidden.shape[1],device=hidden.device)
        for i,block in enumerate(lm.layers):
            hidden=block(hidden,attention_mask=attn,position_ids=positions[0],cache_position=cache_position,
                         position_embeddings=rotary,use_cache=False)
            if i<len(deepstack):hidden=lm._deepstack_process(hidden,visual_mask,deepstack[i])
        hidden=lm.norm(hidden)
        current,current_valid=remove_old_queries(hidden[:,:length],valid,am,fm)
        memory=torch.cat((current,hidden[:,length:]),1)
        memory_valid=F.pad(current_valid,(0,16),value=True)
        return self.flow.conditions(memory,memory_valid)

    def forward(self,batch,predict=False,noise=None):
        conditions=self.encode(batch)
        if predict:
            b=len(batch['instruction'])
            actions=self.flow.sample(conditions,torch.full((b,),20.,device=self.device),
                torch.full((b,),25,device=self.device,dtype=torch.long),noise=noise,steps=10)
            return self.normalizer.denormalize(actions)
        loss=self.flow.loss(batch['actions'].to(self.device),batch['actions_mask'].to(self.device),
            batch['action_hz'].to(self.device),batch['embodiment_id'].to(self.device),
            batch['remaining_actions'].to(self.device),conditions,repeats=self.cfg['repeated_diffusion_steps'])
        return dict(loss=loss,flow=loss)

    @torch.no_grad()
    def predict_action(self,main,wrist,instructions,noise=None):
        self.eval()
        if main.shape[1:]!=(3,256,256) or wrist.shape!=main.shape or main.dtype!=torch.uint8 or wrist.dtype!=torch.uint8:
            raise ValueError('Expected uint8 RGB [B,3,256,256] current images')
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            return self(dict(current_main=main,current_wrist=wrist,instruction=instructions),predict=True,noise=noise)

    def payload(self,state,step,extra):
        return dict(format=FORMAT,mode='student',config=self.cfg,step=step,
                    qwen_config=self.qwen.config.to_dict(),statistics=self.statistics,state_dict=state,extra=extra)

    @classmethod
    def load(cls,path):
        path=Path(path)
        payload=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=False)
        if payload['format']!=FORMAT:raise ValueError('Expected single-latent no-DINO A0 checkpoint')
        qc=Qwen3VLConfig.from_dict(payload['qwen_config'])
        qc._attn_implementation='sdpa';qc.text_config._attn_implementation='sdpa';qc.vision_config._attn_implementation='sdpa'
        prev=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16);qwen=Qwen3VLModel(qc)
            model=cls(payload['config'],qwen,AutoProcessor.from_pretrained(path/'processor',local_files_only=True),payload['statistics'])
        finally:torch.set_default_dtype(prev)
        model.load_state_dict(payload['state_dict'],strict=True,assign=True)
        # Match FSDP BF16 buffers, including nonpersistent rotary frequencies.
        model.bfloat16()
        model.source_step=payload['step'];model.source_extra=payload['extra']
        return model
