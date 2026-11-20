"""W0 -> one latent[16,2048] -> projection768 + current DINO -> original DiT."""
import copy,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig,AutoModel,AutoProcessor,Qwen3VLConfig,Qwen3VLModel
from ablation.aba1.plus_common.model import GoalLocalPolicy
from ablation.aba1.plus_common.roles import role_attention_mask
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from .action import LatentDinoFlow
from .data import ActionNormalizer
from .config import FORMAT

class A0Policy(nn.Module):
    def __init__(self,cfg,qwen,dino,processor,statistics):
        super().__init__()
        self.cfg,self.mode=copy.deepcopy(cfg),'student'
        self.qwen,self.dino,self.processor=qwen,dino,processor
        self.statistics=statistics;self.normalizer=ActionNormalizer(statistics)
        dim=qwen.config.text_config.hidden_size
        self.act_query=nn.Parameter(torch.randn(8,dim)*.02)
        self.flow_action_query=nn.Parameter(torch.randn(8,dim)*.02)
        self.latent_query=nn.Parameter(torch.randn(16,dim)*.02)
        self.flow=LatentDinoFlow(ConditionalFlowMatchingConfig(**cfg['flow']))
        self.qwen.visual.requires_grad_(False).eval();self.dino.requires_grad_(False).eval()

    @property
    def device(self): return self.act_query.device

    def train(self,mode=True):
        super().train(mode);self.qwen.visual.eval();self.dino.eval();return self

    @staticmethod
    def build_qwen(qc):
        qc._attn_implementation='sdpa';qc.text_config._attn_implementation='sdpa';qc.vision_config._attn_implementation='sdpa'
        previous=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16);return Qwen3VLModel(qc)
        finally:torch.set_default_dtype(previous)

    @classmethod
    def from_baseline(cls,cfg):
        if cfg['architecture']!=FORMAT:raise ValueError('Architecture mismatch')
        state=torch.load(cfg['baseline_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
        def take(prefix):return {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
        qs=take('policy_backend.vlm.model.')
        qc=AutoConfig.from_pretrained(cfg['qwen_path'],local_files_only=True)
        qc.text_config.num_hidden_layers=16;qc.text_config.vocab_size=qs['language_model.embed_tokens.weight'].shape[0]
        qwen=cls.build_qwen(qc);qwen.load_state_dict(qs,strict=True,assign=True)
        dino=AutoModel.from_config(AutoConfig.from_pretrained(cfg['dino_path'],local_files_only=True))
        ds=take('policy_backend.lam.vision_encoder.model.');dino.load_state_dict(ds,strict=True,assign=True)
        processor=AutoProcessor.from_pretrained(cfg['qwen_path'],local_files_only=True)
        processor.tokenizer.add_special_tokens({'additional_special_tokens':['<ACT_PH>']})
        if processor.chat_template is None:processor.chat_template=processor.tokenizer.chat_template
        model=cls(cfg,qwen,dino,processor,json.loads(Path(cfg['statistics_path']).read_text()))
        with torch.no_grad():
            model.act_query.copy_(state['policy_backend.act_query'])
            model.flow_action_query.copy_(state['policy_backend.flow_action_query'])
        fs=take('policy_backend.flow.');fs.pop('cfg_embeddings')
        model.flow.load_state_dict(fs,strict=True)
        model.initialization_report=dict(source=cfg['baseline_checkpoint'],strict_retained_load=True,
            qwen_tensors=len(qs),dino_tensors=len(ds),flow_tensors=len(fs),new_parameters=['latent_query'],
            latent_shape=[16,2048],uses_dino=True,old_queries_input_only=True,
            action_condition='latent16_projected768 + current_main_DINO768',
            discarded=['LaWM encoder/decoder/VQ/state decoder','vlm_to_lam','unused flow.cfg_embeddings'],
            teacher=False,distillation=False,plan_tokens=0,gate=False)
        return model

    # Reuse the audited current-view template and frozen visual feature preprocessing.
    encode_current_inputs=GoalLocalPolicy.encode_current_inputs
    current_embeddings=GoalLocalPolicy.current_embeddings
    current_dino=GoalLocalPolicy.current_dino

    def latent_features(self,current):
        hidden,valid,am,fm,positions,visual_mask,deepstack=current
        b,length,_=hidden.shape
        hidden=torch.cat((hidden,self.latent_query[None].expand(b,-1,-1).to(hidden.dtype)),1)
        appended=positions.amax((0,2))[:,None]+1+torch.arange(16,device=hidden.device)[None]
        positions=torch.cat((positions,appended[None].expand(3,-1,-1)),-1)
        visual_mask=F.pad(visual_mask,(0,16),value=False)
        mask=role_attention_mask(valid,goal_slots=16,local_slots=0,dtype=hidden.dtype)
        lm=self.qwen.language_model;rotary=lm.rotary_emb(hidden,positions[-3:])
        cache_position=torch.arange(hidden.shape[1],device=hidden.device)
        for i,block in enumerate(lm.layers):
            hidden=block(hidden,attention_mask=mask,position_ids=positions[0],cache_position=cache_position,
                position_embeddings=rotary,use_cache=False)
            if i<len(deepstack):hidden=lm._deepstack_process(hidden,visual_mask,deepstack[i])
        # Neither old query output nor current-context output is exported to the action head.
        return lm.norm(hidden)[:,length:]

    def encode(self,batch):
        inputs=self.encode_current_inputs(batch['current_main'],batch['current_wrist'],batch['instruction'])
        latent=self.latent_features(self.current_embeddings(inputs))
        return self.flow.conditions(latent,self.current_dino(batch['current_main']))

    def forward(self,batch,predict=False,noise=None):
        conditions=self.encode(batch)
        if predict:
            b=len(batch['instruction'])
            actions=self.flow.sample(conditions,torch.full((b,),20.,device=self.device),
                torch.full((b,),25,device=self.device,dtype=torch.long),noise=noise,steps=10)
            return self.normalizer.denormalize(actions)
        loss=self.flow.loss(batch['actions'].to(self.device),batch['actions_mask'].to(self.device),
            batch['action_hz'].to(self.device),batch['embodiment_id'].to(self.device),
            batch['remaining_actions'].to(self.device),conditions,repeats=self.cfg['repeated_diffusion_steps'],reduction='none')
        return dict(loss=loss.mean(),flow=loss.mean(),per_example_flow=loss)

    @torch.no_grad()
    def predict_action(self,main,wrist,instructions,noise=None):
        self.eval()
        if main.shape[1:]!=(3,256,256) or wrist.shape!=main.shape or main.dtype!=torch.uint8 or wrist.dtype!=torch.uint8:
            raise ValueError('Expected current uint8 RGB dual views')
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            return self(dict(current_main=main,current_wrist=wrist,instruction=instructions),predict=True,noise=noise)

    def payload(self,state,step,extra):
        return dict(format=FORMAT,mode='student',config=self.cfg,step=step,qwen_config=self.qwen.config.to_dict(),
            dino_config=self.dino.config.to_dict(),statistics=self.statistics,state_dict=state,extra=extra)

    @classmethod
    def load(cls,path):
        path=Path(path);payload=torch.load(path/'model.pt',map_location='cpu',mmap=True,weights_only=False)
        if payload['format']!=FORMAT:raise ValueError('Expected single-latent + DINO A0 checkpoint')
        qwen=cls.build_qwen(Qwen3VLConfig.from_dict(payload['qwen_config']))
        dc=dict(payload['dino_config']);kind=dc.pop('model_type')
        dino=AutoModel.from_config(AutoConfig.for_model(kind,**dc))
        model=cls(payload['config'],qwen,dino,AutoProcessor.from_pretrained(path/'processor',local_files_only=True),payload['statistics'])
        model.load_state_dict(payload['state_dict'],strict=True,assign=True)
        # Match FSDP BF16 compute, including nonpersistent rotary buffers.
        model.to(dtype=torch.bfloat16)
        model.source_step=payload['step'];model.source_extra=payload['extra'];return model
