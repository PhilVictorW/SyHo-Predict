"""Original DiT alternating latent and current-DINO cross-attention."""
import torch
from torch.nn import functional as F
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingHead, build_time_grid
from ablation.aba1.plus_common.action import HierarchicalFlow

class LatentDinoFlow(ConditionalFlowMatchingHead):
    def __init__(self,config):
        if config.use_state or config.num_target_vision_tokens>0 or config.cfg_drop_prob:
            raise ValueError('No state, future tokens, or CFG in this A0')
        super().__init__(config)
        # This unused CFG embedding is not a DINO encoder and has no gradient path.
        del self.cfg_embeddings

    def conditions(self,latent,dino):
        if latent.shape[1]!=16: raise ValueError('Exactly 16 latent outputs are required')
        projected=self.enc_vlm(latent)
        base=torch.cat((dino.to(projected.dtype),projected),1)
        image_mask=torch.zeros(base.shape[:2],dtype=torch.bool,device=base.device)
        image_mask[:,:dino.shape[1]]=True
        return base,image_mask,~image_mask

    def timestep_embedding(self,timestep):
        # FSDP replaces child Parameters with tensor views during forward.
        # Read the materialized weight dtype, not next(child.parameters()).
        encoder=self.DiT.timestep_encoder
        projected=encoder.time_proj(timestep.reshape(-1)).to(encoder.timestep_embedder.linear_1.weight.dtype)
        return encoder.timestep_embedder(projected).reshape(*timestep.shape,-1)

    def velocity(self,x,timestep,hz,embodiment,token_valid,conditions):
        base,image_mask,latent_mask=conditions
        grid,hz_valid,_=build_time_grid(self.config.horizon_sec,hz,x.shape[1])
        if torch.any(token_valid & ~hz_valid): raise ValueError('Invalid action time window')
        hidden=self.action_encoder(x,embodiment)*token_valid[...,None]
        pos=self.time_encoder(grid).to(hidden.dtype)*token_valid[...,None]
        dit=self.DiT;temb=self.timestep_embedding(timestep)
        for i,block in enumerate(dit.transformer_blocks):
            hidden=block(hidden,attention_mask=token_valid if i%2 else None,
                encoder_hidden_states=None if i%2 else base,
                encoder_attention_mask=None if i%2 else (latent_mask if i%4==0 else image_mask),
                temb=temb,hidden_positional_embeddings=pos)
        shift,scale=dit.proj_out_1(F.silu(temb)).chunk(2,-1)
        hidden=dit.proj_out_2(dit.norm_out(hidden)*(1+scale[:,None])+shift[:,None])
        return self.action_decoder(hidden,embodiment)*token_valid[...,None]

    def loss(self,actions,mask,hz,embodiment,remaining,conditions,repeats=2,reduction='mean'):
        # Preserve the previous micro_batch=1 objective, including episode tails.
        if reduction not in ('mean','none') or repeats<1:raise ValueError('Invalid loss reduction')
        b=actions.shape[0]
        _,_,count=build_time_grid(self.config.horizon_sec,hz,actions.shape[1])
        expected=torch.arange(actions.shape[1],device=actions.device)[None]<torch.minimum(count,remaining)[:,None]
        if not torch.equal(mask.any(-1),expected) or not torch.all(mask[:,:,:7]==expected[:,:,None]) or mask[:,:,7:].any():
            raise ValueError('Invalid action mask')
        def repeat(x):return x.repeat_interleave(repeats,0)
        actions,mask,hz,embodiment=map(repeat,(actions,mask,hz,embodiment))
        conditions=tuple(repeat(x) for x in conditions)
        noise=self.sample_noise(actions.shape,actions.device,actions.dtype)
        tau=self.sample_time(actions.shape[0],actions.device,actions.dtype)[:,None,None]
        noisy=((1-tau)*noise+tau*actions)*mask.any(-1)[...,None]
        bucket=(tau[:,0,0]*self.config.num_timestep_buckets).long().clamp(0,self.config.num_timestep_buckets-1)
        pred=self.velocity(noisy,bucket,hz,embodiment,mask.any(-1),conditions)
        error=((pred.float()-(actions-noise).float()).square()*mask).sum((1,2))
        per_anchor=(error/mask.sum((1,2)).clamp_min(1)).reshape(b,repeats).mean(1)
        return per_anchor if reduction=='none' else per_anchor.mean()

    sample=HierarchicalFlow.sample
