import torch
from torch.nn import functional as F
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingHead,build_time_grid

class LatentFlow(ConditionalFlowMatchingHead):
    def __init__(self,config):
        if config.use_state or config.num_target_vision_tokens>0:raise ValueError('A0 has no state/future tokens')
        super().__init__(config)
        del self.cfg_embeddings  # Removed visual CFG branch, never optimized or saved.

    def conditions(self,memory,valid):
        return self.enc_vlm(memory),valid.bool()

    def velocity(self,x,timestep,hz,embodiment,token_valid,conditions):
        base,base_valid=conditions
        grid,hz_valid,_=build_time_grid(self.config.horizon_sec,hz,x.shape[1])
        if torch.any(token_valid & ~hz_valid):raise ValueError('Invalid action frequency window')
        hidden=self.action_encoder(x,embodiment)*token_valid[...,None]
        pos=self.time_encoder(grid).to(hidden.dtype)*token_valid[...,None]
        dit=self.DiT
        # Equivalent to the inherited TimestepEncoder.forward, but inspect its
        # weight view directly: FSDP flattening removes nested .parameters().
        encoder=dit.timestep_encoder
        projected=encoder.time_proj(timestep.reshape(-1)).to(encoder.timestep_embedder.linear_1.weight.dtype)
        temb=encoder.timestep_embedder(projected).reshape(*timestep.shape,-1)
        for i,block in enumerate(dit.transformer_blocks):
            hidden=block(hidden,attention_mask=token_valid if i%2 else None,
                encoder_hidden_states=None if i%2 else base,
                encoder_attention_mask=None if i%2 else base_valid,
                temb=temb,hidden_positional_embeddings=pos)
        shift,scale=dit.proj_out_1(F.silu(temb)).chunk(2,-1)
        hidden=dit.proj_out_2(dit.norm_out(hidden)*(1+scale[:,None])+shift[:,None])
        return self.action_decoder(hidden,embodiment)*token_valid[...,None]

    def loss(self, actions, mask, hz, embodiment, remaining, conditions, repeats=2):
        _, hz_valid, count = build_time_grid(self.config.horizon_sec, hz, actions.shape[1])
        expected = torch.arange(actions.shape[1], device=actions.device)[None] < torch.minimum(count, remaining)[:,None]
        if not torch.equal(mask.any(-1), expected) or not torch.all(mask[:, :, :7] == expected[:,:,None]) or mask[:,:,7:].any():
            raise ValueError('Action mask must match each episode tail, 7 dimensions and frequency window')
        def repeat(x): return x.repeat_interleave(repeats, 0)
        actions, mask, hz, embodiment = map(repeat, (actions, mask, hz, embodiment))
        conditions = tuple(repeat(x) for x in conditions)
        noise = self.sample_noise(actions.shape, actions.device, actions.dtype)
        tau = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:,None,None]
        noisy = ((1-tau)*noise+tau*actions)*mask.any(-1)[...,None]
        bucket = (tau[:,0,0]*self.config.num_timestep_buckets).long().clamp(0,self.config.num_timestep_buckets-1)
        pred = self.velocity(noisy, bucket, hz, embodiment, mask.any(-1), conditions)
        return (((pred.float()-(actions-noise).float())**2)*mask).sum()/mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, conditions, hz, embodiment, noise=None, steps=10):
        b = hz.shape[0]
        x = conditions[0].new_empty(b, 50, self.config.action_dim).normal_() if noise is None else noise.clone()
        _, valid, _ = build_time_grid(self.config.horizon_sec, hz, 50)
        x *= valid[...,None]
        for i in range(steps):
            bucket = torch.full((b,), min(int(i/steps*self.config.num_timestep_buckets), self.config.num_timestep_buckets-1), device=x.device, dtype=torch.long)
            x = (x + self.velocity(x, bucket, hz, embodiment, valid, conditions)/steps)*valid[...,None]
        return x[:, :8, :7]
