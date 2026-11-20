import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingHead, build_time_grid


class HierarchicalFlow(ConditionalFlowMatchingHead):
    """Original flow weights retain their keys; new roles are separate 1024-D memory."""
    def __init__(self, config, role_scale=.01):
        if config.use_state or config.num_target_vision_tokens > 0:
            raise ValueError('Goal/Local flow has no state or old future tokens')
        super().__init__(config)
        d = config.hidden_dim
        self.plan = nn.Parameter(torch.randn(4, d)*.02)
        self.goal_proj, self.local_proj = nn.Linear(config.vlm_dim, d), nn.Linear(config.vlm_dim, d)
        self.goal_attn = nn.ModuleDict({str(i): nn.MultiheadAttention(d, config.attention_heads, batch_first=True) for i in range(0, config.num_layers, 4)})
        self.local_attn = nn.ModuleList([nn.MultiheadAttention(d, config.attention_heads, batch_first=True) for _ in range(config.num_layers)])
        self.goal_scale = nn.Parameter(torch.full((config.num_layers,), role_scale))
        self.local_scale = nn.Parameter(torch.full((config.num_layers,), role_scale))
        self.checkpoint_blocks = False

    def conditions(self, current, current_valid, dino, goal, local, role_valid=None):
        b = current.shape[0]
        base = torch.cat((dino, self.enc_vlm(current)), 1)
        vision = torch.ones(b, dino.shape[1], dtype=torch.bool, device=current.device)
        zeros = torch.zeros_like(current_valid)
        image_mask = torch.cat((vision, zeros), 1)
        vlm_mask = torch.cat((~vision, current_valid.bool()), 1)
        if role_valid is None:
            role_valid = torch.ones(b, 2, device=current.device, dtype=torch.bool)
        return base, image_mask, vlm_mask, self.goal_proj(goal), self.local_proj(local), role_valid

    def velocity(self, x, timestep, hz, embodiment, token_valid, conditions, bypass_roles=False):
        base, image_mask, vlm_mask, goal, local, role_valid = conditions
        grid, hz_valid, _ = build_time_grid(self.config.horizon_sec, hz, x.shape[1])
        if torch.any(token_valid & ~hz_valid):
            raise ValueError('Action validity exceeds the frequency window')
        action = self.action_encoder(x, embodiment)*token_valid[..., None]
        pos = self.time_encoder(grid).to(action.dtype)*token_valid[..., None]
        b = x.shape[0]
        prefix = 0 if bypass_roles else 4
        if prefix:
            hidden = torch.cat((self.plan[None].expand(b, -1, -1), action), 1)
            valid = torch.cat((torch.ones(b, 4, dtype=torch.bool, device=x.device), token_valid), 1)
            pos = torch.cat((pos.new_zeros(b, 4, pos.shape[-1]), pos), 1)
        else:
            hidden, valid = action, token_valid
        dit = self.DiT
        temb = dit.timestep_encoder(timestep)
        for i, block in enumerate(dit.transformer_blocks):
            def run(h, temb, base, goal, local, i=i, block=block):
                if not bypass_roles:
                    p, a = h[:, :4], h[:, 4:]
                    if str(i) in self.goal_attn:
                        p = p + self.goal_scale[i]*role_valid[:, 0, None, None]*self.goal_attn[str(i)](p, goal, goal, need_weights=False)[0]
                    a = a + self.local_scale[i]*role_valid[:, 1, None, None]*self.local_attn[i](a, local, local, need_weights=False)[0]
                    h = torch.cat((p, a), 1)
                return block(h, attention_mask=valid if i%2 else None,
                    encoder_hidden_states=None if i%2 else base,
                    encoder_attention_mask=None if i%2 else (vlm_mask if i%4==0 else image_mask),
                    temb=temb, hidden_positional_embeddings=pos)
            if self.checkpoint_blocks and self.training:
                hidden = checkpoint(run, hidden, temb, base, goal, local, use_reentrant=False)
            else:
                hidden = run(hidden, temb, base, goal, local)
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, -1)
        hidden = dit.proj_out_2(dit.norm_out(hidden)*(1+scale[:,None])+shift[:,None])
        return self.action_decoder(hidden[:, prefix:], embodiment)*token_valid[...,None]

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
