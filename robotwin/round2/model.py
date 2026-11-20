"""Frozen original policy plus an exactly zero-initialized Goal/Local residual."""
import copy
import torch
from torch import nn
from robotwin.model import RobotwinPolicy, RobotwinFlow
from robotwin.data import FutureDataset
from goal_local.action import HierarchicalFlow
from starVLA.model.framework.vlas.lawam import VLMToLAMQFormer
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid


class Dataset36(FutureDataset):
    def sample(self, episode, t):
        batch = super().sample(episode, t)
        count = min(36, episode['length'] - t)
        raw = torch.from_numpy(self.actions[episode['dataset_from_index']+t:
                                           episode['dataset_from_index']+t+count].copy())
        batch['actions'].zero_(); batch['actions_mask'].zero_()
        batch['actions'][:count, :16] = self.normalizer.normalize(raw)
        batch['actions_mask'][:count, :16] = True
        batch['action_chunk'] = torch.zeros(36, 16)
        batch['action_chunk'][:count] = raw
        return batch

    def audit(self):
        report = super().audit()
        report.update(horizon=36, horizon_seconds=1.2,
                      future_offsets_note='Offsets stay [8,24,40,56,72,88] frames at 30 Hz; independent of prediction length')
        return report


class CompatibleFlow(RobotwinFlow):
    def __init__(self, config, role_scale):
        super().__init__(config, role_scale)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def velocity(self, x, timestep, hz, embodiment, token_valid, conditions, bypass_roles=False):
        # Frozen original head, no Plan tokens. This is also the residual anchor.
        with torch.no_grad():
            reference = HierarchicalFlow.velocity(self, x, timestep, hz, embodiment,
                                                  token_valid, conditions, bypass_roles=True)
        if bypass_roles or (not torch.is_grad_enabled() and self.residual_gate.item() == 0):
            return reference
        augmented = HierarchicalFlow.velocity(self, x, timestep, hz, embodiment,
                                               token_valid, conditions, bypass_roles=False)
        return reference + self.residual_gate.tanh() * (augmented - reference)

    def loss(self, actions, mask, hz, embodiment, remaining, conditions, repeats=2):
        _, _, count = build_time_grid(self.config.horizon_sec, hz, actions.shape[1])
        valid = torch.arange(actions.shape[1], device=actions.device)[None] < torch.minimum(count, remaining)[:, None]
        if not torch.all(count == 36) or not torch.all(mask[:, :, :16] == valid[:, :, None]) or mask[:, :, 16:].any():
            raise ValueError('Expected 36-step, 16-dimensional episode-tail mask')
        def repeat(x): return x.repeat_interleave(repeats, 0)
        actions, mask, hz, embodiment = map(repeat, (actions, mask, hz, embodiment))
        conditions = tuple(repeat(x) for x in conditions)
        noise = self.sample_noise(actions.shape, actions.device, actions.dtype)
        tau = self.sample_time(actions.shape[0], actions.device, actions.dtype)[:, None, None]
        noisy = ((1-tau)*noise + tau*actions)*mask.any(-1)[..., None]
        bucket = (tau[:, 0, 0]*self.config.num_timestep_buckets).long().clamp(0, self.config.num_timestep_buckets-1)
        pred = self.velocity(noisy, bucket, hz, embodiment, mask.any(-1), conditions)
        return ((pred.float()-(actions-noise).float()).square()*mask).sum()/mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, conditions, noise, bypass=False):
        hz = torch.full((len(noise),), 30., device=noise.device)
        embodiment = torch.ones(len(noise), device=noise.device, dtype=torch.long)
        _, valid, count = build_time_grid(self.config.horizon_sec, hz, noise.shape[1])
        assert torch.all(count == 36)
        x = noise.clone()*valid[..., None]
        for i in range(10):
            bucket = torch.full((len(x),), i*100, device=x.device, dtype=torch.long)
            x = (x + self.velocity(x, bucket, hz, embodiment, valid, conditions, bypass_roles=bypass)/10)*valid[..., None]
        return x[:, :36, :16].float()


class CompatibilityTeacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy = RobotwinPolicy.from_baseline(cfg)
        flow_cfg = copy.deepcopy(self.policy.flow.config)
        flow_cfg.horizon_sec = 1.2
        flow = CompatibleFlow(flow_cfg, cfg['role_scale'])
        missing, unexpected = flow.load_state_dict(self.policy.flow.state_dict(), strict=False)
        assert missing == ['residual_gate'] and not unexpected
        self.policy.flow = flow
        self.mapper = VLMToLAMQFormer(vlm_hidden_dim=2048, lam_code_dim=32, num_layers=1,
                                     num_heads=8, ffn_expansion_factor=4., dropout=0.)
        self.decoder = LAMDecoder_v2(context_dim=1024, input_dim=768, num_queries=1, num_layers=12,
                                    num_heads=16, dropout=0., grid_hw=(16,16), train_in_latent=True,
                                    ffn_expansion_factor=4, num_embodiments=32, code_dim=32, last_ln=True)
        state = torch.load(cfg['baseline_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        for module, prefix in [(self.mapper, 'policy_backend.vlm_to_lam.'), (self.decoder, 'policy_backend.lam.decoder.')]:
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}, strict=True, assign=True)
        self.requires_grad_(False)
        prefixes = ('policy.future.', 'policy.flow.plan', 'policy.flow.goal_', 'policy.flow.local_', 'policy.flow.residual_gate')
        for name, param in self.named_parameters():
            if name.startswith(prefixes): param.requires_grad_(True)
        self.policy.checkpoint_blocks = True
        self.flow.checkpoint_blocks = True

    @property
    def flow(self): return self.policy.flow

    def train(self, mode=True):
        super().train(mode)
        self.policy.qwen.eval(); self.policy.dino.eval(); self.mapper.eval(); self.decoder.eval()
        return self

    @torch.no_grad()
    def prepare(self, batch):
        p = self.policy
        inputs = p.encode_current_inputs(batch['current_main'], batch['current_wrist'], batch['instruction'])
        current = p.current_embeddings(inputs)
        hidden = p.qwen.get_input_embeddings()(inputs['input_ids']).clone()
        hidden[current[2]] = p.act_query[None].expand(len(hidden), -1, -1).reshape(-1, 2048).to(hidden.dtype)
        hidden[current[3]] = p.flow_action_query[None].expand(len(hidden), -1, -1).reshape(-1, 2048).to(hidden.dtype)
        p.qwen.rope_deltas = None
        original = p.qwen(inputs_embeds=hidden, attention_mask=inputs['attention_mask'],
                          pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'], use_cache=False).last_hidden_state
        dino = p.current_dino(batch['current_main'])
        future = self.decoder(dino, self.mapper(original[current[2]].reshape(len(hidden), 8, 2048)))[:, 0]
        visual = torch.cat((dino, future), 1)
        base = torch.cat((visual, self.flow.enc_vlm(original)), 1)
        image_mask = torch.cat((torch.ones(visual.shape[:2], device=base.device, dtype=torch.bool), torch.zeros_like(current[1])), 1)
        text_mask = torch.cat((torch.zeros(visual.shape[:2], device=base.device, dtype=torch.bool), current[1]), 1)
        lv, gv = p.encode_future_images(batch)
        return dict(current=current, base=(base,image_mask,text_mask), future=(lv,gv))

    def conditions(self, batch, prepared, donor=None, roles_off=False):
        p = self.policy
        source_batch, source_prepared = (batch, prepared) if donor is None else donor
        goal, local = p.future(*source_prepared['future'], source_batch['local_valid'].to(p.device),
                               source_batch['goal_frame_valid'].to(p.device))
        encoded = p.language(prepared['current'], goal, local)
        valid = torch.stack((batch['goal_frame_valid'].any(-1), batch['local_valid']), -1).to(p.device)
        if roles_off: valid = torch.zeros_like(valid)
        return (*prepared['base'], self.flow.goal_proj(encoded['goal']), self.flow.local_proj(encoded['local']), valid)

    def forward(self, batch):
        prepared = self.prepare(batch)
        return self.flow.loss(batch['actions'].cuda(), batch['actions_mask'].cuda(), batch['action_hz'].cuda(),
                              batch['embodiment_id'].cuda(), batch['remaining_actions'].cuda(),
                              self.conditions(batch, prepared), repeats=2)
