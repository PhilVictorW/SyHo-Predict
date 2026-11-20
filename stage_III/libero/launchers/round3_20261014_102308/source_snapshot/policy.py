"""Stage-III Student wrapper.

The base Student is loaded through the unchanged ``goal_local.model`` loader.
Only this isolated subclass supplies the optional external multiplier around
the already-trained Stage-II RelevanceGate residual.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from goal_local.model import GoalLocalPolicy
from .gate_actor import GateActor


class Stage3Policy(GoalLocalPolicy):
    """Read-only Stage-II Student plus an inference-time GateActor."""

    @classmethod
    def load_stage2(cls, checkpoint: str | Path, actor: GateActor | None = None,
                    device: str = "cuda:0") -> "Stage3Policy":
        checkpoint = Path(checkpoint).resolve()
        base = GoalLocalPolicy.load(checkpoint)
        marker = checkpoint / "checkpoint_complete.json"
        if not marker.exists():
            raise FileNotFoundError(f"missing completed checkpoint marker: {marker}")
        payload = torch.load(checkpoint / "model.pt", map_location="cpu", mmap=True,
                             weights_only=False)
        if payload.get("mode") != "student" or int(payload.get("step", -1)) != 4000:
            raise ValueError("Stage III requires the completed Stage-II step4000 Student")
        base.__class__ = cls
        base.gate_actor = actor if actor is not None else GateActor()
        base._stage3_eta_override = None
        base._stage3_gate_seed = None
        base._stage3_gate_record = None
        base._stage3_base_checkpoint = str(checkpoint)
        base.to(device).eval().requires_grad_(False)
        base.gate_actor.to(device).eval().requires_grad_(False)
        del payload
        return base

    @staticmethod
    def _pool_features(hidden, valid, length):
        local = hidden[:, length + 8:]
        goal = hidden[:, length:length + 8]
        current = hidden[:, :length]
        mask = valid[:, :length].bool().unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        pooled = [local.float().mean(dim=1), goal.float().mean(dim=1),
                  (current.float() * mask).sum(dim=1) / count]
        return torch.cat([F.layer_norm(x, (x.shape[-1],)) for x in pooled], dim=-1).detach()

    def _sample_eta(self, features):
        if self._stage3_eta_override is not None:
            eta = torch.as_tensor(self._stage3_eta_override, device=features.device,
                                  dtype=torch.float32).reshape(-1)
            if eta.numel() == 1:
                eta = eta.expand(features.shape[0])
            if eta.numel() != features.shape[0] or not ((eta >= 0) & (eta <= 2)).all():
                raise ValueError("eta override must be [B] in [0,2]")
            return dict(z=eta / 2.0, eta=eta, old_logp=None, alpha=None, beta=None)
        return self.gate_actor.collect(features, seed=self._stage3_gate_seed)

    def language(self, current, goal, local):
        """Exact Stage-II language loop with one optional eta multiplier."""
        hidden, valid, am, fm, positions, visual_mask, deepstack = current
        b, length, _ = hidden.shape
        hidden = torch.cat((hidden, goal.to(hidden.dtype), local.to(hidden.dtype)), 1)
        start = positions.amax((0, 2)) + 1
        appended = start[:, None] + torch.arange(16, device=hidden.device)[None]
        positions = torch.cat((positions, appended[None].expand(positions.shape[0], -1, -1)), -1)
        visual_mask = F.pad(visual_mask, (0, 16), value=False)
        attn = self._stage3_role_attention_mask(valid, dtype=hidden.dtype)
        lm = self.qwen.language_model
        rotary = lm.rotary_emb(hidden, positions[-3:])
        layer_roles, gates = {}, {}
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        gate_record = None
        for i, block in enumerate(lm.layers):
            def run(h, i=i, block=block):
                nonlocal gate_record
                h = block(h, attention_mask=attn, position_ids=positions[0],
                          cache_position=cache_position, position_embeddings=rotary,
                          use_cache=False)
                if i < len(deepstack):
                    h = lm._deepstack_process(h, visual_mask, deepstack[i])
                if self.mode == 'student' and i == 7:
                    features = self._pool_features(h, valid, length)
                    gate_record = dict(features=features, **self._sample_eta(features))
                if self.mode == 'student' and i >= 8:
                    g, l = h[:, length:length + 8], h[:, length + 8:]
                    updated, relevance = self.gate(g, l, i - 8)
                    if gate_record is None:
                        raise RuntimeError("Stage III gate decision was not created at block 8")
                    eta = gate_record["eta"]
                    delta = updated - l
                    updated = l + delta * eta.to(delta.dtype)[:, None, None]
                    h = torch.cat((h[:, :length + 8], updated), 1)
                return h
            hidden = checkpoint(run, hidden, use_reentrant=False) if self.checkpoint_blocks and self.training else run(hidden)
            if i + 1 in self.cfg['distill_layers']:
                layer_roles[i + 1] = (hidden[:, length:length + 8], hidden[:, length + 8:])
        hidden = lm.norm(hidden)
        layer_roles[16] = (hidden[:, length:length + 8], hidden[:, length + 8:])
        base, base_valid = self._stage3_remove_old_queries(hidden[:, :length], valid, am, fm)
        result = dict(current=base, current_valid=base_valid,
                      goal=hidden[:, length:length + 8], local=hidden[:, length + 8:],
                      layers=layer_roles)
        if gate_record is None:
            # This occurs only for malformed non-student use; keep diagnostics explicit.
            raise RuntimeError("Stage III did not create a GateActor decision")
        result['stage3_gate'] = gate_record
        return result

    @staticmethod
    def _stage3_role_attention_mask(valid, dtype):
        from goal_local.roles import role_attention_mask
        return role_attention_mask(valid, dtype=dtype)

    @staticmethod
    def _stage3_remove_old_queries(hidden, valid, am, fm):
        from goal_local.roles import remove_old_queries
        return remove_old_queries(hidden, valid, am, fm)

    @torch.inference_mode()
    def predict_action_stage3(self, main, wrist, instructions, noise=None,
                              gate_seed: int | None = None, eta_override=None):
        if self.mode != 'student':
            raise ValueError('Stage III requires a student checkpoint')
        if main.shape[1:] != (3, 256, 256) or wrist.shape != main.shape:
            raise ValueError('expected uint8 [B,3,256,256] for both views')
        if main.dtype != torch.uint8 or wrist.dtype != torch.uint8:
            raise ValueError('Stage III expects uint8 RGB images')
        self._stage3_eta_override = eta_override
        self._stage3_gate_seed = gate_seed
        self._stage3_gate_record = None
        try:
            batch = dict(current_main=main, current_wrist=wrist, instruction=instructions)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == 'cuda'):
                encoded = self.encode(batch)
                conditions = self.action_conditions(batch, encoded)
                b = len(instructions)
                actions = self.flow.sample(
                    conditions,
                    torch.full((b,), 20., device=self.device),
                    torch.full((b,), 25, device=self.device, dtype=torch.long),
                    noise=noise,
                )
            gate = encoded['stage3_gate']
            return self.normalizer.denormalize(actions), gate
        finally:
            self._stage3_eta_override = None
            self._stage3_gate_seed = None

