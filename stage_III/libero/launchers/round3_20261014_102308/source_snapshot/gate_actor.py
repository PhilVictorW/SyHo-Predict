"""Stage-III GateActor and GRPO numerical core.

The actor samples z in (0, 1) and maps it to eta=2*z.  The Student and the
environment are deliberately absent from the optimizer in this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Beta, kl_divergence


class GateActor(nn.Module):
    """6144 -> 256 -> 2 Beta policy, one sample per action chunk."""

    def __init__(self, hidden_dim: int = 256, initial_concentration: float = 20.0,
                 parameter_eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0 or initial_concentration <= 1.0 + parameter_eps:
            raise ValueError("invalid GateActor initialization")
        self.parameter_eps = float(parameter_eps)
        self.trunk = nn.Sequential(nn.Linear(6144, hidden_dim), nn.SiLU())
        self.head = nn.Linear(hidden_dim, 2)
        value = initial_concentration - 1.0 - parameter_eps
        bias = value + math.log(-math.expm1(-value))
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, bias)

    def distribution(self, features: Tensor) -> Beta:
        if features.ndim != 2 or features.shape[-1] != 6144:
            raise ValueError(f"GateActor expects [N,6144], got {tuple(features.shape)}")
        h = self.head(self.trunk(features.detach().float()))
        concentration = 1.0 + F.softplus(h) + self.parameter_eps
        if not torch.isfinite(concentration).all():
            raise FloatingPointError("non-finite GateActor Beta parameters")
        return Beta(concentration[:, 0], concentration[:, 1], validate_args=True)

    @torch.no_grad()
    def collect(self, features: Tensor, seed: int | None = None) -> dict[str, Tensor]:
        dist = self.distribution(features)
        if seed is None:
            z = dist.sample()
        else:
            # Beta.sample() has no generator argument.  Isolate the RNG state so
            # action-noise sampling and GateActor sampling stay independent.
            device = features.device
            devices = [device.index] if device.type == "cuda" and device.index is not None else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                z = dist.sample()
        if not ((z > 0.0) & (z < 1.0)).all():
            raise FloatingPointError("Beta sample reached an endpoint; refusing to clamp")
        logp = dist.log_prob(z)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("non-finite GateActor old_logp")
        return {
            "z": z.detach(),
            "eta": (2.0 * z).detach(),
            "old_logp": logp.detach(),
            "alpha": dist.concentration1.detach(),
            "beta": dist.concentration0.detach(),
        }

    @torch.no_grad()
    def deterministic_eta(self, features: Tensor) -> Tensor:
        return 2.0 * self.distribution(features).mean


def success_efficiency_reward(success: bool, steps: int, max_steps: int,
                              efficiency_weight: float = 0.1) -> float:
    if max_steps <= 0 or not 0 <= steps <= max_steps:
        raise ValueError(f"invalid executed steps={steps}, max_steps={max_steps}")
    if not 0.0 <= efficiency_weight < 1.0:
        raise ValueError("efficiency_weight must be in [0,1)")
    return float(bool(success)) * (1.0 + efficiency_weight * (1.0 - steps / max_steps))


@torch.no_grad()
def group_advantages(rewards: Tensor, std_skip: float = 1e-6,
                     std_floor: float = 1e-2) -> tuple[Tensor, Tensor, Tensor]:
    """Population-standardized group-relative advantages."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must be [groups, trajectories] with >=2 trajectories")
    centered = rewards.detach().float() - rewards.detach().float().mean(dim=1, keepdim=True)
    std = centered.square().mean(dim=1).sqrt()
    active = std > std_skip
    advantages = centered / std.clamp_min(std_floor).unsqueeze(1)
    advantages = torch.where(active[:, None], advantages, torch.zeros_like(advantages))
    return advantages, std, active


@dataclass
class FlatGateBatch:
    features: Tensor
    z: Tensor
    old_logp: Tensor
    old_alpha: Tensor
    old_beta: Tensor
    advantage: Tensor
    trajectory_id: Tensor
    lengths: Tensor

    def validate(self) -> None:
        n = int(self.features.shape[0])
        if self.features.shape != (n, 6144) or n == 0:
            raise ValueError("invalid GateActor feature cache")
        for value in (self.z, self.old_logp, self.old_alpha, self.old_beta,
                      self.advantage, self.trajectory_id):
            if value.shape != (n,):
                raise ValueError("per-decision cache fields must be [N]")
        if self.lengths.ndim != 1 or self.lengths.numel() == 0 or (self.lengths <= 0).any():
            raise ValueError("lengths must be positive")
        if self.trajectory_id.dtype != torch.long:
            raise ValueError("trajectory_id must be int64")
        if (self.trajectory_id < 0).any() or (self.trajectory_id >= self.lengths.numel()).any():
            raise ValueError("invalid trajectory_id")
        counts = torch.bincount(self.trajectory_id, minlength=self.lengths.numel())
        if not torch.equal(counts, self.lengths.to(counts)):
            raise ValueError("trajectory cache length mismatch")
        if not ((self.z > 0) & (self.z < 1)).all():
            raise ValueError("cache must contain z, never eta")
        for value in (self.features, self.z, self.old_logp, self.old_alpha,
                      self.old_beta, self.advantage):
            if not torch.isfinite(value).all():
                raise FloatingPointError("non-finite cached Stage-III data")
        for tid in range(self.lengths.numel()):
            a = self.advantage[self.trajectory_id == tid]
            if not torch.allclose(a, a[:1].expand_as(a)):
                raise ValueError("trajectory advantage must be repeated per decision")

    def weights(self) -> Tensor:
        # Equal trajectory weight, then equal decision weight within trajectory.
        return 1.0 / (self.lengths.numel() * self.lengths[self.trajectory_id].float())


def clipped_gate_loss(actor: GateActor, batch: FlatGateBatch,
                      clip_eps: float = 0.2) -> tuple[Tensor, dict[str, float]]:
    batch.validate()
    dist = actor.distribution(batch.features.detach())
    new_logp = dist.log_prob(batch.z.detach())
    log_ratio = new_logp - batch.old_logp.detach()
    if not torch.isfinite(log_ratio).all() or (log_ratio.abs() > 60).any():
        raise FloatingPointError("invalid Stage-III policy/data version")
    ratio = log_ratio.exp()
    advantage = batch.advantage.detach()
    objective = torch.minimum(
        ratio * advantage,
        ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage,
    )
    weights = batch.weights()
    loss = -(weights * objective).sum()
    with torch.no_grad():
        old_dist = Beta(batch.old_alpha.detach(), batch.old_beta.detach(), validate_args=True)
        kl = kl_divergence(old_dist, dist)
        metrics = {
            "loss": float(loss.detach()),
            "kl_old_new": float((weights * kl).sum()),
            "clip_fraction": float((weights * ((ratio - 1).abs() > clip_eps)).sum()),
            "ratio_max_error": float((ratio - 1).abs().max()),
            "actor_eta_mean": float((weights * (2.0 * dist.mean)).sum()),
            "actor_entropy": float((weights * dist.entropy()).sum()),
        }
    return loss, metrics

