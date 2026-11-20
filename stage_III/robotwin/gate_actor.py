"""FP32 Beta Actor, isolated RNG and a shape-preserving Goal->Local gate."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta

class GateActor(nn.Module):
    def __init__(self, hidden_dim=256, concentration=20.0):
        super().__init__()
        if concentration <= 1.000001:
            raise ValueError("concentration must exceed one")
        self.net = nn.Sequential(nn.Linear(6144, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 2))
        value = concentration - 1.0 - 1e-6
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, value + math.log(-math.expm1(-value)))

    def distribution(self, features):
        if features.ndim != 2 or features.shape[-1] != 6144:
            raise ValueError("Expected [B,6144]")
        with torch.autocast(features.device.type, enabled=False):
            h = self.net(features.detach().float())
            ab = 1.0 + F.softplus(h) + 1e-6
            if not torch.isfinite(ab).all():
                raise FloatingPointError("nonfinite Beta parameters")
            return Beta(ab[:, 0], ab[:, 1], validate_args=True)

    @torch.no_grad()
    def decide(self, features, seed=None, deterministic=False):
        with torch.autocast(features.device.type, enabled=False):
            d = self.distribution(features)
            if deterministic:
                z = d.mean
            else:
                if seed is None:
                    raise ValueError("A separate gate seed is required")
                devices = [features.device.index or 0] if features.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(int(seed))
                    if features.is_cuda:
                        torch.cuda.manual_seed(int(seed))
                    z = d.sample()
            if not ((z > 0) & (z < 1)).all():
                raise FloatingPointError("Beta endpoint; refusing to clamp")
            logp = d.log_prob(z)
            if not torch.isfinite(logp).all():
                raise FloatingPointError("nonfinite log probability")
            return dict(features=features.detach().float().clone(), z=z.clone(),
                        eta=(2*z).clone(), old_logp=logp.clone(),
                        old_alpha=d.concentration1.clone(), old_beta=d.concentration0.clone())

def pool_features(hidden, valid, length):
    with torch.autocast(hidden.device.type, enabled=False):
        l = hidden[:, length+8:].float().mean(1)
        g = hidden[:, length:length+8].float().mean(1)
        mask = valid[:, :length, None].float()
        if (mask.sum(1) == 0).any():
            raise ValueError("No current tokens")
        c = (hidden[:, :length].float()*mask).sum(1)/mask.sum(1)
        return torch.cat([F.layer_norm(x, (2048,)) for x in (l, g, c)], -1).detach()

def goal_local_update(module, goal, local, layer, eta=None):
    """Keep the original gate arithmetic at eta=None or exactly one."""
    g, l = module.norm_g(goal), module.norm_l(local)
    keys = torch.cat((module.null_key[None].expand(g.shape[0], -1, -1), module.proj_g(g)), 1)
    probs = (module.proj_l(l).float() @ keys.float().transpose(-1, -2)
             / math.sqrt(keys.shape[-1])).softmax(-1)
    r = 1-probs[..., :1]
    message = module.messages[layer](l, g, g, need_weights=False)[0]
    delta = module.scales[layer] * r.to(message.dtype) * message
    if eta is not None:
        eta = torch.as_tensor(eta, device=delta.device, dtype=torch.float32).reshape(-1)
        if eta.numel() == 1:
            eta = eta.expand(delta.shape[0])
        if eta.shape != (delta.shape[0],) or not torch.isfinite(eta).all() or not ((eta >= 0) & (eta <= 2)).all():
            raise ValueError("eta must be [B] in [0,2]")
        delta = delta * eta.to(delta.dtype)[:, None, None]
    return local+delta, r
