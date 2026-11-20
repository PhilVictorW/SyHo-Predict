import math
import torch
from torch import nn
from torch.nn import functional as F


def role_attention_mask(current_valid, goal_slots=8, local_slots=8, dtype=torch.float32):
    """Additive SDPA mask: causal C; bidirectional G/L; no implicit role bridge."""
    b, length = current_valid.shape
    n = length + goal_slots + local_slots
    allowed = torch.zeros(b, n, n, dtype=torch.bool, device=current_valid.device)
    allowed[:, :length, :length] = torch.ones(length, length, dtype=torch.bool, device=current_valid.device).tril()
    allowed[:, length:, :length] = True
    allowed[:, length:length+goal_slots, length:length+goal_slots] = True
    allowed[:, length+goal_slots:, length+goal_slots:] = True
    allowed[:, :, :length] &= current_valid[:, None, :].bool()
    # Padding queries need finite attention too; they never become visible keys.
    empty = ~allowed.any(-1)
    allowed |= empty[:, :, None] & torch.eye(n, dtype=torch.bool, device=allowed.device)[None]
    return torch.zeros(b, 1, n, n, dtype=dtype, device=allowed.device).masked_fill(~allowed[:, None], float('-inf'))


def remove_old_queries(hidden, valid, act_mask, flow_mask):
    if torch.any(act_mask & flow_mask) or not torch.all(act_mask.sum(-1) == 8) or not torch.all(flow_mask.sum(-1) == 8):
        raise ValueError('Expected disjoint original 8 action + 8 flow query positions')
    keep = ~(act_mask | flow_mask)
    if not torch.all(valid[~keep]):
        raise ValueError('Original queries cannot occupy padding positions')
    b, n, d = hidden.shape
    return hidden[keep].reshape(b, n-16, d), valid[keep].reshape(b, n-16)


class Resampler(nn.Module):
    def __init__(self, dim=2048, slots=8, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(slots, dim)*.02)
        self.q_norm, self.kv_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, valid):
        # Never run an all-masked softmax. Missing roles are finite, deterministic zeros.
        rows = valid.any(-1)
        out = tokens.new_zeros(tokens.shape[0], self.query.shape[0], self.query.shape[1])
        if rows.any():
            x = self.query[None].expand(int(rows.sum()), -1, -1)
            kv = self.kv_norm(tokens[rows])
            x = x + self.attn(self.q_norm(x), kv, kv, key_padding_mask=~valid[rows], need_weights=False)[0]
            out[rows] = self.norm(x + self.ff(x))
        return out


class FutureResamplers(nn.Module):
    def __init__(self, dim=2048, local_slots=8, goal_slots=8, heads=8):
        super().__init__()
        self.local_slots, self.goal_slots = local_slots, goal_slots
        if local_slots:
            self.local = Resampler(dim, local_slots, heads)
        if goal_slots:
            self.goal = Resampler(dim, goal_slots, heads)
        if local_slots:
            self.view = nn.Parameter(torch.randn(2, dim)*.02)
        if goal_slots:
            self.horizon = nn.Parameter(torch.randn(5, dim)*.02)

    def forward(self, local, goal, local_valid=None, goal_valid=None):
        reference = local if local is not None else goal
        b, _, _, d = reference.shape
        empty = reference.new_empty(b, 0, d)
        lout = gout = empty
        if self.local_slots:
            n = local.shape[2]
            memory = (local + self.view[None, :, None]).flatten(1, 2)
            lout = self.local(memory, local_valid[:, None].expand(-1, 2*n))
        if self.goal_slots:
            memory = (goal + self.horizon[None, :, None]).flatten(1, 2)
            mask = goal_valid[:, :, None].expand(-1, -1, goal.shape[2]).flatten(1)
            gout = self.goal(memory, mask)
        return gout, lout


class RelevanceGate(nn.Module):
    def __init__(self, dim=2048, relevance_dim=256, heads=8, deep_layers=8, scale=.01):
        super().__init__()
        self.norm_g, self.norm_l = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.proj_g, self.proj_l = nn.Linear(dim, relevance_dim), nn.Linear(dim, relevance_dim)
        self.null_key = nn.Parameter(torch.randn(1, relevance_dim)*.02)
        self.messages = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(deep_layers)])
        self.scales = nn.Parameter(torch.full((deep_layers,), scale))

    def forward(self, goal, local, layer):
        g, l = self.norm_g(goal), self.norm_l(local)
        keys = torch.cat((self.null_key[None].expand(g.shape[0], -1, -1), self.proj_g(g)), 1)
        probs = (self.proj_l(l).float() @ keys.float().transpose(-1, -2) / math.sqrt(keys.shape[-1])).softmax(-1)
        relevance = 1-probs[..., :1]
        message = self.messages[layer](l, g, g, need_weights=False)[0]
        return local + self.scales[layer]*relevance.to(message.dtype)*message, relevance


def role_distillation(student, teacher, valid, alpha=1.):
    if not valid.any():
        return student.sum()*0  # Finite and differentiable; don't inspect invalid teacher rows.
    s = F.normalize(student[valid].float(), dim=-1)
    t = F.normalize(teacher[valid].detach().float(), dim=-1)
    return (1-(s*t).sum(-1)).mean() + alpha*((s@s.transpose(-1,-2)-t@t.transpose(-1,-2))**2).mean()
