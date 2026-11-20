import math
import torch
from torch import nn
from torch.nn import functional as F


def latent_attention_mask(current_valid, goal_slots=16, local_slots=0, dtype=torch.float32):
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
