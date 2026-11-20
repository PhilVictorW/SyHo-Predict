"""Fixed, centered Goal/Local residual with a component-balanced auxiliary loss."""
import copy
import torch
from goal_local.action import HierarchicalFlow
from robotwin.round3.model import TerminalHoldFlow
from robotwin.round4.model import GroundedTeacher, FutureActionDataset

COMPONENTS = ('aux_position', 'aux_rotation', 'aux_gripper')


class CenteredFlow(TerminalHoldFlow):
    role_multiplier = 3.

    def velocity(self, x, timestep, hz, embodiment, token_valid, conditions, bypass_roles=False):
        assert not self.plan.requires_grad and not self.residual_gate.requires_grad
        # Off-path gradients are zero for all permitted trainable flow parameters.
        with torch.no_grad():
            reference = HierarchicalFlow.velocity(self, x, timestep, hz, embodiment,
                                                  token_valid, conditions, bypass_roles=True)
            if bypass_roles or self.role_multiplier == 0:
                return reference
            off_cond = (*conditions[:-1], torch.zeros_like(conditions[-1]))
            # The off path executes trainable role projections under no_grad.
            # Do not cache their detached autocast weights for the grad-enabled path.
            with torch.autocast('cuda', enabled=torch.is_autocast_enabled(),
                                dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
                off = HierarchicalFlow.velocity(self, x, timestep, hz, embodiment,
                                                token_valid, off_cond, bypass_roles=False)
        augmented = HierarchicalFlow.velocity(self, x, timestep, hz, embodiment,
                                              token_valid, conditions, bypass_roles=False)
        gate = self.residual_gate.tanh() * self.role_multiplier
        # Preserve the verified audit's arithmetic order.
        return reference + gate * (augmented-reference) - gate * (off-reference)


class BalancedTeacher(GroundedTeacher):
    def __init__(self, cfg, component_weights=(1., 1., 1.)):
        super().__init__(cfg, auxiliary=True)
        with torch.random.fork_rng(devices=[]):
            flow = CenteredFlow(copy.deepcopy(self.flow.config), cfg['role_scale'])
        flow.load_state_dict(self.flow.state_dict(), strict=True)
        flags = {n:p.requires_grad for n,p in self.flow.named_parameters()}
        for n,p in flow.named_parameters():
            p.requires_grad_(flags[n] and n not in ('plan', 'residual_gate'))
        flow.checkpoint_blocks = self.flow.checkpoint_blocks
        self.policy.flow = flow
        self.component_weights = tuple(component_weights)
        for n,p in flow.named_parameters():
            if p.requires_grad:
                assert n.startswith(('goal_', 'local_')), n

    def auxiliary_loss(self, batch, encoded):
        result = super().auxiliary_loss(batch, encoded)
        result['aux_unweighted'] = result['aux']
        result['aux'] = sum(w * result[k] for w,k in zip(self.component_weights, COMPONENTS))/3
        return result


def balanced_weights(means):
    values = torch.tensor([means[k] for k in COMPONENTS], dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('Calibration losses must be finite and nonnegative')
    inv = values.clamp_min(1e-6).reciprocal()
    return (3*inv/inv.sum()).tolist()
