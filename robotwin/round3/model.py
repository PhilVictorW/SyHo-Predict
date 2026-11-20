import copy
import torch
from robotwin.round2.model import Dataset36, CompatibleFlow, CompatibilityTeacher

class TerminalHoldDataset(Dataset36):
    def sample(self, episode, t):
        b=super().sample(episode,t)
        b['observed_actions_mask']=b['actions_mask'].clone()
        n=min(36,episode['length']-t)
        # Absolute EEF actions: reproduce original dataset endpoint clamping.
        b['action_chunk'][n:]=b['action_chunk'][n-1]
        b['actions'][:36,:16]=self.normalizer.normalize(b['action_chunk'])
        b['actions_mask'][:36,:16]=True
        return b
    def audit(self):
        result=super().audit()
        result.update(terminal_padding='repeat last absolute action',attention_tokens=36,
                      supervision_tokens=36,observed_actions_mask='real steps only; used for reported physical errors')
        return result

class TerminalHoldFlow(CompatibleFlow):
    def loss(self,actions,mask,hz,embodiment,remaining,conditions,repeats=2):
        # Dataset supplies a complete absolute-action window, including terminal holds.
        if torch.any(remaining<1):raise ValueError('Anchor must be inside its episode')
        return super().loss(actions,mask,hz,embodiment,remaining.clamp_min(36),conditions,repeats)

class TerminalHoldTeacher(CompatibilityTeacher):
    def __init__(self,cfg):
        super().__init__(cfg)
        # Preserve initialization RNG and all values; only the loss contract changes.
        with torch.random.fork_rng(devices=[]):
            flow=TerminalHoldFlow(copy.deepcopy(self.flow.config),cfg['role_scale'])
        flow.load_state_dict(self.flow.state_dict(),strict=True)
        old_flags={n:p.requires_grad for n,p in self.flow.named_parameters()}
        for n,p in flow.named_parameters():p.requires_grad_(old_flags[n])
        flow.checkpoint_blocks=self.flow.checkpoint_blocks
        self.policy.flow=flow
