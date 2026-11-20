"""Frozen-reference velocity retention along its Euler sampling trajectory."""
import torch
from torch.func import functional_call
from robotwin.round5.model import BalancedTeacher, FutureActionDataset, COMPONENTS
from starVLA.model.framework.vlas.flowmatching_expert import build_time_grid


class RetainedTeacher(BalancedTeacher):
    def __init__(self, cfg):
        super().__init__(cfg, component_weights=(1., 1., 1.))
        self.anchor_seed = 600000
        # A plain dictionary is deliberate: fixed reference tensors are not
        # optimizer parameters, registered submodules, or saved adapter tensors.
        self.__dict__['_reference_parameters'] = None

    def train(self, mode=True):
        super().train(mode)
        # The original frozen DiT defaults to dropout=0.2. Independent masks
        # break the centered residual and immutable-reference equality. Keep its
        # inference behavior in BOTH arms, including checkpoint recomputation.
        # flow.training stays True, so activation checkpointing still works.
        self.flow.DiT.eval()
        return self

    def initialize_reference(self):
        refs = {n:p.detach().clone() for n,p in self.named_parameters()
                if p.requires_grad and n.startswith(('policy.future.', 'policy.flow.'))}
        assert refs and all(not p.requires_grad for p in refs.values())
        self.__dict__['_reference_parameters'] = refs
        self.__dict__['_reference_versions'] = {n:p._version for n,p in refs.items()}

    def check_reference(self):
        refs = self.__dict__['_reference_parameters']
        assert refs and all(not p.requires_grad and p.grad is None and
                            p._version == self.__dict__['_reference_versions'][n]
                            for n,p in refs.items())

    def forward(self, batch, route='train', prepared=None, noise=None, time_index=None):
        if route == 'reference_sample':
            assert not torch.is_grad_enabled()
            return self.flow.sample(self.conditions(batch, prepared), noise)
        if route == 'reference_trace':
            # Called only under no_grad with source adapter tensors substituted.
            assert not torch.is_grad_enabled()
            cond = self.conditions(batch, prepared)
            hz = batch['action_hz'].cuda()
            emb = batch['embodiment_id'].cuda()
            _, valid, count = build_time_grid(self.flow.config.horizon_sec, hz, 50)
            assert torch.all(count == 36)
            x = noise * valid[..., None]
            for i in range(time_index+1):
                bucket = torch.full((len(x),), i*100, device=x.device, dtype=torch.long)
                velocity = self.flow.velocity(x, bucket, hz, emb, valid, cond)
                if i == time_index:
                    return x.detach(), velocity.detach(), bucket, hz, emb, valid
                x = (x + velocity/10) * valid[..., None]
            raise AssertionError('Invalid sampling time')

        assert route == 'train'
        prep = self.prepare(batch)
        conditions, encoded = self.conditions_encoded(batch, prep)
        flow = self.flow.loss(batch['actions'].cuda(), batch['actions_mask'].cuda(),
            batch['action_hz'].cuda(), batch['embodiment_id'].cuda(),
            batch['remaining_actions'].cuda(), conditions, repeats=2)
        aux = self.auxiliary_loss(batch, encoded)
        refs = self.__dict__['_reference_parameters']
        if refs is None:
            raise RuntimeError('Initialize immutable reference after source load, before training')
        # Reference sampling must not alter the action-loss RNG stream or leave
        # detached autocast weights cached for the live branch.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad(), \
                torch.autocast('cuda', enabled=torch.is_autocast_enabled(),
                               dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
            torch.manual_seed(self.anchor_seed)
            noise = torch.randn(len(batch['actions']), 50, 32, device='cuda')
            index = int(torch.randint(10, ()).item())
            trace = functional_call(self, refs, (batch,),
                dict(route='reference_trace', prepared=prep, noise=noise, time_index=index), strict=False)
        x, target, bucket, hz, emb, valid = trace
        predicted = self.flow.velocity(x, bucket, hz, emb, valid, conditions)
        mask = batch['actions_mask'].cuda()
        keep = ((predicted.float()-target.float()).square()*mask).sum()/mask.sum().clamp_min(1)
        self.check_reference()
        return dict(flow=flow, keep=keep, anchor_time=index, **aux)
