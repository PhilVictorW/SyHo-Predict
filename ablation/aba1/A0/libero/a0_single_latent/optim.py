"""FP32 master AdamW with CPU optimizer offload for a shared training GPU."""
import torch


class MasterAdamW:
    def __init__(self, parameters, lr, weight_decay, offload=True):
        self.parameters = list(parameters)
        self.offload = offload
        # Master weights preserve updates below one BF16 quantization interval.
        self.masters = [torch.nn.Parameter(p.detach().to(device='cpu' if offload else p.device, dtype=torch.float32).clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, betas=(.9,.95), eps=1e-8, weight_decay=weight_decay, foreach=False)

    @property
    def param_groups(self):return self.optimizer.param_groups

    @property
    def state(self):return {p:self.optimizer.state[m] for p,m in zip(self.parameters,self.masters)}

    def zero_grad(self,set_to_none=True):
        for p in self.parameters: p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        for p, master in zip(self.parameters, self.masters):
            master.grad = None if p.grad is None else p.grad.detach().to(master.device, dtype=torch.float32)
            p.grad = None
        self.optimizer.step()
        for p, master in zip(self.parameters, self.masters):
            if master.grad is not None: p.copy_(master.to(p.device, dtype=p.dtype))
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(), masters=[p.detach() for p in self.masters])

    def load_state_dict(self, state):
        if len(state['masters']) != len(self.masters): raise ValueError('Optimizer parameter count mismatch')
        self.optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for master, saved, p in zip(self.masters, state['masters'], self.parameters):
                master.copy_(saved);p.copy_(master.to(p.device,dtype=p.dtype))
