"""Synchronous data parallelism for multiple processes sharing one CUDA device.

Gloo collectives operate ONLY on CPU tensors: no duplicate-device NCCL groups.
Gradients are averaged globally; rank 0 owns FP32 AdamW states and broadcasts
the resulting weights. This is one optimizer/model, not six independent runs.
"""
import torch
import torch.distributed as dist
from goal_local.optim import MasterAdamW


class DistributedMaster:
    def __init__(self, parameters, lr, weight_decay, bucket_elements=4_000_000):
        self.params = list(parameters)
        self.cpu_grads = [None] * len(self.params)
        self.seen = [False] * len(self.params)
        self.hooks = [p.register_post_accumulate_grad_hook(self._offload_hook(i)) for i,p in enumerate(self.params)]
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.master = MasterAdamW(self.params, lr, weight_decay, offload=True) if self.rank == 0 else None
        self.buckets = []
        bucket, used = [], 0
        for index, param in enumerate(self.params):
            for offset in range(0, param.numel(), bucket_elements):
                count = min(bucket_elements, param.numel()-offset)
                if bucket and used+count > bucket_elements:
                    self.buckets.append((bucket, used)); bucket, used = [], 0
                bucket.append((index, offset, count)); used += count
        if bucket:
            self.buckets.append((bucket, used))

    def _offload_hook(self, index):
        def hook(param):
            # Each micro-batch gradient is moved off GPU immediately. Preserve
            # the original parameter dtype during accumulation, as p.grad did.
            grad = param.grad.detach()
            if self.cpu_grads[index] is None:
                self.cpu_grads[index] = torch.empty_like(grad, device="cpu")
            if self.seen[index]:
                self.cpu_grads[index].add_(grad.to("cpu"))
            else:
                self.cpu_grads[index].copy_(grad)
                self.seen[index] = True
            param.grad = None
        return hook

    def zero_grad(self):
        self.seen = [False] * len(self.params)
        for param in self.params:
            param.grad = None
        if self.master:
            self.master.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, lr, max_norm):
        present = torch.tensor(self.seen, dtype=torch.int32)
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        if self.master:
            for m, used in zip(self.master.masters, present):
                m.grad = torch.empty_like(m) if used else None
        for entries, count in self.buckets:
            buffer = torch.zeros(count, dtype=torch.float32)
            pos = 0
            for i, offset, size in entries:
                p = self.params[i]
                if self.seen[i]:
                    buffer[pos:pos+size].copy_(self.cpu_grads[i].reshape(-1)[offset:offset+size])
                pos += size
            dist.all_reduce(buffer)
            if self.master:
                buffer.div_(self.world)
                pos = 0
                for i, offset, size in entries:
                    if present[i]:
                        self.master.masters[i].grad.reshape(-1)[offset:offset+size].copy_(buffer[pos:pos+size])
                    pos += size
        for p in self.params:
            p.grad = None
        norm = torch.zeros(1)
        if self.master:
            norm[0] = torch.nn.utils.clip_grad_norm_(self.master.masters, max_norm, error_if_nonfinite=True, foreach=False)
            for group in self.master.optimizer.param_groups:
                group['lr'] = lr
            self.master.optimizer.step()
            self.master.optimizer.zero_grad(set_to_none=True)
        dist.broadcast(norm, src=0)
        self.broadcast_weights()
        return float(norm[0])

    @torch.no_grad()
    def broadcast_weights(self):
        # FP32 transport preserves each parameter's original dtype exactly.
        for entries, count in self.buckets:
            buffer = torch.empty(count, dtype=torch.float32)
            if self.master:
                pos = 0
                for i, offset, size in entries:
                    buffer[pos:pos+size].copy_(self.master.masters[i].reshape(-1)[offset:offset+size])
                    pos += size
            dist.broadcast(buffer, src=0)
            pos = 0
            for i, offset, size in entries:
                self.params[i].reshape(-1)[offset:offset+size].copy_(buffer[pos:pos+size])
                pos += size

    def state_dict(self):
        if not self.master:
            raise RuntimeError('Only rank zero owns optimizer state')
        return self.master.state_dict()

    def load_state_dict(self, state):
        if self.master:
            self.master.load_state_dict(state)
        self.broadcast_weights()
