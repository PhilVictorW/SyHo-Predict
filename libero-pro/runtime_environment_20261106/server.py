"""Reuse the historical student forward pass for the two approved checkpoints."""
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers.initialization import no_init_weights
from goal_local import eval_server
from goal_local.eval_common import atomic_json

CHECKPOINTS = {
    ROOT / 'libero-plus/runs/stage_II_20260925/step_004000',
    ROOT / 'runs/goal_local/stage_II_warm250_20260921/step_004000',
}

class Engine(eval_server.Engine):
    def load(self, path):
        p = Path(path).resolve()
        if p not in CHECKPOINTS or not (p / 'checkpoint_complete.json').is_file():
            raise ValueError('Checkpoint is not one of the two approved students')
        if self.checkpoint == str(p):
            return self.info()
        if self.model is not None:
            raise ValueError('A server is fixed to one student')
        started = time.monotonic()
        # Strict loading overwrites parameters; real rotary buffers are retained.
        with no_init_weights():
            self.model = eval_server.GoalLocalPolicy.load(p)
        self.model = self.model.to('cuda:0').eval().requires_grad_(False)
        if self.model.mode != 'student' or self.model.source_step != 4000 or self.model.cfg['flow']['use_state']:
            raise ValueError('Expected step-4000 student with no state input')
        self.checkpoint = str(p)
        self.step = self.model.source_step
        self.calls = 0
        gc.collect()
        self.load_seconds = time.monotonic() - started
        info = self.info()
        atomic_json(self.ready, info)
        print(json.dumps(info), flush=True)
        return info

    def info(self):
        value = super().info()
        if self.model is not None:
            value.update(allocated_mib=torch.cuda.memory_allocated()/1024**2,
                         reserved_mib=torch.cuda.memory_reserved()/1024**2,
                         peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                         peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                         load_seconds=getattr(self, 'load_seconds', None))
        return value

    def infer(self, request):
        result = super().infer(request)
        if self.calls <= 3 or self.calls % 100 == 0:
            atomic_json(self.ready, self.info())
        return result

eval_server.Engine = Engine
if __name__ == '__main__':
    eval_server.main()
