"""Numerical checks and benchmark sidecars for the RoboTwin student."""
import json
from pathlib import Path
import torch

@torch.no_grad()
def finite_clip(parameters, max_norm):
    # Avoid computing the total gradient norm in BF16. Only clip finite gradients.
    norms = [p.grad.detach().float().norm().double() for p in parameters if p.grad is not None]
    norm = torch.stack(norms).norm() if norms else parameters[0].new_tensor(0., dtype=torch.float64)
    if torch.isfinite(norm):
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale.to(p.grad.dtype))
    return norm

def write_eval_sidecars(run, statistics):
    run = Path(run)
    config = dict(framework=dict(name='RobotwinGoalLocalStudent',
        action_model=dict(flow_cfg=dict(use_state=False))),
        datasets=dict(vla_data=dict(data_mix='robotwin_eef_30hz',image_size=[256,256])))
    (run/'config.yaml').write_text(json.dumps(config,indent=2))
    (run/'dataset_statistics.json').write_text(json.dumps(statistics,indent=2))
