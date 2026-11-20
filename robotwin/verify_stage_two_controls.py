"""Regression checks for nonfinite updates and the episode-progress watchdog."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import subprocess
import torch
from .stage_two_utils import finite_clip,write_eval_sidecars
from goal_local.roles import role_distillation
from examples.Robotwin.eval_files import batched_eval_runner as runner

def main():
    p=torch.nn.Parameter(torch.ones(4,dtype=torch.bfloat16));p.grad=torch.full_like(p,2.)
    norm=finite_clip([p],1.);assert float(norm)==4. and torch.allclose(p.grad.float(),torch.full((4,),.5),atol=.01)
    p.grad=torch.tensor([float('nan'),2,3,4],dtype=torch.bfloat16);before=p.grad.clone()
    assert not torch.isfinite(finite_clip([p],1.));assert torch.equal(p.grad[1:],before[1:])
    s=torch.randn(1,8,2048,requires_grad=True);t=torch.full_like(s,float('nan'))
    loss=role_distillation(s,t,torch.tensor([False]));loss.backward();assert loss==0 and torch.isfinite(s.grad).all()
    class Task:
        pid=42
        def wait(self,timeout):raise subprocess.TimeoutExpired('fake',timeout)
    task=Task()
    with patch.object(runner.time,'time',side_effect=[100.,2401.]),patch.object(runner,'read_task_progress',return_value=None),patch.object(runner,'heartbeat_claimed_task'),patch.object(runner,'terminate_process') as terminate:
        result=runner.wait_for_task_completion(task,run_dir='unused',task_name='task',worker_index=0,heartbeat_sec=30,stall_timeout_sec=2400,attempt_started_at=0.)
        assert result.stalled and result.return_code==124;terminate.assert_called_once_with(task)
    # An episode completion resets the inactivity deadline.
    with patch.object(runner.time,'time',side_effect=[2300.,2401.,4701.]),patch.object(runner,'read_task_progress',side_effect=[{'completed_episodes':1},{'completed_episodes':1},{'completed_episodes':1}]),patch.object(runner,'heartbeat_claimed_task'),patch.object(runner,'terminate_process'):
        result=runner.wait_for_task_completion(task,run_dir='unused',task_name='task',worker_index=0,heartbeat_sec=30,stall_timeout_sec=2400,attempt_started_at=0.)
        assert result.stalled
    report=dict(fp32_norm_and_clip=True,nonfinite_not_applied=True,missing_future_distillation_finite=True,stall_timeout_seconds=2400,episode_progress_resets_deadline=True)
    dest=Path(__file__).resolve().parent/'verification/stage_II_controls.json';dest.write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
