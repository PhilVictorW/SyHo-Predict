"""GPU forward-only admission diagnostics; never executed by check.sh."""
import argparse
from .common import (DEFAULT_CONFIG,FORMAT,atomic_json,configure_gpu,load_config,local_output,
                     require_execute,sha256,ROOT)

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--student",required=True)
    p.add_argument("--output",required=True);p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    a=p.parse_args();require_execute(a);configure_gpu(a.gpu);cfg=load_config(a.config)
    import numpy as np
    import torch
    from .base_loader import load_student
    from .common import read_json
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    model,payload=load_student(cfg,a.student)
    fixture=np.load(ROOT/"robotwin/round10/fixture.npz")
    info=read_json(ROOT/"robotwin/round10/fixture.json")
    image=lambda x:torch.from_numpy(x.copy()).permute(2,0,1)
    obs=dict(current_main=image(fixture["main"])[None],
             current_wrist=torch.stack([image(fixture[k]) for k in ("left","right")])[None],
             instruction=[info["instruction"]])
    results={};outputs={};roles={}
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
        torch.manual_seed(290001)
        noise=torch.randn(1,50,32,device="cuda",dtype=torch.bfloat16).float()
        prep=model.prepare(obs)
        old_cond=model.conditions(prep)
        old_actions=model.flow.sample(old_cond,noise)
        for eta in cfg["acceptance"]["sensitivity_fixed_etas"]:
            cond,enc,_=model.encode_conditions(obs,prepared=prep,eta=eta)
            outputs[eta]=model.flow.sample(cond,noise).float()
            roles[eta]=(enc["goal"].float(),enc["local"].float())
        ident=float((old_actions-outputs[1.]).abs().max())
        ref=model.flow.sample(model.encode_conditions(obs,prepared=prep,eta=1.)[0],noise)
        repeat=float((ref-outputs[1.]).abs().max())
        action_delta=float((outputs[.8]-outputs[1.2]).abs().max())
        local_delta=float((roles[.8][1]-roles[1.2][1]).abs().max())
        goal_delta=float((roles[.8][0]-roles[1.2][0]).abs().max())
    nonzero=bool(model.gate.scales.count_nonzero())
    all_finite=all(torch.isfinite(v).all().item() for v in outputs.values())
    passed=nonzero and all_finite and max(ident,repeat,goal_delta)<=cfg["acceptance"]["identity_atol"] and min(action_delta,local_delta)>cfg["acceptance"]["sensitivity_min_abs"]
    result=dict(format=FORMAT,kind="interaction_sensitivity",student_sha256=sha256(a.student),
                control_site="vlm_goal_local",passed=passed,gate_nonzero=nonzero,
                same_observation_goal_delta=goal_delta,local_delta=local_delta,action_delta=action_delta,
                eta_one_original_max_abs=ident,repeat_max_abs=repeat,
                gate_scales=model.gate.scales.detach().cpu().tolist(),
                scope="fixed diagnostic observation; not online performance evidence")
    atomic_json(local_output(a.output),result);print(result)
    if not passed:raise SystemExit("Interaction sensitivity gate failed")
if __name__=="__main__":main()
