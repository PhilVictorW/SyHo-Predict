"""Supervised Goal->Local preparation. Nothing runs without --execute."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from .common import (DEFAULT_CONFIG, FORMAT, WORK, atomic_json, base_snapshot,
                     canonical_hash, configure_gpu, load_checkpoint, load_config,
                     local_output, require_execute, torch_save_atomic, validate_base_snapshot)

def flow_pair(student, batch, conditions, reference, repeats=2, teacher_conditions=None):
    """One noise/time draw shared by action, retention and teacher screening."""
    import torch
    device = next(student.parameters()).device
    rep = lambda x: x.to(device).repeat_interleave(repeats,0)
    actions, mask, hz, emb = [rep(batch[k]) for k in
                             ("actions","actions_mask","action_hz","embodiment_id")]
    if mask[:,36:].any() or mask[:,:,16:].any() or not mask[:,:36,:16].all():
        raise ValueError("Expected terminal-held 36x16 supervision")
    cond, ref = tuple(rep(x) for x in conditions), tuple(rep(x) for x in reference)
    noise = student.flow.sample_noise(actions.shape,device,actions.dtype)
    tau = student.flow.sample_time(len(actions),device,actions.dtype)[:,None,None]
    valid = mask.any(-1)
    x = ((1-tau)*noise+tau*actions)*valid[...,None]
    bucket = (tau[:,0,0]*1000).long().clamp(0,999)
    v = student.flow.velocity(x,bucket,hz,emb,valid,cond)
    target = actions-noise
    mse = lambda a,b: ((a.float()-b.float()).square()*mask).sum()/mask.sum()
    with torch.no_grad():
        old = student.flow.velocity(x,bucket,hz,emb,valid,ref)
        teacher_loss = None
        if teacher_conditions is not None:
            tc = tuple(rep(y) for y in teacher_conditions)
            tv = student.flow.velocity(x,bucket,hz,emb,valid,tc)
            teacher_loss = mse(tv,target)
    return mse(v,target), mse(v,old), mse(old,target), teacher_loss

def run(args, cfg):
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from goal_local.roles import role_distillation
    from goal_local.optim import MasterAdamW
    from robotwin.round6.model import FutureActionDataset
    from robotwin.round6.train import physical_metrics
    from .base_loader import load_base
    from .runtime import Progress
    import robotwin.data as data_module

    out = local_output(args.run_dir); out.mkdir(parents=True,exist_ok=True)
    from .common import claim_run
    run_lock = claim_run(out)
    if (out/"status.json").exists() and not args.resume:
        raise ValueError("Run exists; explicit --resume checkpoint required")
    s = cfg["interaction"]
    if shutil.disk_usage(out).free < s["minimum_free_gib"]*2**30:
        raise RuntimeError("Insufficient disk reserve for complete interaction checkpoints")
    # Isolate all indexing/decode outputs from historical robotwin experiments.
    data_module.CACHE = WORK/"cache/offline"
    data_module.WORK = WORK
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(s["seed"]); np.random.seed(s["seed"])
    torch.use_deterministic_algorithms(True)
    progress = Progress(out/"status.json")
    progress.report("loading_base",meaningful=True)
    snapshot = base_snapshot(cfg,full_hash=True)
    atomic_json(out/"base_snapshot.json",snapshot)
    student = load_base(cfg)
    frozen = {n:p._version for n,p in student.teacher.named_parameters()}
    frozen_buffers = {n:b.detach().clone() for n,b in student.teacher.named_buffers()}
    def check_frozen():
        if any(p.requires_grad or p.grad is not None or p._version!=frozen[n]
               for n,p in student.teacher.named_parameters()):
            raise RuntimeError("Frozen backbone changed")
        for n,b in student.teacher.named_buffers():
            if not torch.equal(b,frozen_buffers[n]):
                raise RuntimeError("Frozen buffer changed: "+n)
    train = FutureActionDataset(cfg["base"]["dataset"],student.teacher.policy.statistics,
                               "train",seed=s["seed"],val_ratio=s["val_ratio"])
    val = FutureActionDataset(cfg["base"]["dataset"],student.teacher.policy.statistics,
                             "val",seed=s["seed"],val_ratio=s["val_ratio"])
    if {e["episode_index"] for e in train.selected}&{e["episode_index"] for e in val.selected}:
        raise RuntimeError("Episode split overlap")
    rng = np.random.default_rng(s["seed"])
    indices = rng.integers(0,len(train),s["steps"]*s["accumulation"]*s["micro_batch"]).tolist()
    vi = rng.choice(len(val),min(s["validation_samples"],len(val)),replace=False).tolist()
    atomic_json(out/"sampling.json",dict(train_indices=indices,val_indices=vi,
                                      train_episodes=[e["episode_index"] for e in train.selected],
                                      val_episodes=[e["episode_index"] for e in val.selected]))
    vb = [default_collate([val[i]]) for i in vi]
    ac = lambda: torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False)
    def obs(b):
        return {k:b[k] for k in ("current_main","current_wrist","instruction")}
    def losses(b, step, screen=False):
        with ac():
            cond, enc, prepared = student.encode_conditions(obs(b))
            with torch.no_grad():
                ref,_,_ = student.encode_conditions(obs(b),prepared=prepared,baseline=True)
                tc, te = student.posterior(b,prepared)
            action, keep, old_loss, t_loss = flow_pair(student,b,cond,ref,s["flow_repeats"],
                                                      tc if screen else None)
            lv, gv = b["local_valid"].cuda(), b["goal_frame_valid"].any(-1).cuda()
            kd = []
            for role, valid, idx in (("local",lv,1),("goal",gv,0)):
                terms = [role_distillation(enc["layers"][layer][idx],te["layers"][layer][idx],valid)
                         for layer in (4,8,12,16)]
                kd.append(torch.stack(terms).mean())
            ramp = min(1.,step/max(1,s["kd_warmup_steps"]))
            total = action+s["lambda_keep"]*keep+ramp*(s["lambda_local"]*kd[0]+s["lambda_goal"]*kd[1])
        return total,dict(action=action,keep=keep,kd_local=kd[0],kd_goal=kd[1],
                          base_action=old_loss,teacher_action=t_loss)

    # Same forward implementation as Round7 at step0; no optimizer yet.
    student.set_training(False)
    with torch.no_grad(),ac():
        c,_,prep = student.encode_conditions(obs(vb[0]))
        old = student.conditions(prep)
        delta = max(float((x.float()-y.float()).abs().max()) for x,y in zip(c,old))
    if delta > cfg["acceptance"]["identity_atol"]:
        raise RuntimeError("Round10 step0 condition mismatch: "+str(delta))
    params = student.set_training(True,s["variant"])
    if s["variant"]=="query_gate" and sum(p.numel() for p in params.values()) != 135373576:
        raise RuntimeError("Unexpected query/gate trainable parameter count")
    loss,parts = losses(vb[0],1,screen=True)
    # Prove action supervision itself reaches the queries/gate, independently of KD.
    parts["action"].backward()
    grad = {n:float(p.grad.float().norm()) if p.grad is not None else 0. for n,p in params.items()}
    if not all(grad.get(n,0)>0 for n in ("q_goal","q_local")):
        raise RuntimeError("Action supervision cannot reach queries")
    if s["variant"]=="query_gate" and grad.get("gate.scales",0)==0:
        raise RuntimeError("Action supervision cannot open the zero-scale gate")
    check_frozen()
    for p in params.values(): p.grad=None
    # Freeze/eval during teacher quality screening, not a claim of online ability.
    student.set_training(False)
    ratios = []
    with torch.no_grad():
        for b in vb:
            _,m = losses(b,0,screen=True)
            ratios.append((float(m["teacher_action"]),float(m["base_action"])))
    tr = sum(x[0] for x in ratios)/max(sum(x[1] for x in ratios),1e-12)
    atomic_json(out/"preflight.json",dict(step0_max_abs=delta,query_gradient_norms=grad,
                teacher_to_baseline_action_loss_ratio=tr,teacher_screen="offline_only",
                trainable_count=sum(p.numel() for p in params.values()),frozen_passed=True))
    if (s["lambda_local"] or s["lambda_goal"]) and tr > s["teacher_screen_ratio"]:
        raise RuntimeError("Posterior target screen failed; do not enable KD or train automatically")
    params = student.set_training(True,s["variant"])
    optim = MasterAdamW(params.values(),lr=s["learning_rate"],weight_decay=s["weight_decay"],
                        offload=s["optimizer_offload"])
    start,regressions = 0,0
    def evaluate(step):
        student.set_training(False)
        rows=[]
        with torch.no_grad(),torch.random.fork_rng(devices=[0]),ac():
            for i,b in enumerate(vb):
                torch.manual_seed(771000+i)
                noise=torch.randn(1,50,32,device="cuda",dtype=torch.bfloat16).float()
                pred,_=student.predict(obs(b),noise)
                rows.append(physical_metrics(pred,b,student.teacher.policy.statistics))
        means={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        atomic_json(out/("validation_%06d.json"%step),dict(step=step,mean=means,rows=rows))
        return means
    initial = evaluate(0)
    if args.resume:
        saved=load_checkpoint(args.resume)
        validate_base_snapshot(cfg,saved["base_snapshot"])
        if saved["config"]!=cfg or saved["sampling_hash"]!=canonical_hash(dict(train=indices,val=vi)):
            raise ValueError("Resume configuration/sampling mismatch")
        student.load_state_dict(saved["student"],strict=True)
        optim.load_state_dict(saved["optimizer"])
        start=saved["step"]; initial=saved["initial_validation"]; regressions=saved["regressions"]
        torch.set_rng_state(saved["torch_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
    def save(step):
        if shutil.disk_usage(out).free < s["minimum_free_gib"]*2**30:
            raise RuntimeError("Disk reserve exhausted; last complete checkpoint is retained")
        check_frozen()
        path=out/"checkpoints"/("student_%06d.pt"%step)
        payload=dict(format=FORMAT,kind="interaction",control_site="vlm_goal_local",step=step,
            student={k:v.detach().cpu() for k,v in student.state_dict().items()},
            optimizer=optim.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),config=cfg,base_snapshot=snapshot,
            sampling_hash=canonical_hash(dict(train=indices,val=vi)),initial_validation=initial,
            regressions=regressions,variant=s["variant"])
        torch_save_atomic(path,payload)
        atomic_json(out/"last_checkpoint.json",dict(path=str(path),step=step))
    if not args.resume: save(0)
    for step in range(start+1,s["steps"]+1):
        student.set_training(True,s["variant"]);optim.zero_grad()
        values={k:0. for k in ("action","keep","kd_local","kd_goal")}
        for micro in range(s["accumulation"]):
            offset=((step-1)*s["accumulation"]+micro)*s["micro_batch"]
            b=default_collate([train[i] for i in indices[offset:offset+s["micro_batch"]]])
            total,parts=losses(b,step)
            if not torch.isfinite(total): raise FloatingPointError("Nonfinite interaction loss")
            (total/s["accumulation"]).backward()
            for k in values: values[k]+=float(parts[k].detach())/s["accumulation"]
            progress.report("interaction",meaningful=True,step=step-1,micro=micro+1,total=s["steps"])
        norm=torch.nn.utils.clip_grad_norm_(list(params.values()),s["max_grad_norm"])
        if not torch.isfinite(norm): raise FloatingPointError("Nonfinite interaction gradient")
        lr=s["learning_rate"]*min(1.,step/max(s["warmup_steps"],1))
        for g in optim.optimizer.param_groups:g["lr"]=lr
        optim.step();check_frozen()
        row=dict(step=step,**values,lr=lr,grad_norm=float(norm),
                 gate_scales=student.gate.scales.detach().cpu().tolist(),time=time.time())
        with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if step%s["validate_every"]==0 or step==s["steps"]:
            measured=evaluate(step)
            bad=any(measured[k]>s["validation_regression_ratio"]*max(initial[k],1e-8) for k in initial)
            regressions=regressions+1 if bad else 0
        if step%s["save_every"]==0 or step==s["steps"] or regressions>=s["regression_patience"]:
            save(step)
        if regressions>=s["regression_patience"]:
            progress.report("stopped_by_offline_regression",meaningful=True,step=step)
            return
    progress.report("interaction_complete_pending_online_acceptance",meaningful=True,step=s["steps"])

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--config",default=str(DEFAULT_CONFIG));p.add_argument("--run-dir",required=True)
    p.add_argument("--resume");p.add_argument("--gpu");p.add_argument("--execute",action="store_true")
    a=p.parse_args();cfg=load_config(a.config);require_execute(a);configure_gpu(a.gpu)
    try:run(a,cfg)
    except BaseException as e:
        atomic_json(local_output(a.run_dir)/"failure.json",dict(error=repr(e),time=time.time()))
        raise
if __name__=="__main__":main()
