"""Complete-group GRPO; only Actor parameters are optimized."""
from __future__ import annotations
import copy
import torch
from torch.distributions import Beta, kl_divergence

def reward(success, steps, limit, efficiency=0.1):
    if limit <= 0 or not 0 <= steps <= limit or not 0 <= efficiency < 1:
        raise ValueError("Invalid reward inputs")
    return float(bool(success))*(1.0+efficiency*(1.0-steps/limit))

def advantages(rewards, floor=.01, skip=1e-6):
    x = torch.as_tensor(rewards, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] < 2 or not torch.isfinite(x).all():
        raise ValueError("Need finite [groups, trajectories]")
    centered = x-x.mean(1, keepdim=True)
    std = centered.square().mean(1).sqrt()
    return torch.where((std>skip)[:, None], centered/std.clamp_min(floor)[:, None],
                       torch.zeros_like(centered)), std

def make_batch(groups, cfg):
    if not groups or any(len(g) != cfg["group_size"] for g in groups):
        raise ValueError("Only complete groups can be trained")
    versions = {r["actor_sha256"] for g in groups for r in g}
    students = {r["student_sha256"] for g in groups for r in g}
    if len(versions) != 1 or len(students) != 1:
        raise ValueError("Mixed behavior/base policy versions")
    for g in groups:
        identities = {(r["scene_id"], r["initial_state_hash"], r["instruction"],
                       r["task"], r["mode"], r["group_id"], r["env_config_hash"], r["max_steps"]) for r in g}
        if len(identities) != 1 or len({r["trajectory_idx"] for r in g}) != len(g):
            raise ValueError("Group initial condition / trajectory identity mismatch")
        if any(r["termination"] not in ("success", "step_limit") or r["policy"] != "actor_sample" for r in g):
            raise ValueError("Infrastructure/invalid-policy errors are not RL failures")
    adv, std = advantages([[r["reward"] for r in g] for g in groups],
                          cfg["std_floor"], cfg["std_skip"])
    rows, weights, aa = [], [], []
    for j, g in enumerate(groups):
        for i, r in enumerate(g):
            decisions = r["decisions"]
            if not decisions or sum(d["executed_steps"] for d in decisions) != r["env_steps"]:
                raise ValueError("Invalid trajectory action accounting")
            if bool(r["success"]) != (r["termination"] == "success") or (
                not r["success"] and r["env_steps"] != r["max_steps"]
            ):
                raise ValueError("Incomplete failure or inconsistent termination")
            position = 0
            for decision in decisions:
                if decision["chunk_start_step"] != position:
                    raise ValueError("Action chunks overlap or leave an execution gap")
                position += decision["executed_steps"]
            expected = reward(r["success"], r["env_steps"], r["max_steps"], cfg["efficiency_weight"])
            if abs(expected-r["reward"]) > 1e-6:
                raise ValueError("Reward is inconsistent with execution")
            for d in decisions:
                if not 1 <= d["executed_steps"] <= 36:
                    raise ValueError("Invalid chunk execution count")
                rows.append(d)
                weights.append(1/(len(groups)*len(g)*len(decisions)))
                aa.append(float(adv[j, i]))
    b = {k: torch.tensor([d[k] for d in rows], dtype=torch.float32)
         for k in ("features", "z", "old_logp", "old_alpha", "old_beta")}
    b["weight"] = torch.tensor(weights)
    b["advantage"] = torch.tensor(aa)
    if b["features"].shape != (len(rows), 6144):
        raise ValueError("Invalid feature cache shape")
    if not ((b["z"]>0)&(b["z"]<1)).all() or any(not torch.isfinite(v).all() for v in b.values()):
        raise ValueError("Invalid probability cache")
    return b, dict(groups=len(groups), trajectories=sum(map(len, groups)),
                   active_groups=int((std>cfg["std_skip"]).sum()), group_std=std.tolist(),
                   decisions=len(rows), env_steps=sum(r["env_steps"] for g in groups for r in g))

def objective(actor, b, cfg):
    with torch.autocast(b["features"].device.type, enabled=False):
        d = actor.distribution(b["features"])
        lr = d.log_prob(b["z"])-b["old_logp"]
        if not torch.isfinite(lr).all() or (lr.abs()>60).any():
            raise FloatingPointError("Invalid behavior/data version")
        ratio = lr.exp()
        a, w = b["advantage"], b["weight"]
        loss = -(w*torch.minimum(ratio*a, ratio.clamp(1-cfg["clip"], 1+cfg["clip"])*a)).sum()
        old = Beta(b["old_alpha"], b["old_beta"])
        m = dict(loss=float(loss.detach()), kl=float((w*kl_divergence(old, d)).sum().detach()),
                 ratio_max_error=float((ratio-1).abs().max().detach()),
                 clip_fraction=float((w*((ratio-1).abs()>cfg["clip"])).sum().detach()),
                 eta_mean=float((w*2*d.mean).sum().detach()))
    return loss, m

def update_actor(actor, optimizer, groups, cfg):
    b, stats = make_batch(groups, cfg)
    device = next(actor.parameters()).device
    b = {k:v.to(device) for k,v in b.items()}
    loss, before = objective(actor, b, cfg)
    if before["ratio_max_error"] > cfg["ratio_tolerance"]:
        raise RuntimeError("Before-update density ratio differs from one: " + str(before))
    if not stats["active_groups"]:
        # Never run Adam on zero advantages: stored momentum could still move it.
        optimizer.zero_grad(set_to_none=True)
        return {**stats, **before, "updated": False, "updates": 0, "grad_norm": 0.}
    old_actor, old_opt = copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())
    updates = 0
    try:
        for _ in range(cfg["max_updates"]):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = objective(actor, b, cfg)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite Actor gradients")
            optimizer.step(); updates += 1
            with torch.no_grad():
                _, after = objective(actor, b, cfg)
            if after["kl"] > cfg["target_kl"]:
                break
        if after["kl"] > cfg["rollback_kl"]:
            raise FloatingPointError("Actor exceeded rollback KL budget")
    except BaseException:
        actor.load_state_dict(old_actor); optimizer.load_state_dict(old_opt)
        raise
    return {**stats, **after, "updated": True, "updates": updates,
            "grad_norm": float(norm), "before_ratio_max_error": before["ratio_max_error"]}
