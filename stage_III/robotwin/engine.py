"""Explicit process orchestration shared by evaluation and online learning."""
from __future__ import annotations
import os
import time
from pathlib import Path
from .common import ROOT,atomic_json,canonical_hash,read_json,sha256
from .runtime import spawn,child_env,stop_owned,wait_owned
from .rpc import call

def start_server(cfg,config_path,student,gpu,out,port=None):
    port=cfg["runtime"]["port"] if port is None else port
    run=Path(out)/("server_"+str(time.time_ns()));run.mkdir(parents=True)
    ready=run/"ready.json"
    cmd=[cfg["runtime"]["model_python"],"-m","stage_III.robotwin.server",
         "--config",str(config_path),"--gpu",str(gpu),"--port",str(port),
         "--ready",str(ready),"--execute"]
    if student:cmd+=["--student",str(student)]
    proc=spawn(cmd,run/"server.log",ROOT,child_env(cfg,gpu,run))
    deadline=time.monotonic()+cfg["runtime"]["load_timeout_seconds"]
    try:
        while not ready.exists():
            if proc.poll() is not None:raise RuntimeError("Model server failed; see "+str(run/"server.log"))
            if time.monotonic()>deadline:raise TimeoutError("Model load timeout")
            time.sleep(.5)
        info=read_json(ready)
        if info["pid"]!=proc.pid:raise RuntimeError("Server ready PID mismatch")
        url=f"http://127.0.0.1:{port}"
        if call(url,"/health",{})["student_sha256"]!=info["student_sha256"]:
            raise RuntimeError("Connected to a different model server")
        return proc,url,info
    except BaseException:
        stop_owned(proc);raise

def collect(cfg,config_path,row,request,group,n,gpu,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity=dict(row=row,request={k:v for k,v in request.items() if k!="url"},group=group,n=n)
    identity_hash=canonical_hash(identity)
    marker=out/"collection.json"
    if marker.exists() and read_json(marker)["identity_hash"]!=identity_hash:
        raise RuntimeError("Cached collection belongs to a different scene/behavior policy")
    atomic_json(marker,dict(identity_hash=identity_hash,identity=identity))
    result=out/"result.json"
    if result.exists():
        saved=read_json(result)
        if saved.get("status")=="completed" and len(saved["trajectories"])==n:
            return saved
        if saved.get("status")=="skipped":return saved
        raise RuntimeError("Invalid cached group result")
    atomic_json(out/"row.json",row);atomic_json(out/"request.json",request)
    cmd=[cfg["runtime"]["sim_python"],"-m","stage_III.robotwin.rollout",
         "--config",str(config_path),"--row",str(out/"row.json"),"--request",str(out/"request.json"),
         "--group",str(group),"--trajectories",str(n),"--output",str(out),"--gpu",str(gpu),"--execute"]
    attempts=cfg["runtime"]["max_infrastructure_retries"]+1
    for attempt in range(attempts):
        # Never reuse partial trajectories as if they made a complete group.
        proc=spawn(cmd,out/("rollout_%02d.log"%attempt),cfg["runtime"]["sim_root"],child_env(cfg,gpu,out))
        try:
            wait_owned(proc,out/"progress.json",cfg["runtime"]["stall_seconds"],
                       cfg["runtime"]["hard_trajectory_seconds"]*n)
            value=read_json(result)
            if value["status"]!="completed" or len(value["trajectories"])!=n:
                raise RuntimeError("Missing complete group")
            return value
        except TimeoutError as exc:
            reason=repr(exc)
            break  # User requested skip after 30 minutes, not restart the same stalled task.
        except RuntimeError as exc:
            failure=read_json(out/"failure.json") if (out/"failure.json").exists() else {}
            if failure.get("kind") in ("invalid_policy","model_error"):
                raise RuntimeError("Policy error must not be discarded as an infrastructure skip: "+str(failure)) from exc
            reason=repr(exc)+" "+str(failure)
    value=dict(status="skipped",reason=reason,group=group,trajectories_used=0)
    atomic_json(result,value)
    return value
