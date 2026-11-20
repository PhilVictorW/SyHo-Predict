"""CPU integration: real Actor/GRPO/save/resume; mocked model service and environment."""
from pathlib import Path
import sys,importlib.util,tempfile,json,copy
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from stage_III.robotwin.common import WORK,atomic_json,read_json,load_checkpoint,sha256
from stage_III.robotwin.gate_actor import GateActor
from stage_III.robotwin.tests.test_core import cache_group
s=importlib.util.spec_from_file_location("direct",Path(__file__).with_name("train_direct_gpu3.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=Path(__file__).parent
record=read_json(b.parent/"latest_gpu1.json");source=Path(record["run_dir"])
cfg=m.load_config(b/"workflow_direct_gpu3.json");cfg["rl"].update(iterations=2,eval_every=1)
loaded={};crash={"enabled":False}
def startup(a,c,out,manifest):
 atomic_json(out/"startup.json",{"kind":"synthetic_test_only"})
def rpc(url,endpoint,request):
 payload=load_checkpoint(request["path"])
 with torch.random.fork_rng(devices=[]):
  actor=GateActor();actor.load_state_dict(payload["actor"])
 loaded["actor"]=actor
 return {"actor_sha256":sha256(request["path"])}
def collect(c,config,row,request,gid,n,gpu,out):
 if crash["enabled"] and gid==4:raise RuntimeError("intentional CPU test interruption")
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(40000+gid)
  g=cache_group(loaded["actor"])
 for t in g:t.update(actor_sha256=request["actor_sha256"],student_sha256=request["student_sha256"],group_id=gid)
 return {"status":"completed","trajectories":g}
with tempfile.TemporaryDirectory(prefix="direct_driver_cpu_",dir=WORK/"runs") as td:
 base=Path(td);student=base/"student";student.write_text("synthetic identity")
 def args(out):return SimpleNamespace(train_manifest=str(source/"manifests/prepared_train.json"),run_dir=str(out),student=str(student),resume=None,config=str(b/"workflow_direct_gpu3.json"),gpu="3")
 with patch.object(m,"verify_startup",startup),patch("stage_III.robotwin.engine.start_server",return_value=(SimpleNamespace(pid=0),"fake",{"pid":0})),patch("stage_III.robotwin.engine.collect",collect),patch("stage_III.robotwin.rpc.call",rpc),patch("stage_III.robotwin.runtime.stop_owned"):
  a=args(base/"continuous");m.run(a,cfg)
  x=load_checkpoint(base/"continuous/checkpoints/actor_0002.pt")
  crash["enabled"]=True;a=args(base/"resumed")
  try:m.run(a,cfg)
  except RuntimeError as e:
   assert str(e)=="intentional CPU test interruption"
  else:raise AssertionError("interruption not exercised")
  assert read_json(base/"resumed/last_checkpoint.json")["iteration"]==1
  crash["enabled"]=False;a.resume=str(base/"resumed/checkpoints/actor_0001.pt");m.run(a,cfg)
  y=load_checkpoint(base/"resumed/checkpoints/actor_0002.pt")
  assert x["total_episodes"]==y["total_episodes"]==32
  assert x["update_count"]==y["update_count"]==2
  assert all(torch.equal(x["actor"][k],y["actor"][k]) for k in x["actor"])
  assert y["admission_sha256"] is None and y["evaluation_policy"]=="deferred"
  assert read_json(base/"resumed/evaluation_requests/iteration_0002.json")["status"]=="pending_held_out_evaluation"
  assert not torch.cuda.is_initialized()
print("PASS: CPU GRPO updates, atomic checkpoints, identical interrupted/resumed Actor, deferred evaluations; no GPU initialized.")
