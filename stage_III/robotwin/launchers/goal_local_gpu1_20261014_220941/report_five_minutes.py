"""每十分钟的只读报告；同时展示GPU3训练与GPU7场景验证。"""
import argparse,json,time,datetime
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (OSError,ValueError):return {} if default is None else default

def snapshot():
 record=read(HERE.parent/"latest_gpu1.json");run=Path(record["run_dir"]);route=read(HERE/"gpu_route.json");cfg=read(record["config"])
 cells=route.get("task_mode_cells",16);scenes={}
 for split,n in (("train",2),("val",3),("test",3)):
  m=read(run/f"manifests/prepared_{split}.json");counts=Counter((r["task"],r["mode"]) for r in m.get("rows",[]))
  scenes[split]=dict(accepted=len(m.get("rows",[])),target=cells*n,complete_cells=sum(v>=n for v in counts.values()),complete=m.get("complete",False) and len(counts)==cells,events=dict(Counter(x["status"] for x in m.get("events",[]))))
 w=read(run/"workflow_status.json");evaluation=read(run/"paired_validation/progress.json");report=read(run/"paired_validation/report.json");rl=read(run/"rl/status.json")
 active=[]
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/"cmdline").read_bytes().split(bytes([0]))
   if (b"stage_III.robotwin.rollout" in args or any(x.endswith(b"/rollout_timeout_v2.py") for x in args)) and any(str(run).encode() in x for x in args):
    out=Path(args[args.index(b"--output")+1].decode());state=read(out/"progress.json")
    active.append(dict(pid=int(p.name),gpu=args[args.index(b"--gpu")+1].decode(),scene=read(out/"row.json").get("scene_id",out.name),phase=state.get("phase"),seconds_since_progress=round(time.time()-state.get("last_progress_at",time.time()))))
  except (OSError,ValueError):pass
 return dict(time=time.time(),local_time=datetime.datetime.now().isoformat(timespec="seconds"),run=str(run),workflow=w,scenes=scenes,evaluation_progress=evaluation,evaluation_complete=bool(report.get("complete")),evaluation_successes={k:dict(successes=sum(bool(x["success"]) for x in v),episodes=len(v)) for k,v in report.get("arms",{}).items()},admission=read(run/"admission.json").get("passed"),rl=rl,rl_total=cfg["rl"]["iterations"],active_rollouts=active,checks_gpu=route["checks_gpu"],training_gpu=route["training_gpu"])

def main():
 p=argparse.ArgumentParser();p.add_argument("--peek",action="store_true");a=p.parse_args();s=snapshot()
 if not a.peek:
  with (HERE/"five_minute_reports.jsonl").open("a") as f:f.write(json.dumps(s,ensure_ascii=False)+"\n")
 print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
