"""Bounded parallel scene checks using unchanged per-cell preparation and rollout code."""
import argparse,json,sys,time,subprocess,concurrent.futures
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from stage_III.robotwin.common import atomic_json,read_json,sha256,load_config,FORMAT,require_execute
from stage_III.robotwin.runtime import child_env
from stage_III.robotwin.make_manifests import validate_manifest,assert_disjoint

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--candidates",required=True)
 p.add_argument("--split",choices=["train","val","test"],required=True);p.add_argument("--output",required=True)
 p.add_argument("--limit-cells",type=int,required=True);p.add_argument("--scenes-per-cell",type=int,required=True)
 p.add_argument("--workers",type=int,default=3);p.add_argument("--gpu",required=True);p.add_argument("--execute",action="store_true")
 a=p.parse_args();require_execute(a);assert 1<=a.workers<=4
 cfg=load_config(a.config);candidate=read_json(a.candidates);out=Path(a.output).resolve()
 rows=[r for r in candidate["rows"] if r["split"]==a.split]
 cells=list(dict.fromkeys((r["task"],r["mode"]) for r in rows))[:a.limit_cells]
 manifest_files={};errors=[]
 for task,mode in cells:
  cell_dir=out.parent/"parallel_cells"/(task+"__"+mode);cell_dir.mkdir(parents=True,exist_ok=True)
  cell_candidate=dict(candidate,rows=[r for r in candidate["rows"] if (r["task"],r["mode"])==(task,mode)])
  cp=cell_dir/"candidates.json"
  if cp.exists():assert read_json(cp)==cell_candidate,"Per-cell candidates changed"
  else:atomic_json(cp,cell_candidate)
  manifest_files[task,mode]=cell_dir/("prepared_"+a.split+".json")
 def merge():
  accepted=[];events=[]
  for cell in cells:
   file=manifest_files[cell]
   if file.exists():
    m=read_json(file);assert m["split"]==a.split
    accepted.extend(m["rows"]);events.extend(m["events"])
  from collections import Counter
  counts=Counter((r["task"],r["mode"]) for r in accepted)
  complete=all(counts[c]==a.scenes_per_cell for c in cells)
  value=dict(format=FORMAT,kind="prepared_scenes",prepared=True,split=a.split,
    candidate_sha256=sha256(a.candidates),requested_cells=cells,quota_per_cell=a.scenes_per_cell,
    complete=complete,rows=accepted,events=events,updated_at=time.time(),preparation_workers=a.workers)
  # Verify measured rows came from the declared candidates, and reject duplicates.
  expected={r["scene_id"]:r for r in candidate["rows"]};physical=set()
  for row in accepted:
   assert all(row[k]==expected[row["scene_id"]][k] for k in ("task","mode","split","reset_seed"))
   key=(row["task"],row["initial_state_hash"])
   assert key not in physical,"Duplicate actual scene across cells"
   physical.add(key)
  for other in out.parent.glob("prepared_*.json"):
   if other==out:continue
   for row in read_json(other).get("rows",[]):
    assert (row["task"],row["initial_state_hash"]) not in physical,"Scene leaks across splits"
  if complete:validate_manifest(value,a.split)
  atomic_json(out,value)
  return value
 def cell_check(cell):
  file=manifest_files[cell]
  if file.exists() and read_json(file).get("complete"):
   validate_manifest(read_json(file),a.split);return cell,0
  cell_dir=file.parent
  cmd=[cfg["runtime"]["model_python"],"-m","stage_III.robotwin.prepare_scenes",
       "--config",str(Path(a.config).resolve()),"--candidates",str(cell_dir/"candidates.json"),
       "--split",a.split,"--output",str(file),"--limit-cells","1",
       "--scenes-per-cell",str(a.scenes_per_cell),"--gpu",str(a.gpu),"--execute"]
  with (cell_dir/(a.split+".log")).open("ab") as log:
   code=subprocess.run(cmd,cwd=ROOT,env=child_env(cfg,a.gpu,cell_dir),stdout=log,stderr=subprocess.STDOUT).returncode
  return cell,code
 merge()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(cell_check,c):c for c in cells}
  pending=set(futures)
  while pending:
   done,pending=concurrent.futures.wait(pending,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    cell,code=f.result()
    if code:errors.append(dict(cell=cell,returncode=code))
    print("已检查任务模式：",cell,"返回码",code,flush=True)
   value=merge()
 value=merge();atomic_json(out.with_suffix(".parallel_status.json"),dict(complete=value["complete"],errors=errors,workers=a.workers))
 if errors or not value["complete"]:raise SystemExit("场景覆盖未完成，保持停止，不进入训练")
 print("场景准备完成：",a.split,len(value["rows"]),flush=True)
if __name__=="__main__":main()
