"""Bounded parallel scene checks using unchanged per-cell preparation and rollout code."""
import argparse,json,sys,time,subprocess,concurrent.futures
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from stage_III.robotwin.common import atomic_json,read_json,sha256,load_config,FORMAT,require_execute
from stage_III.robotwin.runtime import child_env
from stage_III.robotwin.make_manifests import validate_manifest,assert_disjoint

def extend_candidates(original,split,target,stride):
 rows=[r for r in original["rows"] if r["split"]==split]
 assert rows and len({(r["task"],r["mode"]) for r in rows})==1
 assert len(rows)<target<stride//3
 first=rows[0];prefix=first["scene_id"].rsplit("__",1)[0]
 assert all(r["reset_seed"]==first["reset_seed"]+i and r["scene_id"]==f"{prefix}__{i:04d}" for i,r in enumerate(rows))
 extra=[dict(first,scene_id=f"{prefix}__{i:04d}",reset_seed=first["reset_seed"]+i) for i in range(len(rows),target)]
 result=dict(original,rows=original["rows"]+extra,
  candidate_extension=dict(split=split,original_count=len(rows),expanded_count=target,
   rule="Continue original disjoint split seed range; select by expert/reset validity only"))
 seeds=[r["reset_seed"] for r in result["rows"]]
 assert len(seeds)==len(set(seeds))
 return result

def inherit_manifest(previous,source_manifest,source_candidates,extended_candidates):
 assert previous["candidate_sha256"]==sha256(source_candidates)
 old={r["scene_id"]:r for r in read_json(source_candidates)["rows"]}
 new={r["scene_id"]:r for r in read_json(extended_candidates)["rows"]}
 assert all(new[k]==v for k,v in old.items()),"Candidate extension changed an original scene"
 for r in previous["rows"]:
  assert r["prepared"] and all(r[k]==new[r["scene_id"]][k] for k in ("task","mode","split","reset_seed"))
 inherited=dict(previous,candidate_sha256=sha256(extended_candidates),
  candidate_extension_provenance=dict(source_manifest=str(source_manifest),
   source_manifest_sha256=sha256(source_manifest),source_candidates=str(source_candidates),
   source_candidates_sha256=sha256(source_candidates)))
 assert inherited["rows"]==previous["rows"] and inherited["events"]==previous["events"]
 return inherited

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
  manifest_files[task,mode]=(cell_dir/("prepared_"+a.split+".json"),cp)
  # Existing certificates remain intact; only a separately recorded candidate pool grows.
  for size in (80,200):
   extra_cp=cell_dir/(f"candidates_{a.split}_pool{size:04d}.json")
   extra_out=cell_dir/(f"prepared_{a.split}_pool{size:04d}.json")
   if extra_out.exists():
    extra=read_json(extra_cp);old=read_json(extra_out)
    assert old["candidate_sha256"]==sha256(extra_cp)
    assert extra==extend_candidates(cell_candidate,a.split,size,cfg["scenes"]["seed_stride"])
    manifest_files[task,mode]=(extra_out,extra_cp)
 def merge():
  accepted=[];events=[];extensions=[];expected={}
  for cell in cells:
   file,cp=manifest_files[cell]
   declared=read_json(cp)
   expected.update({r["scene_id"]:r for r in declared["rows"]})
   if cp.name!="candidates.json":
    extensions.append(dict(cell=cell,path=str(cp),sha256=sha256(cp),
      inheritance=str(file.with_suffix(".inheritance.json"))))
   if file.exists():
    m=read_json(file);assert m["split"]==a.split
    accepted.extend(m["rows"]);events.extend(m["events"])
  from collections import Counter
  counts=Counter((r["task"],r["mode"]) for r in accepted)
  complete=all(counts[c]==a.scenes_per_cell for c in cells)
  value=dict(format=FORMAT,kind="prepared_scenes",prepared=True,split=a.split,
    candidate_sha256=sha256(a.candidates),requested_cells=cells,quota_per_cell=a.scenes_per_cell,
    complete=complete,rows=accepted,events=events,updated_at=time.time(),preparation_workers=a.workers,
    candidate_extensions=extensions)
  # Verify measured rows came from the declared candidates, and reject duplicates.
  physical=set()
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
  while True:
   file,cp=manifest_files[cell]
   if file.exists() and read_json(file).get("complete"):
    validate_manifest(read_json(file),a.split);return cell,0
   cell_dir=file.parent
   cmd=[cfg["runtime"]["model_python"],"-m","stage_III.robotwin.prepare_scenes",
        "--config",str(Path(a.config).resolve()),"--candidates",str(cp),
        "--split",a.split,"--output",str(file),"--limit-cells","1",
        "--scenes-per-cell",str(a.scenes_per_cell),"--gpu",str(a.gpu),"--execute"]
   with (cell_dir/(a.split+".log")).open("ab") as log:
    code=subprocess.run(cmd,cwd=ROOT,env=child_env(cfg,a.gpu,cell_dir),stdout=log,stderr=subprocess.STDOUT).returncode
   if code:return cell,code
   previous=read_json(file)
   if previous.get("complete"):
    validate_manifest(previous,a.split);return cell,0
   declared=read_json(cp)
   pool=[r for r in declared["rows"] if r["split"]==a.split]
   attempted={e["scene_id"] for e in previous["events"]}
   if not all(r["scene_id"] in attempted for r in pool):
    raise ValueError("Preparation incomplete before its candidate pool was exhausted")
   sizes=[n for n in (80,200) if n>len(pool)]
   if not sizes:return cell,2
   size=sizes[0]
   original=read_json(cell_dir/"candidates.json")
   extra=extend_candidates(original,a.split,size,cfg["scenes"]["seed_stride"])
   extra_cp=cell_dir/(f"candidates_{a.split}_pool{size:04d}.json")
   extra_out=cell_dir/(f"prepared_{a.split}_pool{size:04d}.json")
   if extra_cp.exists():assert read_json(extra_cp)==extra
   else:atomic_json(extra_cp,extra)
   inherited=inherit_manifest(previous,file,cp,extra_cp)
   if extra_out.exists():raise ValueError("Unexpected existing candidate continuation")
   atomic_json(extra_out.with_suffix(".inheritance.json"),inherited["candidate_extension_provenance"])
   atomic_json(extra_out,inherited)
   # One atomic dictionary assignment publishes the matching source/manifest pair.
   manifest_files[cell]=(extra_out,extra_cp)
   print("候选不足，保留真实验收记录并扩充候选：",cell,a.split,len(pool),"→",size,flush=True)

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
