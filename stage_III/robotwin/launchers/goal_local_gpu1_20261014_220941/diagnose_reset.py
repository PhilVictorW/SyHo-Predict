"""只记录同一候选场景的重置差异，不改变训练实现或验收条件。"""
import argparse,json,sys,traceback,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from stage_III.robotwin.common import configure_gpu,load_config,read_json,atomic_json
p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"workflow.json"));p.add_argument("--gpu",required=True);p.add_argument("--row",required=True);p.add_argument("--out",required=True)
a=p.parse_args();configure_gpu(a.gpu)
from stage_III.robotwin import env_adapter
from stage_III.robotwin.runtime import Progress
import numpy as np
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=load_config(a.config);row=read_json(a.row)
original=env_adapter.fingerprint;records=[];images=[]
def capture(env,obs,decimals):
    result=original(env,obs,decimals)
    records.append(result)
    images.append({k:np.asarray(obs["observation"][k]["rgb"]).copy() for k in ("head_camera","left_camera","right_camera")})
    atomic_json(out/f"reset_{len(records)}.json",result)
    print("已记录第",len(records),"次重置：",result["state_hash"],flush=True)
    return result
env_adapter.fingerprint=capture
progress=Progress(out/"progress.json")
try:
    result=env_adapter.probe_scene(cfg,row,out,progress)
    atomic_json(out/"probe_result.json",result)
except BaseException as exc:
    atomic_json(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
finally:
    summary=dict(scene_id=row["scene_id"],snapshots=len(records),time=time.time(),gpu=int(a.gpu))
    if len(records)>=2:
        summary["physical_hash_equal"]=records[0]["state_hash"]==records[1]["state_hash"]
        summary["rgb_hash_equal"]=records[0]["first_rgb_hash"]==records[1]["first_rgb_hash"]
        summary["image_differences"]={}
        for k in images[0]:
            delta=images[0][k].astype(np.float64)-images[1][k].astype(np.float64)
            summary["image_differences"][k]=dict(mean_abs=float(np.abs(delta).mean()),max_abs=float(np.abs(delta).max()),fraction_changed=float((delta!=0).mean()))
    atomic_json(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
