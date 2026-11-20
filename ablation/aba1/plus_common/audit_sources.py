"""Read-only real-weight audit; no policy construction, CUDA, training or rollout."""
import argparse
import json
from pathlib import Path
from .config import ROOT,ABA,load_config,file_sha256,atomic_json,digest_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    if not output.is_relative_to(ABA/"plus_common"):raise ValueError("Audit output must be in plus_common")
    import torch
    torch.set_num_threads(2)
    cfg=load_config(ABA/"A3/libero-plus/config.json")
    paths=dict(baseline=Path(cfg["baseline_checkpoint"]),teacher=Path(cfg["full_teacher"])/"model.pt",
               student_init=Path(cfg["paired_student_init"])/"model.pt")
    teacher=torch.load(paths["teacher"],map_location="cpu",mmap=True,weights_only=False)
    student=torch.load(paths["student_init"],map_location="cpu",mmap=True,weights_only=False)
    if teacher["mode"]!="posterior" or teacher["step"]!=5000 or student["mode"]!="student" or student["step"]!=0:
        raise ValueError("Unexpected historical stages")
    shared={k:v for k,v in teacher["state_dict"].items() if not k.startswith("future.")}
    for key,value in shared.items():
        other=student["state_dict"][key]
        if value.dtype!=other.dtype or not torch.equal(value,other):
            raise ValueError("Historical Teacher/step0 mismatch: "+key)
    extras=set(student["state_dict"])-set(shared)
    if any(k not in ("q_goal","q_local") and not k.startswith("gate.") for k in extras):
        raise ValueError("Unexpected Student additions")
    for name in ("q_goal","q_local"):
        if tuple(student["state_dict"][name].shape)!=(8,2048):raise ValueError("Expected paired8x2048")
    baseline=torch.load(paths["baseline"],map_location="cpu",mmap=True,weights_only=False)
    prefix="policy_backend."
    for key in ("act_query","flow_action_query"):
        if tuple(baseline[prefix+key].shape)!=(8,2048):raise ValueError("Wrong W0 query shape")
    for old in ("vlm.model.","lam.vision_encoder.model.","flow."):
        if not any(k.startswith(prefix+old) for k in baseline):raise ValueError("Missing W0 retained prefix")
    result=dict(status="passed",teacher_step=5000,student_init_step=0,
        paired_shared_tensors_equal=len(shared),teacher_new_future_tensors=sum(k.startswith("future.") for k in teacher["state_dict"]),
        paired_student_extra_keys=sorted(extras),q_goal_shape=[8,2048],q_local_shape=[8,2048],
        real_forward_performed=False,training_started=False,experimental_inference_started=False,sources={})
    for name,path in paths.items():
        print("Hashing",name,str(path),flush=True)
        result["sources"][name]=dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha256(path))
    del baseline,teacher,student,shared
    from .data import IndexedFutureDataset
    statistics=json.loads(Path(cfg["statistics_path"]).read_text())
    samples=[]
    for variant,roles in (("A0",(False,False)),("A1",(True,False)),("A2",(False,True)),("A3",(True,True))):
        ds=IndexedFutureDataset(cfg["dataset_root"],statistics,local=roles[0],goal=roles[1])
        # Fixed first sufficiently long train episode plus its final anchor.
        episode=next(e for e in ds.selected if e["length"]>88)
        for t in (0,episode["length"]-1):
            row=ds.sample(episode,t)
            if row["current_main"].shape!=(3,256,256) or row["current_main"].dtype!=torch.uint8:
                raise ValueError("Invalid current image")
            if not torch.isfinite(row["actions"]).all():raise ValueError("Nonfinite actions")
            if ("local_main" in row)!=roles[0] or ("goal_main" in row)!=roles[1]:raise ValueError("Role leakage")
            samples.append(dict(variant=variant,episode_id=episode["episode_index"],t=t,
                current_sha256=digest_json(row["current_main"][:,0,0].tolist()),
                action_valid_steps=int(row["actions_mask"].any(-1).sum()),
                fields=sorted(row),local_valid=bool(row["local_valid"]) if roles[0] else None,
                goal_valid=row["goal_frame_valid"].tolist() if roles[1] else None))
        ds.close()
    result["real_dataset_samples"]=samples
    atomic_json(output,result);print(json.dumps({k:v for k,v in result.items() if k not in ("real_dataset_samples","paired_student_extra_keys")},indent=2))

if __name__=="__main__":main()
