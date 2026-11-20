"""Read-only tmux dashboard for A1 LIBERO-Plus; no torch/CUDA initialization."""
import argparse,json,math,os,shutil,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
RUN=WORK/"protocol_plus_v1/seed_2026"

def read_json(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {} if default is None else default

def metrics(stage):
    result=[]
    try:lines=(RUN/stage/"metrics.jsonl").read_text().splitlines()
    except OSError:return result
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:continue
    return result

def alive(pid):
    try:
        p=Path("/proc")/str(pid)
        return b"a1_local.training" in (p/"cmdline").read_bytes() and (p/"stat").read_text().split(") ",1)[1][0]!="Z"
    except (OSError,ValueError):return False

def duration(seconds):
    if seconds is None or not math.isfinite(seconds):return "--"
    minutes=int(max(seconds,0))//60
    days,minutes=divmod(minutes,1440);hours,minutes=divmod(minutes,60)
    return (f"{days}d " if days else "")+f"{hours:02d}h {minutes:02d}m"

def bar(step,total,width):
    ratio=min(1,max(0,step/total));filled=int(ratio*width)
    body="#"*filled
    if 0<ratio<1:body+=">"
    body+="-"*(width-len(body))
    return "["+body+f"] {step:>4}/{total}  {ratio:6.2%}"

def render():
    settings=read_json(WORK/"config.json")["a1_training"]
    teacher_steps=settings["stage_I_steps"];student_steps=settings["stage_II_steps"];gpu=settings["physical_gpus"][0]
    pipeline=read_json(RUN/"pipeline_status.json");stage=pipeline.get("stage","II" if pipeline.get("phase")=="completed" else "I")
    contract=read_json(RUN/stage/"contract.json");launch=read_json(RUN/stage/"launch.json")
    world=contract.get("world",launch.get("plan",{}).get("world_size",6))
    states=[read_json(p) for p in sorted((RUN/stage).glob("status_rank*.json"))]
    workers=[x for x in states if alive(x.get("pid",0))]
    rows={s:metrics(s) for s in ("I","II")}
    train={s:[x for x in rows[s] if x.get("split")=="train"] for s in rows}
    steps={s:max([x.get("step",0) for x in train[s]]+[0]) for s in train}
    phase=pipeline.get("phase","waiting")
    if phase=="training" and states and not workers:phase="WORKERS EXITED"
    width=max(12,min(38,shutil.get_terminal_size((100,24)).columns-38))
    out=[f"A1 LIBERO-Plus | GPU {gpu} | Local-only 16 queries",time.strftime("%Y-%m-%d %H:%M:%S %Z"),
         f"Pipeline: {phase} | Stage {stage} | live workers: {len(workers)}/{world}","",
         "Teacher I  "+bar(steps["I"],teacher_steps,width),"Student II "+bar(steps["II"],student_steps,width),""]
    if states:
        accumulation=contract.get("accumulation_per_rank",[len(range(r,32,world)) for r in range(world)])
        out.append("Microbatches: "+"  ".join(f"r{x.get('rank','?')}={x.get('micro',0)}/{accumulation[x.get('rank',0)]}" for x in states))
        out.append("Workers: "+", ".join(str(x.get("pid")) for x in workers))
    latest=train[stage][-1] if train[stage] else None
    if latest:
        out += [f"Latest completed update: {latest['step']} | loss {latest['loss']:.6f} | flow {latest.get('flow',0):.6f}",
                f"Gradient norm: {latest['grad_norm']:.6f} | LR: {latest['lr_used']:.3g} | effective batch: 32"]
        if "local_distill" in latest:out.append(f"Local distillation loss: {latest['local_distill']:.6f}")
        recent=train[stage][-20:];mean=sum(x["update_seconds"] for x in recent)/len(recent)
        remaining=(teacher_steps if stage=="I" else student_steps)-steps[stage]
        out.append(f"Recent {len(recent)} updates: {mean:.1f} s/update | stage ETA ~{duration(remaining*mean)}")
        out.append(f"Last update: {time.strftime('%H:%M:%S',time.localtime(latest['time']))} ({int(time.time()-latest['time'])} s ago)")
    else:out.append("Waiting for the first optimizer update (loading / validation).")
    val=[x for x in rows[stage] if x.get("split")=="validation"]
    if val:out.append(f"Validation loss: {val[-1]['loss']:.6f} at step {val[-1]['step']}")
    if phase in ("failed","WORKERS EXITED"):
        out.append("ERROR: "+str(pipeline.get("error") or next((x.get("error") for x in states if x.get("error")),"See logs window"))[:180])
    auto=read_json(WORK/"logs/after_training_full_eval_status.json")
    if auto:
        out.append(f"Auto-eval: {auto.get('phase')} | GPU{gpu} x{auto.get('workers',6)} | Student{student_steps}")
        if auto.get("phase") in ("evaluating","completed"):
            out.append("Full eval  "+bar(auto.get("completed",0),10030,width))
    out += ["",f"Teacher: save {teacher_steps} only -> Student: auto-start, save every 500, keep latest 2",
            "Refresh: 2s | Ctrl-b n: next window | Ctrl-b d: detach (training continues)"]
    return "\n".join(out)

def main():
    p=argparse.ArgumentParser();p.add_argument("--once",action="store_true");a=p.parse_args()
    try:
        while True:
            text=render()
            sys.stdout.write(("" if a.once else "\x1b[2J\x1b[H")+text+"\n");sys.stdout.flush()
            if a.once:return
            time.sleep(2)
    except KeyboardInterrupt:pass

if __name__=="__main__":main()
