"""Observe the current run until the requested number of real optimizer updates are stable."""
import argparse, json, math, os, subprocess, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
def read(p):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return {}
parser=argparse.ArgumentParser(__doc__);parser.add_argument('--stage',choices=['I','II'],default='I');parser.add_argument('--steps',type=int,default=6);args=parser.parse_args()
assert args.steps >= 6
record = read(W/'nohup/current.json')
run=Path(record['run']); folder=run/('stage_'+args.stage); control=W/'nohup'/run.name
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
plan=read(run/'run_plan.json')
devices=plan['physical_gpus'];world=plan['world_size']
def alive(pid):
    try:
        p=Path('/proc')/str(pid)
        return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except OSError: return False
while True:
    now=time.time();pipeline=read(run/'pipeline.json');status=read(folder/'status.json')
    loads=[read(folder/f'initialization_{r}.json') for r in range(world)]
    workers=[read(folder/f'worker_{r}.json') for r in range(world)]
    beats=[read(folder/f'heartbeat_{r}.json') for r in range(world)]
    metrics=[]
    try:
        for line in (folder/'metrics.jsonl').read_text().splitlines():
            try: row=json.loads(line)
            except ValueError: continue
            if row.get('split')=='train': metrics.append(row)
    except OSError: pass
    recent=metrics[-args.steps:]
    snapshot=dict(time=time.strftime('%H:%M:%S'),phase=pipeline.get('phase'),step=status.get('step',0),ranks=[dict(gpu=g,phase=('training' if beats[r] else loads[r].get('phase')),pid=workers[r].get('pid') or loads[r].get('pid'),alive=alive(workers[r].get('pid') or loads[r].get('pid')),step=beats[r].get('step'),micro=beats[r].get('micro'),heartbeat_age=round(now-beats[r]['time'],1) if beats[r].get('time') else None) for r,g in enumerate(devices)],recent_steps=[dict(step=x['step'],loss=x['loss'],grad_norm=x['grad_norm'],seconds=round(x['seconds'],2)) for x in recent])
    print(json.dumps(snapshot),flush=True)
    with (control/'stability_poll.jsonl').open('a') as f: f.write(json.dumps(snapshot)+'\n')
    if boot!=record['boot_id'] or pipeline.get('phase')=='failed' or not alive(record['processes']['training']['pid']) or list(folder.glob('failure_rank_*.json')):
        raise SystemExit('Training stopped or failed; inspect logs')
    stable=(len(recent)==args.steps and [x['step'] for x in recent]==list(range(recent[-1]['step']-args.steps+1,recent[-1]['step']+1)) and all(math.isfinite(x[k]) for x in recent for k in (['loss','flow','grad_norm','seconds','local_distill'] if args.stage=='II' else ['loss','flow','grad_norm','seconds'])) and all(x['grad_norm']>0 for x in recent) and all(x['alive'] and (x['step'] or 0)>recent[-1]['step'] for x in snapshot['ranks']) and now-recent[-1]['time']<max(180,2*max(x['seconds'] for x in recent)))
    if stable:
        assert all(loads[r]['load_started_at']>=loads[r-1]['load_finished_at']+plan['load_gap_seconds'] for r in range(1,world))
        gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        gpu_index={line.split(',')[1].strip():int(line.split(',')[0]) for line in gpu_csv.splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
        bindings={int(line.split(',')[1]):gpu_index[line.split(',')[0].strip()] for line in apps.splitlines()}
        assert [bindings[w['pid']] for w in workers]==devices
        for rank,w in enumerate(workers):
            assert w['logical_gpu']==rank
            env=dict(x.split(b'=',1) for x in (Path('/proc')/str(w['pid'])/'environ').read_bytes().split(b'\0') if b'=' in x)
            assert env[b'CUDA_VISIBLE_DEVICES']==','.join(map(str,devices)).encode() and int(env[b'RANK'])==rank and int(env[b'WORLD_SIZE'])==world
        proof=dict(verified_at=now,run=str(run),passed=True,stage=args.stage,consecutive_completed_optimizer_steps=args.steps,last_completed_step=recent[-1]['step'],metrics=recent,workers=workers,heartbeats=beats,loading_records=loads,gpu_bindings=bindings,boot_id=boot)
        (control/'stability_check.json').write_text(json.dumps(proof,indent=2)+'\n')
        print(f'STABLE: {args.steps} consecutive finite optimizer updates and all ranks continuing; '+str(control/'stability_check.json'),flush=True)
        break
    time.sleep(20)
