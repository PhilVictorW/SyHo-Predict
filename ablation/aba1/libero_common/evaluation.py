"""Sequential, resumable 40-task evaluation using the unchanged LIBERO episode core."""
import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .config import ABA, ROOT, atomic_json, file_sha256, load_config, digest_json
from goal_local.eval_client import rpc
from goal_local.eval_common import SUITES, SIM_PYTHON, BASELINE, model_metadata


def episode_rows(path, task, count):
    file=Path(path)/'episodes.checkpoint.jsonl'
    if not file.exists(): return {}
    rows={}
    for line in file.read_text().splitlines():
        if not line.strip(): continue
        # Fail on a truncated record: do not append to a broken JSON line on resume.
        row=json.loads(line)
        index=row['episode_idx']
        if row['task_id']!=task or not 0<=index<count:
            raise ValueError(f'Unexpected evaluation record: {file}')
        if row.get('error'): continue
        if index in rows and rows[index]!=row: raise ValueError('Conflicting episode records')
        rows[index]=row
    return rows


def summarize(output,count,checkpoint_sha):
    tasks=[]
    for suite in SUITES:
        for task in range(10):
            rows=episode_rows(output/'tasks'/f'{suite}__task{task:02d}',task,count)
            n=len(rows); k=sum(bool(row['success']) for row in rows.values())
            steps=[row['num_actions'] for row in rows.values() if row['success']]
            tasks.append(dict(suite=suite,task_id=task,episodes=n,successes=k,
                         success_rate=k/n if n else None,complete=n==count,
                         successful_steps_mean=sum(steps)/len(steps) if steps else None))
    n=sum(t['episodes'] for t in tasks); k=sum(t['successes'] for t in tasks)
    suites={}
    for suite in SUITES:
        group=[t for t in tasks if t['suite']==suite]
        total=sum(t['episodes'] for t in group); passed=sum(t['successes'] for t in group)
        suites[suite]=dict(episodes=total,expected_episodes=10*count,successes=passed,success_rate=passed/total if total else None)
    summary=dict(status='complete' if all(t['complete'] for t in tasks) else 'incomplete',
                 checkpoint_sha256=checkpoint_sha,expected_episodes=40*count,episodes_completed=n,
                 successes=k,success_rate=k/n if n else None,suites=suites,tasks=tasks)
    atomic_json(output/'summary.json',summary)
    with (output/'task_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(tasks[0]));writer.writeheader();writer.writerows(tasks)
    return summary


def stop_child(process):
    if process is None or process.poll() is not None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def main(variant,config,argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--protocol',choices=['quick','full'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--sim-python',default=SIM_PYTHON)
    parser.add_argument('--first-success-video',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    cfg=load_config(config,variant)
    path=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    if not path.is_relative_to((ABA/variant/'libero/runs').resolve()):
        parser.error('Checkpoint must belong to this variant/libero/runs')
    allowed=(ABA/variant/'libero/evaluation').resolve()
    if not output.is_relative_to(allowed) or output==allowed:
        parser.error('Output must be a subdirectory of this variant/libero/evaluation')
    count=10 if args.protocol=='quick' else 50
    protocol=dict(variant=variant,checkpoint=str(path),protocol=args.protocol,
        episodes_per_task=count,expected_episodes=40*count,suites=list(SUITES),task_ids=list(range(10)),
        initial_state_indices=list(range(count)),simulator_seed=0,num_steps_wait=10,
        action_chunk=8,flow_steps=10,max_steps=dict(zip(SUITES,[250,300,320,550])),
        uses_state=False,uses_future=False,teacher_loaded=False,
        noise_seed='(suite_index+1)*10000000+task_id*10000+episode_idx*100+chunk_index',
        first_success_video=args.first_success_video,
        rollout_sha256=file_sha256(ROOT/'examples/LIBERO/eval_files/libero_eval_core.py'))
    if args.dry_run:
        print(json.dumps(dict(protocol,checkpoint_exists=(path/'model.pt').is_file(),starts_processes=False),indent=2));return
    if not (path/'checkpoint_complete.json').is_file(): raise ValueError('Incomplete checkpoint')
    meta=model_metadata(path/'model.pt')
    if meta['mode']!='student' or meta['step']<1 or meta['config']['ablation']!=variant:
        raise ValueError('Expected this ablation’s trained student')
    protocol['checkpoint_sha256']=file_sha256(path/'model.pt')
    protocol['config_sha256']=digest_json(meta['config'])
    protocol['step']=meta['step']; protocol['extra']=meta['extra']
    if not Path(args.sim_python).is_file(): raise FileNotFoundError(args.sim_python)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/'protocol.json'
        if manifest.exists() and json.loads(manifest.read_text())!=protocol:
            raise ValueError('Output belongs to another checkpoint/protocol')
        if not manifest.exists() and (output/'tasks').exists():
            raise ValueError('Unowned existing task results')
        atomic_json(manifest,protocol)
        evaluate(args,variant,path,output,count,protocol)


def evaluate(args,variant,path,output,count,protocol):
    env=os.environ.copy()
    env.update(LIBERO_HOME=str(BASELINE/'runtime/LIBERO'),
               LIBERO_CONFIG_PATH=str(BASELINE/'runtime/libero_config'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
               TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT),env['LIBERO_HOME'],env.get('PYTHONPATH','')])
    ready=output/'server_ready.json'; server=child=None
    def interrupted(signum,frame): raise KeyboardInterrupt(f'Signal {signum}')
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    try:
        summary=summarize(output,count,protocol['checkpoint_sha256'])
        if summary['status']=='complete': return
        with (output/'server.log').open('a') as log:
            server=subprocess.Popen([sys.executable,'-m','ablation.aba1.libero_common.server',
                '--variant',variant,'--device',args.device,'--ready',str(ready)],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+900
            while True:
                if server.poll() is not None: raise RuntimeError('Inference server exited; see server.log')
                info=json.loads(ready.read_text()) if ready.exists() else {}
                if info.get('pid')==server.pid and info.get('port'): break
                if time.monotonic()>deadline: raise TimeoutError('Inference server startup')
                time.sleep(1)
            url=f"http://127.0.0.1:{info['port']}"
            loaded=rpc(url,'/load',dict(checkpoint=str(path)))
            if loaded['model_sha256']!=protocol['checkpoint_sha256'] or loaded['variant']!=variant:
                raise ValueError('Server loaded the wrong checkpoint')
            atomic_json(output/'loaded_policy.json',loaded)
            for suite in SUITES:
                for task in range(10):
                    out=output/'tasks'/f'{suite}__task{task:02d}'
                    if len(episode_rows(out,task,count))==count: continue
                    out.mkdir(parents=True,exist_ok=True)
                    command=[args.sim_python,'-m','goal_local.eval_task','--checkpoint',str(path),
                             '--url',url,'--suite',suite,'--task-id',str(task),'--episodes',str(count),'--output',str(out)]
                    if args.first_success_video: command.append('--first-success-video')
                    atomic_json(output/'status.json',dict(phase='evaluating',suite=suite,task_id=task))
                    with (out/'run.log').open('a') as task_log:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=task_log,stderr=subprocess.STDOUT,start_new_session=True)
                        code=child.wait(timeout=14400); child=None
                    summary=summarize(output,count,protocol['checkpoint_sha256'])
                    if code or len(episode_rows(out,task,count))!=count:
                        raise RuntimeError(f'Incomplete task {suite}/{task}; inspect run.log and resume same command')
            if summary['status']!='complete': raise RuntimeError('Incomplete evaluation')
            atomic_json(output/'status.json',dict(phase='completed'))
            print(json.dumps(summary,indent=2))
    except BaseException as error:
        atomic_json(output/'status.json',dict(phase='interrupted_or_failed',error=repr(error)))
        raise
    finally:
        stop_child(child);stop_child(server)
        for sig,handler in previous.items(): signal.signal(sig,handler)
