"""Independent FIFO evaluation of every 500-step student, two GPUs, 50x2x10."""
import argparse,csv,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from .train_distributed import atomic_json
from examples.Robotwin.eval_files.robotwin_eval_common import DEFAULT_TASKS,derive_ckpt_alias,build_run_group

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/'examples/Robotwin/eval_files/batched_eval_runner.py'
MODES=('demo_clean','demo_randomized')

def eval_environment(output,gpus='0,5',episodes=10,slots=2,tasks=None):
    env=os.environ.copy()
    for k in ('CUDA_VISIBLE_DEVICES','ROBOTWIN_TASKS','ROBOTWIN_RUN_GROUP','ROBOTWIN_RUN_TAG','ROBOTWIN_RESUME_RUN_DIR','RESUME_RUN_DIR','ROBOTWIN_CKPT_ALIAS'):
        env.pop(k,None)
    env.update(PYTHONPATH=str(ROOT),CUDA_DEVICE_ORDER='PCI_BUS_ID',GPU_IDS=gpus,NUM_WORKERS=str(len(gpus.split(','))),
        STAR_VLA_PYTHON=sys.executable,ROBOTWIN_PYTHON='/home/wangchaoyang/envs/RoboTwin4LaWAM/bin/python',
        ROBOTWIN_PATH='/mnt/hdd/wangchaoyang/results/LaWAM/runtime/RoboTwin',
        OUTPUT_ROOT=str(output),ROBOTWIN_EVAL_ROOT=str(output),ROBOTWIN_NUM_SLOTS=str(slots),ROBOTWIN_TEST_NUM=str(episodes),
        ROBOTWIN_SERVER_MODULE='robotwin.eval_server',ROBOTWIN_DISABLE_LAWM='0',ROBOTWIN_REMOVE_QUERY_TOKENS='0',USE_BF16='1',
        ROBOTWIN_REPLAN_STEPS='8',ROBOTWIN_ACTION_ENSEMBLE='0',ROBOTWIN_ACTION_ENSEMBLE_ALPHA='0',
        ROBOTWIN_SAVE_VIDEO='0',ROBOTWIN_SUCCESS_VIDEO_ONLY='0',ROBOTWIN_RESUME_INCOMPLETE='1',
        ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN='1',ROBOTWIN_TASK_HEARTBEAT_SEC='30',ROBOTWIN_TASK_LEASE_TIMEOUT_SEC='3600',
        ROBOTWIN_TASK_STALL_TIMEOUT_SEC='2400',ROBOTWIN_SERVER_IDLE_TIMEOUT_SEC='-1',SERVER_STARTUP_TIMEOUT_SEC='1800',
        PORT_BASE='24940',PORT_SEARCH_LIMIT='128',LAUNCH_DELAY_SEC='5',SEED='0',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        ROBOTWIN_SHARED_WARP_CACHE_DIR=str(output/'cache/warp'))
    if tasks:env['ROBOTWIN_TASKS']=' '.join(tasks)
    (output/'cache/warp').mkdir(parents=True,exist_ok=True)
    return env

def progress(run_dir):
    records=[]
    for p in sorted((run_dir/'tasks').glob('*/progress.json')):
        try:row=json.loads(p.read_text())
        except (OSError,ValueError):continue
        records.append(dict(task=p.parent.name,**row))
    skipped=[]
    for p in sorted((run_dir/'tasks').glob('*/skipped_summary.json')):
        try:skipped.append(json.loads(p.read_text()))
        except (OSError,ValueError):pass
    successes=sum(int(x.get('successes',0)) for x in records)
    episodes=sum(int(x.get('completed_episodes',0)) for x in records)
    return dict(episodes=episodes,successes=successes,success_rate=successes/episodes if episodes else None,
        completed_tasks=sum(int(x.get('completed_episodes',0))>=int(x.get('target_episodes',10)) for x in records),
        skipped_tasks=len(skipped),tasks=records,skipped=skipped)

def summarize(evaluation):
    reports=[]
    for path in sorted(evaluation.glob('step_*_demo_*.json')):
        if '.retry.' in path.name:continue
        try:reports.append(json.loads(path.read_text()))
        except (OSError,ValueError):pass
    rows=[]
    for report in reports:
        step=int(Path(report['checkpoint']).name.split('_')[1])
        skipped={x['task_name'] for x in report['skipped']}
        by_task={x['task']:x for x in report['tasks']}
        for task in DEFAULT_TASKS:
            record=by_task.get(task,{})
            episodes=int(record.get('completed_episodes',0));successes=int(record.get('successes',0))
            rows.append(dict(step=step,mode=report['mode'],task=task,episodes=episodes,successes=successes,
                success_rate=successes/episodes if episodes else '',
                status='skipped_timeout' if task in skipped else 'completed' if episodes>=10 else 'incomplete'))
    if rows:
        with (evaluation/'per_task_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    elif (evaluation/'per_task_results.csv').exists():
        (evaluation/'per_task_results.csv').unlink()
    atomic_json(evaluation/'summary.json',dict(reports=[{k:v for k,v in x.items() if k not in ('tasks','skipped')} for x in reports],updated_at=time.time()))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--smoke-checkpoint',type=Path);a=p.parse_args();run=a.run.resolve()
    evaluation=run/('smoke_eval' if a.smoke_checkpoint else 'quick_eval');evaluation.mkdir(parents=True,exist_ok=True)
    lock=(evaluation/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert len(DEFAULT_TASKS)==50
    active=None
    def stop(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while True:
            pause_file=evaluation/'paused.json'
            pause=json.loads(pause_file.read_text()) if pause_file.exists() else {}
            if not a.smoke_checkpoint and pause.get('paused',False):
                atomic_json(evaluation/'status.json',dict(phase='paused_by_user',paused=True,
                    reason=pause.get('reason','Online evaluation paused'),updated_at=time.time()))
                time.sleep(10)
                continue
            checkpoints=[a.smoke_checkpoint.resolve()] if a.smoke_checkpoint else [p.parent for p in sorted(run.glob('step_*/checkpoint_complete.json')) if int(p.parent.name.split('_')[1])%500==0]
            cancellation_file=evaluation/'cancelled_steps.json'
            cancellation=json.loads(cancellation_file.read_text()) if cancellation_file.exists() else {}
            cancelled={int(step) for step in cancellation.get('steps',[])}
            work=None
            for ckpt in checkpoints:
                if not a.smoke_checkpoint and int(ckpt.name.split('_')[1]) in cancelled:continue
                for mode in MODES:
                    report=evaluation/f'{ckpt.name}_{mode}.json'
                    if not report.exists():work=(ckpt,mode,report);break
                if work:break
            if work is None:
                state=dict(phase='waiting_for_checkpoint',gpu_ids=[0,5],target_per_checkpoint=1000,stall_timeout_sec=2400,cancelled_steps=sorted(cancelled),updated_at=time.time())
                if a.smoke_checkpoint:state['phase']='smoke_completed'
                else:
                    try:finished=json.loads((run/'status.json').read_text()).get('phase')=='completed'
                    except (OSError,ValueError):finished=False
                    if finished and len(checkpoints)==20:state['phase']='completed'
                atomic_json(evaluation/'status.json',state)
                if a.smoke_checkpoint or state['phase']=='completed':return
                time.sleep(10);continue
            ckpt,mode,report=work;model=ckpt/'model.pt';tag=ckpt.name
            tasks=[DEFAULT_TASKS[0]] if a.smoke_checkpoint else list(DEFAULT_TASKS)
            gpus=('0' if mode=='demo_clean' else '5') if a.smoke_checkpoint else '0,5'
            episodes=1 if a.smoke_checkpoint else 10
            env=eval_environment(evaluation,gpus,episodes,1 if a.smoke_checkpoint else 2,tasks)
            run_dir=evaluation/build_run_group(derive_ckpt_alias(model),mode)/tag
            command=[sys.executable,str(RUNNER),'--mode','master','--ckpt_path',str(model),'--task_config',mode,'--run_tag',tag,'--num_workers',str(len(gpus.split(',')))]
            if (run_dir/'.task_scheduler.json').exists():command+=['--resume_run_dir',str(run_dir)]
            with (evaluation/f'{tag}_{mode}.log').open('a') as log:
                active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while active.poll() is None:
                    atomic_json(evaluation/'status.json',dict(phase='evaluating',checkpoint=str(ckpt),step=int(ckpt.name.split('_')[1]),
                        mode=mode,gpu_ids=[int(x) for x in gpus.split(',')],pid=active.pid,updated_at=time.time(),
                        target_tasks=len(tasks),episodes_per_task=episodes,stall_timeout_sec=2400,progress=progress(run_dir)))
                    time.sleep(10)
                rc=active.returncode;active=None
            result=progress(run_dir)
            # A server/launcher failure is not a completed benchmark: retain diagnostics,
            # retry once. Task timeouts are intentionally skipped by the task scheduler.
            if not result['tasks'] and not result['skipped']:
                retry=evaluation/f'{tag}_{mode}.retry.json'
                if not retry.exists():
                    atomic_json(retry,dict(exit_code=rc,time=time.time()));time.sleep(15);continue
            atomic_json(report,dict(checkpoint=str(ckpt),mode=mode,exit_code=rc,target_tasks=len(tasks),
                target_episodes=len(tasks)*episodes,finished_at=time.time(),**result))
            if not a.smoke_checkpoint:summarize(evaluation)
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL)

if __name__=='__main__':main()
