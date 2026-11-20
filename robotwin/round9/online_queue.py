"""Bounded matched closed-loop pilot, isolated from old stage-II queues."""
import fcntl,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'robotwin/round9';OUT=R/'online'
PY='/home/wangchaoyang/envs/LaWAM/bin/python';SIMPY='/home/wangchaoyang/envs/RoboTwin4LaWAM/bin/python'
SIM='/mnt/hdd/wangchaoyang/results/LaWAM/runtime/RoboTwin'
CKPT='/mnt/hdd/wangchaoyang/CKPT/LaWAM/results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt'
TASKS=['click_bell','handover_block','lift_pot','stack_blocks_two'];MODES=['demo_clean','demo_randomized'];ARMS=['original','compatible']
def read(p):
    try:return json.loads(p.read_text())
    except (ValueError,OSError):return {}
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def stop(p):
    if p is None:return
    # Every child is launched in its own session. Never signal unrelated processes.
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    p.wait()
def listening(port):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=1):return True
    except OSError:return False
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'manifest.json').exists(),'Refusing to overwrite prior pilot'
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('ROBOTWIN_') or k in ('OUTPUT_ROOT','RESUME_RUN_DIR','GPU_IDS'):env.pop(k)
    env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ROBOTWIN_PATH=SIM,ROBOTWIN_NUM_SLOTS='1',ROBOTWIN_TEST_NUM='5',ROBOTWIN_SAVE_VIDEO='0',ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN='1',ROBOTWIN_SHARED_WARP_CACHE_DIR=str(OUT/'cache/warp'))
    (OUT/'cache/warp').mkdir(parents=True,exist_ok=True)
    ports={'original':26490,'compatible':26491}
    assert not any(listening(x) for x in ports.values()),'Pilot ports already occupied'
    jobs=[]
    for ti,task in enumerate(TASKS):
        for mi,mode in enumerate(MODES):
            for arm in ARMS[::1 if (ti+mi)%2==0 else -1]:jobs.append(dict(task=task,mode=mode,arm=arm,key=f'{task}__{mode}__{arm}'))
    manifest=dict(started_at=time.time(),pid=os.getpid(),gpu=0,tasks=TASKS,modes=MODES,episodes_per_task_mode_arm=5,total_episodes=80,slots=1,replan_steps=36,action_hz=30,instruction_type='seen',seed=0,noise='SHA256(task|mode|episode_seed|query_index), BF16 Gaussian',stall_timeout_sec=2400,ports=ports,jobs=jobs,baseline_checkpoint=CKPT,candidate_recipe=str(ROOT/'robotwin/round7/candidate_recipe.json'),candidate='fixed_queries initialization, frozen zero relevance gate; no round8 updates',precision_note='Original native BF16 policy; repaired native FP32 parameters/Euler state with BF16 autocast. This is an end-to-end deployment comparison, not a precision-isolated architecture test.')
    write(OUT/'manifest.json',manifest);servers={};active=None;results=[];pairs=[]
    def state(phase,**kw):write(OUT/'status.json',dict(phase=phase,updated_at=time.time(),pid=os.getpid(),target_episodes=80,results=results,pairs=pairs,**kw))
    def interrupted(*args):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def launch(arm):
        ready=OUT/f'{arm}_ready.json';ready.unlink(missing_ok=True)
        with (OUT/f'{arm}_server.log').open('a') as log:
            p=subprocess.Popen([PY,'-m','robotwin.round9.server','--arm',arm,'--ckpt_path',CKPT,'--port',str(ports[arm]),'--ready',str(ready)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        servers[arm]=p;started=time.time()
        while True:
            if p.poll() is not None:raise RuntimeError(f'{arm} server failed ({p.returncode}); see log')
            if ready.exists() and listening(ports[arm]):break
            if time.time()-started>1800:raise TimeoutError(f'{arm} startup timeout')
            state('loading_and_checking_policy',arm=arm,server_pid=p.pid,elapsed=time.time()-started);time.sleep(5)
        return p
    try:
        for arm in ARMS:launch(arm)
        for j in jobs:
            arm=j['arm']
            if servers[arm].poll() is not None:launch(arm)
            jobdir=OUT/'jobs'/j['key'];jobdir.mkdir(parents=True,exist_ok=True)
            je=env.copy();je.update(PAIR_TASK=j['task'],PAIR_MODE=j['mode'],PAIR_ARM=arm,ROBOTWIN_EVAL_ROOT=str(jobdir),ROBOTWIN_RUN_TAG='pilot',PYTHONPATH=f'{SIM}:{ROOT}:{ROOT}/examples/Robotwin/eval_files')
            cmd=[SIMPY,str(R/'bridge.py'),'--config',str(ROOT/'examples/Robotwin/eval_files/deploy_policy.yml'),'--overrides','--task_name',j['task'],'--task_config',j['mode'],'--ckpt_setting','round9_paired','--seed','0','--instruction_type','seen','--policy_name','model2robotwin_interface','--host','127.0.0.1','--port',str(ports[arm]),'--policy_ckpt_path',CKPT,'--replan_steps','36','--action_ensemble','0','--action_ensemble_alpha','0']
            with (jobdir/'bridge.log').open('a') as log:active=subprocess.Popen(cmd,cwd=SIM,env=je,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=last=time.time();count=0;timedout=False;progress={}
            while active.poll() is None:
                paths=list(jobdir.glob('**/progress.json'));progress=read(paths[0]) if paths else {}
                done=int(progress.get('completed_episodes',0))
                if done>count:count=done;last=time.time()
                state('evaluating',current=j,bridge_pid=active.pid,progress=progress,no_episode_progress_seconds=time.time()-last,completed_jobs=len(results))
                if time.time()-last>2400:
                    timedout=True;stop(active);stop(servers.pop(arm));launch(arm);break
                if servers[arm].poll() is not None:stop(active);break
                time.sleep(5)
            rc=active.poll();stop(active);active=None
            paths=list(jobdir.glob('**/summary.json'));summary=read(paths[0]) if paths else {}
            complete=rc==0 and len(summary.get('episodes',[]))==5
            row=dict(**j,status='completed' if complete else 'skipped_timeout' if timedout else 'incomplete_error',exit_code=rc,summary=summary,partial_progress=progress,elapsed=time.time()-started)
            results.append(row);write(jobdir/'result.json',row)
            peer=next((x for x in results[:-1] if x['task']==j['task'] and x['mode']==j['mode']),None)
            if peer and complete and peer['status']=='completed':
                def signature(x):return [(e['seed'],e.get('instruction')) for e in x['summary']['episodes']]
                match=signature(row)==signature(peer)
                paired=dict(task=j['task'],mode=j['mode'],matched=match,original=next(x['summary']['successes'] for x in [peer,row] if x['arm']=='original'),compatible=next(x['summary']['successes'] for x in [peer,row] if x['arm']=='compatible'),episodes=5)
                pairs.append(paired);write(OUT/'paired_results.json',pairs)
                if not match:raise RuntimeError('Seed/instruction pairing mismatch; refusing to continue unmatched benchmark')
            state('between_jobs',completed_jobs=len(results))
            if not complete and not timedout:raise RuntimeError(f'Infrastructure failure in {j["key"]}; see bridge.log')
        state('completed' if all(x['status']=='completed' for x in results) else 'completed_with_incomplete_jobs',completed_jobs=len(results))
    except BaseException as e:
        state('failed' if not isinstance(e,KeyboardInterrupt) else 'interrupted',error=repr(e));traceback.print_exc();raise
    finally:
        stop(active)
        for p in servers.values():stop(p)
if __name__=='__main__':main()
