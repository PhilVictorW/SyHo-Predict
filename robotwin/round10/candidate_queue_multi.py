"""Candidate-only RoboTwin screening against immutable historical results."""
import fcntl,json,os,signal,subprocess,time,traceback
from pathlib import Path
from robotwin.round10.parallel_queue import ROOT,R,SIM,PY,SIMPY,CKPT,PAIRS,base_env,read,write,listening
OUT=R/'candidate_online'
BASE=Path('/mnt/hdd/wangchaoyang/results/LaWAM/RoboTwin')
LANES=('0','3','5a','5b');GPUS={'0':'0','3':'3','5a':'5','5b':'5'};PORTS={'0':26700,'3':26691,'5a':26701,'5b':26702};TARGET=10;STALL=1800
SERVER_MIN_FREE=20000
RETRY_BACKOFF=5

def alive(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return stat[0] != 'Z'
    except OSError:return False

def owned(pid,kind):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return ('robotwin.round10.'+kind in cmd if kind=='server' else str(R/'bridge.py') in cmd)
    except OSError:return False

def stop_owned(pid,kind):
    if not alive(pid) or not owned(pid,kind):return
    assert os.getpgid(pid)==pid,('unexpected process group',pid)
    os.killpg(pid,signal.SIGTERM)
    end=time.time()+12
    while alive(pid) and time.time()<end:time.sleep(.2)
    if alive(pid) and owned(pid,kind):os.killpg(pid,signal.SIGKILL)

def task_json(job,name):
    paths=list(Path(job).glob('*/*/tasks/*/'+name))
    if not paths:paths=list(Path(job).glob('tasks/*/'+name))
    return read(paths[0]) if paths else {}

def heartbeat(job,start):
    paths=[Path(job)/'bridge.log',Path(job)/'requests.jsonl']
    paths+=list(Path(job).glob('*/*/tasks/*/progress.json'))
    times=[start]
    for p in paths:
        try:times.append(p.stat().st_mtime)
        except OSError:pass
    return max(times)

def memory_free():
    try:
        v=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=8)
        return {a.strip():int(b) for a,b in (line.split(',') for line in v.splitlines())}
    except Exception:return {}

def compare(summary,baseline):
    old=baseline.get('episodes',[]);new=summary.get('episodes',[])
    by_seed={int(e['seed']):e for e in old}
    overlap=[e for e in new if int(e['seed']) in by_seed]
    return dict(comparison='historical baseline; instructions and policy noise not matched',
                baseline_path=baseline.get('path'),baseline_successes=baseline.get('successes'),
                baseline_episodes=baseline.get('n_episodes',0),baseline_rate=baseline.get('success_rate'),
                overlapping_seeds=len(overlap),historical_successes_on_overlap=sum(bool(by_seed[int(e['seed'])]['success']) for e in overlap),
                candidate_successes_on_overlap=sum(bool(e['success']) for e in overlap),strict_pair=False)

def load_baselines():
    data={}
    for task,mode in PAIRS:
        key=task+'__'+mode
        path=BASE/f'lawam_robotwin_sft_release__{mode}'/'full_100x50_20260903/tasks'/task/'summary.json'
        d=read(path)
        if d and len(d.get('episodes',[]))==d.get('n_episodes') and d.get('n_episodes')==50:
            data[key]=dict(d,path=str(path))
        else:data[key]=dict(path=None,episodes=[],n_episodes=0)
    return data

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline=load_baselines();write(OUT/'historical_baseline.json',baseline)
    migration=read(OUT/'migration.json');previous=read(OUT/'status.json')
    results=previous.get('results',[]);events=previous.get('events',[])
    active={g:None for g in LANES};servers={};children={};retry_after={};server_failures={g:0 for g in LANES}
    source_active=previous.get('active',{}) or migration.get('active',{})
    # Existing GPU0/GPU3 jobs are adopted by lane id; GPU5 lanes start empty.
    source_active={k:v for k,v in source_active.items() if k in LANES}
    for gpu,a in source_active.items():
        if gpu in active and a:
            if alive(a['pid']) and owned(a['pid'],'bridge'):active[gpu]=a
            elif task_json(a['job'],'summary.json').get('n_episodes')==TARGET:
                sm=task_json(a['job'],'summary.json');results.append(dict(key=a['key'],task=a['task'],mode=a['mode'],gpu=gpu,status='completed',summary=sm,job=a['job'],comparison=compare(sm,baseline[a['key']])))
    for gpu,sv in {**migration.get('servers',{}),**previous.get('servers',{})}.items():
        if gpu in LANES and alive(sv['pid']) and owned(sv['pid'],'server') and listening(sv['port']):
            servers[gpu]=dict(sv,phase='ready');PORTS[gpu]=sv['port']
    manifest=dict(started_at=previous.get('started_at',time.time()),protocol='candidate-only historical comparison; GPU0+GPU3 existing lanes plus GPU5a/GPU5b',
        tasks=50,modes=2,episodes_per_job=TARGET,total_episodes=1000,lanes=LANES,gpus=GPUS,ports=PORTS,stall_sec=STALL,
        candidate_recipe=read(ROOT/'robotwin/round7/candidate_recipe.json'),
        candidate='round7 fixed_queries initialization with round4/round6 adapters; round8 updates NOT used',
        seeds='first 10 recorded episodes sorted by episode_id, independent of success; adopted job retains its existing seeds',
        missing_baseline='evaluate candidate; mark baseline unavailable; do not rerun original automatically',
        historical_baseline_jobs=sum(bool(x['path']) for x in baseline.values()),strict_pair=False)
    write(OUT/'manifest.json',manifest)
    def state(phase='evaluating',**kw):
        aa={}
        for gpu,a in active.items():
            aa[gpu]=dict(a,progress=task_json(a['job'],'progress.json'),no_activity_seconds=max(0,time.time()-heartbeat(a['job'],a['started']))) if a else None
        write(OUT/'status.json',dict(phase=phase,updated_at=time.time(),started_at=manifest['started_at'],target_episodes=1000,
              active=aa,servers=servers,results=results,events=events[-100:],historical_baseline_jobs=manifest['historical_baseline_jobs'],
              server_retry_policy='unlimited_with_5s_backoff',server_failures=server_failures,**kw))
    def start_server(gpu):
        ready=OUT/f'gpu{gpu}_ready.json';ready.unlink(missing_ok=True)
        e=base_env();e.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],ROUND10_SERVER_OUTPUT=str(OUT),ROUND10_SERVER_LABEL='_gpu'+gpu)
        log=(OUT/f'gpu{gpu}_server.log').open('a')
        p=subprocess.Popen([PY,'-m','robotwin.round10.server','--arm','compatible','--ckpt_path',CKPT,'--port',str(PORTS[gpu]),'--ready',str(ready)],cwd=ROOT,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children[p.pid]=p
        servers[gpu]=dict(pid=p.pid,port=PORTS[gpu],phase='loading',ready=str(ready),started=time.time())
    def start_job(gpu,key):
        task,mode=key.split('__');attempt=1+sum(x['key']==key for x in results)
        while (OUT/'jobs'/key/f'attempt_{attempt:02d}_gpu{gpu}').exists(): attempt += 1
        job=OUT/'jobs'/key/f'attempt_{attempt:02d}_gpu{gpu}';job.mkdir(parents=True,exist_ok=False)
        e=base_env();e.update(CUDA_VISIBLE_DEVICES=GPUS[gpu],PAIR_TASK=task,PAIR_MODE=mode,PAIR_ARM='compatible',
            ROBOTWIN_EVAL_ROOT=str(job),ROBOTWIN_RUN_TAG='round10_candidate',
            ROBOTWIN_SHARED_WARP_CACHE_DIR=str(OUT/f'cache/gpu{gpu}/warp'),PYTHONPATH=f'{SIM}:{ROOT}:{ROOT}/examples/Robotwin/eval_files')
        eps=sorted(baseline[key].get('episodes',[]),key=lambda x:x['episode_id'])[:TARGET]
        if len(eps)==TARGET:e['ROBOTWIN_FIXED_SEEDS']=','.join(str(x['seed']) for x in eps)
        write(job/'protocol.json',dict(key=key,gpu=gpu,seeds=[x['seed'] for x in eps],historical_baseline=baseline[key]['path'],strict_pair=False))
        cmd=[SIMPY,str(R/'bridge.py'),'--config',str(ROOT/'examples/Robotwin/eval_files/deploy_policy.yml'),'--overrides',
             '--task_name',task,'--task_config',mode,'--ckpt_setting','round10_candidate','--seed','0','--instruction_type','seen',
             '--policy_name','model2robotwin_interface','--host','127.0.0.1','--port',str(PORTS[gpu]),'--policy_ckpt_path',CKPT,
             '--replan_steps','36','--action_ensemble','0','--action_ensemble_alpha','0']
        log=(job/'bridge.log').open('a');p=subprocess.Popen(cmd,cwd=SIM,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children[p.pid]=p
        active[gpu]=dict(pid=p.pid,key=key,task=task,mode=mode,gpu=gpu,job=str(job),started=time.time(),attempt=attempt)
    try:
        while True:
            free=memory_free()
            for gpu in LANES:
                a=active[gpu];sv=servers.get(gpu)
                if a:
                    p=children.get(a['pid']);running=p.poll() is None if p else alive(a['pid'])
                    timed=running and time.time()-heartbeat(a['job'],a['started'])>STALL
                    if timed:stop_owned(a['pid'],'bridge');running=False
                    if not running:
                        sm=task_json(a['job'],'summary.json');pr=task_json(a['job'],'progress.json')
                        log=Path(a['job'])/'bridge.log'
                        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-150000));tail=f.read().decode(errors='replace')
                        oom='out of memory' in tail.lower()
                        complete=sm.get('n_episodes')==TARGET and len(sm.get('episodes',[]))==TARGET
                        status='completed' if complete else 'skipped_timeout' if timed else 'error_oom' if oom else 'error'
                        row=dict(a,status=status,summary=sm,partial_progress=pr,elapsed=time.time()-a['started'],comparison=compare(sm,baseline[a['key']]))
                        write(Path(a['job'])/'candidate_result.json',row);results.append(row);active[gpu]=None
                        if p:p.wait();children.pop(a['pid'],None)
                        if oom:retry_after[gpu]=time.time()+RETRY_BACKOFF
                if sv:
                    p=children.get(sv['pid']);live=p.poll() is None if p else alive(sv['pid'])
                    if not live:
                        events.append(dict(time=time.time(),gpu=gpu,event='server_exited',pid=sv['pid']))
                        servers.pop(gpu);server_failures[gpu]+=1;retry_after[gpu]=time.time()+RETRY_BACKOFF
                        if p:p.wait();children.pop(sv['pid'],None)
                    elif sv['phase']=='loading':
                        if read(Path(sv['ready'])).get('ready') and listening(sv['port']):sv['phase']='ready'
                        elif time.time()-sv['started']>STALL:
                            stop_owned(sv['pid'],'server');events.append(dict(time=time.time(),gpu=gpu,event='server_load_timeout'))
            terminal={x['key'] for x in results if x['status'] in ('completed','skipped_timeout')}
            counts={k:sum(x['key']==k for x in results) for k in baseline}
            terminal|={k for k,n in counts.items() if n>=2}
            busy={a['key'] for a in active.values() if a}
            todo=[t+'__'+m for t,m in PAIRS if t+'__'+m not in terminal|busy]
            if not todo and not any(active.values()):state('completed');break
            for gpu in LANES:
                if active[gpu] or time.time()<retry_after.get(gpu,0):continue
                if gpu not in servers:
                    # Server exits are retryable indefinitely. GPU memory remains the
                    # resource gate; the short backoff prevents a tight crash loop.
                    if todo and free.get(GPUS[gpu],0)>=SERVER_MIN_FREE:start_server(gpu)
                    continue
                if servers[gpu]['phase']!='ready' or free.get(GPUS[gpu],0)<10000 or not todo:continue
                # Prefer a different GPU after a failed attempt; permit fallback when needed.
                preferred=[k for k in todo if not any(x['key']==k and x['gpu']==gpu and x['status']!='completed' for x in results)]
                key=(preferred or todo)[0];todo.remove(key);start_job(gpu,key)
            state(free_memory_mib=free,pending_jobs=len(todo),terminal_jobs=len(terminal),
                  lane_status={g:('evaluating' if active[g] else 'loading' if servers.get(g,{}).get('phase')=='loading' else 'waiting_for_memory_or_work') for g in GPUS})
            time.sleep(5)
    except BaseException as exc:
        state('failed',error=repr(exc));traceback.print_exc();raise
    finally:
        # Never terminate adopted GPU0/GPU3 work on a supervisor restart; only
        # clean up children launched by this queue instance.
        for a in active.values():
            if a and a['pid'] in children:stop_owned(a['pid'],'bridge')
        for sv in servers.values():
            if sv['pid'] in children:stop_owned(sv['pid'],'server')

if __name__=='__main__':main()
