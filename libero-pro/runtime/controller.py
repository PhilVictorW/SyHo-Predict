"""Nohup supervisor: memory-aware replicas, durable retries, separate model queues."""
import concurrent.futures
import fcntl
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from common import *
sys.path.insert(0,str(PROJECT))
from goal_local.eval_client import rpc

RESERVE_MIB = 8192
WORKER_BUDGET_MIB = 7500
MAX_LOADERS = 3
slots = []
serial = 0
cooldown = {}
stop = False
pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_LOADERS)


def database(variant):
    db = sqlite3.connect(RUN/variant/'queue.sqlite',timeout=60,isolation_level=None)
    db.execute('PRAGMA busy_timeout=60000')
    return db


def live(pid):
    if not pid:
        return False
    try:
        value = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
        return value != 'Z'
    except OSError:
        return False


def end(pid):
    if not live(pid):
        return
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(BASE/'runtime') not in command:
            raise RuntimeError(f'Refusing to signal unrelated process {pid}')
        os.killpg(pid,signal.SIGTERM)
        deadline=time.monotonic()+8
        while live(pid) and time.monotonic()<deadline:
            time.sleep(.2)
        if live(pid):
            os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:
        pass


def recover(slot, reason):
    with database(slot['variant']) as db:
        db.execute("UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END,error=? WHERE state='running' AND worker=?",(reason,slot['name']))


def retire(slot, reason):
    end(slot.get('sim_pid'))
    end(slot.get('server_pid'))
    recover(slot,reason)
    slot['phase']='retired'
    slot['reason']=reason
    slot['retired_at']=time.time()
    print(json.dumps(dict(event='retire',worker=slot['name'],gpu=slot['gpu'],reason=reason)),flush=True)


def gpu_free():
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in data.splitlines()}


def host_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    return 0


def start_server(slot):
    out=Path(slot['directory']);out.mkdir(parents=True,exist_ok=True)
    ready=out/'ready.json'
    with (out/'server.log').open('a') as log:
        proc=subprocess.Popen([MODEL_PYTHON,'-B','-u',str(BASE/'runtime/server.py'),'--ready',str(ready)],
            cwd=PROJECT,env=environment(slot['gpu']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    slot['server_pid']=proc.pid
    slot['_server']=proc
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        info=read(ready,{})
        if info.get('pid')==proc.pid:
            url=f'http://127.0.0.1:{info["port"]}'
            loaded=rpc(url,'/load',dict(checkpoint=CHECKPOINTS[slot['variant']]),timeout=900)
            if loaded['step']!=4000 or loaded['checkpoint']!=CHECKPOINTS[slot['variant']]:
                raise ValueError(loaded)
            slot['url']=url
            return loaded
        if proc.poll() is not None:
            raise RuntimeError(f'Server exited {proc.returncode}; see {out}/server.log')
        time.sleep(1)
    raise TimeoutError('No inference server ready file')


def start_sim(slot):
    folder=RUN/slot['variant'];out=Path(slot['directory'])
    stopfile=folder/'workers'/f'{slot["name"]}.stop'
    stopfile.unlink(missing_ok=True)
    with (out/'simulation.log').open('a') as log:
        proc=subprocess.Popen([SIM_PYTHON,'-B','-u',str(BASE/'runtime/worker.py'),
            '--variant',slot['variant'],'--gpu',str(slot['gpu']),'--worker',slot['name'],'--url',slot['url']],
            cwd=PROJECT,env=environment(slot['gpu']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    slot.update(sim_pid=proc.pid,phase='running',sim_started=time.time(),_sim=proc)
    print(json.dumps(dict(event='worker_started',worker=slot['name'],gpu=slot['gpu'],variant=slot['variant'],server_pid=slot['server_pid'],sim_pid=proc.pid)),flush=True)


def new_slot(variant,gpu,adopt=None):
    global serial
    serial+=1
    name=f'{variant}_g{gpu}_w{serial:03d}'
    out=RUN/variant/'processes'/name
    out.mkdir(parents=True,exist_ok=True)
    slot=dict(name=name,variant=variant,gpu=gpu,directory=str(out),phase='loading',created=time.time())
    slots.append(slot)
    if adopt:
        info=read(RUN/'preflight'/adopt/'ready.json')
        if info and live(info['pid']):
            slot.update(server_pid=info['pid'],url=f'http://127.0.0.1:{info["port"]}',phase='ready',adopted=adopt)
            return slot
    slot['_future']=pool.submit(start_server,slot)
    return slot


def summarize():
    result={}
    for variant,gpus in GPUS.items():
        db=database(variant)
        counts=dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        successes=db.execute("SELECT COALESCE(SUM(success),0) FROM jobs WHERE state='done'").fetchone()[0]
        done=counts.get('done',0)
        per_gpu=[]
        for gpu in gpus:
            rows=db.execute('SELECT state,COUNT(*),COALESCE(SUM(success),0) FROM jobs WHERE gpu=? GROUP BY state',(gpu,)).fetchall()
            c={state:n for state,n,k in rows};n=c.get('done',0);k=sum(k for state,_,k in rows if state=='done')
            active=[s for s in slots if s['gpu']==gpu and s['phase']!='retired']
            per_gpu.append(dict(gpu=gpu,total=sum(c.values()),completed=n,successes=k,success_rate=k/n if n else None,
                pending=c.get('pending',0),running=c.get('running',0),errors=c.get('failed',0),
                workers=sum(s['phase']=='running' for s in active),loading=sum(s['phase']=='loading' for s in active)))
        suites=[]
        for category in CATEGORIES:
            n,k=db.execute("SELECT COUNT(*),COALESCE(SUM(success),0) FROM jobs WHERE state='done' AND json_extract(payload,'$.category')=?",(category,)).fetchone()
            suites.append(dict(category=category,completed=n,successes=k,success_rate=k/n if n else None,total=2000))
        db.close()
        phase='completed' if done==8000 else 'completed_with_errors' if not counts.get('pending',0) and not counts.get('running',0) else 'running'
        row=dict(variant=variant,phase=phase,counts=counts,total=8000,completed=done,successes=successes,
            success_rate=successes/done if done else None,gpus=per_gpu,categories=suites,updated=time.time())
        atomic(RUN/variant/'summary.json',row);result[variant]=row
    return result


def handle_signal(*_):
    global stop
    stop=True


def main():
    lock=(RUN/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,handle_signal);signal.signal(signal.SIGINT,handle_signal)
    assert read(RUN/'preflight/fastload_equivalence.json',{}).get('passed')
    for variant in GPUS:
        # Recover only abandoned jobs of this run; never erase completed results.
        with database(variant) as db:
            db.execute("UPDATE jobs SET state=CASE WHEN attempts>=3 THEN 'failed' ELSE 'pending' END WHERE state='running'")
    new_slot('plus',0,'plus');new_slot('plus',3,'fastload');new_slot('libero',5,'libero')
    try:
        while not stop:
            summary=summarize()
            for slot in list(slots):
                if slot['phase']=='retired':continue
                variant=slot['variant'];pending=summary[variant]['counts'].get('pending',0)
                if slot['phase']=='loading':
                    future=slot['_future']
                    if future.done():
                        try:
                            slot['load_info']=future.result();slot['phase']='ready'
                        except Exception as exc:
                            traceback.print_exc();retire(slot,repr(exc));cooldown[slot['gpu']]=time.time()+90
                    continue
                if not live(slot.get('server_pid')):
                    retire(slot,'inference_server_exited');cooldown[slot['gpu']]=time.time()+30;continue
                if slot['phase']=='ready':
                    if pending:start_sim(slot)
                    else:retire(slot,'queue_drained')
                    continue
                status=read(RUN/variant/'workers'/f'{slot["name"]}.json',{})
                process=slot.get('_sim')
                code=process.poll() if process else None
                if not live(slot.get('sim_pid')) or code is not None:
                    recover(slot,f'simulation_exit_{code}')
                    if slot.get('retiring') or 'out of memory' in status.get('error','').lower():
                        retire(slot,'memory_headroom');cooldown[slot['gpu']]=time.time()+90
                    elif pending:
                        slot['phase']='ready'
                    else:retire(slot,'queue_drained')
                elif status.get('pid')==slot['sim_pid'] and time.time()-status.get('updated',time.time())>600:
                    end(slot['sim_pid']);recover(slot,'no_progress_600_seconds');slot['phase']='ready'
                elif not status and time.time()-slot['sim_started']>600:
                    end(slot['sim_pid']);recover(slot,'startup_timeout');slot['phase']='ready'
            free=gpu_free()
            # Mark one worker per pressured GPU to retire after its current episode.
            for gpu,available in free.items():
                active=[s for s in slots if s['gpu']==gpu and s['phase']=='running']
                if available<4096 and len(active)>1 and not any(s.get('retiring') for s in active):
                    slot=active[-1];slot['retiring']=True
                    (RUN/slot['variant']/'workers'/f'{slot["name"]}.stop').touch()
            # Give every requested GPU one worker before increasing any GPU's replicas.
            candidates=[(variant,gpu) for variant,gpus in GPUS.items() for gpu in gpus]
            candidates.sort(key=lambda item:sum(s['gpu']==item[1] and s['phase']!='retired' for s in slots))
            for variant,gpu in candidates:
                if not summary[variant]['counts'].get('pending',0):continue
                loading=[s for s in slots if s['phase']=='loading']
                if len(loading)>=MAX_LOADERS:break
                if any(s['gpu']==gpu for s in loading) or cooldown.get(gpu,0)>time.time():continue
                if free[gpu]>=RESERVE_MIB+WORKER_BUDGET_MIB and host_available_mib()>24000:
                    new_slot(variant,gpu)
            public=[{k:v for k,v in s.items() if not k.startswith('_')} for s in slots]
            atomic(RUN/'controller.json',dict(pid=os.getpid(),updated=time.time(),phase='running',slots=public,
                gpu_free_mib=free,reserve_mib=RESERVE_MIB,worker_budget_mib=WORKER_BUDGET_MIB))
            if all(x['phase'].startswith('completed') for x in summary.values()):break
            time.sleep(5)
    finally:
        for slot in slots:
            if slot['phase']!='retired':retire(slot,'controller_stop' if stop else 'evaluation_finished')
        pool.shutdown(wait=True,cancel_futures=True)
        result=summarize()
        atomic(RUN/'controller_final.json',dict(pid=os.getpid(),updated=time.time(),stopped=stop,summary=result))

if __name__=='__main__':main()
