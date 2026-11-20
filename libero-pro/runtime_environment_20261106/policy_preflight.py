"""One bounded LIBERO policy preflight on two validated Environment tasks."""
import signal, subprocess, sys, traceback, fcntl
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'runtime_environment_20261106'))
import controller as c
PREFLIGHT=BASE/'runs/20261106_environment_preflight'
PREFLIGHT_RUNTIME=BASE/'runtime_environment_preflight_20261106'
def main():
    PREFLIGHT.mkdir(exist_ok=True)
    lock=(PREFLIGHT/'preflight.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:c.stop.set())
    assert c.predecessor_pending('libero',3)==0 and not c.foreign_gpu_processes(3)
    assert c.gpu_free()[3]>=c.SLOT_MIB+c.HEADROOM_MIB
    PREFLIGHT_RUNTIME.mkdir(exist_ok=True)
    for name in ('common.py','worker.py','server.py'):
        text=(BASE/'runtime_environment_20261106'/name).read_text()
        if name=='common.py':text=text.replace("runs/20261106_environment'","runs/20261106_environment_preflight'")
        (PREFLIGHT_RUNTIME/name).write_text(text)
    jobs=[]
    for suite,ti in [('libero_spatial',9),('libero_10',0)]:
        rows=c.read(BASE/'resources/environment_20261106'/('probe_'+suite+'.json'))
        r=next(x for x in rows if x['task_id']==ti)
        job={k:r[k] for k in ('suite','task_id','name','instruction','bddl','init','initial_states','state_shape','fixtures','bddl_sha256','init_sha256','fixtures_sha256')}
        job.update(id=len(jobs),episode_idx=0,category='env',pro_suite=suite+'_env',
            noise_seed=5_000_000_000+(c.SUITES.index(suite)+1)*10_000_000+ti*10000)
        jobs.append(job)
    c.atomic(PREFLIGHT/'manifest.json',dict(jobs=jobs))
    c.RUNTIME=PREFLIGHT_RUNTIME;c.RUN=PREFLIGHT
    directory=PREFLIGHT/'libero/processes/preflight'
    directory.mkdir(parents=True,exist_ok=True)
    slot=dict(name='libero_environment_preflight',variant='libero',gpu=3,rank=1,directory=str(directory))
    c.atomic(PREFLIGHT/'status.json',dict(phase='loading',pid=c.os.getpid(),gpu=3,started=c.time.time()))
    proc=None
    try:
        c.start_server(slot)
        for index in range(len(jobs)):
            c.atomic(PREFLIGHT/'status.json',dict(phase='evaluating',pid=c.os.getpid(),gpu=3,server_pid=slot['server_pid'],job=index,updated=c.time.time()))
            with (directory/f'smoke_{index}.log').open('w') as log:
                proc=subprocess.Popen([c.SIM_PYTHON,'-B','-u',str(PREFLIGHT_RUNTIME/'worker.py'),
                    '--variant','libero','--gpu','3','--worker',f'preflight_{index}',
                    '--url',slot['url'],'--smoke-job',str(index)],env=c.environment(3),cwd=c.PROJECT,
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            begin=c.time.time()
            while proc.poll() is None:
                if c.stop.wait(2) or c.time.time()-begin>900:raise TimeoutError('Preflight interrupted or exceeded 900s')
            assert proc.returncode==0,(index,proc.returncode)
            result=c.read(PREFLIGHT/'preflight/libero/episodes'/f'{index:05d}.json')
            assert result and not result.get('error')
        c.atomic(PREFLIGHT/'status.json',dict(phase='passed',pid=c.os.getpid(),gpu=3,episodes=len(jobs),updated=c.time.time()))
    except Exception as exc:
        c.atomic(PREFLIGHT/'status.json',dict(phase='failed',error=repr(exc),updated=c.time.time()));raise
    finally:
        c.terminate(proc);c.terminate(slot.get('_server'));c.pool.shutdown(wait=True)
if __name__=='__main__':main()
