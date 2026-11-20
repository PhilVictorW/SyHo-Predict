"""Launch the supplemental evaluation only after data, render, and policy gates pass."""
import fcntl, subprocess, sqlite3, traceback
from common import *
ROOT=BASE/'resources/environment_20261106'
def main():
    lock=(RUN/'bootstrap.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time()
    try:
        while time.time()-started<3600:
            reports=[read(ROOT/('validated_'+s+'.json')) for s in SUITES]
            for suite,rows in zip(SUITES,reports):
                if rows is not None:
                    assert len(rows)==10 and all(r['status']=='validated' and r['initial_states']==50 for r in rows),(suite,'Generation failed')
            render=read(ROOT/'render_validation.json',{})
            policy=read(BASE/'runs/20261106_environment_preflight/status.json',{})
            assert policy.get('phase')!='failed',policy
            ready=all(x is not None for x in reports) and render.get('status')=='passed' and policy.get('phase')=='passed'
            atomic(RUN/'bootstrap.json',dict(pid=os.getpid(),phase='validating',updated=time.time(),
                generation_suites=sum(x is not None for x in reports),render=render.get('status','pending'),policy=policy.get('phase','pending')))
            if ready:break
            time.sleep(5)
        else:raise TimeoutError('Validation exceeded one hour')
        runtime=Path(__file__).resolve().parent
        assert not (RUN/'manifest.json').exists(),'Supplemental queues already exist; refusing duplicate preparation'
        subprocess.run([MODEL_PYTHON,'-B',str(runtime/'prepare.py')],check=True,cwd=PROJECT)
        payloads=[]
        for variant,gpus in GPUS.items():
            with sqlite3.connect('file:'+str(RUN/variant/'queue.sqlite')+'?mode=ro',uri=True) as db:
                assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
                assert dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))=={'pending':2000}
                assert set(x[0] for x in db.execute('SELECT DISTINCT gpu FROM jobs'))==set(gpus)
                rows=db.execute('SELECT id,payload FROM jobs ORDER BY id').fetchall()
                assert [x[0] for x in rows]==list(range(8000,10000))
                payloads.append(rows)
        assert payloads[0]==payloads[1]
        atomic(RUN/'queue_validation.json',dict(updated=time.time(),rows_per_variant=2000,identical_payloads=True,
               identical_initial_states_fixture_poses_and_noise=True,gpu_assignment=GPUS,integrity_check='ok'))
        with (RUN/'controller.log').open('a') as log:
            proc=subprocess.Popen([MODEL_PYTHON,'-B','-u',str(runtime/'controller.py')],cwd=PROJECT,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (RUN/'controller.pid').write_text(str(proc.pid))
        atomic(RUN/'launch.json',dict(started=time.time(),pid=proc.pid,gpu_assignment=GPUS,phase='launched',
            data_validation='passed',render_validation='passed',policy_preflight='passed'))
        time.sleep(3);assert proc.poll() is None,('Controller exited',proc.returncode)
        atomic(RUN/'bootstrap.json',dict(pid=os.getpid(),phase='launched',controller_pid=proc.pid,updated=time.time()))
        print('Environment controller launched',proc.pid,flush=True)
    except Exception as exc:
        atomic(RUN/'bootstrap.json',dict(pid=os.getpid(),phase='failed',error=repr(exc),updated=time.time()))
        traceback.print_exc();raise
if __name__=='__main__':main()
