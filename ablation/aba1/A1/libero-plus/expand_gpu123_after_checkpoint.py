"""No-loss, nohup supervisor handoff to add one rank on GPU1, GPU2 and GPU3."""
import argparse,fcntl,json,os,signal,sys,time
from pathlib import Path
import run_stage_ii_auto_gpu as base
from run_stage_ii_requested_gpu123 import RequestedScheduler
from a1_local.config_runtime import atomic_json,code_identity,digest_json
from a1_local.preflight import check_complete

STATE=base.LOGS/'gpu123_expansion_status.json'

def record(phase,**kw):
    row=dict(time=time.time(),pid=os.getpid(),monitor_script=str(Path(__file__).resolve()),phase=phase,
        requested_rank_to_gpu=[5,4,5,1,2,3],target_checkpoint_step=50,**kw)
    atomic_json(STATE,row)
    return row

def matches(pid,name):return base.evaluation.live_process(pid,name)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--capacity-report',type=Path,required=True);a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(target=[5,4,5,1,2,3],checkpoint=50,training_started=False)));return
    paused=False;handed=False;old_pid=0
    lock=(base.LOGS/'.gpu123_expansion.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=base.load_config(base.WORK/'config.json','A1');out=base.run_root(cfg)/'II'
    original=base.read(base.LOGS/'gpu_scheduler_status.json');old_pid=int(original['pid'])
    if not matches(old_pid,'run_stage_ii_auto_gpu.py'):raise RuntimeError('Original scheduler is not live')
    contract=base.read(out/'contract.json')
    if contract['code_sha256']!=digest_json(code_identity()):raise RuntimeError('Active training code changed')
    request=dict(time=time.time(),minimum_step=50,rank_to_gpu=[5,4,5,1,2,3],reason='user requested GPU1/2/3 expansion',owner_pid=os.getpid())
    atomic_json(out/'elastic_reconfigure.json',request)
    try:
        while True:
            states=base.worker_states(out);step=max([x.get('step',0) for x in states]+[0])
            flow=base.read(base.LOGS/'gpu_scheduler_status.json')
            if not paused:
                if not matches(old_pid,'run_stage_ii_auto_gpu.py'):
                    raise RuntimeError('Original scheduler exited before the checkpoint handoff')
                # Pause only the old coordinator, never a training worker. The
                # handoff monitor takes over pressure checks for the final two updates.
                if step>=48 or (out/'elastic_ready.json').exists():
                    original=flow
                    os.kill(old_pid,signal.SIGSTOP);paused=True
                    atomic_json(out/'elastic_reconfigure.json',request)
                else:
                    record('waiting_checkpoint',completed_step=step,old_scheduler_pid=old_pid,
                        training_pids=[x['pid'] for x in states],capacity_report=str(a.capacity_report))
                    time.sleep(5);continue
            resource=base.sample(out)
            states=resource['workers'];step=max([x.get('step',0) for x in states]+[0])
            ready=base.read(out/'elastic_ready.json');latest=base.read(out/'latest.json')
            row=record('waiting_checkpoint_exit',completed_step=step,old_scheduler_pid=old_pid,
                training_pids=[x['pid'] for x in states],resources=resource)
            current=dict(original,time=time.time(),pid=os.getpid(),monitor_script=str(Path(__file__).resolve()),
                phase='waiting_expansion_checkpoint',stage_II_step=step,resources=resource,
                pending_reconfiguration=request,candidate_rank_to_gpu=request['rank_to_gpu'])
            atomic_json(base.LOGS/'gpu_scheduler_status.json',current)
            atomic_json(base.LOGS/'auto_continuation_status.json',current)
            launcher=int(base.read(base.LOGS/'launch_record.json').get('launcher_pid',0))
            if not states and not matches(launcher,'a1_local.training'):
                if ready.get('step')!=50 or latest.get('step')!=50:
                    raise RuntimeError('Training exited without the requested complete step-50 handoff')
                check_complete(Path(latest['checkpoint']),require_resume=True,verify_hash=True)
                break
            mapping=original.get('rank_to_gpu',[5,4,5])
            if any(resource['gpus'][g]['free_mib']<1024 for g in set(mapping)) or resource['host_available_mib']<4096:
                # Restore the existing pressure/retry controller rather than
                # leave it paused during an unrelated resource emergency.
                os.kill(old_pid,signal.SIGCONT);paused=False
                record('deferred_pressure',completed_step=step,old_scheduler_pid=old_pid)
                time.sleep(10);continue
            time.sleep(5)
        # Its training child has fully exited and the checkpoint is verified.
        # Killing this paused monitor cannot affect the independent training session.
        if matches(old_pid,'run_stage_ii_auto_gpu.py'):os.kill(old_pid,signal.SIGKILL)
        for _ in range(20):
            if not matches(old_pid,'run_stage_ii_auto_gpu.py'):break
            time.sleep(.5)
        scheduler=RequestedScheduler(argparse.Namespace(capacity_report=a.capacity_report,execute=True,poll_seconds=5))
        record('starting_six_ranks',resume_step=50,old_scheduler_pid=old_pid)
        handed=True
        scheduler.run_forever()
        record('completed')
    except BaseException as e:
        record('failed',error=repr(e),old_scheduler_pid=old_pid)
        raise
    finally:
        if paused and not handed and matches(old_pid,'run_stage_ii_auto_gpu.py'):
            os.kill(old_pid,signal.SIGCONT)
        lock.close()

if __name__=='__main__':main()
