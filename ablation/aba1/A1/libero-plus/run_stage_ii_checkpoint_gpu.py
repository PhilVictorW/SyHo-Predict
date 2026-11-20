"""Five initial ranks, additions after 50-step checkpoints; no proactive group stop."""
import argparse,collections,fcntl,json,os,subprocess,time
from pathlib import Path
import run_stage_ii_auto_gpu as base
from a1_local import autoscale as policy
from a1_local.config_runtime import atomic_json
base.RESERVE=policy.GPU_RESERVE
INITIAL=[0,2,4,7,7]
EVALUATION_GPUS=[0,1,2,3,6,7]

class CheckpointScheduler(base.Scheduler):
    def __init__(self,args):
        super().__init__(args);self.mapping=INITIAL.copy();self.target=INITIAL.copy();self.recovering=False
    def record(self,phase,**kw):
        state=dict(time=time.time(),pid=os.getpid(),phase=phase,monitor_script=str(Path(__file__).resolve()),armed=True,
            poll_seconds=self.args.poll_seconds,physical_gpus=sorted(set(self.mapping)),rank_to_gpu=self.mapping,
            training_ranks=len(self.mapping),generation=self.generation,stage_I_steps=3000,stage_II_steps=2500,
            evaluation_workers=6,evaluation_tasks=10030,evaluation_gpus=EVALUATION_GPUS,
            checkpoint_policy='every 50 steps, retain latest complete only',mode='checkpoint_expand',
            automatic_resize=True,active_group_termination=False,initial_rank_to_gpu=INITIAL,excluded_gpus=sorted(policy.EXCLUDED_GPUS),**kw)
        for name in ('gpu_scheduler_status.json','auto_continuation_status.json'):atomic_json(base.LOGS/name,state)
    def launch(self,mapping,detail):
        super().launch(mapping,detail)
        atomic_json(self.out/'autoscale_control.json',dict(enabled=True,generation=self.generation,
            controller_pid=os.getpid(),interval_steps=50,active_group_termination=False,excluded_gpus=sorted(policy.EXCLUDED_GPUS)))
    def full_eval(self):
        previous=os.environ.get('A1_EVALUATION_GPUS')
        os.environ['A1_EVALUATION_GPUS']=','.join(map(str,EVALUATION_GPUS))
        try:super().full_eval()
        finally:
            if previous is None:os.environ.pop('A1_EVALUATION_GPUS',None)
            else:os.environ['A1_EVALUATION_GPUS']=previous
    def window(self,s):
        self.history.append(s)
        while self.history and self.history[0]['time']<s['time']-60:self.history.popleft()
        window=dict(time=s['time'],capacity_mib={str(g):min(x['gpus'][g]['capacity_mib'] for x in self.history) for g in range(8)},
            host_capacity_mib=min(x['host_available_mib']+x['own_host_mib'] for x in self.history),window_seconds=60)
        atomic_json(base.LOGS/'checkpoint_gpu_capacity.json',window);return window
    def run_forever(self):
        self.locks=[(base.LOGS/'.auto_continuation.lock').open('a'),(self.run/'.pipeline.lock').open('a')]
        for lock in self.locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.owns_locks=True
        if base.worker_states(self.out):raise RuntimeError('Existing workers must be inspected before launch')
        for name in ('gpu_scheduler.pid','auto_continuation.pid'):(base.LOGS/name).write_text(str(os.getpid())+'\n')
        self.record('verifying_teacher');base.check_teacher(self.teacher,self.cfg)
        # Stage II uses teacher model.pt only; rank0 verifies its hash before training.
        # Do not reread the unused 17 GiB Stage I optimizer on each launch.
        self.event('checkpoint_monitor_started',initial_rank_to_gpu=INITIAL,active_group_termination=False)
        retry_after=0
        while True:
            try:s=policy.snapshot(self.out)
            except (OSError,subprocess.SubprocessError,ValueError) as error:
                self.event('sample_error',error=repr(error));time.sleep(self.args.poll_seconds);continue
            window=self.window(s);caps,host=policy.capacities(s,window)
            candidate,detail=policy.choose(caps,host,self.mapping,preserve=True)
            if self.child and self.child.poll() is not None:
                code=self.child.returncode;checkpoint,step=self.latest()
                if code==0 and step==2500:self.full_eval();return
                ready=base.read(self.out/'elastic_ready.json')
                if code==0 and ready.get('step')==step and step>0:
                    base.check_complete(checkpoint,require_resume=True,verify_hash=True)
                    self.target=list(ready['request']['rank_to_gpu']);self.recovering=True
                    self.event('checkpoint_expansion_ready',step=step,target=self.target)
                    self.child=None;self.history.clear();continue
                log=self.log_path.read_text(errors='replace')[-30000:]
                memory_failure=any(word in log.lower() for word in ('out of memory','memory admission floor','cannot allocate memory','memoryerror'))
                # Cleanup only after the training launcher has already failed.
                self.stop_owned()
                if not memory_failure:raise RuntimeError(f'Training exited {code}; inspect {self.log_path}')
                self.event('actual_memory_failure_retry',exit_code=code,resume_step=step)
                self.child=None;self.recovering=True;self.history.clear();retry_after=time.time()+30;continue
            if self.child is None:
                _,step=self.latest()
                if step==2500:self.full_eval();return
                target=self.target
                if self.recovering and not policy.admitted(target,s):
                    target,detail=policy.choose(caps,host,self.mapping,preserve=False)
                if time.time()>=retry_after and policy.admitted(target,s):
                    self.launch(target,detail);self.recovering=False;self.target=list(target)
                else:
                    self.record('waiting_capacity',resources=s,capacity=detail,candidate_rank_to_gpu=target,
                        required_host_mib=policy.required_host(len(target)),retry_after=retry_after)
                    time.sleep(self.args.poll_seconds);continue
            states=s['workers'];step=max([x.get('step',0) for x in states]+[0])
            self.record('stage_II',stage_II_step=step,child_pid=self.child.pid,training_pids=[x['pid'] for x in states],
                resources=s,capacity=detail,candidate_rank_to_gpu=candidate,pending_reconfiguration=None,
                training_log=str(self.log_path),next_capacity_check_step=(step//50+1)*50,
                low_free_gpu_warning=[g for g in set(self.mapping) if s['gpus'][g]['free_mib']<1024])
            time.sleep(self.args.poll_seconds)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--poll-seconds',type=float,default=5)
    args=parser.parse_args();scheduler=CheckpointScheduler(args)
    if args.poll_seconds<2:parser.error('Polling interval must be at least two seconds')
    if not args.execute:
        s=policy.snapshot(scheduler.out)
        print(json.dumps(dict(initial_rank_to_gpu=INITIAL,admitted=policy.admitted(INITIAL,s),required_host_mib=policy.required_host(5),snapshot=s),indent=2));return
    try:scheduler.run_forever()
    except BaseException as error:
        if scheduler.owns_locks:scheduler.record('failed',error=repr(error))
        raise
if __name__=='__main__':main()
