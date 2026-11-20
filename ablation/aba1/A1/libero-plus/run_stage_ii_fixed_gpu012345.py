"""Fixed six-rank Stage II on physical GPUs 0..5, followed by full evaluation."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import run_stage_ii_auto_gpu as base
from a1_local.config_runtime import atomic_json
TARGET=list(range(6))

class FixedScheduler(base.Scheduler):
    def __init__(self,args):
        super().__init__(args)
        report=base.read(base.LOGS/'requested_gpu123_capacity.json')
        if len(report.get('requested_mapping',[]))!=6:raise ValueError('Six-rank measured host-memory report required')
        self.required_host=int(report['required_host_after_exit_mib']);self.mapping=TARGET.copy()
    def record(self,phase,**kw):
        super().record(phase,**kw)
        for name in ('gpu_scheduler_status.json','auto_continuation_status.json'):
            path=base.LOGS/name;state=base.read(path)
            state.update(monitor_script=str(Path(__file__).resolve()),mode='fixed',requested_rank_to_gpu=TARGET,automatic_resize=False)
            atomic_json(path,state)
    def admission(self,s):
        caps={g:v['capacity_mib'] for g,v in s['gpus'].items()};host=s['host_available_mib']+s['own_host_mib']
        gpu_ok=all(s['gpus'][g]['free_mib']>=base.BUDGET+base.RESERVE for g in TARGET);host_ok=host>=self.required_host
        return gpu_ok and host_ok,dict(capacity_mib=caps,gpu_slots={g:int(g in TARGET and caps[g]>=base.BUDGET+base.RESERVE) for g in caps},
            host_limit=6 if host_ok else 0,host_capacity_mib=host,required_host_mib=self.required_host,host_reserve_mib=base.HOST_RESERVE,
            rank_budget_mib=base.BUDGET,gpu_reserve_mib=base.RESERVE,requested_gpu_ready=gpu_ok,requested_host_ready=host_ok)
    def run_forever(self):
        self.locks=[(base.LOGS/'.auto_continuation.lock').open('a'),(self.run/'.pipeline.lock').open('a')]
        for lock in self.locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.owns_locks=True
        if base.worker_states(self.out):raise RuntimeError('Previous training workers still alive')
        for name in ('gpu_scheduler.pid','auto_continuation.pid'):(base.LOGS/name).write_text(str(os.getpid())+'\n')
        self.record('verifying_teacher');base.check_teacher(self.teacher,self.cfg);base.check_complete(self.teacher,verify_hash=True)
        self.event('fixed_monitor_started',rank_to_gpu=TARGET,automatic_resize=False)
        critical_count=0;retry_after=0
        while True:
            try:s=base.sample(self.out)
            except (OSError,subprocess.SubprocessError,ValueError) as e:
                self.event('sample_error',error=repr(e));time.sleep(self.args.poll_seconds);continue
            admitted,detail=self.admission(s)
            with (base.LOGS/'gpu_scheduler_samples.jsonl').open('a') as f:f.write(json.dumps(s)+'\n')
            if self.child and self.child.poll() is not None:
                code=self.child.returncode;_,step=self.latest()
                if code==0 and step==2500:self.full_eval();return
                log=self.log_path.read_text(errors='replace')[-30000:]
                memory_failure=any(word in log.lower() for word in ('out of memory','memory admission floor','cannot allocate memory','memoryerror'))
                self.stop_owned()
                if not memory_failure:raise RuntimeError(f'Training exited {code}; inspect {self.log_path}')
                self.event('fixed_memory_retry',exit_code=code,resume_step=step,rank_to_gpu=TARGET)
                self.child=None;retry_after=time.time()+60;critical_count=0
            if self.child is None:
                _,step=self.latest()
                if step==2500:self.full_eval();return
                if admitted and time.time()>=retry_after:self.launch(TARGET,detail)
                else:
                    self.record('waiting_capacity',resources=s,capacity=detail,candidate_rank_to_gpu=TARGET,retry_after=retry_after)
                    time.sleep(self.args.poll_seconds);continue
            states=s['workers'];step=max([state.get('step',0) for state in states]+[0])
            critical=any(s['gpus'][g]['free_mib']<1024 for g in TARGET) or s['host_available_mib']<4096
            critical_count=critical_count+1 if critical else 0
            if critical_count>=2:
                self.event('fixed_pressure_stop',step=step,resources=s);self.stop_owned()
                self.child=None;retry_after=time.time()+60;critical_count=0
            self.record('stage_II' if self.child else 'waiting_capacity',stage_II_step=step,child_pid=self.child.pid if self.child else None,
                training_pids=[state['pid'] for state in states],resources=s,capacity=detail,candidate_rank_to_gpu=TARGET,
                pending_reconfiguration=None,training_log=str(self.log_path))
            time.sleep(self.args.poll_seconds)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--poll-seconds',type=float,default=5)
    a=p.parse_args()
    if a.poll_seconds<2:p.error('Poll interval must be at least two seconds')
    scheduler=FixedScheduler(a)
    if not a.execute:
        admitted,detail=scheduler.admission(base.sample(scheduler.out))
        print(json.dumps(dict(rank_to_gpu=TARGET,admitted=admitted,capacity=detail,training_started=False),indent=2));return
    try:scheduler.run_forever()
    except BaseException as e:
        if scheduler.owns_locks:scheduler.stop_owned();scheduler.record('failed',error=repr(e))
        raise
if __name__=='__main__':main()
