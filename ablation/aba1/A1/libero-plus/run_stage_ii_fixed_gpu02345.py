"""Resume step 50 on five fixed GPUs, then run full evaluation; no GPU monitor."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import run_stage_ii_auto_gpu as base
from a1_local.config_runtime import atomic_json
TARGET=[0,2,3,4,5]
base.RESERVE=0

class FixedRun(base.Scheduler):
    def __init__(self,args):
        super().__init__(args)
        self.mapping=TARGET.copy();self.resume_checkpoint=self.out/'step_000050'
    def latest(self):
        return self.resume_checkpoint,50
    def record(self,phase,**values):
        state=dict(time=time.time(),pid=os.getpid(),phase=phase,
            monitor_script=str(Path(__file__).resolve()),armed=True,mode='fixed',
            physical_gpus=TARGET,rank_to_gpu=TARGET,requested_rank_to_gpu=TARGET,
            training_ranks=5,generation=self.generation,stage_I_steps=3000,stage_II_steps=2500,
            resume_step=50,evaluation_workers=5,evaluation_tasks=10030,evaluation_gpus=TARGET,
            checkpoint_policy='every 50 steps, retain latest complete only',
            automatic_resize=False,gpu_monitoring=False,automatic_restart=False,
            active_group_termination=False,pending_reconfiguration=None,
            training_log=str(self.log_path) if self.log_path else None,**values)
        for name in ('gpu_scheduler_status.json','auto_continuation_status.json'):
            atomic_json(base.LOGS/name,state)
    def full_eval(self):
        base.check_complete(self.out/'step_002500',require_resume=True,verify_hash=True)
        receipt=base.LOGS/'stage_II_exit_code.txt';receipt.write_text('0\n')
        atomic_json(self.run/'pipeline_status.json',dict(phase='completed',
            completed_stages=['I','II'],teacher_step=3000,student_step=2500,
            time=time.time(),rank_to_gpu=TARGET))
        env=os.environ.copy();env['A1_EVALUATION_GPUS']=','.join(map(str,TARGET))
        command=[self.cfg['model_python'],'-B','-u',str(base.WORK/'after_training_full_eval.py'),
            '--execute','--workers','5','--training-exit-file',str(receipt)]
        self.record('starting_full_evaluation')
        with open(os.devnull) as stdin:
            self.child=subprocess.Popen(command,cwd=base.WORK,env=env,stdin=stdin,
                start_new_session=True,pass_fds=(self.locks[0].fileno(),))
        self.record('full_evaluation',child_pid=self.child.pid)
        code=self.child.wait();result=base.read(base.evaluation.STATE)
        if code or result.get('phase')!='completed' or result.get('completed')!=10030:
            raise RuntimeError('Full evaluation incomplete; inspect after_training_full_eval.log')
        self.record('completed',completed=10030,success_rate=result.get('success_rate'))
    def run_forever(self):
        self.locks=[(base.LOGS/'.auto_continuation.lock').open('a'),(self.run/'.pipeline.lock').open('a')]
        for lock in self.locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.owns_locks=True
        if base.worker_states(self.out):raise RuntimeError('Previous training workers still alive')
        meta,_=base.check_complete(self.resume_checkpoint,require_resume=True)
        if meta['step']!=50 or meta['config']!=self.cfg or meta['extra']['stage']!='II':
            raise ValueError('Requires this run complete Stage II step 50')
        base.check_teacher(self.teacher,self.cfg)
        for name in ('auto_continuation.pid','fixed_gpu02345.pid'):
            (base.LOGS/name).write_text(str(os.getpid())+'\n')
        self.record('starting_stage_II')
        # The reused launch helper archives stale control JSON and serializes loads.
        # Its dynamic scheduler loop is never called.
        self.launch(TARGET,dict(fixed=True,gpu_monitoring=False,automatic_resize=False))
        for name in ('autoscale_control.json','elastic_reconfigure.json','elastic_ready.json'):
            if (self.out/name).exists():raise RuntimeError('Stale dynamic control file: '+name)
        launch=base.read(base.LOGS/'launch_record.json')
        launch.update(mode='fixed',automatic_resize=False,gpu_monitoring=False,
            automatic_restart=False,evaluation_gpus=TARGET,evaluation_workers=5)
        atomic_json(base.LOGS/'launch_record.json',launch)
        self.record('stage_II',child_pid=self.child.pid,stage_II_step=50)
        # A blocking child wait only: no GPU polling, resizing, migration or retry.
        code=self.child.wait()
        (base.LOGS/'stage_II_exit_code.txt').write_text(str(code)+'\n')
        if code:
            atomic_json(self.run/'pipeline_status.json',dict(phase='failed',stage='II',
                time=time.time(),exit_code=code,rank_to_gpu=TARGET,training_log=str(self.log_path)))
            raise RuntimeError(f'Training exited {code}; automatic retry disabled: {self.log_path}')
        self.full_eval()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run=FixedRun(args)
    if not args.execute:
        print(json.dumps(dict(rank_to_gpu=TARGET,resume=str(run.resume_checkpoint),
            automatic_resize=False,gpu_monitoring=False,automatic_restart=False,evaluation_workers=5),indent=2));return
    try:run.run_forever()
    except BaseException as error:
        if run.owns_locks:run.record('failed',error=repr(error))
        raise
if __name__=='__main__':main()
