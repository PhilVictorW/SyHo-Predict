"""GPU0 Stage II with measured capacity, serialized loading, and full-eval handoff."""
import argparse,fcntl,json,math,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a1_local.config_runtime import load_config,run_root,checkpoint_path,stage_steps,atomic_json,runtime_placement,code_identity
from a1_local.preflight import plan,check_complete,check_teacher
from ablation.aba1.plus_common.environment import subprocess_environment
import after_training_full_eval as evaluation
LOGS=WORK/'logs'

def select_capacity(free_mib,peak_mib,host_available_mib,host_peak_mib,reserve_mib=8192):
    if min(free_mib,peak_mib,host_available_mib,host_peak_mib)<=0:raise ValueError('Missing resource measurement')
    budget=math.ceil(peak_mib*1.15+512)
    gpu_limit=max(0,math.floor((free_mib-reserve_mib)/budget))
    # Rank zero holds the CPU optimizer. Additional ranks still need CPU gradients,
    # model construction space, one data worker, and pinned activations.
    extra_host=max(8192,host_peak_mib-18*1024)+1024
    host_limit=max(0,1+math.floor((host_available_mib-16*1024-host_peak_mib)/extra_host))
    ranks=min(32,gpu_limit,host_limit)
    return dict(processes=ranks,gpu_limit=gpu_limit,host_limit=host_limit,
        measured_peak_mib=peak_mib,per_rank_budget_mib=budget,reserve_mib=reserve_mib,
        admission_mib=budget+reserve_mib,free_mib=free_mib,host_available_mib=host_available_mib,
        host_peak_mib=host_peak_mib,incremental_host_budget_mib=extra_host)

def free_gpu_mib():
    return int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-report',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    cfg=load_config(WORK/'config.json','A1');run=run_root(cfg)
    capacity=json.loads(args.capacity_report.read_text());n=int(capacity['processes'])
    if not 1<=n<=32:raise ValueError('Measured capacity does not admit a rank')
    spec=plan(cfg,'II',require_sources=True,placement=runtime_placement(cfg,'II',[0],n))
    command=[cfg['model_python'],'-B','-u','-m','a1_local.training','--config',str(WORK/'config.json'),
        '--stage','II','--gpus','0','--processes-per-gpu',str(n),'--workers','1','--cpu-threads','2',
        '--offload-saved-tensors','--share-frozen-encoders','--keep-models','1',
        '--minimum-free-mib',str(capacity['admission_mib']),'--execute']
    print(json.dumps(dict(plan=spec,command=command,capacity=capacity,training_started=args.execute),indent=2),flush=True)
    if not args.execute:return
    phase='preparing_stage_II'
    def record(current,**details):
        atomic_json(LOGS/'auto_continuation_status.json',dict(phase=current,pid=os.getpid(),time=time.time(),armed=True,
            monitor_script=str(Path(__file__).resolve()),physical_gpus=[0],training_ranks=n,
            stage_I_steps=3000,stage_II_steps=2500,evaluation_workers=6,evaluation_tasks=10030,
            evaluation_gpus=cfg['a1_training']['physical_gpus'],**details))
    with (LOGS/'.auto_continuation.lock').open('a') as flow_lock,(run/'.pipeline.lock').open('a') as pipeline_lock:
        fcntl.flock(flow_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(pipeline_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'II/arguments.json').exists() or (LOGS/'stage_II_exit_code.txt').exists():
            raise RuntimeError('Existing Stage II/receipt requires explicit inspection before restart')
        record(phase,capacity=capacity)
        atomic_json(LOGS/'auto_continuation_launch.json',dict(time=time.time(),pid=os.getpid(),sid=os.getsid(0),
            monitor_script=str(Path(__file__).resolve()),launch_method='nohup',training_ranks=n,physical_gpus=[0],
            capacity=capacity,stage_I_steps=3000,stage_II_steps=2500,evaluation_tasks=10030,evaluation_workers=6,
            evaluation_gpus=cfg['a1_training']['physical_gpus']))
        (LOGS/'auto_continuation.pid').write_text(str(os.getpid())+'\n')
        try:
            teacher=checkpoint_path(cfg,'I',3000)
            check_teacher(teacher,cfg);check_complete(teacher,verify_hash=True)
            needed=n*capacity['per_rank_budget_mib']+capacity['reserve_mib']
            while free_gpu_mib()<needed:
                record('waiting_gpu0_memory',required_free_mib=needed,capacity=capacity)
                time.sleep(10)
            env=subprocess_environment(cfg,'0',run/'runtime_cache')
            env.update(PYTHONPATH=str(WORK)+os.pathsep+env['PYTHONPATH'],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_GLOO_LAZY_INIT='1',MUJOCO_EGL_DEVICE_ID='0',TMPDIR='/tmp')
            atomic_json(run/'pipeline_plan.json',dict(plan=spec,command=command,capacity=capacity,config=cfg,code_identity=code_identity()))
            atomic_json(run/'pipeline_status.json',dict(phase='training',stage='II',time=time.time(),command=command,teacher_step=3000,physical_gpus=[0],world_size=n))
            with open(os.devnull) as stdin,(LOGS/'train_stageII_gpu0.log').open('a') as log:
                child=subprocess.Popen(command,cwd=WORK,env=env,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(flow_lock.fileno(),pipeline_lock.fileno()))
            (LOGS/'nohup_training.pid').write_text(str(child.pid)+'\n')
            atomic_json(LOGS/'launch_record.json',dict(time=time.time(),stage='II',physical_gpus=[0],world_size=n,
                processes_per_gpu=n,rank_to_gpu=[0]*n,launcher_pid=child.pid,launcher_sid=child.pid,launch_method='nohup',
                stdin='/dev/null',resume_step=0,teacher=str(teacher),stage_I_steps=3000,stage_II_steps=2500,
                model_loading='serialized_file_lock',capacity=capacity,save_every=50,keep_checkpoints_per_stage=1,
                training_log=str(LOGS/'train_stageII_gpu0.log'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                continuation=dict(enabled=True,pid=os.getpid(),stages=['II:2500','LIBERO-Plus full:10030'])))
            phase='stage_II'
            while child.poll() is None:
                states=[evaluation.read_json(p) for p in (run/'II').glob('status_rank*.json')]
                record(phase,child_pid=child.pid,stage_II_step=max([s.get('step',0) for s in states]+[0]),
                    training_pids=[s['pid'] for s in states if evaluation.live_process(s.get('pid',0),'a1_local.training')])
                time.sleep(10)
            receipt=LOGS/'stage_II_exit_code.txt';tmp=receipt.with_suffix('.tmp');tmp.write_text(str(child.returncode)+'\n');tmp.replace(receipt)
            if child.returncode!=0:raise RuntimeError('Stage II exited with code '+str(child.returncode))
            check_complete(checkpoint_path(cfg,'II',2500),verify_hash=True)
            atomic_json(run/'pipeline_status.json',dict(phase='completed',completed_stages=['I','II'],teacher_step=3000,student_step=2500,time=time.time()))
            phase='full_evaluation'
            eval_command=[cfg['model_python'],'-B','-u',str(WORK/'after_training_full_eval.py'),'--execute','--workers','6','--poll-seconds','10','--training-exit-file',str(receipt)]
            with open(os.devnull) as stdin:
                child=subprocess.Popen(eval_command,cwd=WORK,stdin=stdin,start_new_session=True,pass_fds=(flow_lock.fileno(),))
            while child.poll() is None:
                result=evaluation.read_json(evaluation.STATE)
                record(phase,child_pid=child.pid,evaluation_phase=result.get('phase'),completed=result.get('completed',0))
                time.sleep(10)
            result=evaluation.read_json(evaluation.STATE)
            if child.returncode or result.get('phase')!='completed' or result.get('completed')!=10030:
                raise RuntimeError('Full evaluation did not finish all 10030 tasks')
            record('completed',completed=10030,successes=result.get('successes'),success_rate=result.get('success_rate'))
        except BaseException as error:
            if phase!='full_evaluation':atomic_json(run/'pipeline_status.json',dict(phase='failed',stage='II',time=time.time(),error=repr(error)))
            record('failed',error=repr(error),failed_phase=phase)
            raise
if __name__=='__main__':main()
