"""nohup entry: fixed 10-rank Stage II resume, then the existing full-evaluation handoff."""
import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
LOGS=WORK/'logs';RUN=WORK/'protocol_plus_v1/seed_2026';AUDIT=LOGS/'resume100_fixed10_gpu2567_20261030'
from a2_stage2_fixed.runtime import POLICY,CHECKPOINT_POLICY,atomic_json

def record(phase,**values):
 row=dict(time=time.time(),pid=os.getpid(),phase=phase,armed=True,
          stage_II_physical_gpus=POLICY['physical_gpus'],rank_to_gpu=POLICY['base_rank_to_gpu'],
          training_ranks=10,stage_II_steps=2500,checkpoint_policy=CHECKPOINT_POLICY,
          evaluation_workers=8,evaluation_tasks=10030,**values)
 atomic_json(LOGS/'auto_continuation_status.json',row);print(json.dumps(row),flush=True)

def main():
 with (LOGS/'.auto_continuation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  archive=AUDIT/'previous_state';archive.mkdir(exist_ok=False)
  for p in (RUN/'II').glob('*.json'):shutil.copy2(p,archive/('II_'+p.name))
  for name in ['pipeline_status.json']:
   p=RUN/name
   if p.exists():shutil.copy2(p,archive/name)
  exit_file=LOGS/'stage_II_exit_code.txt'
  if exit_file.exists():shutil.move(exit_file,archive/exit_file.name)
  cp=RUN/'II/step_000100'
  command=['bash',str(WORK/'start_stage_ii_fixed10.sh'),'--execute','--resume',str(cp)]
  record('starting_stage_II',resume=str(cp),command=command,log=str(LOGS/'train_stage_II_fixed10_gpu2567_20261030.log'))
  with (LOGS/'train_stage_II_fixed10_gpu2567_20261030.log').open('a') as stream:
   child=subprocess.Popen(command,cwd=WORK,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
   atomic_json(AUDIT/'launch.json',dict(time=time.time(),driver_pid=os.getpid(),launcher_pid=child.pid,command=command,
       rank_to_gpu=POLICY['base_rank_to_gpu'],resume=str(cp),checkpoint_policy=CHECKPOINT_POLICY,launch_method='nohup',sid=os.getsid(0)))
   while child.poll() is None:
    try:s=json.loads((LOGS/'stage_II_adaptive_status.json').read_text())
    except (OSError,ValueError):s={}
    record('stage_II',launcher_pid=child.pid,admission_phase=s.get('phase'),step=s.get('step'),worker_pids=s.get('worker_pids',s.get('resident_pids',[])))
    time.sleep(20)
  exit_file.write_text(str(child.returncode)+'\n')
  if child.returncode:record('failed_stage_II',exit_code=child.returncode);raise SystemExit(child.returncode)
  record('starting_full_evaluation')
  command=['/home/wangchaoyang/envs/LaWAM/bin/python','-B','-u',str(WORK/'after_training_full_eval.py'),
           '--execute','--workers','8','--poll-seconds','10','--training-exit-file',str(exit_file)]
  result=subprocess.run(command,cwd=WORK,stdin=subprocess.DEVNULL,pass_fds=(lock.fileno(),))
  record('completed' if result.returncode==0 else 'failed_full_evaluation',exit_code=result.returncode)
  raise SystemExit(result.returncode)
if __name__=='__main__':main()
