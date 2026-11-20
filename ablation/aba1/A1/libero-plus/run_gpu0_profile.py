"""Run one isolated Stage II worker through validation and two real updates."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
sys.path[:0]=[str(WORK),str(WORK.parents[3])]
from a1_local.config_runtime import load_config,atomic_json,run_root
from ablation.aba1.plus_common.environment import subprocess_environment

def main():
 cfg=load_config(WORK/'config.json','A1');run=run_root(cfg)
 output=WORK/'capacity_profiles'/time.strftime('gpu0_single_%Y%m%d_%H%M%S');output.mkdir(parents=True)
 env=subprocess_environment(cfg,'0',run/'runtime_cache')
 env.update(PYTHONPATH=str(WORK)+os.pathsep+env['PYTHONPATH'],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TORCH_GLOO_LAZY_INIT='1',MUJOCO_EGL_DEVICE_ID='0',TMPDIR='/tmp')
 command=[cfg['model_python'],'-B','-u','-m','a1_local.training','--config',str(WORK/'config.json'),'--stage','II','--gpus','0','--processes-per-gpu','1','--workers','1','--cpu-threads','2','--offload-saved-tensors','--share-frozen-encoders','--keep-models','1','--profile-updates','2','--profile-output',str(output),'--minimum-free-mib','16384','--execute']
 with (WORK/'logs/.gpu0_capacity.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  previous=WORK/'logs/gpu0_capacity_profile_result.json'
  if previous.exists():
   data=json.loads(previous.read_text());old_output=Path(data['profile_output'])
   previous.rename(old_output/'wrapper_result.json')
  with open(os.devnull) as stdin:
   p=subprocess.Popen(command,env=env,cwd=WORK,stdin=stdin,start_new_session=True)
  record=dict(time=time.time(),pid=os.getpid(),child_pid=p.pid,profile_output=str(output),command=command,launch_method='nohup',formal_training=False)
  atomic_json(WORK/'logs/gpu0_capacity_profile_launch.json',record)
  atomic_json(run/'pipeline_status.json',dict(phase='profiling',stage='II',time=time.time(),profile_output=str(output),command=command))
  peak=0;minimum_free=None;failure=None
  while p.poll() is None:
   try:
    gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
    free,used=map(int,gpu.strip().split(','));minimum_free=free if minimum_free is None else min(minimum_free,free)
    state=json.loads((output/'status_rank0.json').read_text()) if (output/'status_rank0.json').exists() else {}
    pid=state.get('pid');proc_mib=0
    if pid:
     apps=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
     for line in apps.splitlines():
      pieces=line.split(',')
      if int(pieces[0])==pid:proc_mib=int(pieces[1]);peak=max(peak,proc_mib)
    row=dict(time=time.time(),free_mib=free,used_mib=used,pid=pid,worker_mib=proc_mib,peak_worker_mib=peak,minimum_free_mib=minimum_free,phase=state.get('phase'),step=state.get('step'),micro=state.get('micro'))
    with (output/'resource_samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic_json(WORK/'logs/gpu0_capacity_profile_status.json',row)
   except (OSError,ValueError,subprocess.SubprocessError) as e:
    print('monitor:',repr(e),flush=True)
   time.sleep(3)
  result=dict(record,exit_code=p.returncode,ended=time.time(),peak_worker_mib=peak,minimum_free_mib=minimum_free)
  atomic_json(WORK/'logs/gpu0_capacity_profile_result.json',result)
  print(json.dumps(result),flush=True)
  return p.returncode
if __name__=='__main__':raise SystemExit(main())
