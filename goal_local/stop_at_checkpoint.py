"""Stop only the specified training process group after a complete target checkpoint."""
import argparse,ctypes,fcntl,json,os,select,signal,struct,time
from pathlib import Path
from .eval_common import ROOT,atomic_json,read_json,model_metadata

def identity(pid):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text()
  fields=raw[raw.rfind(')')+2:].split()
  if fields[0]=='Z':return None
  return dict(start_ticks=int(fields[19]),pgid=int(fields[2]))
 except (FileNotFoundError,ProcessLookupError):return None

def checkpoint_ready(path,step):
 marker=read_json(path/'checkpoint_complete.json',{})
 try:return marker.get('step')==step and all((path/name).stat().st_size>0 for name in ('model.pt','trainer.pt'))
 except FileNotFoundError:return False

class CheckpointWatch:
 def __init__(self,run,step):
  self.path=Path(run)/f'step_{step:06d}';self.libc=ctypes.CDLL(None,use_errno=True)
  self.libc.inotify_init1.argtypes=[ctypes.c_int];self.libc.inotify_init1.restype=ctypes.c_int
  self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];self.libc.inotify_add_watch.restype=ctypes.c_int
  self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
  if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
  self.root_watch=self.add(Path(run));self.target_watch=None;self.ensure_target()
 def add(self,path):
  wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),0x00000100|0x00000080|0x00000008)
  if wd<0:raise OSError(ctypes.get_errno(),f'inotify watch {path}')
  return wd
 def ensure_target(self):
  if self.target_watch is None and self.path.is_dir():self.target_watch=self.add(self.path)
 def wait(self,timeout=1):
  self.ensure_target()
  if select.select([self.fd],[],[],timeout)[0]:
   try:data=os.read(self.fd,65536)
   except BlockingIOError:data=b''
   offset=0
   while offset+16<=len(data):
    wd,mask,cookie,length=struct.unpack_from('iIII',data,offset)
    name=data[offset+16:offset+16+length].split(b'\0',1)[0].decode()
    offset+=16+length
    if wd==self.root_watch and name==self.path.name:self.ensure_target()
 def close(self):os.close(self.fd)

def halt_after_checkpoint(run,pid,expected,step):
 if identity(pid)!=expected:raise RuntimeError('Training PID identity changed; refusing to signal')
 command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 if b'goal_local.train' not in command or b'II' not in command:raise RuntimeError('Not the specified Stage II trainer')
 if Path(f'/proc/{pid}/cwd').resolve()!=ROOT.resolve():raise RuntimeError('Unexpected training workspace')
 target=run/f'step_{step:06d}'
 if not checkpoint_ready(target,step):raise RuntimeError('Target checkpoint is not complete')
 # Freeze the trainer before inspecting weights; never interrupt a checkpoint write.
 os.kill(pid,signal.SIGSTOP);committed=False
 try:
  meta=model_metadata(target/'model.pt')
  if meta['mode']!='student' or meta['step']!=step:raise RuntimeError('Target student weight metadata mismatch')
  if identity(pid)!=expected:raise RuntimeError('Training identity changed during final checkpoint verification')
  status=read_json(run/'status.json',{})
  os.killpg(expected['pgid'],signal.SIGTERM);committed=True
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
  limit=time.monotonic()+30
  while identity(pid)==expected and time.monotonic()<limit:time.sleep(.2)
  if identity(pid)==expected:raise RuntimeError('Trainer did not exit after SIGTERM')
  status.update(phase='completed',step=step,micro=0,eta_seconds=0,updated_at=time.time(),
   checkpoint=str(target),configured_total_steps=status.get('total_steps',10000),total_steps=step,
   stop_reason='User requested stop after complete step-4000 checkpoint; original LR schedule retained',
   stopped_by='goal_local.stop_at_checkpoint')
  atomic_json(run/'status.json',status)
  return dict(target_step=step,checkpoint=str(target),model_step=meta['step'],training_stopped=True)
 finally:
  if not committed and identity(pid)==expected:
   try:os.kill(pid,signal.SIGCONT)
   except ProcessLookupError:pass

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args()
 run=Path(a.run).resolve()
 if not run.is_relative_to(ROOT/'runs/goal_local') or a.step<1 or a.step%250:raise ValueError('Expected local run and a 250-step checkpoint boundary')
 if run.name!='stage_II_warm250_20260921':raise ValueError('This guard is restricted to the requested student run')
 lock=(run/'stop_guard.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 expected=identity(a.pid)
 if expected is None:raise RuntimeError('Training process is not alive')
 command=Path(f'/proc/{a.pid}/cmdline').read_bytes().split(b'\0')
 if b'goal_local.train' not in command or b'II' not in command:raise RuntimeError('Not a Stage II training PID')
 request=dict(target_step=a.step,training_pid=a.pid,training_identity=expected,guard_pid=os.getpid(),created_at=time.time(),
  keep_original_steps_argument=True,keep_lr_schedule=True,save_every=250,eval_every=250,stop_only_after_complete_checkpoint=True)
 atomic_json(run/'stop_at_checkpoint.json',request)
 watch=CheckpointWatch(run,a.step);last_report=0
 try:
  while True:
   if identity(a.pid)!=expected:
    atomic_json(run/'stop_guard_status.json',dict(**request,phase='training_exited_before_guard_stop',time=time.time()))
    raise RuntimeError('Training process exited or changed before target stop')
   if checkpoint_ready(watch.path,a.step):
    result=halt_after_checkpoint(run,a.pid,expected,a.step)
    atomic_json(run/'stop_guard_status.json',{**request,**result,'phase':'completed','time':time.time()})
    print(json.dumps(result),flush=True);return
   if time.monotonic()-last_report>=15:
    state=read_json(run/'status.json',{});payload=dict(**request,phase='waiting_for_complete_checkpoint',time=time.time(),current_step=state.get('step'),training_phase=state.get('phase'))
    atomic_json(run/'stop_guard_status.json',payload);print(json.dumps({k:payload[k] for k in ('phase','current_step','target_step','training_pid','time')}),flush=True)
    last_report=time.monotonic()
   watch.wait()
 finally:watch.close()
if __name__=='__main__':main()

