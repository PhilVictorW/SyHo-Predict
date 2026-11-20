"""Archive completed student model weights without touching the training process."""
import argparse,fcntl,os,shutil,time,traceback
from .eval_common import *
def archive(src):
 step=int(src.name.split('_')[-1]);dst=SNAPSHOTS/src.name
 if not (src/'checkpoint_complete.json').exists():return
 dst.mkdir(parents=True,exist_ok=True)
 marker=dst/'eval_snapshot_complete.json'
 # First pin the immutable source inode. Training may then prune its own name safely.
 if not (dst/'model.pt').exists():os.link(src/'model.pt',dst/'model.pt')
 if not (dst/'processor').exists():shutil.copytree(src/'processor',dst/'processor',copy_function=os.link)
 atomic_json(marker,dict(step=step,source=str(src),archived_at=time.time(),weights_only=True,model_bytes=(dst/'model.pt').stat().st_size))
def move_to_ssd(dst):
 if (dst/'model.pt').is_symlink():return
 step=int(dst.name.split('_')[-1]);cache=SSD/'weights'/dst.name;cache.mkdir(parents=True,exist_ok=True)
 target=cache/'model.pt'
 if shutil.disk_usage(SSD).free < (dst/'model.pt').stat().st_size+20*1024**3:raise RuntimeError('SSD free space below snapshot reserve')
 if not target.exists():
  tmp=cache/'model.tmp';shutil.copyfile(dst/'model.pt',tmp)
  if tmp.stat().st_size!=(dst/'model.pt').stat().st_size:raise RuntimeError('Snapshot copy size mismatch')
  meta=model_metadata(tmp)
  if meta['mode']!='student' or meta['step']!=step:raise RuntimeError('Snapshot is not the requested student step')
  tmp.replace(target)
 else:
  meta=model_metadata(target)
  if meta['mode']!='student' or meta['step']!=step:raise RuntimeError('Existing SSD snapshot mismatch')
 # Replace only our own hardlink, never any file in the training directory.
 link=dst/'model.ssd';link.unlink(missing_ok=True);link.symlink_to(target);link.replace(dst/'model.pt')
 marker=read_json(dst/'eval_snapshot_complete.json',{});marker.update(ssd_path=str(target),ssd_verified_at=time.time());atomic_json(dst/'eval_snapshot_complete.json',marker)
def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
 SSD.mkdir(parents=True,exist_ok=True);SNAPSHOTS.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
 lock=(SSD/'archive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  try:
   for src in sorted(TRAIN.glob('step_*')):
    step=int(src.name.split('_')[-1])
    if step>=2000 and step%250==0 and (src/'checkpoint_complete.json').exists():
     if not (SNAPSHOTS/src.name/'eval_snapshot_complete.json').exists():archive(src);print('archived',step,flush=True)
   for dst in sorted(SNAPSHOTS.glob('step_*')):
    if (dst/'eval_snapshot_complete.json').exists():move_to_ssd(dst)
   available=[int(x.parent.name.split('_')[-1]) for x in SNAPSHOTS.glob('step_*/eval_snapshot_complete.json') if (x.parent/'model.pt').exists()]
   status=read_json(TRAIN/'status.json',{});latest=max(available,default=0)
   missing=[s for s in range(2000,latest+1,250) if s not in available]
   atomic_json(OUTPUT/'archive_status.json',dict(time=time.time(),available=available,missing=missing,training_step=status.get('step'),training_pid=status.get('pid'),training_phase=status.get('phase'),ssd=str(SSD)))
  except Exception as e:
   traceback.print_exc();atomic_json(OUTPUT/'archive_error.json',dict(time=time.time(),error=repr(e)))
   if a.once:raise
  if a.once:return
  time.sleep(15)
if __name__=='__main__':main()
