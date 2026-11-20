"""Adopt the current GPU3 model/rollout without stopping or rerunning either."""
import os,sys,time,signal,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from stage_III.robotwin.common import read_json,atomic_json,sha256
from timeout_runtime import process_identity,POLICY
if sys.argv[1:]!=['--execute']:raise SystemExit('--execute required')
assert read_json(HERE/'timeout_tests.json')['passed']
record=read_json(HERE.parent/'latest_gpu1.json');source=Path(record['run_dir']);run=source/'rl'
status=read_json(run/'status.json');old=process_identity(status['pid'])
assert old and old['uid']==os.getuid() and any(Path(x).name=='train_direct_gpu3.py' for x in shlex.split(old['command']))
assert str(old['command']).endswith('--execute ') or '--execute' in old['command']
session='robotwin_stageIII_train_gpu3_v2'
assert subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode!=0
stopped=False;killed=False
try:
 os.kill(old['pid'],signal.SIGSTOP);stopped=True
 assert process_identity(old['pid'])['start']==old['start']
 children=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   f=(p/'stat').read_text().split(') ',1)[1].split()
   if int(f[1])==old['pid']:children.append(process_identity(int(p.name)))
  except (OSError,ValueError):pass
 servers=[p for p in children if 'stage_III.robotwin.server' in shlex.split(p['command'])]
 workers=[p for p in children if 'stage_III.robotwin.rollout' in shlex.split(p['command'])]
 assert len(servers)==1 and len(workers)<=1,children
 server=servers[0];assert server['state']!='Z'
 checkpoint=read_json(run/'last_checkpoint.json');cp=Path(checkpoint['path']);digest=sha256(cp)
 assert read_json(str(cp)+'.complete.json')['sha256']==digest
 startup=read_json(run/'startup.json');assert sha256(HERE/'train_direct_gpu3.py')==startup['driver_sha256']
 adoption=dict(version=2,run_dir=str(run),old_controller=old,server=server,
    student_sha256=startup['student_sha256'],actor_sha256=None,checkpoint=str(cp),checkpoint_sha256=digest,
    rollout=None,time=time.time(),timeout_policy=POLICY)
 if workers:
  worker=workers[0];args=shlex.split(worker['command']);out=Path(args[args.index('--output')+1]).resolve()
  assert out.is_relative_to(run/'groups') and args[args.index('--gpu')+1]=='3'
  collection=read_json(out/'collection.json');progress=read_json(out/'progress.json')
  adoption['actor_sha256']=collection['identity']['request']['actor_sha256']
  adoption['rollout']=dict(process=worker,output=str(out),identity_hash=collection['identity_hash'],started_at=progress['started_at'])
 else:
  # No rollout is using the server: reload the committed Actor lazily for the next group.
  adoption['actor_sha256']=None
 for process in (server,*workers):assert process['uid']==os.getuid() and process['pgid']==process['pid']
 atomic_json(HERE/'adopt_timeout_v2.json',adoption)
 # Replace only the CPU coordinator, preserving both independently owned GPU process groups.
 os.kill(old['pid'],signal.SIGKILL);killed=True
 for _ in range(100):
  current=process_identity(old['pid'])
  if current is None or current['state']=='Z':break
  time.sleep(.05)
 assert process_identity(server['pid'])['start']==server['start']
 cmd=[sys.executable,'-B','-u',str(HERE/'train_direct_gpu3_v2.py'),
      '--config',str(HERE/'workflow_direct_gpu3.json'),'--student',str(source/'interaction/checkpoints/student_000200.pt'),
      '--source-run',str(source),'--train-manifest',str(source/'manifests/prepared_train.json'),'--run-dir',str(run),
      '--gpu','3','--defer-pretraining-evaluation','--execute','--resume',str(cp),
      '--adopt-runtime',str(HERE/'adopt_timeout_v2.json')]
 shell='exec '+shlex.join(cmd)+' >> '+shlex.quote(str(HERE/'train_gpu3.log'))+' 2>&1'
 subprocess.run(['tmux','new-session','-d','-s',session,'-n','train','-c',str(ROOT),shell],check=True)
 record.update(training_tmux_session=session,timeout_policy=POLICY,training_driver=str(HERE/'train_direct_gpu3_v2.py'))
 atomic_json(HERE.parent/'latest_gpu1.json',record)
 script=HERE/'start_direct_gpu3.sh';text=script.read_text()
 (HERE/'start_direct_gpu3_v1.sh').write_text(text)
 script.write_text(text.replace('train_direct_gpu3.py','train_direct_gpu3_v2.py'))
 atomic_json(HERE/'timeout_activation.json',dict(time=time.time(),old_controller=old['pid'],
     gpu3_model_pid_unchanged=server['pid'],gpu3_rollout_pid_unchanged=workers[0]['pid'] if workers else None,
     checkpoint=str(cp),checkpoint_sha256=digest,new_session=session,gpu7_unchanged=True))
 print('已切换CPU训练调度器；保留GPU3模型PID',server['pid'],'、采样PID',workers[0]['pid'] if workers else None,'；GPU7未改动。',flush=True)
except BaseException:
 if stopped and not killed:os.kill(old['pid'],signal.SIGCONT)
 raise
