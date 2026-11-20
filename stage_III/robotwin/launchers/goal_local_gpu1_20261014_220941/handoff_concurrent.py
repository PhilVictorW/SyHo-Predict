"""Replace only the workflow controller; adopt its unchanged running GPU7 scene process."""
import sys,os,signal,time,subprocess,shlex
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from stage_III.robotwin.common import read_json,atomic_json
from resume_ready import process_identity
if sys.argv[1:]!=["--execute"]:raise SystemExit("--execute required")
record=read_json(HERE.parent/"latest_gpu1.json");run=Path(record["run_dir"])
active=read_json(HERE/"active_execution.json");old=process_identity(active["pid"])
assert old and old["uid"]==os.getuid() and str(HERE/"resume_ready.py") in old["command"]
session="robotwin_stageIII_checks_gpu7"
assert subprocess.run(["tmux","has-session","-t",session],capture_output=True).returncode!=0
stopped=False;killed=False
try:
 os.kill(old["pid"],signal.SIGSTOP);stopped=True
 assert process_identity(old["pid"])["start_ticks"]==old["start_ticks"]
 children=[]
 for p in Path("/proc").iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/"stat").read_text().split(") ",1)[1].split()
   if int(fields[1])==old["pid"]:children.append(process_identity(int(p.name)))
  except (OSError,ValueError):pass
 assert len(children)==1,children
 child=children[0];args=shlex.split(child["command"])
 assert str(HERE/"prepare_parallel.py") in args and args[args.index("--gpu")+1]=="7"
 split=args[args.index("--split")+1];assert split in ("val","test")
 assert child["pgid"]==child["pid"] and child["uid"]==os.getuid()
 adoption=dict(run_dir=str(run),phase="prepare_"+split,gpu=7,process=child,
               old_controller=old,created_at=time.time(),purpose="preserve GPU7 scenes; independent GPU3 RL")
 atomic_json(HERE/"adopt_checks_concurrent.json",adoption)
 route=read_json(HERE/"gpu_route.json")
 route.update(training_gpu=3,checks_gpu=7,workflow_mode="concurrent_training",updated_at=time.time())
 atomic_json(HERE/"gpu_route.json",route)
 record.update(training_gpu=3,checks_gpu=7,training_config=str(HERE/"workflow_direct_gpu3.json"),
    training_tmux_session="robotwin_stageIII_train_gpu3",tmux_session=session,workflow_mode="concurrent_training")
 atomic_json(HERE.parent/"latest_gpu1.json",record)
 # SIGKILL targets only the paused CPU controller, bypassing its child-killing signal handler.
 os.kill(old["pid"],signal.SIGKILL);killed=True
 for _ in range(40):
  now=process_identity(old["pid"])
  if now is None or now["state"]=="Z":break
  time.sleep(.05)
 now=process_identity(child["pid"])
 assert now and now["start_ticks"]==child["start_ticks"] and now["state"]!="Z"
 cmd=[sys.executable,"-B","-u",str(HERE/"resume_ready.py"),"--gpu","7","--execute","--worker","--adopt-state",str(HERE/"adopt_checks_concurrent.json")]
 shell="exec "+shlex.join(cmd)+" >> "+shlex.quote(str(HERE/"checks_concurrent.log"))+" 2>&1"
 subprocess.run(["tmux","new-session","-d","-s",session,"-n","checks","-c",str(ROOT),shell],check=True)
 atomic_json(HERE/"concurrent_handoff.json",dict(old_controller=old,new_session=session,adopted_child=child,time=time.time(),gpu7_child_unchanged=True))
 print("GPU7场景验证进程保持运行，PID",child["pid"],"；仅替换CPU调度器，后续不会重复启动RL。",flush=True)
except BaseException:
 if stopped and not killed:os.kill(old["pid"],signal.SIGCONT)
 raise
