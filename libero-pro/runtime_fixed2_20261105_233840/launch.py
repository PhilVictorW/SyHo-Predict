"""Launch a nohup supervisor in its own session; reuse an already live run."""
import fcntl
import subprocess
from common import *

runtime = Path(__file__).resolve().parent
lock = (RUN/'launch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX)
pidfile = RUN/'controller.pid'
pid = int(pidfile.read_text()) if pidfile.exists() else 0
try:
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    active = str(runtime/'controller.py') in cmd and Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','T')
except OSError:
    active = False
if not active:
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with (RUN/'nohup.log').open('a') as log:
        proc=subprocess.Popen(['nohup',MODEL_PYTHON,'-B','-u',str(runtime/'controller.py')],
            cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid=proc.pid
    pidfile.write_text(str(pid)+'\n')
    time.sleep(1)
    if proc.poll() is not None:
        raise RuntimeError(f'Controller exited {proc.returncode}; inspect {RUN}/nohup.log')
if subprocess.run(['tmux','has-session','-t','libero_pro_progress'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
    subprocess.run(['tmux','new-session','-d','-s','libero_pro_progress','-x','170','-y','42',
                    f'{MODEL_PYTHON} -B -u {runtime}/dashboard.py'],check=True)
    subprocess.run(['tmux','set-option','-t','libero_pro_progress','remain-on-exit','on'],check=True)
print(f'Controller PID: {pid} (existing={active})\nView: tmux attach -t libero_pro_progress')
