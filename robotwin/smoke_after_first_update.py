import json,os,sys,time
from pathlib import Path
run=Path(sys.argv[1]).resolve()
while not (run/'step_000001/model_complete.json').exists():
    try:status=json.loads((run/'status.json').read_text())
    except (OSError,ValueError):status={}
    if status.get('phase')=='failed':raise RuntimeError('Training failed before first student snapshot')
    time.sleep(5)
os.execv(sys.executable,[sys.executable,'-u','-m','robotwin.eval_queue','--run',str(run),'--smoke-checkpoint',str(run/'step_000001')])
