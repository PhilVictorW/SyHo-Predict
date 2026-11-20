"""Preserve every completed periodic model checkpoint without touching training."""
import argparse,json,os,time
from pathlib import Path

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def retain_one(src):
    src=Path(src)
    if not (src/'checkpoint_complete.json').is_file():return False
    dest=src.parent/'retained_checkpoints'/src.name
    dest.mkdir(parents=True,exist_ok=True)
    files=[src/'model.pt',*[p for p in (src/'processor').rglob('*') if p.is_file()]]
    for p in files:
        q=dest/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(p,q)
        except FileExistsError:pass
        assert p.stat().st_size==q.stat().st_size, f'Archive mismatch: {q}'
    atomic(dest/'retention_complete.json',dict(step=int(src.name.split('_')[1]),model_bytes=(dest/'model.pt').stat().st_size,retained='model.pt and processor; optimizer state is not required for weights',updated=time.time()))
    return True

def run(root,once=False):
    root=Path(root)
    while True:
        try:
            for p in sorted(root.glob('step_*')):
                if not (root/'retained_checkpoints'/p.name/'retention_complete.json').exists():
                    if retain_one(p):print('Preserved',p.name,flush=True)
            saved=sorted(p.parent.name for p in (root/'retained_checkpoints').glob('step_*/retention_complete.json'))
            atomic(root/'retention_status.json',dict(phase='running',pid=os.getpid(),updated=time.time(),saved=saved,interval_seconds=5))
        except Exception as exc:
            atomic(root/'retention_status.json',dict(phase='error',pid=os.getpid(),updated=time.time(),error=repr(exc)))
            print('Retention error:',repr(exc),flush=True)
            if once:raise
        if once:break
        time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();run(a.run,a.once)
