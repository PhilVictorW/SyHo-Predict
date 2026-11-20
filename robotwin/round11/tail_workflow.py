"""Sequential uniform-vs-tail continuation supervisor on GPU2."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'robotwin/round11';OUT=R/'tail_workflow';PY='/home/wangchaoyang/envs/LaWAM/bin/python'
MOD='robotwin.round11.tail_adapter_train'
def read(p):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {}
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def ident(pid):
    try:
        q=Path('/proc')/str(pid);stat=(q/'stat').read_text().rsplit(') ',1)[1].split();cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode();
        return dict(pid=pid,start_ticks=stat[19],command=cmd) if stat[0]!='Z' else None
    except OSError:return None
def same(rec):
    x=ident(rec.get('pid')) if rec else None
    return bool(x and x['start_ticks']==rec.get('start_ticks') and x['command']==rec.get('command'))
def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'plan.json',dict(gpu=2,base='action_adapter_200/step_000200',modes=['uniform','tail'],steps=200,lr=5e-6,
      effective_batch=4,tail_sampling='50 percent of anchors from final 36 frames; remaining 50 percent uniform',
      order='uniform then tail',save_every=50,after='three-way independent confirmation',no_online_before_tail_safety=True))
    state=dict(pid=os.getpid(),phase='starting',mode=None,step=0,total_steps=200,updated_at=time.time())
    child=None
    def update(**kw):state.update(kw);state['updated_at']=time.time();write(OUT/'status.json',state)
    def launch(mode):
        nonlocal child
        d=R/f'tail_adapter_200_{mode}';d.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
          HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=str(ROOT),ADAPTER_SAMPLING_MODE=mode)
        with (d/'run.log').open('a') as log:
            child=subprocess.Popen(['nice','-n','10','ionice','-c','2','-n','7',PY,'-u','-m',MOD],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rec=None
        for _ in range(20):
            rec=ident(child.pid)
            if rec and ('tail_adapter_train' in rec['command'] or child.poll() is not None): break
            time.sleep(.25)
        if not rec:
            raise RuntimeError('Worker exited before identity was observable')
        write(OUT/f'{mode}_process.json',rec)
        return child
    modes=['uniform','tail']
    try:
        for mode in modes:
            d=R/f'tail_adapter_200_{mode}';s=read(d/'status.json')
            if s.get('phase')=='completed':
                update(phase='mode_already_completed',mode=mode,step=s.get('step',200));continue
            if (d/'manifest.json').exists() and s.get('phase') not in ('completed',):
                update(phase='mode_has_existing_run',mode=mode,step=s.get('step',0),action='refusing duplicate launch');return
            child=launch(mode);update(phase='training',mode=mode,step=0,child=read(OUT/f'{mode}_process.json'))
            while child.poll() is None:
                s=read(d/'status.json');update(phase='training',mode=mode,step=s.get('step',0),worker_phase=s.get('phase'),micro=s.get('micro',0),child=read(OUT/f'{mode}_process.json'))
                time.sleep(15)
            rc=child.wait();s=read(d/'status.json');child=None
            if rc!=0 or s.get('phase') not in ('completed','stopped_by_regression_guard'):
                update(phase='failed',mode=mode,returncode=rc,worker=s);return
            update(phase='mode_completed',mode=mode,step=s.get('step',200))
        update(phase='training_completed',mode=None,step=200,next='three_way_confirmation')
    except BaseException as e:
        update(phase='failed',error=repr(e));(OUT/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
