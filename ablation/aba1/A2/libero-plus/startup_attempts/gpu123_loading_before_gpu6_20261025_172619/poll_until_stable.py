"""Observe startup until six synchronized updates; never changes or stops training."""
import argparse,json,math,os,subprocess,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
RUN=WORK/'protocol_plus_v1/seed_2026'
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}
def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1][0] not in ('Z','X')
    except OSError:return False
def snapshot():
    states=[read(RUN/'I'/f'status_rank{r}.json') for r in range(6)]
    metrics=[]
    try:
        for line in (RUN/'I/metrics.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row.get('split')=='train':metrics.append(row)
    except FileNotFoundError:pass
    continuation=read(WORK/'logs/auto_continuation_status.json')
    return dict(time=time.time(),steps=metrics[-1]['step'] if metrics else 0,
        live_ranks=sum(alive(s.get('pid',0)) for s in states),
        ranks=[{k:s.get(k) for k in ('rank','pid','physical_gpu','phase','step','micro','validation_batch','validation_batches')} for s in states],
        continuation=continuation.get('phase'),continuation_alive=alive(continuation.get('pid',0)),
        metrics=metrics,pipeline=read(RUN/'pipeline_status.json'))
def validate(s):
    if s['pipeline'].get('phase')=='failed' or str(s['continuation']).startswith('failed'):
        raise RuntimeError('Training or automatic continuation failed')
    if s['live_ranks']!=6 or s['steps']<6:return False
    if any(r.get('physical_gpu')!=[1,1,2,2,3,3][i] for i,r in enumerate(s['ranks'])):
        raise ValueError('Incorrect rank placement')
    if any((r.get('step') or 0)<6 for r in s['ranks']):return False
    if not s['continuation_alive']:raise ValueError('Continuation exited')
    for row in s['metrics']:
        if not all(math.isfinite(row[k]) for k in ('loss','flow','grad_norm','lr_used')):
            raise ValueError('Nonfinite training metric')
    contract=read(RUN/'I/contract.json');audit=read(RUN/'I/parameter_audit.json');sync=read(RUN/'I/sync_audit.json')
    if contract.get('rank_to_gpu')!=[1,1,2,2,3,3] or contract.get('accumulation_per_rank')!=[6,6,5,5,5,5]:
        raise ValueError('Actual run contract mismatch')
    if audit.get('goal_slots')!=16 or audit.get('local_slots')!=0 or audit.get('plan_tokens')!=4:
        raise ValueError('Actual model is not Goal-only16 with Plan4')
    if sync.get('max_difference')!=0:raise ValueError('Ranks diverged')
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuid={r.split(',')[1].strip():int(r.split(',')[0]) for r in gpus.splitlines()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    actual={int(p):dict(gpu=uuid[u.strip()],used_mib=int(m)) for p,u,m in (r.split(',') for r in apps.splitlines())}
    for row in s['ranks']:
        if actual.get(row['pid'],{}).get('gpu')!=row['physical_gpu']:raise ValueError('CUDA process/device mismatch')
    s['gpu_allocations']={str(r['pid']):actual[r['pid']] for r in s['ranks']}
    s['sync_audit']=sync;s['parameter_audit']=audit
    return True
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        s=snapshot();done=validate(s)
        printable={k:v for k,v in s.items() if k not in ('metrics','pipeline')}
        with (WORK/'logs/startup_poll.jsonl').open('a') as f:f.write(json.dumps(printable)+'\n')
        print(json.dumps(printable),flush=True)
        if done:
            s['status']='stable_six_updates';s['observed_at']=time.time()
            (WORK/'logs/stage_I_stability_result.json').write_text(json.dumps(s,indent=2)+'\n')
            return
        if args.once:return
        time.sleep(20)
if __name__=='__main__':main()
