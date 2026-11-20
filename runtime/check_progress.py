import json, time, sys
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[1]
run=root/'results/ablation/inference_without_lawm_and_queries/full_40x50_20260915'
state=json.loads((run/'queue/state.json').read_text())
report=json.loads((run/'ablation_summary.json').read_text())
print('status',state['status'],'heartbeat_age_sec',round(time.time()-state['updated_at'],1))
print('episodes',report['total_episodes'],'successes',report['total_successes'],'errors',report['errors'],'tasks',dict(Counter(t['status'] for t in state['tasks'].values())))
for gpu,info in state['gpus'].items():
    task=info.get('task'); n=0
    if task:
        p=run/'tasks'/task/'episodes.checkpoint.jsonl'
        n=len(p.read_text().splitlines()) if p.exists() else 0
    log=run/'queue'/('gpu'+gpu)/'server.log'
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    audit=[l for l in lines if 'VERIFIED' in l]
    print('GPU',gpu,info['status'],task,n,'/50','audit:',(audit[:2] if '--audit' in sys.argv else 'PASS') if audit else 'pending')
    if info['status'] in ['loading','error']: print(' last:',lines[-1][:250] if lines else 'no log')
