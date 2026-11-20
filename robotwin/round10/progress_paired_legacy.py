import json,time,os
from pathlib import Path
P=Path(__file__).resolve().parent/'online/status.json'
def score(k,n):return f'{k}/{n} success'+(f' ({100*k/n:.1f}%)' if n else ' (pending)')
while True:
    try:s=json.loads(P.read_text())
    except (OSError,ValueError):s={}
    os.system('clear');print('RoboTwin round10 | Original vs Compatible | GPU0')
    print('Phase:',s.get('phase','waiting'),' | Updated:',time.strftime('%H:%M:%S',time.localtime(s.get('updated_at',time.time()))))
    current=s.get('current',{});progress=s.get('progress',{});results=s.get('results',[]);active=s.get('active',{})
    done=sum(len(r.get('summary',{}).get('episodes',[])) for r in results)+progress.get('completed_episodes',0)
    print('['+'#'*(done//25)+'-'*(40-done//25)+f'] {done}/1000 episodes')
    if active:
        for arm,a in active.items():
            if a: print('Lane',arm,'GPU',a.get('gpu'),'|',a.get('task'),'|',a.get('mode'),'|',a.get('progress',{}).get('completed_episodes',0),'/10')
    if current:
        print('Current:',current.get('task'),'|',current.get('mode'),'|',current.get('arm'))
        print('Task progress:',str(progress.get('completed_episodes',0))+'/5 episodes |',score(progress.get('successes',0),progress.get('completed_episodes',0)))
        trace=P.parent/'jobs'/current['key']/'requests.jsonl'
        try:
            q=json.loads(trace.read_text().splitlines()[-1]);print('Latest query: seed',q['episode_seed'],'chunk',q['query_index'],'at',time.strftime('%H:%M:%S',time.localtime(q['time'])))
        except (OSError,ValueError,IndexError):pass
        print('No completed episode for:',round(s.get('no_episode_progress_seconds',0)),'sec / timeout 2400 sec')
    elif s.get('arm'):print('Loading/checking:',s['arm'])
    print()
    for arm in ['original','compatible']:
        rows=[r for r in results if r['arm']==arm and r['status']=='completed'];n=sum(len(r['summary']['episodes']) for r in rows);k=sum(r['summary']['successes'] for r in rows)
        if current.get('arm')==arm:n+=progress.get('completed_episodes',0);k+=progress.get('successes',0)
        print(f'{arm:12s}: '+score(k,n))
    print('Completed jobs:',sum(r['status']=='completed' for r in results),'/100 | Incomplete/skipped:',sum(r['status']!='completed' for r in results))
    print('Matched task/mode pairs:')
    for row in s.get('pairs',[]):print(row['task'],row['mode'],'| original',str(row['original'])+'/5 | compatible',str(row['compatible'])+'/5 | matched',row['matched'])
    if s.get('error'):print('ERROR:',s['error'])
    print('Results: robotwin/round10/online');time.sleep(5)
