import json,time,os
from pathlib import Path
P=Path(__file__).resolve().parent/'candidate_online/status.json'
def score(k,n):return f'{k}/{n} ({100*k/n:.1f}%)' if n else '--'
while True:
    try:s=json.loads(P.read_text())
    except (OSError,ValueError):s={}
    rows=s.get('results',[]);active=s.get('active',{})
    good=[r for r in rows if r['status']=='completed']
    complete=sum(r['summary']['n_episodes'] for r in good)
    live=sum(a.get('progress',{}).get('completed_episodes',0) for a in active.values() if a)
    done=complete+live;bars=min(40,done*40//1000)
    os.system('clear')
    print('RoboTwin Round10 | repaired candidate only | GPU0 + GPU3')
    print('Phase:',s.get('phase','waiting'),'Updated:',time.strftime('%F %T',time.localtime(s.get('updated_at',time.time()))))
    print('['+'#'*bars+'-'*(40-bars)+f'] {done}/1000 episodes (completed jobs + active)')
    print('Historical original: clean 2126/2300 = 92.43%; randomized 2017/2250 = 89.64%')
    print('Historical baseline available: 91/100 task-mode jobs. NOT strict paired comparison.')
    print('Candidate: round7 fixed_queries + round4/6 adapters; no Stage II restart.')
    for gpu in ('0','3','5a','5b'):
        a=active.get(gpu)
        if a:
            p=a.get('progress',{});n=p.get('completed_episodes',0)
            print(f"GPU{gpu}: {a['task']} {a['mode']} | {n}/10 | success {score(p.get('successes',0),n)} | no activity {a.get('no_activity_seconds',0):.0f}/1800s")
        else:print(f"GPU{gpu} (physical {gpu[0] if gpu.startswith('5') else gpu}): {s.get('lane_status',{}).get(gpu,s.get('servers',{}).get(gpu,{}).get('phase','waiting'))} | free {s.get('free_memory_mib',{}).get(gpu[0] if gpu.startswith('5') else gpu,'?')} MiB")
    print('Completed jobs:',len(good),'/100 | Terminal jobs:',s.get('terminal_jobs',0),'| Error/timeout attempts:',sum(r['status']!='completed' for r in rows))
    for mode in ('demo_clean','demo_randomized'):
        rr=[r for r in good if r['mode']==mode];n=sum(r['summary']['n_episodes'] for r in rr);k=sum(r['summary']['successes'] for r in rr)
        print(mode,'candidate completed subset:',score(k,n),'| not full-benchmark accuracy')
    print('Recent results:')
    for row in rows[-16:]:
        sm=row.get('summary',{});pr=row.get('partial_progress',{});k=sm.get('successes',pr.get('successes',0));n=sm.get('n_episodes',pr.get('completed_episodes',0));c=row.get('comparison',{})
        print(row['task'],row['mode'],row['status'],score(k,n),'historical:',score(c.get('baseline_successes') or 0,c.get('baseline_episodes',0)))
    try:review=json.loads((P.parent/'reviews/status.json').read_text())
    except (OSError,ValueError):review={}
    print('Immediate comparison: reviewed',review.get('completed_comparisons',0),'completed jobs | pending diagnoses/repairs',len(review.get('pending_repairs',[])))
    for c in review.get('cases',[])[-5:]:
        delta=c.get('delta_percentage_points');delta_text=f'{delta:+.1f}pp' if delta is not None else '--'
        print('Review:',c['key'],c['decision'],delta_text,'|',c.get('repair_stage'))
    if s.get('error'):print('ERROR:',s['error'])
    print('Results: robotwin/round10/candidate_online | Refresh: 5s')
    time.sleep(5)
