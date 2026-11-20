"""Read-only tmux dashboard; evaluation is owned by the nohup controller."""
import argparse
import sqlite3
import sys
from common import *


def bar(n, total, width=24):
    count = min(width, int(width*n/total)) if total else 0
    return '[' + '#' * count + '-' * (width-count) + ']'


def expanded_plus_summary():
    with sqlite3.connect('file:'+str(RUN/'plus'/'queue.sqlite')+'?mode=ro', uri=True, timeout=30) as db:
        counts = dict(db.execute('SELECT state,COUNT(*) FROM jobs GROUP BY state'))
        successes = db.execute("SELECT COALESCE(SUM(success),0) FROM jobs WHERE state='done'").fetchone()[0]
        rows = db.execute('SELECT gpu,state,COUNT(*),COALESCE(SUM(success),0) FROM jobs GROUP BY gpu,state').fetchall()
    per_gpu = []
    for gpu in (0,1,2,3,5,6,7):
        selected = [r for r in rows if r[0]==gpu]
        per_gpu.append(dict(gpu=gpu,total=sum(r[2] for r in selected),
                           completed=sum(r[2] for r in selected if r[1]=='done'),
                           successes=sum(r[3] for r in selected if r[1]=='done')))
    return dict(completed=counts.get('done',0),successes=successes,counts=counts,gpus=per_gpu)


def render():
    c = read(RUN/'controller.json', {})
    refill = read(RUN/'refill.json', {})
    extra = refill.get('slots', [])
    replaced = {(s['variant'], s['gpu']) for s in extra}
    displayed_slots = [s for s in c.get('slots', []) if (s['variant'], s['gpu']) not in replaced] + extra
    handoff = read(RUN/'plus7.json', {})
    managed = handoff.get('slots', [])
    replaced = {(s['variant'],s['gpu']) for s in managed}
    displayed_slots = [s for s in displayed_slots if (s['variant'],s['gpu']) not in replaced] + managed
    assignment = {k:list(v) for k,v in GPUS.items()}
    if handoff.get('activated'):
        assignment['plus'] = handoff['gpu_assignment']['plus']
    free_values = dict(c.get('gpu_free_mib', {}))
    free_values.update(handoff.get('gpu_free_mib', {}))
    launch = read(RUN/'launch.json', {})
    age = time.time()-c.get('updated', time.time())
    lines = [f'LIBERO-PRO | LIBERO 97.95% vs PLUS 89.62% | {time.strftime("%Y-%m-%d %H:%M:%S")}',
             '160 tasks x 50 initial states = 8000 episodes/model | Object / Swap / Language / Task',
             f'nohup controller PID {c.get("pid", "starting")} | heartbeat {age:.0f}s | elapsed {(time.time()-launch.get("started",time.time()))/3600:.2f}h',
             f'Maximum 2 inference replicas/GPU. Pair admission: >= {c.get("required_pair_free_mib",15024)/1024:.2f} GiB free. GPU4 excluded.', '']
    if refill:
        lines.insert(3, f'Refill controller PID {refill.get("pid")} | heartbeat {time.time()-refill.get("updated",0):.0f}s | {refill.get("phase")}')
    if handoff:
        lines.insert(3, f'PLUS seven-GPU handoff PID {handoff.get("pid")} | {handoff.get("phase")} | heartbeat {time.time()-handoff.get("updated",0):.0f}s')
        if not handoff.get('activated'):
            lines.insert(4, f'Trigger: LIBERO {handoff.get("libero_counts",{}).get("done",0)}/8000 complete + GPUs 0/1/2/3 released; then PLUS x2 on 0/1/2/3/5/6/7.')
    for variant in assignment:
        data = expanded_plus_summary() if variant=='plus' and handoff.get('activated') else read(RUN/variant/'summary.json', {})
        n, k = data.get('completed', 0), data.get('successes', 0)
        sr = f'{k/n:.2%}' if n else '--'
        label = 'LIBERO (libro)' if variant == 'libero' else 'PLUS'
        lines.append(f'{label:14s} {bar(n,8000,32)} {n:4d}/8000 ({n/80:5.2f}%) | success {k}/{n} = {sr}')
        if variant=='libero' and handoff.get('activated'):
            lines.extend(['All LIBERO episodes complete; GPUs 0/1/2/3 transferred to PLUS.', ''])
            continue
        lines.append('GPU  active/load  progress                   done/assigned  success   SR      free GiB  status')
        by_gpu = {r['gpu']: r for r in data.get('gpus', [])}
        for gpu in assignment[variant]:
            r = by_gpu.get(gpu, {})
            n, k, total = r.get('completed', 0), r.get('successes', 0), r.get('total', 0)
            pair = [s for s in displayed_slots if s['gpu']==gpu and s['variant']==variant]
            phases = '/'.join(s['phase'] for s in pair) or 'starting'
            sr = f'{k/n:6.2%}' if n else '    --'
            free = free_values.get(str(gpu), 0)/1024
            active = sum(s['phase']=='running' for s in pair)
            loading = sum(s['phase']=='loading' for s in pair)
            lines.append(f'{gpu:3d}   {active}/2 +{loading}    {bar(n,total)} {n:4d}/{total:<5d}   {k:4d}    {sr}   {free:6.2f}   {phases}')
            details=[]
            for s in pair:
                p=s.get('progress', {});m=s.get('model', {})
                if s['phase']=='running':
                    details.append(f'w{s["rank"]}: job {p.get("job_id","--")} step {p.get("action_step",0)} calls {m.get("calls",0)} ({p.get("phase","starting")})')
                else:
                    details.append(f'w{s["rank"]}: {s["phase"]}')
            lines.append('      '+' | '.join(details))
        counts=data.get('counts',{})
        lines.append(f'  Infrastructure errors: {counts.get("failed",0)}; excluded from success-rate denominator. Retries are automatic.')
        lines.append('')
    lines.extend(['Waiting GPUs are monitored by nohup every 10 seconds and start automatically when both replicas fit.',
                  'Each replica has one model process and one simulator helper. Success rates are provisional.',
                  f'Results: {RUN}', 'Detach: Ctrl-b then d. Closing tmux does not stop evaluation.'])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        if not args.once: print('\033[H\033[2J',end='')
        print(render(),flush=True)
        if args.once: break
        time.sleep(3)
