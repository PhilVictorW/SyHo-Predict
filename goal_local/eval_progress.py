"""Lightweight read-only tmux dashboard; never rereads rollout history."""
import argparse,sys,time
from datetime import datetime
from pathlib import Path
from .eval_common import OUTPUT,TRAIN,SUITES,read_json
def bar(done,total,width=24):
 filled=min(width,int(width*done/max(total,1)))
 return '['+'#'*filled+'-'*(width-filled)+f'] {done}/{total} ({100*done/max(total,1):.1f}%)'
def rate(success,total):return f'{100*success/total:.2f}%' if total else 'N/A'
def alive(pid):return bool(pid and Path(f'/proc/{pid}').exists())
def render(fixed_step):
 status=read_json(OUTPUT/'queue_status.json',{});training=read_json(TRAIN/'status.json',{})
 step=fixed_step or status.get('step') or 2500
 row=read_json(OUTPUT/f'step_{step:06d}'/'summary.json',{})
 n=row.get('episodes_completed',0);success=row.get('successes',0)
 lines=[f'LIBERO STUDENT | checkpoint {step} | '+datetime.now().strftime('%m-%d %H:%M:%S'),
  'GPU6 x2 | GPU4 excluded | detach: Ctrl-b then d',
  f'Queue: {"ALIVE" if alive(status.get("pid")) else "STOPPED"} / {status.get("phase","unknown")}',
  f'Queue heartbeat age {time.time()-status.get("time",time.time()):.0f}s | display refresh 3s',
  f'TOTAL {bar(n,400)}',f'Success {success}/{n} = {rate(success,n)} | failures {n-success}']
 tasks={(t['suite'],t['task_id']):t for t in row.get('tasks',[])}
 for w in status.get('workers',[]):
  gpu=w['gpu'];worker=w.get('worker','legacy')
  lines+=['',f'GPU {gpu} / {worker}: {w.get("phase","unknown")}']
  if w.get('step')==step and w.get('phase')=='evaluating':
   suite,task=w['suite'],w['task_id'];t=tasks.get((suite,task),{});count=t.get('episodes',0);passed=t.get('successes',0)
   lines+=[f'  {suite} task {task:02d} | success {passed}/{count} {rate(passed,count)}',f'  {bar(count,10,20)}']
  else:lines +=[f'  Checkpoint {w.get("step","-")}']
 lines+=['SUITE             EPISODES  SUCCESS    RATE']
 for suite in SUITES:
  v=row.get('suites',{}).get(suite,{});count=v.get('episodes',0);passed=v.get('successes',0)
  lines.append(f'{suite:17s} {count:3d}/100  {passed:3d}/{count:<3d}  {rate(passed,count):>6}')
 lines +=[f'Train GPU 0: {"ALIVE" if alive(training.get("pid")) else "STOPPED"} | step {training.get("step","?")} | {training.get("phase","?")}',
  'Worker rates include resumed episodes of their current task.',
  'Final checkpoint score requires 400 episodes.']
 return '\n'.join(lines)
def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int);p.add_argument('--once',action='store_true');a=p.parse_args()
 while True:
  try:view=render(a.step)
  except Exception as e:view=f'Dashboard read error (will retry): {e!r}'
  if not a.once:sys.stdout.write('\033[H\033[2J')
  print(view,flush=True)
  if a.once:return
  time.sleep(3)
if __name__=='__main__':main()

