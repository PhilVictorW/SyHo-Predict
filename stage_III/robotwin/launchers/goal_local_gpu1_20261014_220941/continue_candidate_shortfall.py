"""Resume only an exhausted validation pool after the current owned controller exits."""
import json,os,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
  return dict(pid=pid,start=s[19],uid=p.stat().st_uid,state=s[0])
 except (OSError,ValueError,IndexError):return None
record=read(HERE/'candidate_continuation.json');expected=record['controller']
while True:
 current=identity(expected['pid'])
 if not current or current['state']=='Z' or any(current[k]!=expected[k] for k in ('pid','start','uid')):break
 time.sleep(30)
run=Path(record['run_dir']);status=read(run/'workflow_status.json')
if status.get('phase')!='prepare_val' or status.get('status')!='stopped':
 raise SystemExit('原调度器非候选不足退出；不自动重启：'+str(status))
parallel=read(run/'manifests/prepared_val.parallel_status.json')
assert not parallel['complete'] and not parallel['errors'],'Unexpected scene worker errors'
manifest=read(run/'manifests/prepared_val.json');missing=[]
for task,mode in manifest['requested_cells']:
 cell=run/'manifests/parallel_cells'/(task+'__'+mode)
 m=read(cell/'prepared_val.json')
 if m['complete']:continue
 pool=[r for r in read(cell/'candidates.json')['rows'] if r['split']=='val']
 attempted={e['scene_id'] for e in m['events']}
 assert all(r['scene_id'] in attempted for r in pool),'Missing cell has untried original candidates'
 missing.append((task,mode,len(m['rows']),len(pool)))
assert missing,'No exhausted candidate cells found'
print('原批次全部检测完毕；开始补足独立候选，保留已验收结果：',missing,flush=True)
python=read(record['config'])['runtime']['model_python']
os.execv(python,[python,'-B','-u',str(HERE/'resume_ready.py'),'--gpu','7','--execute','--worker'])
