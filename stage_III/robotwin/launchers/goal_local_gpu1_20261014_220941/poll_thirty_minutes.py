"""Sleep 30 minutes between reports; stop reporting once all scene preparation is complete."""
import json,subprocess,time
from pathlib import Path
from report_five_minutes import snapshot
HERE=Path(__file__).resolve().parent

def scenes_complete(s):
 return all(s['scenes'][split]['complete'] and s['scenes'][split]['accepted']==s['scenes'][split]['target'] for split in ('train','val','test'))

def main():
 print('已改为每30分钟休眠后报告；训练、验证、测试场景全部检验完成后停止轮询。',flush=True)
 while True:
  subprocess.run(['sleep','1800'],check=True)
  s=snapshot()
  with (HERE/'thirty_minute_reports.jsonl').open('a') as f:f.write(json.dumps(s,ensure_ascii=False)+'\n')
  lines=[s['local_time']+' 场景检验与Stage III进度']
  for split,label in (('train','训练'),('val','验证'),('test','测试')):
   row=s['scenes'][split];lines.append(f"{label}场景 {row['accepted']}/{row['target']}，{'已完成' if row['complete'] else '未完成'}")
  r=s['rl'];lines.append(f"GPU3 Stage III：{r.get('phase','未启动')}，已提交{r.get('iteration',0)}/100轮，当前组{r.get('group','-')}，有效轨迹{r.get('episodes',0)}，实际更新{r.get('update_count',0)}")
  lines.append('当前任务：'+str(r.get('task','-'))+' / '+str(r.get('mode','-')))
  print('\n'.join(lines),flush=True)
  if scenes_complete(s):
   proof={'stopped_at':s['local_time'],'reason':'all_scene_preparation_complete','training_continues_in_background':True,'scenes':s['scenes']}
   (HERE/'polling_completed.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
   print('所有场景检验已完成，停止主动轮询；不停止训练或独立评估进程。',flush=True)
   return
if __name__=='__main__':main()
