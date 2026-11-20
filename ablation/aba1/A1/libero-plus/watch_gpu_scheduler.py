"""Read-only eight-GPU scheduler dashboard."""
import collections,json,os,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parent
while True:
 try:
  s=json.loads((WORK/'logs/gpu_scheduler_status.json').read_text())
  counts=collections.Counter(s.get('rank_to_gpu',[]));c=s.get('capacity',{});r=s.get('resources',{})
  if s.get('mode')=='fixed' and s.get('gpu_monitoring') is False:
   out=WORK/'protocol_plus_v1/seed_2026/II'
   workers=[]
   for path in sorted(out.glob('status_rank*.json')):
    try:workers.append(json.loads(path.read_text()))
    except (OSError,ValueError):pass
   try:latest=json.loads((out/'latest.json').read_text()).get('step',0)
   except (OSError,ValueError):latest=0
   lines=['A1 Stage II 固定 GPU'+','.join(map(str,s.get('rank_to_gpu',[])))+'，各1进程',
    time.strftime('%F %T %Z')+' | 只读状态面板',
    '动态显存监控、扩容、换卡、自动重启、主动停训：全部关闭',
    '每50步完整保存后删除上一份；最新完整检查点：'+str(latest),
    'Stage II 2500步成功结束 → 同五卡全量推理10030任务','']
   for worker in workers:
    lines.append(f"R{worker.get('rank')} GPU{worker.get('physical_gpu')} PID{worker.get('pid')} | {worker.get('phase')} | 已完成 {worker.get('step',0)} | micro {worker.get('micro','--')}")
   if sys.stdout.isatty():sys.stdout.write('\033[H\033[J')
   print('\n'.join(lines),flush=True);time.sleep(5);continue
  fixed=s.get('mode')=='fixed'
  checkpoint_expand=s.get('mode')=='checkpoint_expand'
  lines=['A1 Stage II 检查点扩容，排除GPU5，不主动停训' if checkpoint_expand else 'A1 Stage II 固定 GPU0–5，各1进程' if fixed else 'A1 Stage II 八卡后台调度',time.strftime('%F %T %Z')+f" | PID {s.get('pid')} | {s.get('phase')} | 数据 {time.time()-s.get('time',0):.0f}s 前",
   f"每进程 8.5 GiB | 加载前每卡额外预留 {c.get('gpu_reserve_mib',4096)/1024:g} GiB | 当前容量 | 串行错峰加载",
   'GPU  当前空闲   本组显存   稳定可用容量  安全容量/已分配']
  for g in range(8):
   v=r.get('gpus',{}).get(str(g),{});cap=c.get('capacity_mib',{}).get(str(g),0)
   lines.append(f" {g}    {v.get('free_mib',0)/1024:6.1f}G    {v.get('own_mib',0)/1024:6.1f}G      {cap/1024:6.1f}G          {c.get('gpu_slots',{}).get(str(g),0)}/{counts[g]}")
  lines += [f"主机可用内存 {r.get('host_available_mib',0)/1024:.1f} GiB | 预留 16 GiB | " + ('固定6 ranks' if fixed else f"主机限制 {c.get('host_limit','?')} ranks"),
   f"当前 ranks: {s.get('rank_to_gpu',[])} | 候选: {s.get('candidate_rank_to_gpu',[])}",
   f"Stage II: {s.get('stage_II_step',0)}/2500 | " + ("每50步保存，只保留最新完整权重" if fixed else "满50步完整保存后扩缩容"),
   f"待重组请求: {s.get('pending_reconfiguration') or '无'}",
   'Stage II 2500 → LIBERO-Plus 全量 10030 自动续接',
   'Ctrl-b 0 训练进度 | Ctrl-b 1 训练日志 | Ctrl-b 2 调度日志 | Ctrl-b d 脱离']
  if s.get('error'):lines.append(str(s['error']))
  if sys.stdout.isatty():sys.stdout.write('\033[H\033[J')
  print('\n'.join(lines),flush=True)
 except (OSError,ValueError) as e:print('等待调度状态：',e,flush=True)
 time.sleep(5)
