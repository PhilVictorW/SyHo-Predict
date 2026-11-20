"""Live Chinese dashboard for the full classic LIBERO Stage-III tmux run."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path


def read_json(path: Path, default):
    try: return json.loads(path.read_text())
    except Exception: return default


def gpu_rows():
    try:
        out=subprocess.check_output([
            'nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=3)
        rows=[]
        for line in out.strip().splitlines():
            bits=[x.strip() for x in line.split(',')]
            if len(bits)>=4: rows.append(bits)
        return rows
    except Exception as exc:
        return [["?","?","?",f"nvidia-smi error: {exc}"]]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); ap.add_argument('--interval',type=float,default=10); args=ap.parse_args()
    run=Path(args.run); status_path=run/'status.json'; metrics_path=run/'metrics.jsonl'; summary_path=run/'group_summary.jsonl'
    while True:
        status=read_json(status_path,{})
        metrics=[]
        try:
            with metrics_path.open() as f:
                for line in f:
                    if line.strip(): metrics.append(json.loads(line))
        except Exception: pass
        last=metrics[-1] if metrics else {}
        try: summary_count=sum(1 for _ in summary_path.open())
        except Exception: summary_count=0
        print('\033[2J\033[H',end='')
        print('=== classic LIBERO Stage III 全量 GRPO tmux 监控 ===')
        print('更新时间:',time.strftime('%Y-%m-%d %H:%M:%S'),' | run:',run.name)
        print('状态:',status.get('phase','--'),' | batch:',f"{status.get('batch',0)}/{status.get('total_batches',25)}",
              '| groups:',f"{status.get('groups_completed',0)}/800",'| trajectories:',f"{status.get('trajectories_completed',0)}/3200")
        print('Student:',status.get('config',{}).get('base_student','--'))
        print('Actor:',status.get('actor','--'))
        if last:
            print('最近更新: groups=',last.get('groups'),'decisions=',last.get('decisions'),
                  'successes=',last.get('successes'),'reward_mean=',f"{last.get('reward_mean',0):.6f}",
                  'loss=',f"{last.get('loss',0):.6g}",'KL=',f"{last.get('kl_old_new',0):.6g}",
                  'eta_mean=',f"{last.get('actor_eta_mean',0):.6f}",'grad_norm=',f"{last.get('grad_norm',0):.6g}")
        print('已写入 group summary:',summary_count,'/ 800')
        print('\nWorker 状态:')
        workers=status.get('worker_status',{})
        for key in sorted(workers,key=lambda x:int(x)):
            w=workers[key]
            print(f"  worker{key}: GPU{w.get('gpu','?')} 端口{w.get('port','?')}  {w.get('state','--')} "
                  f"group={w.get('last_group',w.get('group_id','--'))} success={w.get('successes','--')}/{w.get('trajectories','--')}")
        print('\nGPU 状态（指定布局：GPU0×1，GPU2×2，GPU5×2，GPU7×2）:')
        for row in gpu_rows(): print('  GPU'+row[0]+': memory '+row[1]+'/'+row[2]+' MiB, util '+row[3]+'%')
        print('\n查看后台训练: tmux attach -t libero_stage3_full800')
        print('训练日志: ',run/'console.log')
        if status.get('phase') in {'completed','failed'}:
            print('\n终态:',status.get('phase'),'；监控仍保持运行，可 Ctrl-C 关闭监控窗口。')
        time.sleep(args.interval)

if __name__=='__main__': main()
