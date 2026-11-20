"""CPU-only tmux display for the durable Round11 experiment supervisor."""
import json
import time
from pathlib import Path
from robotwin.round11.autopilot import R, OUT, read


def bar(n, total):
    return '[' + ('#'*min(40, int(40*n/max(1,total)))).ljust(40, '.') + f'] {n}/{total}'


def main():
    while True:
        s = read(OUT/'status.json')
        c = read(R/'confirmation_500/status.json')
        d = read(OUT/'decision.json')
        print('\033[2J\033[H', end='')
        print('RoboTwin Round11 自动衔接 | GPU2 | ' + time.strftime('%F %T'))
        print('后台监控:', s.get('phase','starting'), '| 心跳', round(time.time()-s.get('updated_at',time.time()),1), '秒前')
        print('独立离线验证:', bar(c.get('completed',0),128), c.get('phase','pending'))
        print('权重复现:', read(R/'confirmation_500/preflight.json').get('passed','pending'))
        print('下一步:', d.get('next','等待完整结果'), '|', d.get('reason','验证结束后自动判定'))
        train = read(R/'action_adapter_200/status.json')
        if train:
            print('\n动作适配器短训:', bar(train.get('step',0),200), train.get('phase'), '| micro', train.get('micro',0))
            print('LR 1e-5 | batch4 | 保存/诊断每50步 | 固定兼容模型初始化')
            path = R/'action_adapter_200/metrics.jsonl'
            try: rows = [json.loads(x) for x in path.read_text().splitlines()]
            except (OSError,ValueError): rows = []
            losses = [r for r in rows if r.get('split') == 'train']
            if losses:
                print('最新 action loss:', losses[-1]['action_loss'])
                print('分段 action loss 均值（不同批次，仅供趋势参考）:')
                for start in range(0,200,25):
                    part = [r['action_loss'] for r in losses if start<r['step']<=start+25]
                    if part: print(f'  {start+1:3d}-{start+len(part):3d}: {sum(part)/len(part):.8f}')
            for path in sorted((R/'action_adapter_200').glob('validation_*.json')):
                row = read(path)
                print('固定诊断 step', row.get('step'), row.get('mean'))
        online = read(R/'online_pilot_500/status.json')
        if online:
            print('\n小规模闭环:', bar(online.get('completed_jobs',0),8), online.get('phase'), online.get('active',''))
            print('匹配 episode:', online.get('matched_episodes',0), '| 候选成功:', online.get('candidate_successes',0),
                  '| 已记录兼容模型成功:', online.get('reference_successes',0))
        if s.get('error'): print('\n异常:', s['error'])
        print('\n仅展示本轮；离线误差不等于在线成功率。Ctrl-b d 退出查看，任务继续。', flush=True)
        time.sleep(5)


if __name__ == '__main__': main()
