"""CPU-only live report; never starts training or online evaluation."""
import argparse,json,time
from pathlib import Path
from statistics import mean
RUN=Path(__file__).resolve().parent/'runs/teacher_compat36_20260925'

def generate():
    status=json.loads((RUN/'status.json').read_text())
    rows=[]
    if (RUN/'metrics.jsonl').exists():
        for line in (RUN/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    validations=[]
    for path in sorted(RUN.glob('validation_*.json')):
        data=json.loads(path.read_text())
        selected=[r for r in data['rows'] if not r['donor_same_episode']]
        paired={v:{k:mean(r['metrics'][v][k] for r in selected) for k in selected[0]['metrics'][v]} for v in selected[0]['metrics']}
        validations.append(dict(step=data['step'],mean=data['mean'],donor_excluded_mean=paired,
                               paired_samples=len(selected),gate=data['gate'],
                               mean_wrong_future_relative_l2=mean(r['wrong_future_relative_l2'] for r in selected)))
    summary=dict(status=status,validations=validations,student_started=False,online_evaluation_started=False,
                 interpretation='Offline teacher screening only; review before starting a student')
    tmp=RUN/'review_summary.tmp';tmp.write_text(json.dumps(summary,indent=2)+'\n');tmp.replace(RUN/'review_summary.json')
    text=['# 第二轮教师预实验动态结果', '',f"阶段：{status['phase']}；有效更新：{status['step']}/{status['total_steps']}。",'',
          '完整36步物理动作误差；下表排除与错配donor同轨迹的样本。不是闭环成功率。','',
          '| 步数 | 条件 | 位置cm | 姿态角° | 夹爪MAE |','|---|---|---:|---:|---:|']
    for item in validations:
        for v,m in item['donor_excluded_mean'].items():
            text.append(f"| {item['step']} | {v} | {m['position_cm']:.6f} | {m['rotation_deg']:.6f} | {m['gripper_mae']:.7f} |")
    text+=['','验收关注真实未来相对roles_off及wrong_future是否稳定改善，同时保持原始base路径水平。小样本微小变化需要扩大样本确认。',
           '学生action-only/低蒸馏对照尚未启动；自动在线评测继续暂停。', '',
           'adapter检查点依赖原始冻结权重，不能直接交给旧推理服务器。', '', '[完整loss和验证趋势](curves.png)']
    tmp=RUN/'REVIEW.tmp';tmp.write_text('\n'.join(text)+'\n');tmp.replace(RUN/'REVIEW.md')
    train=[r for r in rows if r['split']=='train']
    if train:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        axes[0,0].plot([r['step'] for r in train],[r['flow'] for r in train],alpha=.6,label='action loss')
        axes[0,0].plot([r['step'] for r in train],[mean(x['flow'] for x in train[max(0,i-9):i+1]) for i in range(len(train))],label='10-update mean')
        axes[0,0].set_title('Training flow-matching loss');axes[0,0].legend()
        for ax,key in zip([axes[0,1],axes[1,0],axes[1,1]],['position_cm','rotation_deg','gripper_mae']):
            for v in ['full','base','roles_off','wrong_future']:
                ax.plot([r['step'] for r in validations],[r['donor_excluded_mean'][v][key] for r in validations],marker='o',label=v)
            ax.set_title(key+' (fixed paired validation)');ax.legend(fontsize=8)
        for ax in axes.flat:ax.set_xlabel('Effective updates');ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(RUN/'curves.tmp.png',dpi=150);plt.close(fig)
        (RUN/'curves.tmp.png').replace(RUN/'curves.png')
    return status['phase']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        try:phase=generate()
        except FileNotFoundError:phase='waiting'
        if not args.watch or phase in ('completed_teacher_pilot','failed'):break
        time.sleep(30)
