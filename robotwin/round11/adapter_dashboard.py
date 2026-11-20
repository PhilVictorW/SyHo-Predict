import time
from robotwin.round11.autopilot import R,read

def main():
    while True:
        w=read(R/'adapter_workflow/status.json');s=read(R/'adapter_confirmation_200/status.json')
        o=read(R/'online_adapter_200/status.json');r=read(R/'adapter_confirmation_200/results.json')
        print('\033[2J\033[H',end='')
        print('RoboTwin | 动作适配器200步独立验证 → 条件分流 | GPU2 |',time.strftime('%F %T'))
        print('后台:',w.get('phase','starting'),'| 心跳',round(time.time()-w.get('updated_at',time.time()),1),'秒前')
        done=s.get('completed',0);print('['+('#'*min(40,int(40*done/128))).ljust(40,'.')+f'] {done}/128 成对测试')
        print('验证:',s.get('phase','pending'),s.get('cohort',''),s.get('anchor',''),s.get('variant',''))
        print('32个新episode +8个尾部episode；同噪声比较初始/200步/Goal关闭/Local关闭/两者关闭。')
        print('权重复现:',read(R/'adapter_confirmation_200/preflight.json').get('passed','pending'))
        if r:
            for g in ['fresh','tail']:print(g,r['summary'][g]['comparisons']['adapter200_vs_initial'])
            print('满足闭环测试门槛:',r['online_eligible'])
        if o:
            print('\n闭环:',o.get('phase'),o.get('completed_jobs',0),'/8任务模式',o.get('active',''))
            print('当前任务:',o.get('progress',{}))
            print('匹配episode:',o.get('matched_episodes',0),'候选成功:',o.get('candidate_successes',0),'历史兼容对照成功:',o.get('reference_successes',0))
        if w.get('error'):print('异常:',w['error'])
        if s.get('error'):print('验证异常:',s['error'])
        print('\n通过→自动闭环；未通过→自动生成误差/消融诊断。旧StageII未重启。Ctrl-b d退出查看。',flush=True)
        time.sleep(5)

if __name__=='__main__':main()
