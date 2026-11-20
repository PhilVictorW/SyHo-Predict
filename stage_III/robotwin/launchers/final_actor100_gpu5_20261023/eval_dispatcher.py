"""Two independent GPU5 policy servers and dynamically assigned task-mode jobs."""
import argparse, fcntl, os, signal, subprocess, sys, time, traceback
from pathlib import Path
from eval_common import ROOT, HERE, read, write, episodes, last_progress, timed_out, aggregate
from stage_III.robotwin.runtime import spawn, stop_owned, child_env
from stage_III.robotwin.rpc import call

def log(message):
    print(time.strftime('%m-%d %H:%M:%S')+' '+message,flush=True)

def run_queue(run):
    os.environ['NO_PROXY']=os.environ['no_proxy']='127.0.0.1,localhost,::1'
    run=Path(run);plan=read(run/'manifest.json');cfg=read(run/'config.json')
    lock=(run/'queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    jobs=plan['jobs'];waiting=[j for j in jobs if not (run/'jobs'/j['key']/'result.json').exists()]
    lanes={str(i):dict(server=None,task=None,phase='pending',restarts=0,retry_at=0) for i in range(2)}
    for name,lane in lanes.items():
        lane['restarts']=max((int(x.name.rsplit('_',1)[1]) for x in (run/'servers').glob(f'lane_{name}_attempt_*')),default=0)
    def status(phase='running',**kw):
        snapshot={}
        for name,lane in lanes.items():
            job=lane.get('job');task=lane.get('task');server=lane.get('server')
            active=dict(phase=lane['phase'],server_pid=server.pid if server and server.poll() is None else None,
                        task_pid=task.pid if task and task.poll() is None else None,port=plan['ports'][int(name)],
                        server_restarts=lane['restarts'])
            if job:
                folder=run/'jobs'/job['key'];ep=episodes(folder)
                active.update(task=job['task'],mode=job['mode'],episodes=len(ep),
                              successes=sum(bool(x['success']) for x in ep),activity=read(folder/'activity.json'),
                              no_progress_seconds=int(time.time()-last_progress(folder,lane['task_started'])))
            snapshot[name]=active
        stats=aggregate(run,jobs)
        write(run/'status.json',dict(phase=phase,pid=os.getpid(),updated_at=time.time(),
                  gpu=5,parallel=2,lanes=snapshot,remaining_jobs=len(waiting),**stats,**kw))
        return stats
    def launch_server(name,lane):
        lane['restarts']+=1
        out=run/'servers'/f'lane_{name}_attempt_{lane["restarts"]:03d}';out.mkdir(parents=True)
        env=child_env(cfg,5,out);env['PYTHONPATH']=str(ROOT)
        command=[cfg['runtime']['model_python'],'-B','-u','-m','stage_III.robotwin.server',
                 '--config',str(run/'config.json'),'--student',plan['student'],'--gpu','5',
                 '--port',str(plan['ports'][int(name)]),'--ready',str(out/'ready.json'),'--execute']
        lane.update(server=spawn(command,out/'server.log',ROOT,env),server_out=out,
                    phase='loading',load_started=time.time())
        log(f'通道{name}：GPU5加载冻结学生和基座，随后加载Actor100；PID {lane["server"].pid}')
    def start_task(name,lane,job):
        folder=run/'jobs'/job['key'];folder.mkdir(parents=True,exist_ok=True)
        env=child_env(cfg,5,run/'cache'/f'lane_{name}')
        for key in list(env):
            if key.startswith('ROBOTWIN_') or key.startswith('PAIR_'): env.pop(key,None)
        env.update(ROBOTWIN_PATH=cfg['runtime']['sim_root'],ROBOTWIN_NUM_SLOTS='1',
                   ROBOTWIN_TEST_NUM='50',ROBOTWIN_SAVE_VIDEO='0',ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN='1',
                   ROBOTWIN_EVAL_ROOT=str(folder),ROBOTWIN_RUN_TAG='stageIII_final_actor100',
                   ROBOTWIN_SHARED_WARP_CACHE_DIR=str(run/'cache'/f'lane_{name}'/'warp'),
                   PAIR_TASK=job['task'],PAIR_MODE=job['mode'],PAIR_ARM='compatible',
                   STAGE3_POLICY_IDENTITY=str(lane['server_out']/'policy_identity.json'),
                   PYTHONPATH=os.pathsep.join((str(HERE),str(ROOT),cfg['runtime']['sim_root'],
                                             str(ROOT/'examples/Robotwin/eval_files'))))
        if job.get('seeds'):env['ROBOTWIN_FIXED_SEEDS']=','.join(map(str,job['seeds']))
        command=[cfg['runtime']['sim_python'],'-B','-u',str(HERE/'bench_bridge.py'),
                 '--config',str(ROOT/'examples/Robotwin/eval_files/deploy_policy.yml'),'--overrides',
                 '--task_name',job['task'],'--task_config',job['mode'],
                 '--ckpt_setting','stageIII_final_actor100','--seed','0','--instruction_type','seen',
                 '--policy_name','model2robotwin_interface','--host','127.0.0.1',
                 '--port',str(plan['ports'][int(name)]),'--policy_ckpt_path',cfg['base']['checkpoint'],
                 '--replan_steps','36','--action_ensemble','0','--action_ensemble_alpha','0']
        write(folder/'protocol.json',dict(**job,student_sha256=plan['student_sha256'],
                    actor_sha256=plan['actor_sha256'],policy='actor_mean',stall_seconds=2400,
                    instruction_rule='native seen descriptions, Python random seeded by episode seed',
                    historical_comparison='seeds only; not a strict paired result'))
        lane.update(task=spawn(command,folder/'bridge.log',cfg['runtime']['sim_root'],env),
                    job=job,task_started=time.time(),phase='evaluating')
        log(f'通道{name}：开始 {job["task"]} / {job["mode"]}，目标50条')
    def finish(name,lane,timeout=False):
        process=lane['task'];folder=run/'jobs'/lane['job']['key']
        if timeout: stop_owned(process)
        eps=episodes(folder);complete=process.poll()==0 and len(eps)==50
        state='completed' if complete else 'skipped_timeout' if timeout else 'skipped_error'
        tail=''
        if state=='skipped_error':
            with (folder/'bridge.log').open('rb') as f:
                f.seek(max(0,(folder/'bridge.log').stat().st_size-6000));tail=f.read().decode(errors='replace')
        result=dict(key=lane['job']['key'],task=lane['job']['task'],mode=lane['job']['mode'],
                    status=state,episodes=len(eps),successes=sum(bool(x['success']) for x in eps),
                    missing_episodes=50-len(eps),episode_errors=sum(bool(x.get('error')) for x in eps),
                    exit_code=process.poll(),elapsed=time.time()-lane['task_started'],finished_at=time.time(),
                    reason='40分钟没有实际任务进展' if timeout else tail)
        write(folder/'result.json',result)
        log(f'通道{name}：{result["key"]} {state}，完成{len(eps)}/50，成功{result["successes"]}')
        # A timeout must never block the remaining task queue.
        lane.update(task=None,job=None,phase='ready')
        if timeout or not complete:
            try: call(lane['url'],'/health',{},timeout=5)
            except Exception:
                stop_owned(lane['server']);lane.update(server=None,phase='recovering',retry_at=time.time()+5)
    def shutdown(*args):
        raise KeyboardInterrupt('Supervisor stop requested')
    signal.signal(signal.SIGTERM,shutdown)
    try:
        log('开始全量评测：GPU5双通道；50任务×2模式×50条=5000条；40分钟无进展跳过当前任务模式')
        while waiting or any(x.get('task') for x in lanes.values()):
            for name,lane in lanes.items():
                server=lane.get('server');task=lane.get('task')
                if task:
                    timeout=timed_out(run/'jobs'/lane['job']['key'],lane['task_started'])
                    if task.poll() is not None or timeout: finish(name,lane,timeout)
                    else: continue
                server=lane.get('server')
                if server and server.poll() is not None:
                    lane.update(server=None,phase='recovering',retry_at=time.time()+30)
                    log(f'通道{name}：模型服务退出，稍后重新加载；不会影响另一通道')
                if not waiting: continue
                if lane.get('server') is None:
                    if time.time()>=lane['retry_at']:launch_server(name,lane)
                    continue
                if lane['phase']=='loading':
                    ready=lane['server_out']/'ready.json'
                    if not ready.exists():
                        if time.time()-lane['load_started']>3600:
                            stop_owned(lane['server']);lane.update(server=None,phase='recovering',retry_at=time.time()+30)
                        continue
                    info=read(ready)
                    if info.get('pid')!=lane['server'].pid:raise RuntimeError('Wrong model server PID')
                    url=f'http://127.0.0.1:{plan["ports"][int(name)]}'
                    info=call(url,'/health',{},timeout=30)
                    if info['student_sha256']!=plan['student_sha256']:raise RuntimeError('Student identity mismatch')
                    identity=call(url,'/actor',{'path':plan['actor']},timeout=120)
                    if identity['actor_sha256']!=plan['actor_sha256']:raise RuntimeError('Actor identity mismatch')
                    identity['base_checkpoint']=cfg['base']['checkpoint']
                    write(lane['server_out']/'policy_identity.json',identity)
                    lane.update(phase='ready',url=url)
                    log(f'通道{name}：冻结学生与Actor100均加载完成，身份校验通过')
                if lane['phase']=='ready' and waiting:start_task(name,lane,waiting.pop(0))
            status()
            time.sleep(5)
        stats=status('finished')
        write(run/'summary.json',dict(**stats,complete=stats['episodes']==5000 and stats['skipped_cells']==0,
                                     policy='actor_mean',finished_at=time.time(),
                                     student=plan['student'],actor=plan['actor']))
        log('全部任务模式已处理，结果已保存；跳过项目单独记录，不计为已完成episode')
    except BaseException as exc:
        status('stopped',error=repr(exc));traceback.print_exc();raise
    finally:
        for lane in lanes.values():
            for key in ('task','server'):
                if lane.get(key):stop_owned(lane[key])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    run_queue(parser.parse_args().run)
