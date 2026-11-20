from pathlib import Path
import json,os,sys,time,subprocess
W=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change/ablation/aba1/A1/libero')
sys.path.insert(0,str(W))
from watch_full_eval import read
from tmux_progress import process
record=read(W/'nohup/current.json');run=Path(record['run']);control=W/'nohup'/run.name
backup=Path(Path('/tmp/a1_parallel_eval_backup').read_text());before=read(backup/'preserved_episodes.json')
while True:
    state=read(W/'auto_eval'/run.name/'status.json')
    workers=state.get('workers',[])
    view=dict(time=time.strftime('%H:%M:%S'),phase=state.get('phase'),episodes=state.get('episodes_completed'),
        workers=[{k:x.get(k) for k in ('gpu','phase','server_pid','server_alive','suite','task_id','episodes_at_start','episodes_completed','calls')} for x in workers])
    print(json.dumps(view),flush=True)
    with (control/'parallel_stability_poll.jsonl').open('a') as f:f.write(json.dumps(view)+'\n')
    if not process(record['processes']['auto_eval']['pid'])['alive'] or state.get('phase')=='watcher_stopped_or_failed':
        raise SystemExit('Parallel evaluator stopped; inspect log')
    if (len(workers)==4 and {x['gpu'] for x in workers}=={0,1,2,3}
            and all(x.get('phase')=='evaluating' and x.get('server_alive') and x.get('calls',0)>0
                    and x.get('episodes_completed',0)>=x.get('episodes_at_start',0)+2 for x in workers)):
        output=Path(state['output']);preserved=0
        from ablation.aba1.libero_common.evaluation import episode_rows
        for name,rows in before.items():
            task=int(name.rsplit('task',1)[1]);now=episode_rows(output/'tasks'/name,task,50)
            for i,row in rows.items():
                assert now[int(i)]==row,(name,i)
                preserved+=1
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        indices={x.split(',')[1].strip():int(x.split(',')[0]) for x in raw.splitlines()}
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
        binding={int(x.split(',')[1]):indices[x.split(',')[0].strip()] for x in raw.splitlines()}
        loads=[]
        for x in workers:
            assert binding[x['server_pid']]==x['gpu']
            env=dict(v.split(b'=',1) for v in (Path('/proc')/str(x['server_pid'])/'environ').read_bytes().split(b'\0') if b'=' in v)
            assert env[b'CUDA_VISIBLE_DEVICES']==str(x['gpu']).encode()
            loaded=read(output/'.parallel_workers'/f'gpu{x["gpu"]}'/'loaded_policy.json')
            assert loaded['step']==2000 and loaded['variant']=='A1' and loaded['physical_gpu']==str(x['gpu'])
            loads.append(loaded)
        loads.sort(key=lambda x:x['load_started_at'])
        assert all(b['load_started_at']-a['load_finished_at']>=10 for a,b in zip(loads,loads[1:]))
        proof=dict(passed=True,verified_at=time.time(),physical_gpus=[0,1,2,3],state=state,
            preserved_episodes_verified=preserved,minimum_new_episodes_per_active_worker=2,model_loads=loads)
        (control/'parallel_stability_check.json').write_text(json.dumps(proof,indent=2)+'\n')
        print('STABLE: all four GPUs completed at least two additional episodes; '+str(control/'parallel_stability_check.json'),flush=True)
        break
    time.sleep(20)
