"""Diagnostic-only paired episodes: same student, only gripper polarity changes."""
import argparse,json,os,sys,time,dataclasses
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];WORK=ROOT/'libero-plus'
sys.path[:0]=[str(ROOT),str(WORK/'runtime'),str(ROOT.parent/'LaWAM_aba2/libero-plus/runtime')]
from student_eval_common import BENCH,SUITES,atomic
from sim_compat import enable
enable()
import libero.libero as libero
assert Path(libero.__file__).resolve().is_relative_to(BENCH)
import numpy as np
from examples.LIBERO.eval_files import libero_eval_core as core
from goal_local.eval_client import StudentClient

class PairedClient(StudentClient):
    def __init__(self,*a,flip=False,**kw):super().__init__(*a,**kw);self.flip=flip;self.trace=[]
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        if self.flip:result['raw_action']['open_gripper']=1-result['raw_action']['open_gripper']
        if len(self.trace)<24:
            a=result['raw_action'];self.trace.append(dict(xyz=np.asarray(a['world_vector']).tolist(),gripper=float(a['open_gripper'][0])))
        return result

p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);p.add_argument('--url',required=True);args=p.parse_args()
protocol=json.loads((HERE/'paired_protocol.json').read_text());folder=HERE/'paired_episodes';folder.mkdir(exist_ok=True)
adapter=core.get_benchmark_adapter('libero_plus');benchmark=core._get_benchmark_dict();suites={}
for idx,task in enumerate(protocol['tasks']):
    if idx%2!=args.worker:continue
    s=task['suite'];tid=task['task_id']
    if s not in suites:suites[s]=benchmark[s]()
    suite=suites[s];actual=suite.get_task(tid);assert actual.name==task['name']
    states=suite.get_task_init_states(tid)
    # Alternate order across pairs, avoiding a single fixed run-order effect.
    for flip in ([False,True] if idx%2==0 else [True,False]):
        label='flipped' if flip else 'unchanged';output=folder/f'{idx:02d}_{label}.json'
        if output.exists():continue
        np.random.seed(0)
        client=PairedClient(protocol['checkpoint'],args.url,flip=flip);client.configure_episode(SUITES.index(s),tid,0)
        ea=core.EvalArgs(pretrained_path=protocol['checkpoint'],benchmark_variant='libero_plus',task_suite_name=s,task_id=tid,
            num_trials_per_task=1,num_workers=1,seed=0,num_steps_wait=10,save_videos=False,video_out_path=str(folder),eval_action_chunk_len=8)
        core._configure_mujoco_backend(ea,adapter)
        atomic(HERE/f'paired_worker{args.worker}.json',dict(phase='running',task=task,variant=label,time=time.time()))
        print(json.dumps(dict(event='start',task=task,variant=label)),flush=True)
        result=core._run_single_episode(args=ea,adapter=adapter,task=actual,task_id=tid,episode_idx=0,
            init_state=states[core._task_init_state_index(states,0)],client_model=client,run_output_dir=folder,env_seed=0)
        row=dict(dataclasses.asdict(result),pair=idx,suite=s,category=task['category'],variant=label,checkpoint=protocol['checkpoint'],
            trace_first24=client.trace,completed=time.time())
        atomic(output,row);print(json.dumps(dict(event='result',pair=idx,suite=s,category=task['category'],variant=label,success=result.success)),flush=True)
        client.close()
atomic(HERE/f'paired_worker{args.worker}.json',dict(phase='complete',time=time.time()))
