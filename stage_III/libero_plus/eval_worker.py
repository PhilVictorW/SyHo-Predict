"""Persistent simulator worker using the baseline evaluation seeds and core."""
from __future__ import annotations
import argparse
import dataclasses
import faulthandler
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from .eval_common import ROOT, SUITES, atomic, read, digest, connect, claim, finish
from .runtime_safety import patch_reset


class Heartbeat:
    def __init__(self, folder, worker, job):
        self.path = folder / 'workers' / f'worker_{worker}.json'
        self.worker, self.job = worker, job
        self.last = 0

    def report(self, phase, force=False, step=None):
        now = time.time()
        if not force and now - self.last < 5:
            return
        budget = 600 if phase == '导入仿真环境' else 180
        atomic(self.path, dict(worker=self.worker, job=self.job['id'], attempt=self.job['attempts'],
                              pid=os.getpid(), phase=phase, step=step, updated_at=now,
                              started=self.job.get('started'), timeout_seconds=budget))
        self.last = now
        faulthandler.dump_traceback_later(max(1,budget-15))
        if force:
            print(f'[Stage III评估] 路{self.worker} 样本{self.job["id"]}: {phase}', flush=True)


def imports():
    sys.path.insert(0, str(ROOT.parent / 'LaWAM_aba2'))
    from sim_compat import enable
    enable()
    import numpy as np
    if not hasattr(np, 'float_'):
        np.float_ = np.float64
    np.random.seed(0)
    import libero.libero as libero
    from libero.libero.envs.env_wrapper import ControlEnv
    patch_reset(ControlEnv)
    from examples.LIBERO.eval_files import libero_eval_core as core
    expected = ROOT.parent / 'LaWAM_aba2/examples/LIBERO/eval_files/libero_eval_core.py'
    if Path(core.__file__).resolve() != expected.resolve():
        raise RuntimeError(f'wrong baseline simulator core: {core.__file__}')
    from .client import Stage3Client, rpc

    class EvaluationClient(Stage3Client):
        def configure_episode(self, suite_index, task_id, episode_idx):
            self.episode_seed = (suite_index+1)*10_000_000 + task_id*10_000 + episode_idx*100

        def step(self, example, step=0, return_intermediates=False, **kwargs):
            if return_intermediates:
                raise ValueError('evaluation does not use similarity features')
            if self.actions is None or self.cursor == len(self.actions):
                result = rpc(self.url, '/infer', dict(
                    checkpoint=self.checkpoint, actor_checkpoint=self.actor_checkpoint,
                    main=self._encode(example['primary_image']), wrist=self._encode(example['wrist_image']),
                    instruction=str(example['lang']), noise_seed=self.episode_seed+self.chunk, gate_seed=0))
                action = np.asarray(result['actions'],dtype=np.float32)
                if (result.get('normalized') is not False or result.get('gate_mode') != 'beta_mean'
                        or action.shape != (8,7) or not np.isfinite(action).all()
                        or not np.isin(action[:,6],[0.,1.]).all()):
                    raise ValueError('invalid evaluation action or gate mode')
                self.actions, self.cursor = action, 0
                self.decision_records.append(dict(result['gate'], noise_seed=self.episode_seed+self.chunk))
                self.chunk += 1
            action = self.actions[self.cursor].copy(); self.cursor += 1
            return {'raw_action': dict(world_vector=action[:3], rotation_delta=action[3:6],
                                       open_gripper=np.asarray([1-float(action[6])],dtype=np.float32))}
    return np, libero, core, EvaluationClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True);p.add_argument('--url',required=True)
    args=p.parse_args();folder=args.folder.resolve();protocol=read(folder/'protocol.json')
    bundle=read(folder/'model_bundle.json')
    if digest(protocol['actor_checkpoint']) != bundle['actor_sha256']:
        raise ValueError('actor snapshot changed')
    faulthandler.enable();db=connect(folder);modules=None;suites={}
    try:
        while True:
            job=claim(db,args.worker)
            if job is None: break
            task=job['payload'];progress=Heartbeat(folder,args.worker,job)
            begin=time.time()
            try:
                progress.report('导入仿真环境',True)
                if modules is None: modules=imports()
                np,libero,core,Client=modules
                suite_name=task['suite'];tid=int(task['task_id'])
                if suite_name not in suites:
                    suites[suite_name]=core._get_benchmark_dict()[suite_name]()
                suite=suites[suite_name];actual=suite.get_task(tid)
                if actual.name != task['name']:raise ValueError('manifest task name mismatch')
                initial=suite.get_task_init_states(tid)
                idx=core._task_init_state_index(initial,0);state=initial[idx]
                adapter=core.get_benchmark_adapter('libero_plus');original=adapter.build_env
                def build(task,resolution,seed):
                    progress.report('创建环境',True)
                    env,description=original(task,resolution,seed)
                    for name,label in (('reset','重置环境'),('set_init_state','载入初始状态')):
                        call=getattr(env,name)
                        def wrapped(*a,_call=call,_label=label,**kw):
                            progress.report(_label,True);v=_call(*a,**kw)
                            progress.report(_label+'完成',True);return v
                        setattr(env,name,wrapped)
                    step_call=env.step
                    def step(*a,**kw):
                        progress.report('仿真执行动作');return step_call(*a,**kw)
                    env.step=step
                    return env,description
                adapter=dataclasses.replace(adapter,build_env=build)
                client=Client(protocol['base_student'],protocol['actor_checkpoint'],args.url,0)
                client.configure_episode(SUITES.index(suite_name),tid,0)
                step_call=client.step
                def policy_step(*a,**kw):
                    progress.report('模型推理',step=kw.get('step'));return step_call(*a,**kw)
                client.step=policy_step
                eval_args=core.EvalArgs(pretrained_path=protocol['base_student'],benchmark_variant='libero_plus',
                    task_suite_name=suite_name,task_id=tid,num_trials_per_task=1,num_workers=1,seed=0,
                    num_steps_wait=10,save_videos=False,video_out_path=str(folder),eval_action_chunk_len=8)
                core._configure_mujoco_backend(eval_args,adapter)
                result=core._run_single_episode(args=eval_args,adapter=adapter,task=actual,task_id=tid,
                    episode_idx=0,init_state=state,client_model=client,run_output_dir=folder,env_seed=0)
                if result.error:raise RuntimeError(result.error)
                record=dataclasses.asdict(result)
                etas=[float(v['eta']) for v in client.decision_records]
                record.update(id=job['id'],suite=suite_name,category=task['category'],task_name=task['name'],
                    worker=args.worker,attempt=job['attempts'],gate_mode='beta_mean',base_student=protocol['base_student'],
                    actor_checkpoint=protocol['actor_checkpoint'],actor_sha256=bundle['actor_sha256'],
                    initial_state_idx=idx,initial_state_hash=hashlib.sha256(pickle.dumps(state,protocol=4)).hexdigest(),
                    env_seed=0,noise_seed_base=client.episode_seed,action_protocol_id='libero_plus_open_to_legacy_close_v1',
                    decisions=len(etas),eta_mean=sum(etas)/len(etas) if etas else None,
                    eta_min=min(etas) if etas else None,eta_max=max(etas) if etas else None,
                    elapsed=time.time()-begin,completed_at=time.time())
                atomic(folder/'episodes'/f'{suite_name}_{tid:05d}.json',record)
                finish(db,job,args.worker,'done',record)
                print(f'[Stage III评估][完成] 样本{job["id"]} {suite_name}/{task["category"]} '
                      f'success={result.success} steps={result.num_actions}',flush=True)
            except Exception as exc:
                traceback.print_exc()
                record=dict(id=job['id'],task=task,worker=args.worker,attempt=job['attempts'],
                            error=repr(exc),elapsed=time.time()-begin,time=time.time())
                atomic(folder/'errors'/f'{job["id"]:05d}_attempt{job["attempts"]}.json',record)
                finish(db,job,args.worker,'error',record,repr(exc))
                print(f'[Stage III评估][错误记录] 样本{job["id"]}，继续下一项：{exc}',flush=True)
            finally:
                faulthandler.cancel_dump_traceback_later()
        atomic(folder/'workers'/f'worker_{args.worker}.json',dict(worker=args.worker,pid=os.getpid(),
               phase='本次队列完成',updated_at=time.time(),job=None))
    finally:
        db.close()


if __name__=='__main__':main()
