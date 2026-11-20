"""Renderer and restored-state audit; no policy/model process."""
import sys, time, json, traceback
from common import *
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv
ROOT=BASE/'resources/environment_20261106'
rows=[]
deadline=time.time()+1800
for suite in SUITES:
    while not (ROOT/('validated_'+suite+'.json')).exists():
        if time.time()>deadline: raise TimeoutError('State generation deadline')
        time.sleep(2)
    reports=read(ROOT/('validated_'+suite+'.json'))
    for task in reports:
        assert task['status']=='validated',task
        env=None
        try:
            env=OffScreenRenderEnv(bddl_file_name=task['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            env.seed(0);env.reset()
            states=torch.load(task['init'],map_location='cpu',weights_only=False)
            fixtures=read(task['fixtures'])[0]
            for f in fixtures.values():
                bid=env.sim.model.body_name2id(f['body'])
                env.sim.model.body_pos[bid]=f['pos'];env.sim.model.body_quat[bid]=f['quat']
            obs=env.set_init_state(states[0])
            for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
            false=[]
            for state in env.env.parsed_problem['initial_state']:
                if all(x in env.env.object_states_dict for x in state[1:]):
                    if not env.env._eval_predicate(state):false.append(state)
            critical_false=[]
            for pred in false:
                if len(pred)!=3 or pred[0] not in ('on','in'):continue
                support=pred[2]
                if support in env.env.parsed_problem['regions']:
                    support=env.env.parsed_problem['regions'][support]['target']
                if support in env.env.objects_dict or support in env.env.fixtures_dict:
                    critical_false.append(pred)
            assert not critical_false, (task['name'],critical_false)
            images={}
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                im=obs[camera]
                assert im.shape==(256,256,3) and np.isfinite(im).all() and im.std()>5,(task['name'],camera)
                images[camera]=dict(shape=list(im.shape),std=float(im.std()))
            path=ROOT/'render_samples'/suite/(task['name']+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.ascontiguousarray(obs['agentview_image'][::-1,::-1])).save(path)
            result=dict(suite=suite,task_id=task['task_id'],name=task['name'],status='passed',images=images,initial_predicates_false=false,sample=str(path))
            rows.append(result);print(json.dumps(result),flush=True)
        finally:
            if env is not None:env.close()
        atomic(ROOT/'render_validation_progress.json',rows)
assert len(rows)==40
atomic(ROOT/'render_validation.json',dict(status='passed',tasks=len(rows),rows=rows,updated=time.time()))
