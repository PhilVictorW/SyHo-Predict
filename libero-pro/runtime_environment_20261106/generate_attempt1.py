"""Deterministic local Environment extension; CPU-only state generation."""
import argparse, hashlib, json, random, re, runpy, signal, sys, time, traceback
from collections import Counter
from common import *
sys.path.insert(0, str(REPO))
import numpy as np
import torch
import yaml
from perturbation import BDDLParser, EnvironmentReplacePerturbator
from libero.libero.envs.env_wrapper import ControlEnv

ROOT = BASE / 'resources/environment_20261106'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timeout(*_): raise TimeoutError('Environment initialization exceeded 90 seconds')
signal.signal(signal.SIGALRM, timeout)

def generate(suite, count, probe):
    mapping = runpy.run_path(str(REPO/'libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']
    cfg = REPO/'libero_ood/ood_environment.yaml'
    si = SUITES.index(suite)
    reports=[]
    for ti,name in enumerate(mapping[suite]):
        src=REPO/'libero/libero/bddl_files'/suite/(name+'.bddl')
        original=src.read_text()
        p=EnvironmentReplacePerturbator(BDDLParser(original),str(cfg))
        old=p._extract_current_env(suite,name)
        # Official code fixes living_room_table; avoid identity transformations on living-room tasks.
        new='main_table' if old=='living_room_table' else 'living_room_table'
        assert old in p.ALLOWED_ENVS and old != new
        content=original.replace(old,new)
        content=p._rewrite_problem_env_token(content,new)
        content=p._rewrite_fixtures_type(content,new,p.ENV_FIXTYPE[new])
        assert content != original
        lang=lambda t:' '.join(re.search(r'\(:language\s+(.*?)\)',t,re.S).group(1).split()).lower()
        assert lang(original)==lang(content)
        folder=ROOT/('probe' if probe else 'validated')/suite
        folder.mkdir(parents=True,exist_ok=True)
        bddl=folder/(name+'.bddl');bddl.write_text(content)
        report=dict(suite=suite,task_id=ti,name=name,source=str(src),source_sha256=sha(src),
                    source_environment=old,target_environment=new,bddl=str(bddl),
                    bddl_sha256=sha(bddl),instruction=lang(content),requested_states=count,
                    protocol='official_environment_replacement_with_nonidentity_living_room_fallback_v1')
        env=None; validator=None; states=[]; seeds=[]; rejects=Counter(); maxdrift=0.; minz=100.
        try:
            signal.alarm(90)
            env=ControlEnv(bddl_file_name=str(bddl),has_offscreen_renderer=False,
                           use_camera_obs=False,hard_reset=False)
            signal.alarm(0)
            validator=ControlEnv(bddl_file_name=str(bddl),has_offscreen_renderer=False,
                                 use_camera_obs=False,hard_reset=False)
            names=list(validator.env.objects_dict)
            ids=[validator.sim.model.body_name2id(validator.env.objects_dict[n].root_body) for n in names]
            table_z=float(env.env.workspace_offset[2])
            unique=set()
            for attempt in range(max(200,count*10)):
                if len(states)==count: break
                seed=20261106+si*100000+ti*1000+attempt
                random.seed(seed);np.random.seed(seed);env.seed(seed)
                signal.alarm(90)
                env.env.reset() # Bypass wrapper's unbounded retry loop.
                initial=env.get_sim_state().copy()
                validator.seed(0);validator.env.reset()
                validator.set_init_state(initial)
                before=validator.sim.data.body_xpos[ids].copy()
                initially_done=bool(validator.check_success())
                for _ in range(10):
                    validator.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
                after=validator.sim.data.body_xpos[ids].copy()
                final=validator.get_sim_state()
                signal.alarm(0)
                if not np.isfinite(initial).all() or not np.isfinite(final).all():
                    rejects['nonfinite']+=1;continue
                if initially_done or validator.check_success():
                    rejects['already_successful']+=1;continue
                if after[:,2].min()<table_z-.08:
                    rejects['object_below_workspace']+=1;continue
                drift=float(np.linalg.norm(after[:,:2]-before[:,:2],axis=1).max())
                if drift>.15:
                    rejects['large_uncommanded_xy_motion']+=1;continue
                digest=hashlib.sha256(initial.tobytes()).hexdigest()
                if digest in unique: rejects['duplicate_state']+=1;continue
                unique.add(digest);states.append(initial);seeds.append(seed)
                maxdrift=max(maxdrift,drift);minz=min(minz,float(after[:,2].min()))
            assert len(states)==count, f'Only {len(states)}/{count} valid states; {rejects}'
            init=folder/(name+'.pruned_init')
            torch.save(np.stack(states),init)
            restored=torch.load(init,map_location='cpu',weights_only=False)
            assert np.array_equal(restored,np.stack(states))
            report.update(status='validated',init=str(init),init_sha256=sha(init),
                initial_states=len(states),state_shape=list(states[0].shape),unique_states=len(unique),
                generation_seeds=seeds,rejected=dict(rejects),max_settle_xy_drift=maxdrift,
                min_settled_object_z=minz,workspace_z=table_z,
                validation='CPU reset and restore into separate seed-0 evaluator; finite state; 10 zero-action settling steps; no solved initial goal; no fallen objects; XY drift <= 0.15m; unique states; torch load roundtrip')
        except Exception as exc:
            signal.alarm(0);traceback.print_exc()
            report.update(status='failed',error=repr(exc),accepted=len(states),rejected=dict(rejects))
        finally:
            signal.alarm(0)
            if env is not None: env.close()
            if validator is not None: validator.close()
        report['updated']=time.time()
        atomic(folder/(name+'.json'),report)
        reports.append(report)
        print(json.dumps(report),flush=True)
    atomic(ROOT/(('probe_' if probe else 'validated_')+suite+'.json'),reports)
    return 0 if all(r['status']=='validated' for r in reports) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=SUITES)
    p.add_argument('--count',type=int,default=50);p.add_argument('--probe',action='store_true')
    a=p.parse_args();raise SystemExit(generate(a.suite,a.count,a.probe))
