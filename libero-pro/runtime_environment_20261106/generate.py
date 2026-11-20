"""Build paired Environment initial states from released base-suite states.
Preserve local task layout and robot joint states; translate object/furniture supports
to the replacement workspace. The official fresh reset sampler can misplace objects
on articulated fixtures after environment replacement, so do not use it for saved states.
"""
import argparse, hashlib, re, runpy, sys, traceback
from collections import Counter
from common import *
sys.path.insert(0,str(REPO))
import numpy as np
import torch
from perturbation import BDDLParser, EnvironmentReplacePerturbator
from libero.libero.envs.env_wrapper import ControlEnv
ROOT=BASE/'resources/environment_20261106'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def positions(env,names):
    return np.array([env.sim.data.body_xpos[env.sim.model.body_name2id(env.env.objects_dict[n].root_body)].copy() for n in names])
def step10(env):
    for _ in range(10): env.step([0.]*6+[-1.])
def generate(suite,count,probe):
    mapping=runpy.run_path(str(REPO/'libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']
    reports=[]
    for ti,name in enumerate(mapping[suite]):
        src=REPO/'libero/libero/bddl_files'/suite/(name+'.bddl')
        srcinit=REPO/'libero/libero/init_files'/suite/(name+'.pruned_init')
        text=src.read_text();p=EnvironmentReplacePerturbator(BDDLParser(text),str(REPO/'libero_ood/ood_environment.yaml'))
        old=p._extract_current_env(suite,name);new='main_table' if old=='living_room_table' else 'living_room_table'
        changed=p._rewrite_fixtures_type(p._rewrite_problem_env_token(text.replace(old,new),new),new,p.ENV_FIXTYPE[new])
        assert changed!=text and new!=old
        folder=ROOT/('probe' if probe else 'validated')/suite;folder.mkdir(parents=True,exist_ok=True)
        bddl=folder/(name+'.bddl');bddl.write_text(changed)
        lang=lambda x:' '.join(re.search(r'\(:language\s+(.*?)\)',x,re.S).group(1).split()).lower()
        assert lang(text)==lang(changed)
        report=dict(suite=suite,task_id=ti,name=name,source=str(src),source_sha256=sha(src),
            source_init=str(srcinit),source_init_sha256=sha(srcinit),source_environment=old,target_environment=new,
            bddl=str(bddl),bddl_sha256=sha(bddl),instruction=lang(changed),requested_states=count,
            protocol='official_scene_replacement_with_transferred_validated_base_states_v3')
        source_env=None;target=None;states=[];fixtures_list=[];origins=[];rejects=Counter();diagnostics=[]
        try:
            source_env=ControlEnv(bddl_file_name=str(src),has_offscreen_renderer=False,use_camera_obs=False,hard_reset=False)
            target=ControlEnv(bddl_file_name=str(bddl),has_offscreen_renderer=False,use_camera_obs=False,hard_reset=False)
            source_env.seed(0);source_env.env.reset()
            target.seed(0);target.env.reset()
            source_states=np.asarray(torch.load(srcinit,map_location='cpu',weights_only=False))
            assert source_states.shape[1]==len(target.get_sim_state())
            assert source_env.sim.model.nq==target.sim.model.nq
            names=list(source_env.env.objects_dict);assert names==list(target.env.objects_dict)
            workspace_delta=np.array(target.env.workspace_offset)-np.array(source_env.env.workspace_offset)
            fixture_delta=workspace_delta.copy();fixture_delta[2]+=target.env.z_offset-source_env.env.z_offset
            fixtures={}
            for fname,fobj in source_env.env.fixtures_dict.items():
                bid=source_env.sim.model.body_name2id(fobj.root_body)
                fixtures[fname]=dict(body=fobj.root_body,pos=(source_env.sim.model.body_pos[bid]+fixture_delta).tolist(),
                                    quat=source_env.sim.model.body_quat[bid].tolist())
            parents={}
            for predicate in source_env.env.parsed_problem['initial_state']:
                if len(predicate)!=3 or predicate[0] not in ('on','in'):continue
                obj,support=predicate[1:]
                if support in source_env.env.parsed_problem['regions']:
                    support=source_env.env.parsed_problem['regions'][support]['target']
                parents[obj]=support
            def delta_for(obj,seen=()):
                assert obj not in seen, (obj,seen)
                support=parents.get(obj,old)
                if support in fixtures:return fixture_delta
                if support in names:return delta_for(support,seen+(obj,))
                return workspace_delta
            shifts={n:delta_for(n) for n in names}
            addresses={}
            for n in names:
                joint=source_env.env.objects_dict[n].joints[-1]
                addr=source_env.sim.model.get_joint_qpos_addr(joint)
                assert addr==target.sim.model.get_joint_qpos_addr(joint) and isinstance(addr,tuple) and addr[1]-addr[0]==7
                addresses[n]=addr[0]
            unique=set()
            for attempt in range(max(100,count*4)):
                if len(states)>=count:break
                index=attempt%len(source_states)
                initial=source_states[index].copy()
                for n,start in addresses.items():initial[1+start:1+start+3]+=shifts[n]
                if attempt>=len(source_states):
                    rng=np.random.default_rng(20261106+SUITES.index(suite)*100000+ti*1000+attempt)
                    initial[1:8]+=rng.normal(0,.003,7)
                target.seed(0);target.env.reset()
                for f in fixtures.values():
                    bid=target.sim.model.body_name2id(f['body'])
                    target.sim.model.body_pos[bid]=f['pos'];target.sim.model.body_quat[bid]=f['quat']
                target.set_init_state(initial)
                before=positions(target,names)
                initial_success=target.check_success()
                step10(target);after=positions(target,names);first_success=target.check_success()
                step10(target);later=positions(target,names)
                drift=float(np.linalg.norm(later-after,axis=1).max())
                moved=float(np.linalg.norm(after-before,axis=1).max())
                reason=None
                if not np.isfinite(target.get_sim_state()).all():reason='nonfinite'
                elif initial_success or first_success or target.check_success():reason='already_successful'
                elif later[:,2].min()<target.env.workspace_offset[2]-.08:reason='object_below_workspace'
                elif drift>.03:reason='unstable_after_settling'
                elif moved>.15:reason='large_initial_motion'
                # All initial support relationships to objects/fixtures must hold.
                critical_false=[]
                for pred in target.env.parsed_problem['initial_state']:
                    if len(pred)!=3 or pred[0] not in ('on','in'):continue
                    support=pred[2]
                    if support in target.env.parsed_problem['regions']:
                        support=target.env.parsed_problem['regions'][support]['target']
                    if support in target.env.objects_dict or support in target.env.fixtures_dict:
                        if not target.env._eval_predicate(pred):critical_false.append(pred)
                if critical_false:reason='fixture_or_object_support_mismatch'
                digest=hashlib.sha256(initial.tobytes()).hexdigest()
                if digest in unique:reason='duplicate_state'
                if reason:
                    rejects[reason]+=1
                    if rejects[reason]==1:print(json.dumps(dict(event='candidate_rejected',suite=suite,task_id=ti,reason=reason,critical_false=critical_false,drift=drift,moved=moved)),flush=True)
                    continue
                unique.add(digest);states.append(initial);fixtures_list.append(fixtures)
                origins.append(dict(source_index=index,robot_jitter=attempt>=len(source_states),attempt=attempt))
                diagnostics.append(dict(settling_motion=moved,post_settle_motion=drift))
            assert len(states)==count,f'Only {len(states)}/{count} validated states: {rejects}'
            init=folder/(name+'.pruned_init');torch.save(np.stack(states),init)
            fp=folder/(name+'.fixtures.json');atomic(fp,fixtures_list)
            assert np.array_equal(torch.load(init,map_location='cpu',weights_only=False),np.stack(states))
            report.update(status='validated',init=str(init),init_sha256=sha(init),fixtures=str(fp),fixtures_sha256=sha(fp),
                initial_states=len(states),unique_states=len(unique),state_shape=list(states[0].shape),state_origins=origins,
                workspace_translation=workspace_delta.tolist(),fixture_translation=fixture_delta.tolist(),
                rejected=dict(rejects),max_settle_motion=max(d['settling_motion'] for d in diagnostics),
                max_post_settle_motion=max(d['post_settle_motion'] for d in diagnostics),
                validation='Released base states transferred by support relationships; complete fixed-fixture poses restored; finite; unique; no solved initial goal; no fallen objects; 10+10 dummy steps; post-settle movement <=3cm; initial On/In object-and-fixture relationships hold; torch roundtrip')
        except Exception as exc:
            traceback.print_exc();report.update(status='failed',error=repr(exc),accepted=len(states),rejected=dict(rejects))
        finally:
            if source_env is not None:source_env.close()
            if target is not None:target.close()
        report['updated']=time.time();atomic(folder/(name+'.json'),report);reports.append(report)
        print(json.dumps(report),flush=True)
    atomic(ROOT/(('probe_' if probe else 'validated_')+suite+'.json'),reports)
    return 0 if all(r['status']=='validated' for r in reports) else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True);p.add_argument('--count',type=int,default=50);p.add_argument('--probe',action='store_true')
    a=p.parse_args();raise SystemExit(generate(a.suite,a.count,a.probe))
