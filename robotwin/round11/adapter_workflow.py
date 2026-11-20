"""Adapter confirmation -> qualified online pilot or component/role diagnosis."""
import fcntl,hashlib,math,os,subprocess,time,traceback,json
from robotwin.round11.autopilot import ROOT,R,PY,read,write,identity,live,find_module,free_memory
OUT=R/'adapter_workflow';CONFIRM=R/'adapter_confirmation_200';ONLINE=R/'online_adapter_200'

def decision(result):
    assert result.get('completed') is True and len(result['rows'])==128
    assert result['frozen_audit']['passed'] is True and result['preflight_max_abs']<1e-6
    assert result['future_access_guard_anchors']==64
    assert len({(r['cohort'],r['episode'],r['t'],r['noise_seed']) for r in result['rows']})==128
    groups=['fresh','tail','tail_remaining_1','tail_remaining_8','tail_remaining_24','tail_remaining_36']
    s=result['summary'];assert s['fresh']['episodes']==32 and s['tail']['episodes']==8
    assert s['fresh']['pairs']==s['tail']['pairs']==64
    for g in groups:
        c=s[g]['comparisons']['adapter200_vs_initial']
        assert all(math.isfinite(x) for x in list(c['ratios'].values())+c['bootstrap_95'])
    primary=s['fresh']['comparisons']['adapter200_vs_initial']
    safe=all(s[g]['comparisons']['adapter200_vs_initial']['nonregression_passed'] for g in groups)
    eligible=bool(primary['improvement_supported'] and safe);assert result['online_eligible']==eligible
    return dict(next='online_pilot' if eligible else 'diagnosis',primary=primary,all_nonregression_passed=safe,
        reason='Fresh benefit and all safety checks pass' if eligible else 'Benefit/safety gate not met; analyze role ablations and per-episode errors')

def diagnose(result):
    grouped={}
    for r in result['rows']:
        key=(r['cohort'],r['episode'])
        grouped.setdefault(key,dict(cohort=key[0],episode=key[1],task=r['task'],rows=[]))['rows'].append(r)
    entries=[]
    for item in grouped.values():
        rows=item.pop('rows');keys=['position_cm','rotation_deg','gripper_mae']
        base={k:sum(r['metrics']['initial'][k] for r in rows)/len(rows) for k in keys}
        new={k:sum(r['metrics']['adapter200'][k] for r in rows)/len(rows) for k in keys}
        item.update(initial=base,adapter200=new,relative_change={k:new[k]/max(base[k],1e-10)-1 for k in keys})
        entries.append(item)
    write(OUT/'diagnosis.json',dict(completed=True,evidence=result['summary'],episodes=entries,
        kind='Independent component error and Goal/Local intervention diagnosis',
        limitation='Role injection ablations do not prove privileged-future distillation is useful',
        next_research_step='Design matched-budget training control using the measured failing component and role sensitivity; do not blindly extend training'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'plan.json',dict(physical_gpu=2,poll_seconds=15,checkpoint='action_adapter_200/step_000200/adapter.pt',
        confirmation='32 fresh episodes +8 tail episodes at remaining1/8/24/36; two noises;128 matched cases',
        ablations=['Goal off','Local off','both off'],after_pass='4 tasks x2 modes x5 episodes recorded-reference online screening',
        after_fail='automatic component/role diagnosis; retain all checkpoints',online_stall_seconds=1800,
        unrelated_processes='never signaled or restarted'))
    child=None
    def update(phase,**kw):write(OUT/'status.json',dict(pid=os.getpid(),phase=phase,updated_at=time.time(),**kw))
    def ensure(module,target,required):
        nonlocal child
        state=read(target/'status.json');rp=OUT/(target.name+'_process.json');old=read(rp)
        found=old if old and live(old) else find_module(module)
        if found:
            if found!=old:write(rp,found)
            return 'running',state
        if state.get('phase') in ('completed','failed'):return state['phase'],state
        if old or (target/'manifest.json').exists():return 'unexpected_exit',state
        free=free_memory()
        if free<required:return 'waiting_gpu2_memory',dict(free_mib=free,required_mib=required)
        target.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='2',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',
            MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NO_ALBUMENTATIONS_UPDATE='1',PYTHONPATH=str(ROOT))
        cmd=['nice','-n','10','ionice','-c','2','-n','7',PY,'-u','-m',module]
        if target==ONLINE:cmd+=['--candidate','action_adapter']
        with (target/'run.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(1);record=identity(child.pid)
        assert record and f'-m {module}' in record['command'],'Worker exited before startup'
        write(rp,record);print(json.dumps(dict(event='started',module=module,process=record)),flush=True)
        return 'started',record
    while True:
        try:
            if child is not None:child.poll()
            status=read(CONFIRM/'status.json')
            if status.get('phase')=='completed':
                path=CONFIRM/'results.json';result=read(path)
                selected=dict(decision(result),confirmation_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                prior=read(OUT/'decision.json')
                if prior:assert prior==selected,'Decision input changed'
                else:write(OUT/'decision.json',selected)
                if find_module('robotwin.round11.adapter_confirm'):update('waiting_confirmation_exit',decision=selected)
                elif selected['next']=='online_pilot':
                    stage,state=ensure('robotwin.round11.online_pilot',ONLINE,32768)
                    update('online_'+stage,decision=selected,online=state)
                    if stage=='completed' and not (OUT/'online_review.json').exists():
                        write(OUT/'online_review.json',dict(results=read(ONLINE/'results.json'),
                            conclusion='Screening complete; inspect matched coverage and success before broader claims'))
                else:
                    if not (OUT/'diagnosis.json').exists():diagnose(result)
                    update('diagnosis_completed',decision=selected,diagnosis=str(OUT/'diagnosis.json'),
                        next='evidence-based next training design; no automatic blind step extension')
            else:
                stage,state=ensure('robotwin.round11.adapter_confirm',CONFIRM,24576)
                update('confirmation_'+stage,confirmation=state)
        except Exception:
            error=traceback.format_exc();update('workflow_error',error=error);print(error,flush=True)
        time.sleep(15)

if __name__=='__main__':main()
