import json
from pathlib import Path
import time

WORK=Path(__file__).resolve().parents[1]
TRAIN=WORK/'runs/stage_II_20260925'
ROOT=WORK/'evaluation/stage_II_20260925'
BASE=Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2')
BENCH=Path('/mnt/hdd/wangchaoyang/CKPT/LaWAM/dataset/libero-plus/inference/LIBERO-plus')
PY='/home/wangchaoyang/envs/LaWAM/bin/python'
SIMPY='/home/wangchaoyang/envs/libero-plus/bin/python'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
CATEGORIES=['Camera','Robot','Language','Light','Background','Noise','Layout']
GPUS=[6,6,7,7]
PORTS=[24880,24881,24882,24883]
ACTION_PROTOCOL=dict(id='libero_plus_open_to_legacy_close_v1',source_gripper='one_is_open',
    client_gripper='one_is_closed',simulator_gripper='minus_one_open_plus_one_closed',
    conversion='client_gripper = 1 - predicted_gripper',motion_dimensions_unchanged=True)
_COLLECT_CACHE={}

def evaluation_spec(step):
    override_file=ROOT/'sampling_overrides.json'
    overrides=json.loads(override_file.read_text()) if override_file.exists() else {}
    selected=overrides.get(str(step))
    if selected is None:
        return dict(folder=ROOT/f'step_{step:06d}_gripper_v1',manifest=ROOT/'full_manifest.json',scope='full_libero_plus')
    if step!=1000 or selected['scope']!='step500_guided_stratified_subset':
        raise ValueError('Only the explicitly requested step-1000 sampling override is supported')
    folder=ROOT/selected['folder'];manifest=ROOT/selected['manifest']
    if folder.parent!=ROOT or manifest.parent!=ROOT:raise ValueError('Sampling paths must remain in evaluation directory')
    return dict(folder=folder,manifest=manifest,scope=selected['scope'])

def eval_folder(step):
    return evaluation_spec(step)['folder']

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def collect(folder):
    import csv
    folder=Path(folder); protocol=json.loads((folder/'protocol.json').read_text())
    if protocol.get('action_protocol')!=ACTION_PROTOCOL:
        raise ValueError('Refuse to mix evaluation results with a different gripper protocol')
    wanted={(r['suite'],r['task_id']):r for r in protocol['tasks']}
    # Completed episode files are immutable; avoid rereading all 10030 every refresh.
    cache=_COLLECT_CACHE.setdefault(str(folder),{})
    rows={}
    for f in (folder/'episodes').glob('*.json'):
        if f.name not in cache:cache[f.name]=json.loads(f.read_text())
        r=cache[f.name];key=(r['suite'],r['task_id'])
        if key not in wanted or r.get('error'):raise ValueError(f'Unexpected result {f}')
        assert r['episode_idx']==0 and r['checkpoint']==protocol['checkpoint']
        if r.get('action_protocol_id')!=ACTION_PROTOCOL['id']:
            raise ValueError(f'Unverified gripper semantics in {f}')
        rows[key]=r
    cells={s:{c:dict(episodes=0,successes=0,expected=sum(t['suite']==s and t['category']==c for t in wanted.values())) for c in CATEGORIES} for s in SUITES}
    for key,r in rows.items():
        cell=cells[key[0]][wanted[key]['category']];cell['episodes']+=1;cell['successes']+=int(r['success'])
    suites={};categories={}
    for s in SUITES:
        suites[s]=dict(episodes=sum(c['episodes'] for c in cells[s].values()),successes=sum(c['successes'] for c in cells[s].values()))
        for c in CATEGORIES:
            cell=cells[s][c];cell['success_rate']=cell['successes']/cell['episodes'] if cell['episodes'] else None
    for c in CATEGORIES:categories[c]=dict(episodes=sum(cells[s][c]['episodes'] for s in SUITES),successes=sum(cells[s][c]['successes'] for s in SUITES))
    for r in list(suites.values())+list(categories.values()):r['success_rate']=r['successes']/r['episodes'] if r['episodes'] else None
    n=len(rows);success=sum(int(r['success']) for r in rows.values())
    result=dict(step=protocol['step'],smoke=protocol['smoke'],scope=protocol.get('scope','sampled'),checkpoint=protocol['checkpoint'],episodes=n,expected=len(wanted),
        successes=success,success_rate=success/n if n else None,complete=n==len(wanted),cells=cells,suites=suites,categories=categories,action_protocol=ACTION_PROTOCOL,updated=time.time())
    if 'sampling' in protocol:
        result['sampling']=protocol['sampling']
        result['full_benchmark_comparable']=False
    atomic(folder/'results.json',result)
    with (folder/'matrix.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['suite','category','completed','expected','successes','success_rate'])
        for s in SUITES:
            for c in CATEGORIES:
                r=cells[s][c];writer.writerow([s,c,r['episodes'],r['expected'],r['successes'],r['success_rate']])
    return result
