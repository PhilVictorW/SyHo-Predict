"""Admission for additions only; never stop live ranks because memory is low."""
import collections,json,math,subprocess,time
from pathlib import Path
from .config_runtime import atomic_json
WORK=Path(__file__).resolve().parents[1]
BUDGET=8704
EXCLUDED_GPUS={5}
GPU_RESERVE=2048
HOST_RESERVE=16384
SHARED_HOST=6144

def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}

def host_budgets():
    rows=read(WORK/'logs/requested_gpu123_capacity.json')['measurements']
    private=lambda row:row['rss_peak_mib']-row['file_mib']+row['swap_mib']
    master=math.ceil(private(next(row for row in rows if row['rank']==0))*1.10)
    extra=math.ceil(max(private(row) for row in rows if row['rank']!=0)*1.10)
    return master,extra

def required_host(world):
    master,extra=host_budgets()
    return HOST_RESERVE+SHARED_HOST+master+max(0,world-1)*extra

def snapshot(output):
    workers=[];own_host=0
    for path in Path(output).glob('status_rank*.json'):
        row=read(path);pid=row.get('pid',0);proc=Path('/proc')/str(pid)
        try:
            cmd=(proc/'cmdline').read_bytes()
            if b'a1_local.training' not in cmd or str(WORK/'config.json').encode() not in cmd or b'--worker' not in cmd:continue
            fields=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines())
            if fields['State'].strip().startswith(('Z','X')):continue
            own_host+=int(fields.get('RssAnon','0 kB').split()[0])/1024
            workers.append(row)
        except (OSError,ValueError):pass
    pids={row['pid'] for row in workers};owned=collections.defaultdict(int)
    command=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits']
    for line in subprocess.check_output(command,text=True,timeout=15).splitlines():
        fields=[v.strip() for v in line.split(',')]
        if len(fields)==3 and fields[1].isdigit() and int(fields[1]) in pids and fields[2].isdigit():owned[fields[0]]+=int(fields[2])
    gpus={}
    command=['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits']
    for line in subprocess.check_output(command,text=True,timeout=15).splitlines():
        gpu,uuid,free,used=[v.strip() for v in line.split(',')]
        gpus[int(gpu)]=dict(free_mib=int(free),used_mib=int(used),own_mib=owned[uuid],capacity_mib=int(free)+owned[uuid])
    if set(gpus)!=set(range(8)):raise ValueError('Expected eight GPUs')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024
    return dict(time=time.time(),gpus=gpus,host_available_mib=available,own_host_mib=own_host,workers=workers)

def capacities(s,window=None):
    caps={int(g):v['capacity_mib'] for g,v in s['gpus'].items()}
    host=s['host_available_mib']+s['own_host_mib']
    if window and 0<=s['time']-window.get('time',0)<=20:
        caps={g:min(v,window.get('capacity_mib',{}).get(str(g),v)) for g,v in caps.items()}
        host=min(host,window.get('host_capacity_mib',host))
    return caps,host

def choose(caps,host,current=(),preserve=True):
    caps={int(g):v for g,v in caps.items()};counts=collections.Counter();mapping=[]
    if set(current)&EXCLUDED_GPUS:raise ValueError('Running placement includes an excluded GPU')
    limits={g:0 if g in EXCLUDED_GPUS else max(0,int((v-GPU_RESERVE)//BUDGET)) for g,v in caps.items()}
    master,extra=host_budgets();host_limit=max(0,1+int((host-HOST_RESERVE-SHARED_HOST-master)//extra)) if host>=required_host(1) else 0
    for gpu in current:
        if preserve or (counts[gpu]<limits.get(gpu,0) and len(mapping)<host_limit):mapping.append(gpu);counts[gpu]+=1
    while len(mapping)<min(32,host_limit):
        options=[g for g in limits if counts[g]<limits[g]]
        if not options:break
        gpu=max(options,key=lambda g:(caps[g]-counts[g]*BUDGET,-g));mapping.append(gpu);counts[gpu]+=1
    return mapping,dict(capacity_mib=caps,host_capacity_mib=host,host_limit=host_limit,gpu_slots=limits,
        rank_budget_mib=BUDGET,gpu_reserve_mib=GPU_RESERVE,host_reserve_mib=HOST_RESERVE,
        host_master_mib=master,host_extra_mib=extra,shared_host_mib=SHARED_HOST)

def admitted(mapping,s):
    counts=collections.Counter(mapping)
    return bool(mapping) and not (set(mapping)&EXCLUDED_GPUS) and s['host_available_mib']>=required_host(len(mapping)) and all(
        s['gpus'][g]['free_mib']>=count*BUDGET+GPU_RESERVE for g,count in counts.items())

def checkpoint_decision(output,step,current):
    output=Path(output);control=read(output/'autoscale_control.json')
    if not control.get('enabled') or step%50:return None
    latest=read(output/'latest.json');marker=read(output/f'step_{step:06d}'/'checkpoint_complete.json')
    if latest.get('step')!=step or not marker.get('complete') or not marker.get('resumable'):
        raise ValueError('Expansion requires a committed resumable checkpoint')
    s=snapshot(output);window=read(WORK/'logs/checkpoint_gpu_capacity.json');caps,host=capacities(s,window)
    mapping,detail=choose(caps,host,current,preserve=True)
    decision=dict(time=time.time(),step=step,checkpoint=str(output/f'step_{step:06d}'),
        previous_rank_to_gpu=list(current),rank_to_gpu=mapping,capacity=detail,
        added_ranks=len(mapping)-len(current),decision='expand' if mapping!=list(current) else 'continue',
        active_group_termination=False,excluded_gpus=sorted(EXCLUDED_GPUS))
    atomic_json(output/'autoscale_last_decision.json',decision)
    with (WORK/'logs/checkpoint_gpu_decisions.jsonl').open('a') as f:f.write(json.dumps(decision)+'\n')
    if mapping!=list(current):
        request=dict(time=time.time(),minimum_step=step,rank_to_gpu=mapping,reason='additional capacity after complete checkpoint',checkpoint=str(output/f'step_{step:06d}'))
        atomic_json(output/'elastic_reconfigure.json',request)
        return request
    return None
