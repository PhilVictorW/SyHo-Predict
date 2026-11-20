"""Requested GPU1/2/3 expansion using live Stage II memory measurements."""
import argparse,collections,json,math,sys,time
from pathlib import Path
import run_stage_ii_auto_gpu as base
from a1_local.config_runtime import atomic_json

class RequestedScheduler(base.Scheduler):
    def __init__(self,args):
        super().__init__(args)
        self.report=base.read(args.capacity_report)
        self.target=list(self.report['requested_mapping']);self.monitor_script=str(Path(sys.argv[0]).resolve())
        if self.target!=[5,4,5,1,2,3]:raise ValueError('Expected the reviewed six-rank placement')
        if not self.report['host_admitted']:raise ValueError('Host memory report does not admit six ranks')
        self.required_host=int(self.report['required_host_after_exit_mib'])
        self.extra_host=int(self.report['incremental_host_budget_mib'])
        self.base_choose=base.choose_mapping
        base.choose_mapping=self.choose_requested
    def capacities(self,s):
        cap=super().capacities(s)
        # Newly requested GPUs use at least 60s of fresh observations, plus the
        # same per-load admission guard. Old transients remain visible in logs.
        recent=[x for x in self.history if x['time']>=s['time']-60]
        for g in (1,2,3):cap[g]=min(x['gpus'][g]['capacity_mib'] for x in recent)
        return cap
    def choose_requested(self,capacities,host_mib,current=(),blocked=()):
        limits={int(g):max(0,int((free-base.RESERVE)//base.BUDGET)) if int(g) not in blocked else 0 for g,free in capacities.items()}
        counts=collections.Counter(self.target)
        gpu_ready=all(limits.get(g,0)>=n for g,n in counts.items())
        host_ready=host_mib>=self.required_host
        detail=dict(gpu_slots=limits,host_limit=(6+max(0,int((host_mib-self.required_host)//self.extra_host))) if host_ready else 0,
            rank_budget_mib=base.BUDGET,gpu_reserve_mib=base.RESERVE,host_reserve_mib=base.HOST_RESERVE,
            host_capacity_mib=host_mib,capacity_mib=capacities,required_host_mib=self.required_host,
            incremental_host_budget_mib=self.extra_host,requested_rank_to_gpu=self.target,
            requested_gpu_ready=gpu_ready,requested_host_ready=host_ready,capacity_report=str(self.args.capacity_report))
        if not gpu_ready or not host_ready:
            # Keep an existing healthy group until its checkpoint boundary.
            # Admission failures at startup wait without discarding progress.
            return [],detail
        mapping=list(current) if current else list(self.target)
        # Retain every requested rank; automatically add further safe capacity.
        if any(mapping.count(g)<n for g,n in counts.items()):mapping=list(self.target)
        limit=min(32,detail['host_limit'],sum(limits.values()))
        if any(mapping.count(g)>limits.get(g,0) for g in set(mapping)):mapping=list(self.target)
        mapping=mapping[:max(6,limit)]
        while len(mapping)<limit:
            options=[g for g in limits if mapping.count(g)<limits[g]]
            if not options:break
            g=max(options,key=lambda x:(capacities[x]-mapping.count(x)*base.BUDGET,-x));mapping.append(g)
        return mapping,detail
    def record(self,phase,**kw):
        super().record(phase,**kw)
        for path in (base.LOGS/'gpu_scheduler_status.json',base.LOGS/'auto_continuation_status.json'):
            s=base.read(path);s.update(monitor_script=self.monitor_script,requested_rank_to_gpu=self.target,
                requested_capacity_report=str(self.args.capacity_report))
            atomic_json(path,s)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--poll-seconds',type=float,default=5);p.add_argument('--capacity-report',type=Path,required=True)
    a=p.parse_args()
    if a.poll_seconds<2:p.error('Polling interval must be at least two seconds')
    scheduler=RequestedScheduler(a)
    if not a.execute:
        s=base.sample(scheduler.out);cap={g:v['capacity_mib'] for g,v in s['gpus'].items()}
        m,d=scheduler.choose_requested(cap,s['host_available_mib']+s['own_host_mib'])
        print(json.dumps(dict(requested=scheduler.target,admitted=m,capacity=d,training_started=False),indent=2));return
    try:scheduler.run_forever()
    except BaseException as e:
        if scheduler.owns_locks:scheduler.record('failed',error=repr(e))
        raise
if __name__=='__main__':main()
