"""Review each completed or failed task immediately; keep diagnoses separate from fixes."""
import argparse,csv,fcntl,hashlib,io,json,math,re,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;OUT=R/'candidate_online';REV=OUT/'reviews'

def read(p):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {}

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def records(p):
    try:lines=Path(p).read_text(errors='replace').splitlines()
    except OSError:return []
    out=[]
    for line in lines:
        try:out.append(json.loads(line))
        except ValueError:pass
    return out

def interval(k,n):
    if not n:return None
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-w),min(1,c+w)]

def compare(row,baseline):
    sm=row.get('summary',{});eps=sm.get('episodes',[])
    b=baseline.get(row['key'],{});old={int(e['seed']):e for e in b.get('episodes',[])}
    done=row['status']=='completed' and len(eps)==10 and sm.get('n_episodes')==10
    d=dict(key=row['key'],job=row['job'],gpu=row['gpu'],evaluated_at=time.time(),run_status=row['status'],
           completed=done,baseline_path=b.get('path'),baseline_successes=b.get('successes'),baseline_episodes=b.get('n_episodes',0),
           strict_pair=False,caveat='Historical instructions, policy noise and simulator concurrency are not matched; 10-episode screening only.',
           model_fix_applied=False)
    if not done:
        pr=row.get('partial_progress',{});d.update(decision='runtime_repair',partial_completed=pr.get('completed_episodes',0),partial_successes=pr.get('successes',0));return d
    k=sum(bool(e['success']) for e in eps);n=len(eps);overlap=[e for e in eps if int(e['seed']) in old]
    losses=[int(e['seed']) for e in overlap if not e['success'] and old[int(e['seed'])]['success']]
    gains=[int(e['seed']) for e in overlap if e['success'] and not old[int(e['seed'])]['success']]
    old_rate=b.get('success_rate');delta=(k/n-old_rate)*100 if old_rate is not None else None
    decision='baseline_missing' if not old else 'performance_diagnosis' if losses or (delta is not None and delta < -1e-8) else 'no_regression_signal'
    d.update(decision=decision,candidate_successes=k,candidate_episodes=n,candidate_rate=k/n,candidate_wilson95=interval(k,n),
             baseline_rate=old_rate,delta_percentage_points=delta,overlap_episodes=len(overlap),
             baseline_successes_on_overlap=sum(bool(old[int(e['seed'])]['success']) for e in overlap),
             candidate_successes_on_overlap=sum(bool(e['success']) for e in overlap),
             regression_seeds=losses,improvement_seeds=gains,failed_episodes=[e for e in eps if not e['success']])
    return d

def diagnose(row,d):
    job=Path(row['job']);log=job/'bridge.log'
    try:text=log.read_text(errors='replace')
    except OSError:text=''
    traces=records(job/'requests.jsonl');byseed={}
    for q in traces:byseed.setdefault(int(q['episode_seed']),[]).append(q)
    evidence={}
    targets=d.get('regression_seeds',[]) or [int(e['seed']) for e in d.get('failed_episodes',[])]
    for seed in targets:
        qs=byseed.get(seed,[])
        # Previously generated original traces, if available, are read only.
        old=records(R/'online/jobs'/f"{row['key']}__original"/'requests.jsonl')
        peer=[q for q in old if int(q['episode_seed'])==seed and int(q['query_index'])==0]
        first=next((q for q in qs if int(q['query_index'])==0),None)
        episode=next((e for e in row.get('summary',{}).get('episodes',[]) if int(e['seed'])==seed),{})
        evidence[str(seed)]=dict(query_count=len(qs),instruction=first.get('instruction') if first else None,
            query_indices=[q['query_index'] for q in qs],steps=episode.get('steps'),episode_error=episode.get('error'),
            initial_images_match_prior_original=(first['image_sha256']==peer[0]['image_sha256']) if first and peer else None,
            instruction_matches_prior_original=(first['instruction']==peer[0]['instruction']) if first and peer else None,
            noise_matches_prior_original=(first['noise_seed']==peer[0]['noise_seed']) if first and peer else None)
    vulkan='ErrorInitializationFailed' in text;oom='out of memory' in text.lower()
    seed_error='Fixed evaluation seeds exhausted' in text or ('All slot workers exited' in text and bool(re.search(r'seed=\d+ invalid',text)))
    error_lines=[line.strip() for line in text.splitlines() if any(x in line for x in ['RuntimeError:','AttributeError:','OutOfMemoryError:','[environment retry]','invalid due to error','Fixed evaluation seeds exhausted'])]
    diagnostics=dict(completed_at=time.time(),runtime_error_lines=error_lines[-15:],cuda_oom=oom,vulkan_initialization_error=vulkan,
                     suspected_fixed_seed_exhaustion=seed_error,failed_seed_evidence=evidence,trace_queries=len(traces),
                     unique_episode_seeds=sorted(byseed),evidence_source='saved logs, episode outcomes, requests and historical records')
    if d['decision']=='runtime_repair':
        if vulkan and seed_error:
            action='Bridge patched: bounded same-seed retries after setup exceptions, rebuild partially initialized environment, persist partial episodes. Validate on retry; do not count invalid setup as policy failure.'
            stage='runtime_patch_written_validation_pending'
        elif oom:action='Queue admission guard and bounded retry/cooldown active; no model-weight change justified.';stage='resource_retry_pending'
        else:action='Inspect runtime error and retry separately; retain partial results.';stage='runtime_diagnosis_complete'
    elif d['decision']=='performance_diagnosis':
        action='Failure-seed and deployment evidence audited. Reproduce candidate failures against original only on affected seeds before selecting a shared model fix. No weight change has been applied.'
        stage='diagnosis_complete_controlled_reproduction_needed'
    elif d['decision']=='baseline_missing':action='Baseline unavailable; retain candidate result, no unsupported regression claim.';stage='baseline_missing'
    else:action='No regression signal in this screening sample; retain current model.';stage='no_fix_indicated'
    d.update(diagnostics=diagnostics,repair_stage=stage,next_action=action)
    return d

def run_once():
    status=read(OUT/'status.json');baseline=read(OUT/'historical_baseline.json');all_reviews=[]
    for row in status.get('results',[]):
        ident=hashlib.sha256((row['job']+'|'+row['status']).encode()).hexdigest()[:12]
        path=REV/f"{row['key']}__{ident}.json";d=read(path)
        if not d:
            d=diagnose(row,compare(row,baseline));write(path,d)
            with (REV/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),key=d['key'],decision=d['decision'],repair_stage=d['repair_stage'],review=str(path)),ensure_ascii=False)+'\n')
            print(d['key'],d['decision'],d['repair_stage'],flush=True)
        all_reviews.append(dict(d,review_path=str(path)))
    pending=[d for d in all_reviews if d['decision'] in ('runtime_repair','performance_diagnosis')]
    # Keep runtime attempts visible, but a later completed retry resolves that runtime case.
    completed_keys={d['key'] for d in all_reviews if d['completed']}
    pending=[d for d in pending if not (d['decision']=='runtime_repair' and d['key'] in completed_keys)]
    info=dict(updated_at=time.time(),source_status_age_seconds=time.time()-status.get('updated_at',time.time()),
              reviewed_attempts=len(all_reviews),completed_comparisons=sum(d['completed'] for d in all_reviews),
              review_interval_seconds=5,model_fix_applied=False,
              automatic_work='comparison and log/seed/deployment diagnosis; arbitrary model repair is not automated',
              cases=all_reviews,pending_repairs=pending)
    write(REV/'status.json',info)
    buf=io.StringIO();w=csv.writer(buf);w.writerow(['task_mode','status','candidate_success','candidate_n','historical_success','historical_n','delta_pp','overlap_n','lost_seeds','decision','repair_stage'])
    for d in all_reviews:w.writerow([d['key'],d['run_status'],d.get('candidate_successes'),d.get('candidate_episodes'),d.get('baseline_successes'),d.get('baseline_episodes'),d.get('delta_percentage_points'),d.get('overlap_episodes'),json.dumps(d.get('regression_seeds',[])),d['decision'],d['repair_stage']])
    tmp=REV/'comparison.csv.tmp';tmp.write_text(buf.getvalue());tmp.replace(REV/'comparison.csv')
    return info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    REV.mkdir(parents=True,exist_ok=True)
    lock=(REV/'watcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:run_once()
        except Exception:
            traceback.print_exc();write(REV/'watcher_error.json',dict(time=time.time(),error=traceback.format_exc()))
            if args.once:raise
        if args.once:break
        time.sleep(5)

if __name__=='__main__':main()
