"""Compare aba2 against completed aba1 and original LaWAM, using unique episode records."""
import argparse, csv, json
from pathlib import Path

BASELINE = Path('/mnt/hdd/wangchaoyang/results/LaWAM/LIBERO/full/lawam_libero_sft_release/full_40x50_20260903')
ABA1 = Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba1/results/ablation/inference_without_lawm/full_40x50_20260914')
SUITES = ['libero_10', 'libero_object', 'libero_goal', 'libero_spatial']


def read_episodes(run, suite):
    paths = sorted((run/'tasks').glob(suite+'__task*/episodes.checkpoint.jsonl')) if (run/'tasks').is_dir() else [run/'suites'/suite/'episodes.checkpoint.jsonl']
    rows = {}
    for path in paths:
        if not path.exists(): continue
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try: row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines)-1: continue  # An episode writer may be mid-append.
                raise
            key = (int(row['task_id']), int(row['episode_idx']))
            if key in rows: raise RuntimeError(f'Duplicate episode: {path} {key}')
            if key not in {(t,e) for t in range(10) for e in range(50)}:
                raise RuntimeError(f'Unexpected episode: {path} {key}')
            rows[key] = row
    return list(rows.values())


def metrics(rows):
    n=len(rows); k=sum(bool(r['success']) for r in rows)
    return n,k,100*k/n if n else None


def atomic_text(path, text):
    temp=path.with_name(path.name+'.tmp'); temp.write_text(text); temp.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('run_dir', type=Path); args=parser.parse_args()
    suite_rows=[]; task_rows=[]; video_records=0
    for suite in SUITES:
        rows=read_episodes(args.run_dir,suite); base=read_episodes(BASELINE,suite); aba1=read_episodes(ABA1,suite)
        n,k,rate=metrics(rows); bn,bk,br=metrics(base); an,ak,ar=metrics(aba1)
        if bn != 500 or an != 500: raise RuntimeError(f'Incomplete comparison data for {suite}')
        errors=sum(bool(r.get('error')) for r in rows)
        video_records+=sum(bool(r.get('rollout_video_path')) for r in rows)
        suite_rows.append(dict(suite=suite, episodes=n,successes=k,errors=errors,success_rate_percent=rate,
            aba1_successes=ak,aba1_success_rate_percent=ar,delta_vs_aba1_pp=rate-ar if rate is not None else None,
            baseline_successes=bk,baseline_success_rate_percent=br,delta_vs_baseline_pp=rate-br if rate is not None else None,
            complete=n==500 and errors==0))
        for task_id in range(10):
            task=[r for r in rows if r['task_id']==task_id]; bt=[r for r in base if r['task_id']==task_id]; at=[r for r in aba1 if r['task_id']==task_id]
            tn,tk,tr=metrics(task); _,btk,btr=metrics(bt); _,atk,atr=metrics(at)
            task_rows.append(dict(suite=suite,task_id=task_id,task_description=next((r['task_description'] for r in task+bt),''),
                episodes=tn,successes=tk,errors=sum(bool(r.get('error')) for r in task),success_rate_percent=tr,
                aba1_successes=atk,aba1_success_rate_percent=atr,delta_vs_aba1_pp=tr-atr if tr is not None else None,
                baseline_successes=btk,baseline_success_rate_percent=btr,delta_vs_baseline_pp=tr-btr if tr is not None else None))
    total=sum(r['episodes'] for r in suite_rows); successes=sum(r['successes'] for r in suite_rows)
    rate=100*successes/total if total else None
    aba1_rate=sum(r['aba1_successes'] for r in suite_rows)/20
    baseline_rate=sum(r['baseline_successes'] for r in suite_rows)/20
    mp4_files=list(args.run_dir.rglob('*.mp4'))
    if video_records or mp4_files: raise RuntimeError('Videos must not be generated in aba2')
    report=dict(experiment='inference-only: no LaWM future tokens and no 8 act + 8 flow query outputs in action memory',
        retrained=False,complete=all(r['complete'] for r in suite_rows),expected_episodes=2000,total_episodes=total,
        total_successes=successes,errors=sum(r['errors'] for r in suite_rows),success_rate_percent=rate,
        aba1_success_rate_percent=aba1_rate,baseline_success_rate_percent=baseline_rate,
        delta_vs_aba1_pp=rate-aba1_rate if rate is not None else None,
        delta_vs_baseline_pp=rate-baseline_rate if rate is not None else None,
        videos_saved=0,aba1_run=str(ABA1),baseline_run=str(BASELINE),suites=suite_rows)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    for name,data in [('suite_results.csv',suite_rows),('task_results.csv',task_rows)]:
        path=args.run_dir/name; tmp=path.with_name(path.name+'.tmp')
        with tmp.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        tmp.replace(path)
    atomic_text(args.run_dir/'ablation_summary.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fmt=lambda v: f'{v:.2f}' if v is not None else '—'
    lines=['# LaWAM aba2: inference without LaWM and query outputs','',
        'Status: '+('COMPLETE' if report['complete'] else 'IN PROGRESS / INCOMPLETE'),'',
        'Original LIBERO SFT weights; no retraining. VLM input/forward unchanged. Remove exactly 8 act and 8 flow query output positions and the matching attention-mask positions before the action expert. Keep current DINO features; remove LaWM future features. No MP4 videos.','',
        '| Suite | Completed | Successes | aba2 SR (%) | aba1 SR (%) | Original SR (%) | vs aba1 (pp) | vs original (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in suite_rows:
        lines.append(f"| {r['suite']} | {r['episodes']}/500 | {r['successes']} | {fmt(r['success_rate_percent'])} | {fmt(r['aba1_success_rate_percent'])} | {fmt(r['baseline_success_rate_percent'])} | {fmt(r['delta_vs_aba1_pp'])} | {fmt(r['delta_vs_baseline_pp'])} |")
    lines.extend([f"| **Total** | **{total}/2000** | **{successes}** | **{fmt(rate)}** | **{fmt(aba1_rate)}** | **{fmt(baseline_rate)}** | **{fmt(report['delta_vs_aba1_pp'])}** | **{fmt(report['delta_vs_baseline_pp'])}** |",'',
        f"Errors: {report['errors']}; MP4 files: 0.",'',
        '40 tasks × 50 episodes; simulator seed 0; original initial states and horizon limits; 20 Hz; 8 executed actions/chunk; 10 flow steps; bf16. GPUs 0,1,2,3,6,7. GPU 4/5 excluded. Comparisons use separate runs, without paired policy-noise draws. Incomplete rates are provisional.'])
    atomic_text(args.run_dir/'REPORT.md','\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__': main()
