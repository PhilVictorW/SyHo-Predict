import json, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
run=root/'results/ablation/inference_without_lawm_and_queries/full_40x50_20260915'
while True:
    subprocess.run([sys.executable, str(root/'runtime/summarize_ablation.py'), str(run)], stdout=subprocess.DEVNULL, check=True)
    report=json.loads((run/'ablation_summary.json').read_text())
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'total',report['total_episodes'],'/2000', 'successes',report['total_successes'], flush=True)
    for s in report['suites']:
        print(' ',s['suite'],s['successes'],'/',s['episodes'],'errors',s['errors'],'complete',s['complete'],flush=True)
    if report['complete'] or (run/'exit_code.txt').exists():
        break
    time.sleep(30)
