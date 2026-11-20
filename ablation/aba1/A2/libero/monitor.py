#!/usr/bin/env python3
"""Read-only concise A2 progress."""
import json
from pathlib import Path
WORK=Path(__file__).resolve().parent
plan=json.loads((WORK/'run_plan.json').read_text())
run=WORK/'runs'/plan['run_name']
def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return None
print(json.dumps(dict(run=str(run),pipeline=read(run/'pipeline.json'),
    stage_I=read(run/'stage_I/status.json'),stage_II=read(run/'stage_II/status.json'),
    workers=[read(run/'stage_I'/f'worker_{i}.json') for i in range(3)],
    evaluation=read(WORK/'evaluation'/run.name/'full_40x50_step2000/summary.json')),indent=2))
