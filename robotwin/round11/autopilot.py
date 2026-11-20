"""Durable CPU supervisor for confirmation -> one bounded, evidence-selected experiment."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
R = ROOT/'robotwin/round11'
OUT = R/'autopilot'
CONFIRM = R/'confirmation_500'
PY = '/home/wangchaoyang/envs/LaWAM/bin/python'


def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def identity(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ', 1)[1].split()
        cmd = Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        if stat[0] == 'Z': return None
        return dict(pid=int(pid), start_ticks=stat[19], command=cmd)
    except (OSError, ValueError, TypeError): return None


def live(record):
    actual = identity(record.get('pid'))
    return bool(actual and actual['start_ticks'] == record.get('start_ticks')
                and actual['command'] == record.get('command'))


def decide(report):
    """Pure decision function: never infer success from incomplete or failed verification."""
    assert report.get('completed') is True
    assert report.get('frozen_audit', {}).get('passed') is True
    assert report.get('preflight_max_abs', float('inf')) < 1e-6
    assert report.get('future_access_guard_anchors') == 64
    rows = report['rows']
    assert len(rows) == 128
    assert len({(r['cohort'], r['episode'], r['t'], r['noise_seed']) for r in rows}) == 128
    groups = ['fresh', 'tail', 'tail_remaining_1', 'tail_remaining_8', 'tail_remaining_24', 'tail_remaining_36']
    result = report['summary']
    assert result['fresh']['episodes'] == 32 and result['tail']['episodes'] == 8
    assert result['fresh']['pairs'] == result['tail']['pairs'] == 64
    for group in groups:
        for contrast in result[group]['comparisons'].values():
            values = list(contrast['ratios'].values()) + contrast['bootstrap_95']
            assert all(math.isfinite(x) for x in values), 'Invalid confirmation statistic'
    def safe(comparison):
        return all(result[g]['comparisons'][comparison]['nonregression_passed'] for g in groups)
    fresh = result['fresh']['comparisons']
    if fresh['query_only_vs_initial']['improvement_supported'] and safe('query_only_vs_initial'):
        return dict(next='online_pilot', candidate='query_only',
            reason='Predeclared primary improves on fresh32 and passes all tail nonregression checks')
    if (fresh['query_gate_vs_initial']['improvement_supported'] and
        fresh['query_gate_vs_query_only']['improvement_supported'] and
        fresh['query_gate_vs_gate_off']['improvement_supported'] and
        all(safe(c) for c in ['query_gate_vs_initial', 'query_gate_vs_query_only', 'query_gate_vs_gate_off'])):
        return dict(next='online_pilot', candidate='query_gate',
            reason='Secondary gate contrasts support exploratory closed-loop screening; not multiplicity-adjusted proof')
    return dict(next='action_adapter', candidate=None,
        reason='No candidate meets fresh improvement plus tail safety criteria; switch trainable component, do not extend query/gate training')


def free_memory():
    try:
        raw = subprocess.check_output(['nvidia-smi', '--id=2', '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10)
        return int(raw.strip())
    except (OSError, ValueError, subprocess.SubprocessError): return 0


def find_module(module):
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        record = identity(path.name)
        if record and f'-m {module} ' in record['command']:
            found.append(record)
    assert len(found) <= 1, f'Multiple active processes for {module}'
    return found[0] if found else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'plan.json', dict(poll_seconds=15, physical_gpu=2, minimum_free_mib={'training':24576, 'online':32768},
        input=str(CONFIRM), after_benefit='8 task/mode x5 episodes; replay recorded compatible reference',
        otherwise='current-only goal/local action adapters, 200 steps, LR1e-5, batch4, save/validate50',
        on_failure='retain logs/checkpoints and keep monitoring; no blind retry of failed assertions',
        followup_limit='one bounded next experiment; no old StageII3000 or long-training restart',
        process_scope='observe confirmation; launch followup only after it exits; never terminate unrelated processes'))
    proc = None
    state = dict(pid=os.getpid(), started_at=time.time())
    def update(phase, **kw):
        state.update(phase=phase, updated_at=time.time(), **kw)
        write(OUT/'status.json', state)
    while True:
        try:
            if proc is not None: proc.poll()  # Reap our exited child, if any.
            status = read(CONFIRM/'status.json')
            update('monitoring_confirmation', confirmation=status)
            if status.get('phase') == 'failed':
                update('confirmation_failed', error=status.get('error'), action='await corrected confirmation; no model promotion')
            elif status.get('phase') == 'completed' and (CONFIRM/'results.json').exists():
                result_path = CONFIRM/'results.json'
                digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
                decision = read(OUT/'decision.json')
                if decision:
                    assert decision['confirmation_sha256'] == digest, 'Confirmation changed after decision'
                else:
                    decision = dict(decide(read(result_path)), confirmation_sha256=digest, decided_at=time.time())
                    assert read(CONFIRM/'preflight.json').get('passed') is True
                    write(OUT/'decision.json', decision)
                    print(json.dumps(decision), flush=True)
                train = decision['next'] == 'action_adapter'
                target = R/('action_adapter_200' if train else 'online_pilot_500')
                module = 'robotwin.round11.' + ('action_adapter' if train else 'online_pilot')
                child = read(OUT/'followup.json')
                followup = read(target/'status.json')
                update('monitoring_followup', decision=decision, followup=followup)
                if followup.get('phase') in ('completed', 'stopped_by_regression_guard'):
                    write(OUT/'completion.json', dict(decision=decision, result=followup, completed_at=time.time()))
                    update('followup_completed', result=followup,
                           action='bounded next experiment finished; results retained for review')
                elif followup.get('phase') == 'failed':
                    update('followup_failed', error=followup.get('error'), action='preserve evidence; no blind relaunch')
                elif child and live(child):
                    update('monitoring_followup', child=child, followup=followup,
                           stale_warning=bool(followup and time.time()-followup.get('updated_at', 0)>1800))
                else:
                    # Adopt a worker if a supervisor restart happened between Popen and recording its PID.
                    found = find_module(module)
                    if found:
                        write(OUT/'followup.json', found)
                        update('adopted_followup', child=found)
                    elif identity(status.get('pid')):
                        update('waiting_confirmation_exit')
                    elif child or (target/'manifest.json').exists():
                        update('followup_exited_unexpectedly', action='retain complete checkpoint; no duplicate run', followup=followup)
                    else:
                        free = free_memory()
                        required = 24576 if train else 32768
                        if free < required:
                            update('waiting_gpu2_memory', free_mib=free, required_mib=required)
                        else:
                            target.mkdir(parents=True, exist_ok=True)
                            env = os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES='2', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                                CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                                NO_ALBUMENTATIONS_UPDATE='1', PYTHONPATH=str(ROOT))
                            cmd = ['nice', '-n', '10', 'ionice', '-c', '2', '-n', '7', PY, '-u', '-m', module]
                            if not train: cmd += ['--candidate', decision['candidate']]
                            with (target/'run.log').open('a') as log:
                                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
                            # The nice/ionice wrapper execs python; capture its final stable identity.
                            time.sleep(1)
                            record = identity(proc.pid)
                            assert record and f'-m {module}' in record['command'], 'Worker failed before startup'
                            write(OUT/'followup.json', record)
                            update('followup_started', child=record, command=cmd)
                            print(json.dumps(dict(event='followup_started', decision=decision, child=record)), flush=True)
            elif status and not identity(status.get('pid')):
                update('confirmation_exited_unexpectedly', action='preserve partial evidence; no result inferred')
        except Exception:
            error = traceback.format_exc()
            update('supervisor_error', error=error)
            print(error, flush=True)
        time.sleep(15)


if __name__ == '__main__': main()
