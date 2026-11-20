"""Fixed ten-rank Stage II: two each on GPUs2,5,6 and four on GPU7, loaded sequentially."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .runtime import (WORK, POLICY, CHECKPOINT_POLICY, CapacityError, admission, atomic_json, code_identity,
                      load_config, read_json, resources, resources_many, run_root, selected_spec, stage_root, validate_mapping)
from a2_local.preflight import check_teacher, check_complete

STATE = WORK / 'logs/stage_II_adaptive_status.json'


def record(phase, **values):
    row = dict(phase=phase, pid=os.getpid(), time=time.time(), physical_gpus=POLICY['physical_gpus'],
               stage='II', policy=POLICY, **values)
    atomic_json(STATE, row)
    print(json.dumps(row), flush=True)


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def resources_all():
    return resources_many(POLICY['physical_gpus'])


def prefix_safe(profiles, snapshots):
    if not profiles:
        return False
    used = {p['physical_gpu'] for p in profiles}
    return (all(snapshots[gpu]['gpu_free_mib'] >= POLICY['gpu_headroom_mib'] for gpu in used)
            and min(row['host_available_mib'] for row in snapshots.values()) >=
            max(p['optimizer_headroom_mib'] for p in profiles))


def choose_gpu(snapshots, profiles, blocked=(), required_gpu=None):
    decisions = {gpu: admission(row, profiles) for gpu, row in snapshots.items()}
    eligible = [gpu for gpu, decision in decisions.items() if decision['allowed']
                and gpu not in blocked and (required_gpu is None or gpu == required_gpu)]
    # Give each allowed GPU a resident rank before filling one device.
    selected = min(eligible, key=lambda gpu: (
        sum(p['physical_gpu'] == gpu for p in profiles), -snapshots[gpu]['gpu_free_mib'],
        POLICY['physical_gpus'].index(gpu))) if eligible else None
    return selected, decisions


def observe(children, profiles):
    minimum = resources_all()
    deadline = time.monotonic() + POLICY['observe_seconds']
    while time.monotonic() < deadline:
        time.sleep(min(POLICY['poll_seconds'], max(0, deadline-time.monotonic())))
        if any(p.poll() is not None for p in children):
            raise RuntimeError('A resident worker exited before training')
        current = resources_all()
        for gpu in minimum:
            for key in ('gpu_free_mib', 'host_available_mib'):
                minimum[gpu][key] = min(minimum[gpu][key], current[gpu][key])
    return minimum


def start_candidate(cfg, control, rank, children, resume, physical_gpu):
    env = os.environ.copy()
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(physical_gpu), PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TORCH_GLOO_LAZY_INIT='1')
    command = [cfg['model_python'], '-B', '-u', '-m', 'a2_stage2_fixed.worker',
               '--control', str(control), '--rank', str(rank), '--parent', str(os.getpid())]
    if resume:
        command += ['--resume', str(resume)]
    with (control / f'worker{rank}.log').open('a') as stream, open(os.devnull) as stdin:
        child = subprocess.Popen(command, cwd=WORK, env=env, stdin=stdin,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    children.append(child)
    record('probing_rank', candidate_rank=rank, candidate_pid=child.pid,
           resident_pids=[p.pid for p in children[:-1]], candidate_gpu=physical_gpu, command=command)
    deadline = time.monotonic() + POLICY['worker_timeout_seconds']
    last_record = 0
    while True:
        row = read_json(control / f'worker{rank}.json')
        code = child.poll()
        if row.get('pid') == child.pid and row.get('phase') == 'capacity_rejected':
            terminate(child); children.pop()
            raise CapacityError(row.get('error', 'Candidate exceeded capacity'))
        if code is not None:
            children.pop()
            raise RuntimeError(f'Worker {rank} exited {code}; inspect {control}/worker{rank}.log')
        if any(p.poll() is not None for p in children[:-1]):
            raise RuntimeError('Previously admitted worker exited')
        if row.get('pid') == child.pid and row.get('phase') == 'ready':
            return row['profile']
        if time.monotonic() > deadline:
            raise TimeoutError(f'Worker {rank} admission timed out')
        if time.monotonic() - last_record >= 20:
            record('probing_rank', candidate_rank=rank, candidate_pid=child.pid,
                   candidate_phase=row.get('phase'), candidate_gpu=physical_gpu, resources=resources_all())
            last_record = time.monotonic()
        time.sleep(POLICY['poll_seconds'])


def admit(cfg, control, children, resume=None, required_world=None, required_mapping=None):
    profiles = []
    blocked = set()
    rejections = {gpu: 0 for gpu in POLICY["physical_gpus"]}
    limit = required_world or POLICY['max_ranks']
    while len(profiles) < limit:
        required_gpu = required_mapping[len(profiles)] if required_mapping is not None else None
        gpu, decisions = choose_gpu(resources_all(), profiles, blocked, required_gpu)
        if gpu is None:
            record('capacity_limit' if profiles else 'waiting_memory',
                   admitted_ranks=len(profiles), decisions=decisions, blocked_gpus=sorted(blocked))
            if profiles:
                break
            time.sleep(20)
            blocked.clear()
            continue
        try:
            profile = start_candidate(cfg, control, len(profiles), children, resume, gpu)
        except CapacityError as exc:
            record('candidate_rejected', admitted_ranks=len(profiles), candidate_gpu=gpu, reason=str(exc))
            blocked.add(gpu)
            rejections[gpu] += 1
            continue
        if profile['physical_gpu'] != gpu:
            raise ValueError('Probe physical GPU does not match admission')
        profiles.append(profile)
        observed = observe(children, profiles)
        record('rank_admitted', admitted_ranks=len(profiles), latest_profile=profile,
               observed_minimum=observed)
        if not prefix_safe(profiles, observed):
            terminate(children.pop()); profiles.pop()
            blocked.add(gpu)
            rejections[gpu] += 1
        else:
            # A later candidate supplies a fresh complete observation window on both cards.
            # Temporary pressure must not exclude a recovered card for the whole launch.
            for other in list(blocked):
                if rejections[other] < 2 and admission(observed[other], profiles)['allowed']:
                    blocked.remove(other)
                    record('capacity_recovered', physical_gpu=other, prior_rejections=rejections[other],
                           observed_minimum=observed[other], admitted_ranks=len(profiles))
    if required_world and len(profiles) != required_world:
        raise CapacityError('Resume requires the saved world size and GPU mapping; no updates were made')
    return profiles


def admit_hybrid(cfg, control, children, resume=None, required_mapping=None):
    """Admit the ten explicitly assigned ranks one at a time; no expansion."""
    base = list(required_mapping) if required_mapping is not None else POLICY['base_rank_to_gpu']
    if required_mapping is not None:
        validate_mapping(base)
    limit = len(base) if required_mapping is not None else POLICY['max_ranks']
    profiles = []
    while len(profiles) < limit:
        if any(child.poll() is not None for child in children):
            raise RuntimeError('Resident worker exited during admission')
        rank = len(profiles)
        mandatory = rank < len(base)
        gpu = base[rank] if mandatory else POLICY['additional_gpu']
        snapshots = resources_all()
        decision = admission(snapshots[gpu], profiles)
        if profiles and not prefix_safe(profiles, snapshots):
            record('waiting_memory', candidate_rank=rank, candidate_gpu=gpu,
                   admitted_ranks=rank, reason='Resident GPU or optimizer memory headroom changed',
                   resources=snapshots, rank_policy=POLICY['rank_policy'])
            time.sleep(20)
            continue
        if not decision['allowed']:
            record('waiting_memory' if mandatory else 'capacity_limit', candidate_rank=rank,
                   candidate_gpu=gpu, admitted_ranks=rank, mandatory=mandatory, decision=decision,
                   rank_policy=POLICY['rank_policy'])
            if not mandatory:
                break
            time.sleep(20)
            continue
        try:
            profile = start_candidate(cfg, control, rank, children, resume, gpu)
        except CapacityError as exc:
            record('candidate_rejected', candidate_rank=rank, candidate_gpu=gpu,
                   admitted_ranks=rank, mandatory=mandatory, reason=str(exc))
            if not mandatory:
                break
            time.sleep(20)
            continue
        if profile['physical_gpu'] != gpu:
            raise ValueError('Worker physical GPU differs from required assignment')
        profiles.append(profile)
        observed = observe(children, profiles)
        candidate_safe = (observed[gpu]['gpu_free_mib'] >= POLICY['gpu_headroom_mib'] and
            min(row['host_available_mib'] for row in observed.values()) >=
            max(p['optimizer_headroom_mib'] for p in profiles))
        if not candidate_safe:
            # Keep a successfully loaded worker alive; later admission waits for headroom.
            record('waiting_memory', candidate_rank=rank, candidate_gpu=gpu,
                   admitted_ranks=len(profiles), mandatory=mandatory, observed_minimum=observed,
                   reason='Loaded worker retained while waiting for memory headroom')
        record('rank_admitted', admitted_ranks=len(profiles), latest_profile=profile,
               observed_minimum=observed, rank_policy=POLICY['rank_policy'])
    validate_mapping([p['physical_gpu'] for p in profiles])
    # All mandatory cards must retain headroom. Do not silently discard a base rank.
    while True:
        observed = observe(children, profiles)
        if prefix_safe(profiles, observed):
            break
        record('waiting_memory', admitted_ranks=len(profiles), reason='Final shared-memory observation',
               resources=observed, rank_policy=POLICY['rank_policy'])
        time.sleep(20)
    if required_mapping is not None and [p['physical_gpu'] for p in profiles] != list(required_mapping):
        raise CapacityError('Resume mapping differs; no optimizer updates were made')
    return profiles


def teacher_ready(cfg):
    root = run_root(cfg)
    if (WORK / 'logs/exit_code.txt').read_text().strip() != '0':
        raise ValueError('Stage I must exit successfully before admission')
    teacher = root / 'I/step_003000'
    check_teacher(teacher, cfg)
    from .resume import checked_complete
    checked_complete(teacher, verify_hash=True)
    for path in (root / 'I').glob('status_rank*.json'):
        row = read_json(path)
        proc = Path('/proc') / str(row.get('pid', 0))
        try:
            live = proc.joinpath('stat').read_text().split(') ', 1)[1][0] not in ('Z', 'X')
            command = proc.joinpath('cmdline').read_bytes()
        except OSError:
            continue
        if live and b'a2_local.training' in command and str(WORK / 'config.json').encode() in command:
            raise RuntimeError('Stage I still has live workers; admission is forbidden')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = load_config(WORK / 'config.json', 'A2')
    if args.execute:
        from .resume import require_receipt
        require_receipt()
    if not args.execute:
        print(json.dumps(dict(stage='II', policy=POLICY, checkpoint_policy=CHECKPOINT_POLICY, steps=2500, world_size=10,
              effective_batch=32, base_rank_to_gpu=POLICY['base_rank_to_gpu'], additional_gpu=POLICY['additional_gpu'],
              model_load_order='one_at_a_time', optimizer_updates_during_probe=0,
              teacher=str(run_root(cfg) / 'I/step_003000'), training_started=False), indent=2))
        return
    root = run_root(cfg); output = stage_root(cfg, 'II')
    output.mkdir(parents=True, exist_ok=True)
    children = []
    control = None
    started = False
    def interrupted(signum, frame):
        raise InterruptedError(f'Admission/training supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with ExitStack() as stack:
            for path in (root / '.pipeline.lock', output / '.training.lock'):
                lock = stack.enter_context(path.open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            teacher_ready(cfg)
            required_world = None
            required_mapping = None
            if args.resume:
                args.resume = args.resume.resolve()
                if args.resume.parent != output or not (output / 'contract.json').exists():
                    raise ValueError('Resume must use this Stage II run and its saved contract')
                from .resume import checked_complete
                meta, _ = checked_complete(args.resume, require_resume=True, verify_hash=True)
                if meta['step'] >= 2500 or meta['config'] != cfg:
                    raise ValueError('Invalid Stage II resume checkpoint')
                contract = read_json(output / 'contract.json')
                if contract.get('model_loading') != 'sequential_measured_admission':
                    raise ValueError('Checkpoint does not use the adaptive runtime')
                required_mapping = POLICY['base_rank_to_gpu']
                required_world = len(required_mapping)
                selected_spec(cfg, required_world, required_mapping)
            elif (output / 'arguments.json').exists():
                raise RuntimeError('Existing Stage II requires explicit --resume')
            control = output / 'startup' / f'admission_{time.time_ns()}'
            control.mkdir(parents=True)
            started = True
            atomic_json(root / 'pipeline_status.json', dict(phase='starting', stage='II', time=time.time(),
                        admission=str(control), pid=os.getpid()))
            profiles = admit_hybrid(cfg, control, children, args.resume, required_mapping)
            world = len(profiles)
            mapping = [p['physical_gpu'] for p in profiles]
            selection = dict(world_size=world, physical_gpus=list(dict.fromkeys(mapping)),
                             allowed_physical_gpus=POLICY['physical_gpus'], rank_to_gpu=mapping,
                             pids=[p.pid for p in children],
                             accumulation_per_rank=selected_spec(cfg, world, mapping)['accumulation_per_rank'],
                             effective_batch=32, profiles=profiles, policy=POLICY,
                             rank_policy=POLICY['rank_policy'],
                             checkpoint_policy=CHECKPOINT_POLICY, time=time.time(),
                             rendezvous='file://' + str(control / 'gloo_store'), resume=str(args.resume) if args.resume else None)
            atomic_json(output / 'selected_runtime.json', selection)
            atomic_json(output / 'launch.json', dict(runtime='a2_stage2_fixed', selection=selection,
                        config=str(WORK / 'config.json'), code_identity=code_identity()))
            atomic_json(root / 'pipeline_status.json', dict(phase='training', stage='II', time=time.time(),
                        selected_world_size=world, pid=os.getpid()))
            atomic_json(control / 'selection.json', selection)
            last = 0
            while any(p.poll() is None for p in children):
                failed = [(p.pid, p.returncode) for p in children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Stage II workers failed: {failed}; full checkpoints retained')
                if time.monotonic() - last > 20:
                    worker_state = read_json(output / 'status_rank0.json')
                    phase = 'waiting_checkpoint_space' if worker_state.get('phase') == 'waiting_checkpoint_space' else 'training'
                    record(phase, world_size=world, worker_pids=[p.pid for p in children],
                           step=worker_state.get('step'), worker_phase=worker_state.get('phase'),
                           checkpoint_policy=CHECKPOINT_POLICY,
                           checkpoint_free_bytes=worker_state.get('checkpoint_free_bytes'),
                           checkpoint_required_bytes=worker_state.get('checkpoint_required_bytes'), resources=resources_all())
                    last = time.monotonic()
                time.sleep(2)
            if any(p.returncode != 0 for p in children):
                raise RuntimeError('Stage II did not exit successfully')
            check_complete(output / 'step_002500', verify_hash=True)
            atomic_json(root / 'pipeline_status.json', dict(phase='completed', completed_stages=['II'],
                        student_step=2500, selected_world_size=world, time=time.time()))
            record('completed', world_size=world, step=2500)
    except BaseException as exc:
        if started:
            atomic_json(root / 'pipeline_status.json', dict(phase='failed', stage='II', time=time.time(), error=repr(exc)))
        record('failed', error=repr(exc))
        raise
    finally:
        if control is not None:
            (control / 'STOP').touch()
        for child in children:
            terminate(child)


if __name__ == '__main__':
    main()
