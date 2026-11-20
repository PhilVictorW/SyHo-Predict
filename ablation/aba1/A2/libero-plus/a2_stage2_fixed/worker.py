"""Independently admitted resident model; joins Gloo only after world-size selection."""
import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace
from .runtime import WORK, CapacityError, atomic_json, load_config, read_json, selected_spec, worker_physical_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    control = args.control.resolve()
    if not control.is_relative_to(WORK):
        raise ValueError('Control directory must remain inside A2')
    physical_gpu = worker_physical_gpu()
    cfg = load_config(WORK / 'config.json', 'A2')
    status_path = control / f'worker{args.rank}.json'
    def report(phase, **values):
        atomic_json(status_path, dict(phase=phase, rank=args.rank, pid=os.getpid(), physical_gpu=physical_gpu, time=time.time(), **values))
    try:
        from .probe import prepare
        prepared, profile = prepare(cfg, args.rank, report, args.resume)
        report('ready', profile=profile)
        last = time.monotonic()
        while True:
            if (control / 'STOP').exists() or not Path(f'/proc/{args.parent}').exists():
                raise RuntimeError('Admission controller stopped')
            selection = read_json(control / 'selection.json')
            if selection:
                break
            if time.monotonic() - last > 10:
                report('ready', profile=profile); last = time.monotonic()
            time.sleep(2)
        world = selection['world_size']
        if selection['pids'][args.rank] != os.getpid():
            raise ValueError('Admission PID does not match worker')
        if selection['rank_to_gpu'][args.rank] != physical_gpu:
            raise ValueError('Selected physical GPU does not match worker visibility')
        os.environ.update(RANK=str(args.rank), WORLD_SIZE=str(world), LOCAL_RANK='0')
        runtime_args = SimpleNamespace(config=str(WORK / 'config.json'), stage='II',
            gpus=str(physical_gpu), workers=1, cpu_threads=2, keep_models=1, execute=True,
            offload_saved_tensors=True, share_frozen_encoders=True, resume=args.resume,
            rendezvous=selection['rendezvous'], stop_after_updates=None)
        report('training', world_size=world, profile=profile)
        from .training import run
        run(runtime_args, cfg, selected_spec(cfg, world, selection['rank_to_gpu']), prepared)
        report('completed', world_size=world, step=2500)
    except BaseException as exc:
        import torch
        capacity = isinstance(exc, (CapacityError, torch.cuda.OutOfMemoryError))
        report('capacity_rejected' if capacity else 'failed', error=repr(exc))
        if capacity:
            raise SystemExit(75)
        raise


if __name__ == '__main__':
    main()
