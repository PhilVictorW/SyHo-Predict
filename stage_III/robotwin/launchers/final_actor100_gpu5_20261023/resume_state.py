"""Recover durable episode outcomes without replaying completed seeds."""
import json
import os
from pathlib import Path


def durable_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_resume(save_dir, fixed_seeds, seed_base, target):
    save_dir = Path(save_dir)
    records = []
    source = None
    for name in ('summary.json', 'episodes.partial.json'):
        path = save_dir / name
        if path.exists():
            # Fail closed on malformed outcomes; never silently overwrite them.
            saved = json.loads(path.read_text())
            records = saved['episodes']
            source = name
            break
    ids = [int(x['episode_id']) for x in records]
    seeds = [int(x['seed']) for x in records]
    if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate saved episode IDs or seeds')
    if len(records) > target or any(i < 0 for i in ids):
        raise ValueError('Invalid saved episode count or ID')
    if any(type(x.get('success')) is not bool for x in records):
        raise ValueError('Invalid saved success outcome')
    records = sorted(records, key=lambda x: int(x['episode_id']))
    consumed = set(seeds)
    events = save_dir / 'environment_events.jsonl'
    if events.exists():
        lines = events.read_text().splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    # Power loss can leave the last append incomplete. Retry it.
                    continue
                raise
            if event.get('type') in ('candidate_invalid', 'candidate_setup_invalid') and not event.get('error'):
                consumed.add(int(event['seed']))
    index = 0
    if fixed_seeds:
        if len(fixed_seeds) != len(set(fixed_seeds)) or not consumed.issubset(set(fixed_seeds)):
            raise ValueError('Saved seeds do not match the fixed evaluation protocol')
        while index < len(fixed_seeds) and fixed_seeds[index] in consumed:
            index += 1
        if any(s in consumed for s in fixed_seeds[index:]):
            raise ValueError('Nonsequential seed history: single-lane recovery required')
        next_seed = seed_base
    else:
        if any(s < seed_base for s in consumed):
            raise ValueError('Saved seed outside sequential evaluation protocol')
        next_seed = max(consumed, default=seed_base - 1) + 1
    return dict(records=records, completed=len(records),
                successes=sum(x['success'] for x in records),
                next_episode_id=max(ids, default=-1) + 1,
                fixed_seed_index=index, next_seed=next_seed, source=source)
