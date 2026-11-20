import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

ROOT = Path('/mnt/hdd/wangchaoyang/LatentWAM/LaWAM_aba2_change')
TEACHER = ROOT / 'libero-plus/runs/stage_I_20260923'
STUDENT = ROOT / 'libero-plus/runs/stage_II_20260925'
S3 = ROOT / 'stage_III/libero_plus'
AUDIT = ROOT / 'libero-plus/audits/checkpoint_cleanup_20261024'
DELETE = [TEACHER / 'step_004000'] + [
    TEACHER / 'retained_checkpoints' / f'step_{s:06d}'
    for s in (1000, 2000, 3000, 4000)
] + [STUDENT / f'step_{s:06d}' for s in range(0, 4000, 500)]
KEEP = [TEACHER / 'step_005000', TEACHER / 'retained_checkpoints/step_005000',
        TEACHER / 'best', STUDENT / 'step_004000']
EVAL = ROOT / 'libero-plus/evaluation/stage_II_steps2500_4000_parallel_20261008'
RESULT = EVAL / 'step_004000/step_004000_step500_stratified_gripper_v1/results.json'
MANIFEST = EVAL / 'manifest_same_as_steps1000_1500_2000.json'
ROUND2 = S3 / 'runs/grpo_round2_full8380_20261018_211335'


def emit(*args):
    print(*args, flush=True)


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def fingerprint(path, digest=False):
    s = path.lstat()
    result = dict(size=s.st_size, allocated=s.st_blocks * 512, dev=s.st_dev,
                  inode=s.st_ino, nlink=s.st_nlink, mode=s.st_mode,
                  mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    if stat.S_ISLNK(s.st_mode):
        result['symlink_target'] = os.readlink(path)
        result['resolved'] = str(path.resolve())
        assert all(not path.resolve().is_relative_to(d) for d in DELETE)
        return result
    assert stat.S_ISREG(s.st_mode), f'Not a regular file: {path}'
    if digest and s.st_size <= 32 * 1024 * 1024:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        result['sha256'] = h.hexdigest()
    return result


def tree(path):
    assert path.is_dir() and not path.is_symlink() and path.resolve() == path
    entries = sorted(path.rglob('*'))
    for p in entries:
        assert not p.is_symlink(), f'Symlink found: {p}'
        assert p.resolve().is_relative_to(path), f'Path escaped: {p}'
        assert p.is_file() or p.is_dir(), f'Special file: {p}'
    return entries


def active_training():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if any(b'\n' in a for a in args):
                continue
            cmd = b' '.join(args).decode(errors='replace')
            if ('LaWAM_aba2_change/libero-plus/runtime/train' in cmd or
                    'stage_III.libero_plus.full_train' in cmd):
                found.append(dict(pid=int(p.name), command=cmd))
        except (OSError, PermissionError):
            pass
    assert not found, f'Training process found: {found}'


def facts():
    r = json.loads(RESULT.read_text())
    m = json.loads(MANIFEST.read_text())
    s = json.loads((ROUND2 / 'status.json').read_text())
    assert r['episodes'] == r['expected'] == 8380 and r['successes'] == 7510 and r['complete']
    assert m['total_tasks'] == len(m['tasks']) == 8380
    assert s['phase'] == 'paused'
    source = json.loads((STUDENT / 'teacher_source.json').read_text())
    return dict(baseline_successes=7510, baseline_episodes=8380,
                baseline_success_rate=r['success_rate'], task_count=len(m['tasks']),
                stage_III_phase=s['phase'], stage_III_batch=s['batch'], teacher_source=source)


def prepare():
    active_training()
    assert not AUDIT.exists(), f'Audit already exists: {AUDIT}'
    emit('【清理前校验】检查明确列出的 13 个旧 checkpoint 目录。')
    deleted_files = {}
    metadata = {}
    for d in DELETE:
        assert d not in KEEP and all(not k.is_relative_to(d) for k in KEEP)
        for p in tree(d):
            if p.is_file():
                assert p.suffix in ('.pt', '.json', '.jinja'), f'Unexpected file: {p}'
                deleted_files[str(p)] = fingerprint(p)
                if p.parent == d and p.suffix == '.json':
                    metadata[str(p)] = json.loads(p.read_text())
    protected_paths = set()
    for d in KEEP:
        assert (d / 'model.pt').is_file() and (d / 'processor').is_dir()
        protected_paths.update(p for p in tree(d) if p.is_file())
    for d in (TEACHER, STUDENT):
        protected_paths.update(p for p in d.iterdir() if p.is_file())
    for d in (ROOT / 'goal_local', ROOT / 'libero-plus/runtime', ROOT / 'libero-plus/configs'):
        protected_paths.update(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected_paths.update((
        RESULT, RESULT.with_name('protocol.json'), MANIFEST,
        ROOT / 'libero-plus/evaluation/stage_II_20260925/step_004000_gripper_v1/results.json',
        ROOT / 'libero-plus/evaluation/stage_II_20260925/step_000500_gripper_v1/results.json',
        S3 / 'baseline_snapshot.json', S3 / 'manifest/full_8380.json',
        S3 / 'runs/grpo_full8380_20261013_010306/checkpoints/actor_update_0262.pt',
        ROUND2 / 'checkpoints/actor_current.pt', ROUND2 / 'status.json',
        S3 / 'launchers/round2_full8380_20261018_211335/latest_pause.json',
    ))
    protected = {}
    emit('【清理前校验】记录保留权重的 inode、大小和时间戳；校验配置、任务清单及结果文件 SHA-256。')
    for p in sorted(protected_paths):
        protected[str(p)] = fingerprint(p, digest=True)
    protected_inodes = {(s['dev'], s['inode']) for s in protected.values()}
    inodes = {}
    for name, s in deleted_files.items():
        k = (s['dev'], s['inode'])
        assert k not in protected_inodes, f'Delete shares protected inode: {name}'
        v = inodes.setdefault(k, dict(allocated=s['allocated'], nlink=s['nlink'], paths=[]))
        v['paths'].append(name)
    assert all(v['nlink'] == len(v['paths']) for v in inodes.values()), 'Unexpected external hardlinks'
    baseline = json.loads((S3 / 'baseline_snapshot.json').read_text())
    assert protected[str(RESULT)]['sha256'] == baseline['baseline_result_sha256']
    for filename, key in [('checkpoint_complete.json', 'base_student_marker_sha256'),
                          ('action_semantics.json', 'base_student_action_semantics_sha256')]:
        assert protected[str(STUDENT / 'step_004000' / filename)]['sha256'] == baseline[key]
    dependencies = {}
    for name in (baseline['sft_dataset'], baseline['inference_benchmark'],
                 '/home/wangchaoyang/envs/LaWAM', '/home/wangchaoyang/envs/libero-plus'):
        p = Path(name)
        assert p.exists(), f'Missing dependency: {p}'
        s = p.stat()
        dependencies[name] = dict(resolved=str(p.resolve()), dev=s.st_dev, inode=s.st_ino)
    plan = dict(created_at=time.time(), root=str(ROOT), delete_directories=list(map(str, DELETE)),
                keep_directories=list(map(str, KEEP)), teacher_best_step=5000,
                deleted_files=deleted_files, old_checkpoint_metadata=metadata,
                protected_files=protected, dependencies=dependencies,
                expected_reclaimed_allocated_bytes=sum(v['allocated'] for v in inodes.values()),
                baseline_facts=facts(), recorded_student_model_sha256=baseline['base_student_model_sha256'],
                validation_method='Large weights: unchanged inode/size/mtime/ctime/link count. Files <=32 MiB: SHA-256 also checked. No new full model hash or inference rerun.')
    AUDIT.mkdir(parents=True)
    shutil.copyfile(__file__, AUDIT / 'cleanup.py')
    write_json(AUDIT / 'before.json', plan)
    emit('【清理清单已保存】', AUDIT / 'before.json')
    emit(json.dumps(dict(directories=len(DELETE), files=len(deleted_files), protected_files=len(protected),
                         reclaim_GiB=plan['expected_reclaimed_allocated_bytes'] / 2**30), ensure_ascii=False))


def execute():
    plan = json.loads((AUDIT / 'before.json').read_text())
    assert plan['delete_directories'] == list(map(str, DELETE))
    assert not (AUDIT / 'after.json').exists(), 'Already completed'
    active_training()
    assert facts() == plan['baseline_facts']
    for name, saved in plan['protected_files'].items():
        assert fingerprint(Path(name), digest='sha256' in saved) == saved, f'Protected changed: {name}'
    for d in DELETE:
        current = {str(p): fingerprint(p) for p in tree(d) if p.is_file()}
        saved = {p: s for p, s in plan['deleted_files'].items() if Path(p).is_relative_to(d)}
        assert current == saved, f'Deletion inventory changed: {d}'
    completed = []
    write_json(AUDIT / 'progress.json', dict(phase='deleting', completed=completed))
    for d in DELETE:
        emit(f'【删除旧 checkpoint】{d.relative_to(ROOT)}')
        shutil.rmtree(d)
        assert not d.exists()
        completed.append(str(d))
        write_json(AUDIT / 'progress.json', dict(phase='deleting', completed=completed))
    emit('【删除完成】复核保留权重、配置、8380 任务清单、原结果和 Stage III 暂停状态。')
    for name, saved in plan['protected_files'].items():
        assert fingerprint(Path(name), digest='sha256' in saved) == saved, f'Protected changed: {name}'
    for name, saved in plan['dependencies'].items():
        p = Path(name)
        s = p.stat()
        assert dict(resolved=str(p.resolve()), dev=s.st_dev, inode=s.st_ino) == saved
    assert facts() == plan['baseline_facts']
    assert all(not p.exists() for p in DELETE)
    result = dict(completed_at=time.time(), phase='complete', removed_directories=completed,
                  kept_directories=plan['keep_directories'], teacher_best_step=5000,
                  protected_files_verified=len(plan['protected_files']),
                  reclaimed_allocated_bytes=plan['expected_reclaimed_allocated_bytes'],
                  reclaimed_GiB=plan['expected_reclaimed_allocated_bytes'] / 2**30,
                  baseline_facts=facts(), validation_method=plan['validation_method'],
                  filesystem_available_bytes=shutil.disk_usage(ROOT).free)
    write_json(AUDIT / 'after.json', result)
    write_json(AUDIT / 'progress.json', dict(phase='complete', completed=completed))
    emit('【清理及复核完成】', json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    {'prepare': prepare, 'execute': execute}[sys.argv[1]]()
