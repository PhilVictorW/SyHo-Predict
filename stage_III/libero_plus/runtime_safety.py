"""Process-local simulator guards and resumable trajectory validation."""
from __future__ import annotations
import ast
import faulthandler
import hashlib
import inspect
import json
import os
import subprocess
import textwrap
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_reset(wrapper_class, max_attempts: int = 10) -> None:
    """Keep noise handling; only change the binding in this Stage-III process."""
    original = wrapper_class.reset
    if getattr(original, '_stage3_bounded', False):
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    loops = [n for n in ast.walk(function) if isinstance(n, ast.While)]
    if len(loops) != 1:
        raise RuntimeError('Unexpected LIBERO-plus reset source')
    removed = 0
    for node in ast.walk(loops[0]):
        if isinstance(node, ast.Try) and len(node.finalbody) == 1 and isinstance(node.finalbody[0], ast.Continue):
            node.finalbody = []
            removed += 1
    if removed != 1:
        raise RuntimeError('Expected reset finally/continue guard not found')
    function.body.insert(0, ast.parse('_stage3_attempt = 0').body[0])
    loops[0].body[:0] = ast.parse(
        '_stage3_attempt += 1\n'
        f'if _stage3_attempt > {int(max_attempts)}:\n'
        '    raise RuntimeError("LIBERO-plus reset placement retries exhausted")\n'
    ).body
    ast.fix_missing_locations(tree)
    scope = dict(original.__globals__)
    exec(compile(tree, '<stage3_bounded_reset>', 'exec'), scope)
    patched = scope[original.__name__]
    patched._stage3_bounded = True
    wrapper_class.reset = patched


class Progress:
    def __init__(self, output: Path, group: int, trajectory: int):
        self.path = output / 'progress' / f'group_{group:04d}.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.group, self.trajectory = group, trajectory
        self.last_write = 0.0

    def report(self, phase: str, *, step: int | None = None, force: bool = False):
        now = time.time()
        if not force and now - self.last_write < 5:
            return
        payload = dict(group=self.group, trajectory=self.trajectory, phase=phase,
                       step=step, pid=os.getpid(), updated_at=now,
                       timeout_seconds=float(os.environ.get(
                           'STAGE3_IMPORT_TIMEOUT' if phase == '导入仿真环境' else 'STAGE3_STALL_TIMEOUT',
                           '600' if phase == '导入仿真环境' else '180')))
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(self.path)
        self.last_write = now
        timeout = payload['timeout_seconds']
        faulthandler.dump_traceback_later(max(1, timeout - 15))
        if force:
            print(f'[Stage III][进度] 组 {self.group} / 轨迹 {self.trajectory}: {phase}', flush=True)


def cached_trajectory(path: Path, *, group: int, trajectory: int,
                      task: dict, student: Path, actor: Path,
                      actor_hash: str, initial_state_idx: int = 0) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    expected = dict(group_id=group, trajectory_idx=trajectory, suite=task['suite'],
                    task_id=task['task_id'], task_name=task['name'],
                    initial_state_idx=initial_state_idx,
                    base_student_checkpoint=str(student.resolve()),
                    actor_checkpoint=str(actor.resolve()))
    if any(row.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'Cached trajectory identity mismatch: {path}')
    recorded_hash = row.get('actor_sha256')
    if recorded_hash is not None and recorded_hash != actor_hash:
        raise RuntimeError(f'Cached trajectory belongs to a different actor: {path}')
    if recorded_hash is None and row.get('collected_at', 0) < actor.stat().st_mtime:
        raise RuntimeError(f'Legacy trajectory predates current actor: {path}')
    if not row.get('decisions'):
        raise RuntimeError(f'Cached trajectory has no policy decisions: {path}')
    return row


def wait_with_progress(proc, progress: Path, *, stall_timeout: float,
                       hard_timeout: float, poll_seconds: float = 2,
                       import_timeout: float | None = None) -> int:
    start = time.monotonic()
    wall_start = time.time()
    try:
        while True:
            try:
                return proc.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                last = max(wall_start, progress.stat().st_mtime if progress.exists() else wall_start)
                phase = ''
                try:
                    phase = json.loads(progress.read_text()).get('phase', '')
                except (OSError, ValueError):
                    pass
                budget = (import_timeout if import_timeout is not None and phase in
                          ('启动采样子进程', '导入仿真环境') else stall_timeout)
                if time.time() - last > budget or time.monotonic() - start > hard_timeout:
                    raise TimeoutError(f'rollout timed out: phase={phase}, no progress '
                                       f'{time.time() - last:.0f}s (limit={budget}s), '
                                       f'elapsed={time.monotonic() - start:.0f}s (limit={hard_timeout}s)')
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # SIGKILL can wait for an uninterruptible disk read.
                    # Keep the original timeout classification for skip handling.
                    print(f'[Stage III][清理] PID {proc.pid} 已发 SIGKILL，等待内核结束 I/O。', flush=True)
        raise
