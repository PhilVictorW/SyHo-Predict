"""Isolated, resumable Stage III evaluation records; no training mutations."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / 'stage_III/libero_plus'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
CATEGORIES = ['Camera', 'Robot', 'Language', 'Light', 'Background', 'Noise', 'Layout']
GPUS = [1, 1, 3, 3, 6, 6, 7, 7]
PORTS = list(range(29800, 29808))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def connect(folder):
    db = sqlite3.connect(Path(folder) / 'queue.sqlite', timeout=60, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=60000')
    return db


def initialize_queue(folder, tasks):
    db = connect(folder)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS tasks (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, '
               "state TEXT NOT NULL DEFAULT 'pending', worker INTEGER, attempts INTEGER DEFAULT 0, "
               'started REAL, finished REAL, result TEXT, error TEXT)')
    db.execute('BEGIN IMMEDIATE')
    try:
        for i, task in enumerate(tasks):
            encoded = json.dumps(task, sort_keys=True, ensure_ascii=False)
            old = db.execute('SELECT payload FROM tasks WHERE id=?', (i,)).fetchone()
            if old and old['payload'] != encoded:
                raise ValueError(f'evaluation task identity changed at {i}')
            db.execute('INSERT OR IGNORE INTO tasks (id,payload) VALUES (?,?)', (i, encoded))
        if db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] != len(tasks):
            raise ValueError('unexpected evaluation queue length')
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    finally:
        db.close()


def claim(db, worker):
    db.execute('BEGIN IMMEDIATE')
    try:
        row = db.execute("SELECT * FROM tasks WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
        if row is not None:
            db.execute("UPDATE tasks SET state='running',worker=?,attempts=attempts+1,started=?,finished=NULL "
                       'WHERE id=?', (worker, time.time(), row['id']))
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    if row is None:
        return None
    return dict(row) | {'attempts': row['attempts'] + 1, 'payload': json.loads(row['payload'])}


def finish(db, job, worker, state, result, error=None):
    if state not in ('done', 'timeout', 'error'):
        raise ValueError(state)
    changed = db.execute('UPDATE tasks SET state=?,finished=?,result=?,error=? '
                         "WHERE id=? AND state='running' AND worker=? AND attempts=?",
                         (state, time.time(), json.dumps(result, ensure_ascii=False), error,
                          job['id'], worker, job['attempts'])).rowcount
    if changed != 1:
        raise RuntimeError('refusing stale or duplicate evaluation result')


def requeue_errors(folder, max_attempts=3):
    db = connect(folder)
    try:
        return db.execute("UPDATE tasks SET state='pending',worker=NULL WHERE state IN ('timeout','error') "
                          'AND attempts<?', (max_attempts,)).rowcount
    finally:
        db.close()


def result_summary(folder, protocol):
    cells = {s: {c: dict(expected=0, done=0, successes=0, timeout=0, error=0, running=0, pending=0)
                 for c in CATEGORIES} for s in SUITES}
    workers = {str(i): dict(done=0, successes=0, timeout=0, error=0, active=None) for i in range(8)}
    rows = []
    if (Path(folder) / 'queue.sqlite').exists():
        db = connect(folder)
        try:
            rows = [dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY id')]
        finally:
            db.close()
    if not rows:
        rows = [dict(id=i, payload=json.dumps(t), state='pending', worker=None, result=None, attempts=0)
                for i, t in enumerate(protocol['tasks'])]
    for row in rows:
        task = json.loads(row['payload']); cell = cells[task['suite']][task['category']]
        cell['expected'] += 1; cell[row['state']] += 1
        result = json.loads(row['result']) if row['result'] else {}
        success = int(row['state'] == 'done' and result.get('success') is True)
        cell['successes'] += success
        if row['worker'] is not None:
            worker = workers[str(row['worker'])]
            if row['state'] in ('done','timeout','error'):
                worker[row['state']] += 1; worker['successes'] += success
            elif row['state'] == 'running':
                worker['active'] = dict(id=row['id'], task=task, started=row['started'], attempts=row['attempts'])
    total = {k: sum(cell[k] for cs in cells.values() for cell in cs.values())
             for k in ('expected','done','successes','timeout','error','running','pending')}
    terminal = total['done'] + total['timeout'] + total['error']
    total.update(processed=terminal, complete=total['done'] == total['expected'],
                 all_attempted=terminal == total['expected'],
                 success_rate=total['successes'] / total['done'] if total['done'] else None,
                 success_rate_full_denominator=total['successes'] / total['expected'] if total['expected'] else None,
                 cells=cells, workers=workers, updated_at=time.time(),
                 gate_mode='beta_mean', actor_checkpoint=protocol.get('actor_checkpoint'),
                 base_student=protocol['base_student'], evaluation_scope='same_8380_training_subset',
                 note='超时/错误单独记录；success_rate仅含完成项，完整8380成功率须complete=true。')
    return total


def training_complete(status, expected=8380):
    return (status.get('phase') in ('completed','completed_with_skips')
            and status.get('groups_processed') == expected
            and status.get('batch') == (expected + 31) // 32
            and status.get('groups_completed', 0) + status.get('groups_skipped', 0) == expected)
