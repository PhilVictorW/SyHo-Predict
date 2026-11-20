"""Validate handoff timing and queue integrity using disposable database copies."""
import collections
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import plus7_after_libero as h


def validate():
    production = h.c.RUN
    checks = []
    plus_live = [dict(pid=900+g, gpu=g, kind='server', variant='plus') for g in (5,6,7)]
    assert h.gate_reason({'done': 7999, 'running': 1}, [])
    assert h.gate_reason({'done': 7999, 'failed': 1}, [])
    assert h.gate_reason({'done': 8000}, [dict(pid=1, gpu=0, kind='sim')])
    assert h.gate_reason({'done': 8000}, plus_live) is None
    checks.append('Wait for all 8000 episodes and released GPU 0/1/2/3 processes')

    waiting = [dict(phase='waiting_memory', retry_after=0) for _ in range(2)]
    assert h.admission_reason(0, waiting, [], 15024, 0, 24000) is None
    assert h.admission_reason(0, waiting, [], 15023, 0, 24000)
    assert h.admission_reason(0, waiting, [], 15024, 3, 24000)
    assert h.admission_reason(0, waiting, [], 15024, 0, 23999)
    assert h.admission_reason(0, waiting, [dict(pid=1, gpu=0, kind='server')], 30000, 0, 40000)
    assert h.admission_reason(0, waiting, [dict(pid=1, gpu=0, kind='sim')], 30000, 0, 40000)
    assert h.admission_reason(0, waiting*2, [], 60000, 0, 40000)
    partial = [dict(phase='running', server_pid=2, sim_pid=3), waiting[0]]
    own = [dict(pid=2, gpu=0, kind='server'), dict(pid=3, gpu=0, kind='sim')]
    assert h.admission_reason(0, partial, own, 8024, 0, 24000) is None
    assert h.admission_reason(0, partial, own+[dict(pid=4,gpu=0,kind='server')], 30000, 0, 40000)
    checks.append('Per-GPU cap, old simulator release, GPU/host memory and loader limits')

    with tempfile.TemporaryDirectory(prefix='libero_plus7_validation_', dir='/tmp') as tmp:
        root = Path(tmp)
        for variant in ('libero', 'plus'):
            (root/variant).mkdir()
            with sqlite3.connect('file:'+str(production/variant/'queue.sqlite')+'?mode=ro', uri=True) as src:
                with sqlite3.connect(root/variant/'queue.sqlite') as dst:
                    src.backup(dst)
        h.c.RUN = root
        h.AUDIT = root/'handoff'
        h.STATE = root/'plus7.json'
        h.processes = lambda: plus_live
        h.original_state = lambda: dict(phase='running', slots=[])
        with sqlite3.connect(root/'plus/queue.sqlite') as db:
            before = dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
        try:
            h.activate()
            raise AssertionError('Activated before LIBERO completed')
        except RuntimeError as exc:
            assert 'Waiting for LIBERO' in str(exc)
        with sqlite3.connect(root/'plus/queue.sqlite') as db:
            assert before == dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
            assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='scheduling_events'").fetchone()
        checks.append('Incomplete LIBERO leaves the entire PLUS queue untouched')

        with sqlite3.connect(root/'libero/queue.sqlite') as db:
            db.execute("UPDATE jobs SET state='done'")
        h.activate()
        with sqlite3.connect(root/'plus/queue.sqlite') as db:
            after = dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
            assert len(before)==len(after)==8000
            distribution = collections.Counter()
            for job_id, old in before.items():
                new = after[job_id]
                if old[3] != 'pending':
                    assert new == old
                else:
                    assert new[:2]+new[3:] == old[:2]+old[3:]
                    assert new[2] in h.GPUS
                    distribution[new[2]] += 1
            assert set(distribution)==set(h.GPUS)
            assert max(distribution.values())-min(distribution.values())<=1
            record = json.loads(db.execute('SELECT payload FROM scheduling_events WHERE event=?',(h.EVENT,)).fetchone()[0])
            assert len(record['assignments'])==sum(distribution.values())
        assert len(h.slots)==8 and {s['gpu'] for s in h.slots}==set(h.NEW_GPUS)
        assert all(s['variant']=='plus' for s in h.slots)
        checks.append('Completed/running rows, seeds, payloads, attempts and results preserved exactly')
        checks.append('Pending episodes balanced across seven GPUs; GPU4 excluded; eight new PLUS slots')
        with sqlite3.connect(h.AUDIT/'queue_before.sqlite') as db:
            assert before == dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
        try:
            h.activate()
            raise AssertionError('Duplicate handoff was accepted')
        except RuntimeError as exc:
            assert 'already committed' in str(exc)
        with sqlite3.connect(root/'plus/queue.sqlite') as db:
            assert after == dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
        with sqlite3.connect(h.AUDIT/'queue_before.sqlite') as db:
            assert before == dict((row[0],row) for row in db.execute('SELECT * FROM jobs'))
        checks.append('Duplicate activation rejected; queue and original backup remain intact')
    h.c.RUN = production
    result = dict(time=time.time(), passed=True, checks=checks)
    target = production/'plus7_handoff'/'validation.json'
    h.c.atomic(target, result)
    print(json.dumps(result, indent=2))
    h.c.pool.shutdown(wait=True)


if __name__=='__main__':
    validate()
