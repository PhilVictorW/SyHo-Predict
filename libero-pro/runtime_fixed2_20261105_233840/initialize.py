"""Create both durable queues from the same immutable manifest."""
import hashlib
import sqlite3
from common import *

manifest = read(RUN / 'manifest.json')
for variant, gpus in GPUS.items():
    folder = RUN / variant
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(folder / 'queue.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute("CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY,payload TEXT NOT NULL,gpu INTEGER,state TEXT NOT NULL DEFAULT 'pending',worker TEXT,attempts INTEGER NOT NULL DEFAULT 0,success INTEGER,error TEXT,started REAL,finished REAL)")
    db.execute('CREATE INDEX IF NOT EXISTS job_queue ON jobs(state,gpu,id)')
    for job in manifest['jobs']:
        db.execute('INSERT OR IGNORE INTO jobs(id,payload,gpu) VALUES(?,?,?)',
                   (job['id'],json.dumps(job),gpus[job['id'] % len(gpus)]))
    db.commit()
    assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == manifest['total_episodes']
    db.close()
    checkpoint = Path(CHECKPOINTS[variant])
    stat = (checkpoint / 'model.pt').stat()
    atomic(folder / 'protocol.json', dict(variant=variant, checkpoint=str(checkpoint),
        checkpoint_step=4000, teacher_step=5000 if variant=='plus' else 2000,
        weight_bytes=stat.st_size, weight_mtime_ns=stat.st_mtime_ns, weight_inode=stat.st_ino,
        manifest=str(RUN/'manifest.json'), manifest_sha256=hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),
        episodes=manifest['total_episodes'],tasks=manifest['total_tasks'],gpus=gpus,
        display_name='libro' if variant=='libero' else 'plus',
        inference_replicas_per_gpu=2, simulator_helpers_per_gpu=2,
        pair_admission_free_mib=15024, memory_poll_seconds=10,
        gripper='2*(1-g)-1' if variant=='plus' else '2*g-1',
        simulator_seed=0, horizon=MAX_STEPS, noise_seed='fixed in shared manifest; independent of worker/GPU',
        inference='historical GoalLocalPolicy / StudentClient, current dual RGB + BDDL language, 10 flow steps, chunk 8',
        errors_are_failures=False, infrastructure_errors_excluded_until_successful_retry=True,
        scope=manifest['scope_note']))
    print(variant,manifest['total_episodes'],gpus)
