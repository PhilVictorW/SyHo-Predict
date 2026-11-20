"""Stage II cadence and cooperative disk-space waiting; retain verified snapshots."""
from pathlib import Path
import shutil
import time

from a2_local.checkpointing import commit_checkpoint as original_commit
from a2_local.checkpointing import durable_json
from .runtime import CHECKPOINT_POLICY


def wait_for_space(output, required, rank, dist, status, step, poll_seconds=None):
    """All ranks exchange a heartbeat while rank zero waits for disk capacity."""
    poll = CHECKPOINT_POLICY['poll_seconds'] if poll_seconds is None else poll_seconds
    while True:
        message = [None]
        if rank == 0:
            try:
                free = shutil.disk_usage(output).free
                message[0] = dict(ready=free >= required, free_bytes=free,
                                  required_bytes=required)
            except OSError as error:
                message[0] = dict(error=repr(error))
        dist.broadcast_object_list(message, src=0)
        info = message[0]
        if info.get('error'):
            raise OSError(info['error'])
        if info['ready']:
            status(phase='checkpoint', step=step,
                   checkpoint_free_bytes=info['free_bytes'],
                   checkpoint_required_bytes=info['required_bytes'])
            return
        status(phase='waiting_checkpoint_space', step=step,
               checkpoint_free_bytes=info['free_bytes'],
               checkpoint_required_bytes=info['required_bytes'])
        time.sleep(poll)


def commit_checkpoint(temporary, final, metadata, artifacts, validate):
    cadence = CHECKPOINT_POLICY['save_every']
    metadata = dict(metadata, save_every=cadence)
    # Preserve fsync, hash validation, atomic publication and pruning order.
    original_commit(temporary, final, metadata, artifacts, validate)
    durable_json(Path(final).parent / 'recovery_retention.json',
                 dict(full_resume=str(final), retained_models=[str(final)],
                      keep_checkpoints=1, save_every=cadence))
