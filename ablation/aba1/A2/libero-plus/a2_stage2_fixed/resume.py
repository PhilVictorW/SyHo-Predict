"""Validate the recorded 4-to-10 rank migration without altering checkpoint100."""
import json
from pathlib import Path
from a2_local.preflight import check_complete
from a2_local.config_runtime import digest_json
from a2_stage2.runtime import code_identity as prior_code_identity

WORK=Path(__file__).resolve().parents[1]
AUDIT=WORK/'logs/resume100_fixed10_20261030'

def require_receipt():
    if not json.loads((AUDIT/'verified.json').read_text()).get('complete'):
        raise ValueError('Checkpoint checksum verification has not completed')
    return json.loads((AUDIT/'checkpoint_validation.json').read_text())

def checked_complete(path,require_resume=False,verify_hash=False):
    path=Path(path).resolve()
    receipt=require_receipt()
    entry=receipt['checkpoints'].get(str(path))
    if not verify_hash or entry is None:
        return check_complete(path,require_resume=require_resume,verify_hash=verify_hash)
    if require_resume and not entry['full_resume_verified']:
        return check_complete(path,require_resume=True,verify_hash=True)
    for filename,expected in entry['files'].items():
        s=Path(filename).stat()
        actual=dict(dev=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)
        if actual!=expected:raise ValueError('Verified checkpoint changed: '+filename)
    return check_complete(path,require_resume=require_resume,verify_hash=False)

def validate_resume_contract(recovery,new):
    old=recovery['contract']
    if old==new:return False
    prior=json.loads((AUDIT/'source_recovery_metadata.json').read_text())
    if recovery['step']!=100 or old!=prior['contract']:
        raise ValueError('Only the explicitly verified step100 checkpoint may migrate layouts')
    if old['code_sha256']!=digest_json(prior_code_identity()):
        raise ValueError('Original model/training code changed since checkpoint100')
    if [(s['rank'],s['epoch'],s['consumed']) for s in recovery['rank_states']]!=[(r,0,800) for r in range(4)]:
        raise ValueError('Expected exactly 3200 consumed global anchors at step100')
    if new['world']!=10 or new['rank_to_gpu']!=[2,5,6,7,2,5,6,7,7,7] or new.get('sampler')!='global_batch32_v1':
        raise ValueError('Unexpected new layout or sampler')
    allowed={'world','physical_gpus','rank_to_gpu','accumulation','accumulation_per_rank','rank_policy','code_sha256','sampler'}
    left={k:v for k,v in old.items() if k not in allowed}
    right={k:v for k,v in new.items() if k not in allowed}
    if left!=right:
        differences=[k for k in left.keys()|right.keys() if left.get(k)!=right.get(k)]
        raise ValueError('Migration changed model/data/optimizer schedule contract: '+str(differences))
    return True
