"""Fetch the released original LIBERO-PRO task resources at a pinned revision."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'resources'
revision = json.loads((RES / 'hf_revision.json').read_text())['sha']
tree = json.loads(Path('/tmp/libero_pro_hf_tree.json').read_text())
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
CATEGORIES = ('object', 'swap', 'lan', 'task')
folders = {f'{s}_{c}' for s in SUITES for c in CATEGORIES}
items = [x for x in tree if x['type'] == 'file' and len(x['path'].split('/')) == 3
         and x['path'].split('/')[0] in ('bddl_files', 'init_files')
         and x['path'].split('/')[1] in folders]
assert len(items) == 320, len(items)
(RES / 'hf_tree.json').write_text(json.dumps(tree, indent=2))

def fetch(item):
    dest = RES / 'dataset' / item['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != item['size']:
        url = f'https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro/resolve/{revision}/{item["path"]}'
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as response, dest.with_suffix(dest.suffix+'.part').open('wb') as out:
                    shutil.copyfileobj(response, out)
                assert dest.with_suffix(dest.suffix+'.part').stat().st_size == item['size']
                dest.with_suffix(dest.suffix+'.part').replace(dest)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
    return dict(path=item['path'], size=dest.stat().st_size, sha256=hashlib.sha256(dest.read_bytes()).hexdigest())

rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for row in pool.map(fetch, items):
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f'downloaded {len(rows)}/{len(items)}', flush=True)
(RES / 'dataset_manifest.json').write_text(json.dumps(dict(
    repo='zhouxueyang/LIBERO-Pro', revision=revision, files=rows,
    categories=list(CATEGORIES), suites=list(SUITES),
    excluded='Environment files are not published in the original official dataset; later seven-case extension is a separate protocol.'
), indent=2))
print('complete', revision, len(rows), flush=True)
