"""Enumerate every released task and every provided initial state, without repeats."""
import hashlib
import json
import re
import runpy
from collections import Counter
import torch
from common import *

task_map = runpy.run_path(str(REPO / 'libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']
tasks = []
jobs = []
for ci, category in enumerate(CATEGORIES):
    for si, suite in enumerate(SUITES):
        pro_suite = f'{suite}_{category}'
        for ti, name in enumerate(task_map[pro_suite]):
            bddl = REPO / 'libero/libero/bddl_files' / pro_suite / f'{name}.bddl'
            init = REPO / 'libero/libero/init_files' / pro_suite / f'{name}.pruned_init'
            text = bddl.read_text()
            language = ' '.join(re.search(r'\(:language\s+([^)]*)\)', text, flags=re.S).group(1).split()).lower()
            states = torch.load(init, map_location='cpu', weights_only=False)
            assert len(states) > 0
            task = dict(suite=suite, pro_suite=pro_suite, category=category, task_id=ti,
                        name=name, instruction=language, bddl=str(bddl), init=str(init),
                        initial_states=len(states), state_shape=list(states[0].shape),
                        bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                        init_sha256=hashlib.sha256(init.read_bytes()).hexdigest())
            tasks.append(task)
            for ei in range(len(states)):
                jobs.append(dict(id=len(jobs), **task, episode_idx=ei,
                                 noise_seed=(ci+1)*1_000_000_000+(si+1)*10_000_000+ti*10_000+ei*100))
manifest = dict(benchmark='original LIBERO-PRO official released four perturbation sets',
                github_commit='eafdb809426b13153aa1e4c42d6601844217dfec',
                dataset_revision=read(BASE / 'resources/dataset_manifest.json')['revision'],
                tasks=tasks, jobs=jobs, total_tasks=len(tasks), total_episodes=len(jobs),
                max_steps=MAX_STEPS, settling_steps=10, image_size=256, action_chunk=8,
                flow_steps=10, language_source='BDDL :language, never unperturbed filename',
                seed=0, uses_state=False, uses_future=False, teacher_loaded=False,
                stage_III_actor=False, save_videos=False,
                scope_note='All 160 original released tasks. Environment resources are not published; seven-case extension excluded.')
atomic(RUN / 'manifest.json', manifest)
print(json.dumps(dict(tasks=len(tasks), episodes=len(jobs), states_per_task=dict(Counter(t['initial_states'] for t in tasks))), indent=2))
