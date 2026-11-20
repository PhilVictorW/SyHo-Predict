"""Create disjoint LIBERO Stage-III initial-state manifests.

The historical 98.5% evaluation uses initial-state indices 0..9.  Stage III
uses 10..29 for RL-train, 30..39 for RL-val, and 40..49 for final test.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def build(indices: list[int], split: str) -> dict:
    tasks = []
    for suite in SUITES:
        for task_id in range(10):
            tasks.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "initial_state_indices": indices,
                    "split": split,
                }
            )
    return {
        "format": "stage3_libero_manifest_v1",
        "split": split,
        "suites": list(SUITES),
        "task_count": len(tasks),
        "initial_state_policy": "same index within a group; disjoint across train/val/test",
        "tasks": tasks,
    }


def main() -> None:
    outputs = {
        "rl_train.json": build(list(range(10, 30)), "rl-train"),
        "rl_val.json": build(list(range(30, 40)), "rl-val"),
        "final_test.json": build(list(range(40, 50)), "final-test"),
    }
    for name, payload in outputs.items():
        path = ROOT / "manifest" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"写入 {path}: {payload['task_count']} tasks")


if __name__ == "__main__":
    main()
