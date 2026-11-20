"""Classic LIBERO rollout client with Stage-III gate records."""
from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def rpc(url: str, endpoint: str, payload=None, timeout: float = 1200):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        value = json.load(response)
    if isinstance(value, dict) and "error" in value:
        raise RuntimeError(value["error"])
    return value


class Stage3Client:
    def __init__(self, checkpoint: str, actor_checkpoint: str, url: str,
                 trajectory_seed: int):
        self.checkpoint = str(Path(checkpoint).resolve())
        self.actor_checkpoint = str(Path(actor_checkpoint).resolve())
        self.url = url
        self.trajectory_seed = int(trajectory_seed)
        self.reset("")

    def configure_episode(self, suite_index: int, task_id: int, episode_idx: int,
                          group_id: int | None = None,
                          trajectory_idx: int | None = None):
        if group_id is None or trajectory_idx is None:
            self.trajectory_seed = ((int(suite_index) + 1) * 10_000_000
                                    + int(task_id) * 10_000
                                    + int(episode_idx) * 100)
        else:
            self.trajectory_seed = (int(group_id) * 1_000_000_000
                                    + int(trajectory_idx) * 10_000_000
                                    + (int(suite_index) + 1) * 100_000
                                    + int(task_id) * 10 + int(episode_idx))

    def reset(self, task_description: str, **_kwargs):
        self.task_description = str(task_description)
        self.actions = None
        self.cursor = 0
        self.chunk = 0
        self.decision_records: list[dict] = []

    def close(self):
        return None

    @staticmethod
    def _encode(images):
        x = np.asarray(images[-1], dtype=np.uint8)
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("expected HWC RGB image")
        if x.shape[:2] != (256, 256):
            x = np.asarray(Image.fromarray(x).resize((256, 256), Image.Resampling.BILINEAR))
        return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()

    def step(self, example, step=0, return_intermediates=False, **_kwargs):
        if return_intermediates:
            raise ValueError("Stage III rollout does not request similarity intermediates")
        if self.actions is None or self.cursor == len(self.actions):
            chunk_seed = self.trajectory_seed + self.chunk * 104729
            gate_seed = self.trajectory_seed + self.chunk * 130363 + 17
            result = rpc(self.url, "/infer", {
                "checkpoint": self.checkpoint,
                "actor_checkpoint": self.actor_checkpoint,
                "main": self._encode(example["primary_image"]),
                "wrist": self._encode(example["wrist_image"]),
                "instruction": str(example["lang"]),
                "noise_seed": int(chunk_seed),
                "gate_seed": int(gate_seed),
            })
            actions = np.asarray(result["actions"], dtype=np.float32)
            if result.get("normalized") is not False or actions.shape != (8, 7):
                raise ValueError("invalid Stage III action response")
            if not np.isfinite(actions).all():
                raise ValueError("non-finite Stage III action response")
            self.actions = actions
            self.cursor = 0
            record = dict(result["gate"])
            record.update(chunk=self.chunk, noise_seed=int(chunk_seed), gate_seed=int(gate_seed))
            self.decision_records.append(record)
            self.chunk += 1
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        # Keep exactly the historical classic-LIBERO StudentClient action semantics.
        return {"raw_action": {
            "world_vector": action[:3],
            "rotation_delta": action[3:6],
            "open_gripper": action[6:7],
        }}
