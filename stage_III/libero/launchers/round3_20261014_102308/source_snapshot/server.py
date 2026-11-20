"""Isolated Stage-III policy server.

The server loads exactly the completed Stage-II step4000 Student and a
Stage-III GateActor.  It never writes to the Stage-I/II run directories.
"""
from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goal_local.eval_common import atomic_json  # noqa: E402
from .gate_actor import GateActor  # noqa: E402
from .policy import Stage3Policy  # noqa: E402


EXPECTED_STAGE2 = ROOT / "runs/goal_local/stage_II_warm250_20260921/step_004000"
STAGE3_ROOT = ROOT / "stage_III/libero"


class Engine:
    def __init__(self, device: str, ready: str):
        self.device = torch.device(device)
        self.ready = Path(ready)
        self.lock = threading.Lock()
        self.model: Stage3Policy | None = None
        self.checkpoint: str | None = None
        self.actor_checkpoint: str | None = None
        self.step: int | None = None
        self.calls = 0

    def info(self) -> dict:
        return dict(
            pid=os.getpid(),
            device=str(self.device),
            physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            checkpoint=self.checkpoint,
            actor_checkpoint=self.actor_checkpoint,
            step=self.step,
            calls=self.calls,
            teacher_loaded=False,
            uses_future=False,
            uses_state=False,
            stage="III",
        )

    def load(self, checkpoint: str, actor_checkpoint: str) -> dict:
        student = Path(checkpoint).resolve()
        expected = EXPECTED_STAGE2.resolve()
        if student != expected:
            raise ValueError(f"Stage III is bound to the immutable step4000 Student: {expected}")
        marker = json.loads((student / "checkpoint_complete.json").read_text())
        if marker != {"step": 4000, "owner": "goal_local.train"}:
            raise ValueError(f"unexpected Stage-II marker: {marker}")
        actor_path = Path(actor_checkpoint).resolve()
        if not actor_path.is_relative_to(STAGE3_ROOT.resolve()):
            raise ValueError("GateActor checkpoint must stay in the isolated Stage-III directory")
        actor = GateActor()
        if actor_path.exists():
            payload = torch.load(actor_path, map_location="cpu", weights_only=False)
            actor.load_state_dict(payload["actor_state_dict"], strict=True)
        actor.to(self.device).eval().requires_grad_(False)
        self.model = Stage3Policy.load_stage2(student, actor=actor, device=str(self.device))
        self.model.eval().requires_grad_(False)
        self.checkpoint = str(student)
        self.actor_checkpoint = str(actor_path)
        self.step = 4000
        self.calls = 0
        gc.collect()
        info = self.info()
        atomic_json(self.ready, info)
        print("[Stage III][服务端] 已只读加载 Stage-II step4000 Student，Teacher 未加载，GateActor 已就绪", flush=True)
        return info

    def reload_actor(self, actor_checkpoint: str) -> dict:
        if self.model is None:
            raise RuntimeError("Stage III server has not loaded a checkpoint")
        actor_path = Path(actor_checkpoint).resolve()
        if not actor_path.is_relative_to(STAGE3_ROOT.resolve()):
            raise ValueError("GateActor checkpoint must stay in the isolated Stage-III directory")
        payload = torch.load(actor_path, map_location="cpu", weights_only=False)
        self.model.gate_actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.model.gate_actor.to(self.device).eval().requires_grad_(False)
        self.actor_checkpoint = str(actor_path)
        info = self.info()
        atomic_json(self.ready, info)
        print(f"[Stage III][服务端] 已热更新 GateActor：{actor_path.name}；Student 仍为只读 step4000", flush=True)
        return info

    @staticmethod
    def _image(value: str) -> torch.Tensor:
        raw = base64.b64decode(value, validate=True)
        if len(raw) != 256 * 256 * 3:
            raise ValueError("expected 256x256 uint8 RGB image")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(256, 256, 3).copy()
        return torch.from_numpy(arr).permute(2, 0, 1)[None]

    @torch.inference_mode()
    def infer(self, request: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Stage III server has not loaded a checkpoint")
        expected = {"checkpoint", "actor_checkpoint", "main", "wrist", "instruction",
                    "noise_seed", "gate_seed"}
        if not expected.issubset(request) or set(request) - expected - {"eta_override"}:
            raise ValueError(f"unexpected Stage III request keys: {sorted(request)}")
        if str(Path(request["checkpoint"]).resolve()) != self.checkpoint:
            raise ValueError("Student checkpoint mismatch")
        if str(Path(request["actor_checkpoint"]).resolve()) != self.actor_checkpoint:
            raise ValueError("GateActor checkpoint mismatch")
        main, wrist = self._image(request["main"]), self._image(request["wrist"])
        action_generator = torch.Generator(device=self.device).manual_seed(int(request["noise_seed"]))
        noise = torch.randn((1, 50, 32), generator=action_generator,
                            device=self.device, dtype=torch.bfloat16)
        begin = time.monotonic()
        actions, gate = self.model.predict_action_stage3(
            main.to(self.device), wrist.to(self.device), [str(request["instruction"])],
            noise=noise, gate_seed=int(request["gate_seed"]),
            eta_override=request.get("eta_override"),
        )
        actions = actions[0].float().cpu().numpy()
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid Stage III action output")
        low = np.asarray(self.model.statistics["franka"]["action"]["min"][:6])
        high = np.asarray(self.model.statistics["franka"]["action"]["max"][:6])
        actions[:, :6] = np.clip(actions[:, :6], low, high)
        self.calls += 1
        record = {
            "features": gate["features"][0].float().cpu().tolist(),
            "z": float(gate["z"][0].float().cpu()),
            "eta": float(gate["eta"][0].float().cpu()),
            "old_logp": None if gate["old_logp"] is None else float(gate["old_logp"][0].float().cpu()),
            "old_alpha": None if gate["alpha"] is None else float(gate["alpha"][0].float().cpu()),
            "old_beta": None if gate["beta"] is None else float(gate["beta"][0].float().cpu()),
        }
        return dict(actions=actions.tolist(), step=4000, normalized=False, shape=[8, 7],
                    gate=record, inference_seconds=time.monotonic() - begin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--checkpoint", default=str(EXPECTED_STAGE2))
    parser.add_argument("--actor-checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    engine = Engine(args.device, args.ready)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def send_json(self, code: int, result: dict):
            data = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.send_json(404, {"error": "unknown endpoint"})
            self.send_json(200, engine.info())

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size < 2_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(size))
                with engine.lock:
                    if self.path == "/load":
                        result = engine.load(request["checkpoint"], request["actor_checkpoint"])
                    elif self.path == "/reload_actor":
                        result = engine.reload_actor(request["actor_checkpoint"])
                    elif self.path == "/infer":
                        result = engine.infer(request)
                    else:
                        raise ValueError(f"unknown endpoint {self.path}")
                self.send_json(200, result)
            except Exception as exc:
                traceback.print_exc()
                self.send_json(500, {"error": repr(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    engine.port = server.server_address[1]
    atomic_json(args.ready, engine.info())
    print(f"[Stage III][服务端] 监听端口 {args.port}，设备 {args.device}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

