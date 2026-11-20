"""Real disk rotation/failure tests and a CPU AdamW recovery equivalence test."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parent
sys.path[:0] = [str(WORK), str(WORK.parents[3])]
import numpy as np
import torch
from a1_local.checkpointing import commit_checkpoint, make_recovery, rank_recovery
from a1_local.config_runtime import FORMAT, digest_json
from a1_local.preflight import check_complete
from goal_local.optim import MasterAdamW


class RollingCheckpoints(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cfg = {"fixture": "tiny recovery test"}
        torch.set_num_threads(1)

    def tearDown(self):
        self.directory.cleanup()

    def write(self, step, stage="I", validate=check_complete):
        output = self.root / stage
        output.mkdir(exist_ok=True)
        temporary = output / f".step_{step:06d}.incomplete"
        temporary.mkdir()
        model = dict(format=FORMAT, mode="posterior", config=self.cfg, step=step,
                     extra={"stage": stage}, state_dict={"p": torch.tensor([step])})
        torch.save(model, temporary / "model.pt")
        torch.save({"step": step, "optimizer": torch.tensor([step])}, temporary / "trainer.pt")
        metadata = dict(format=FORMAT, complete=True, resumable=True, step=step,
                        stage=stage, config_sha256=digest_json(self.cfg))
        final = output / f"step_{step:06d}"
        commit_checkpoint(temporary, final, metadata, ["model.pt", "trainer.pt"], validate)
        return final

    def test_50_100_150_rotates_only_after_valid_new_checkpoint(self):
        for step in (50, 100, 150):
            new = self.write(step)
            meta, marker = check_complete(new, require_resume=True, verify_hash=True)
            self.assertEqual(meta["step"], step)
            self.assertTrue(marker["resumable"])
            self.assertEqual(list(new.parent.glob("step_*")), [new])
            self.assertEqual(json.loads((new.parent / "latest.json").read_text())["step"], step)

    def test_corrupt_new_checkpoint_keeps_old_and_latest_pointer(self):
        old = self.write(50)
        def corrupt_then_validate(path, **kwargs):
            (path / "trainer.pt").write_bytes(b"incomplete data")
            check_complete(path, **kwargs)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.write(100, validate=corrupt_then_validate)
        check_complete(old, require_resume=True, verify_hash=True)
        self.assertEqual(json.loads((old.parent / "latest.json").read_text())["step"], 50)

    def test_failed_commit_keeps_old(self):
        old = self.write(50)
        real_rename = Path.rename
        def fail_publish(path, destination):
            if path.name == ".step_000100.incomplete":
                raise OSError("simulated interrupted publication")
            return real_rename(path, destination)
        with patch.object(Path, "rename", fail_publish), self.assertRaises(OSError):
            self.write(100)
        check_complete(old, require_resume=True, verify_hash=True)
        self.assertEqual(json.loads((old.parent / "latest.json").read_text())["step"], 50)

    def test_stage_ii_rotation_preserves_final_teacher(self):
        teacher = self.write(3000)
        self.write(50, "II")
        student = self.write(100, "II")
        check_complete(teacher, require_resume=True, verify_hash=True)
        self.assertEqual(list(student.parent.glob("step_*")), [student])

    def test_real_optimizer_rng_and_per_rank_cursor_recovery(self):
        torch.manual_seed(2026)
        random.seed(2026)
        np.random.seed(2026)
        parameter = torch.nn.Parameter(torch.tensor([.5, -.25], dtype=torch.bfloat16))
        optimizer = MasterAdamW([parameter], .003, .01)
        def update(p, opt):
            opt.zero_grad()
            noise = torch.rand(2) + random.random() + float(np.random.rand())
            ((p.float() - noise)**2).sum().backward()
            opt.step()
        for _ in range(3):
            update(parameter, optimizer)
        rng = dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state())
        states = [dict(rank=r, epoch=0, consumed=3*n, rng=copy.deepcopy(rng))
                  for r,n in enumerate([6,6,5,5,5,5])]
        recovery = make_recovery(3, dict(world=6), optimizer.state_dict(), None, states)
        path = self.root / "trainer.pt"
        torch.save(recovery, path)
        update(parameter, optimizer)
        expected_parameter = parameter.detach().clone()
        expected = copy.deepcopy(optimizer.state_dict())
        restored = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        p2 = torch.nn.Parameter(torch.zeros_like(parameter))
        opt2 = MasterAdamW([p2], .003, .01)
        opt2.load_state_dict(restored["optimizer"])
        state = rank_recovery(restored, 0, 6)
        torch.set_rng_state(state["rng"]["torch"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        update(p2, opt2)
        torch.testing.assert_close(p2, expected_parameter, atol=0, rtol=0)
        actual = opt2.state_dict()
        torch.testing.assert_close(actual["masters"], expected["masters"], atol=0, rtol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual["optimizer"]["state"][0][key],
                                       expected["optimizer"]["state"][0][key], atol=0, rtol=0)
        self.assertEqual([rank_recovery(restored,r,6)["consumed"] for r in range(6)],
                         [18,18,15,15,15,15])
        broken = copy.deepcopy(restored)
        broken["rank_states"].pop()
        with self.assertRaises(ValueError):
            rank_recovery(broken,0,6)

if __name__ == "__main__":
    unittest.main(verbosity=2)
