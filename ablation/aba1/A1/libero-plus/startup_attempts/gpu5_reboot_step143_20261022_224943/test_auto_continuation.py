"""CPU/file-only checks of handoff ordering, exit receipts and failure gates."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import continue_training as chain
import after_training_full_eval as evaluation


class Continuation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.logs = self.work / "logs"
        self.run = self.work / "protocol_plus_v1/seed_2026"
        self.logs.mkdir()
        self.run.mkdir(parents=True)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for module in (chain, evaluation):
            for key, value in (("WORK", self.work), ("RUN", self.run)):
                stack.enter_context(patch.object(module, key, value))
        stack.enter_context(patch.object(chain, "LOGS", self.logs))
        stack.enter_context(patch.object(chain, "STATE", self.logs / "chain.json"))
        stack.enter_context(patch.object(chain, "II_EXIT", self.logs / "stage_II_exit_code.txt"))
        stack.enter_context(patch.object(evaluation, "STATE", self.logs / "eval.json"))
        self.events = []
        stack.enter_context(patch.object(chain, "record"))
        stack.enter_context(patch.object(chain, "verify_teacher", side_effect=lambda: self.events.append("verify_teacher")))
        self.marker("I", 3000)
        self.completed("I")
        (self.logs / "exit_code.txt").write_text("0\n")

    def marker(self, stage, step):
        root = self.run / stage / f"step_{step:06d}"
        root.mkdir(parents=True, exist_ok=True)
        (root / "model.pt").write_bytes(b"fake checkpoint; tests never load a model")
        chain.write_json(root / "checkpoint_complete.json", dict(complete=True, stage=stage, step=step))

    def completed(self, stage):
        chain.write_json(self.run / "pipeline_status.json", dict(phase="completed", completed_stages=[stage]))

    def successful_child(self, command, phase, lock_fd, poll):
        self.events.append(phase)
        if phase == "stage_II":
            self.assertEqual((self.logs / "stage_I_exit_code.txt").read_text(), "0\n")
            self.marker("II", 2500)
            self.completed("II")
            chain.II_EXIT.write_text("0\n")
        else:
            self.assertEqual(command[-2:], ["--training-exit-file", str(chain.II_EXIT)])
            self.assertEqual(evaluation.snapshot(chain.II_EXIT)["phase"], "ready")
            chain.write_json(evaluation.STATE, dict(phase="completed", completed=10030))

    def test_success_runs_teacher_verification_then_student_then_full_evaluation(self):
        with patch.object(chain, "run_child", side_effect=self.successful_child):
            chain.continue_run(0, 2)
        self.assertEqual(self.events, ["verify_teacher", "stage_II", "full_evaluation"])
        self.assertEqual((self.logs / "exit_code.txt").read_text(), "0\n")

    def test_stage_I_must_exit_successfully_and_commit_correct_checkpoint(self):
        pipeline = dict(phase="completed", completed_stages=["I"])
        marker = dict(complete=True, stage="I", step=3000)
        self.assertEqual(chain.teacher_gate(pipeline, marker, 0, [123], True), "waiting_stage_I_exit")
        self.assertEqual(chain.teacher_gate(dict(phase="training"), marker, None, [123], True), "waiting_stage_I")
        for code in (None, 1):
            self.assertTrue(chain.teacher_gate(pipeline, marker, code, [], True).startswith("failed"))
        for bad in (dict(marker, complete=False), dict(marker, step=5000), dict(marker, stage="II")):
            self.assertEqual(chain.teacher_gate(pipeline, bad, 0, [], True), "failed_stage_I_checkpoint")
        self.assertEqual(chain.teacher_gate(pipeline, marker, 0, [], False), "failed_stage_I_checkpoint")
        self.assertEqual(chain.teacher_gate(pipeline, marker, 0, [], True), "ready")

    def test_failed_student_never_launches_evaluation(self):
        def fail(command, phase, lock_fd, poll):
            self.events.append(phase)
            chain.II_EXIT.write_text("1\n")
            raise RuntimeError("simulated training failure")
        with patch.object(chain, "run_child", side_effect=fail), self.assertRaises(RuntimeError):
            chain.continue_run(0, 2)
        self.assertEqual(self.events, ["verify_teacher", "stage_II"])

    def test_incomplete_existing_student_is_never_overwritten(self):
        (self.run / "II").mkdir()
        (self.run / "II/arguments.json").write_text("{}")
        with patch.object(chain, "run_child") as child, self.assertRaisesRegex(RuntimeError, "incomplete Stage II"):
            chain.continue_run(0, 2)
        child.assert_not_called()

    def test_successful_student_receipt_skips_training_on_monitor_relaunch(self):
        self.marker("II", 2500)
        self.completed("II")
        chain.II_EXIT.write_text("0\n")
        with patch.object(chain, "run_child", side_effect=self.successful_child):
            chain.continue_run(0, 2)
        self.assertEqual(self.events, ["full_evaluation"])

    def test_stage_I_receipt_cannot_release_student_evaluation(self):
        self.marker("II", 2500)
        self.completed("II")
        self.assertEqual(evaluation.snapshot(chain.II_EXIT)["phase"], "waiting_successful_pipeline_exit")
        chain.II_EXIT.write_text("0\n")
        self.assertEqual(evaluation.snapshot(chain.II_EXIT)["phase"], "ready")

    def test_stage_II_wrapper_preserves_stage_I_receipt_and_records_failure(self):
        original = Path(__file__).resolve().parent / "run_stage_ii_logged.sh"
        wrapper = self.work / original.name
        wrapper.write_text(original.read_text())
        (self.work / "train.sh").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" > received_args.txt\nexit 7\n')
        proc = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 7, proc.stderr)
        self.assertEqual(chain.II_EXIT.read_text().strip(), "7")
        self.assertEqual((self.logs / "exit_code.txt").read_text().strip(), "0")
        self.assertEqual((self.work / "received_args.txt").read_text().strip(), "--stage II --gpus 5 --execute")
        again = subprocess.run(["bash", str(wrapper)], capture_output=True, text=True)
        self.assertEqual(again.returncode, 64)
        self.assertEqual(chain.II_EXIT.read_text().strip(), "7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
