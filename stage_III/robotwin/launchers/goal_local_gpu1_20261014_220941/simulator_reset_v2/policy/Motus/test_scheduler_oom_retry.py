import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_quantized_eval
import run_quantized_eval_resume


class OomRetrySummaryTests(unittest.TestCase):
    def test_detects_cuda_oom_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "task.log"
            log_path.write_text(
                "Traceback\n"
                "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB.\n",
                encoding="utf-8",
            )

            detector = getattr(run_quantized_eval, "is_oom_log", None)
            self.assertIsNotNone(detector)
            self.assertTrue(detector(log_path))

    def test_oom_waiting_status_does_not_score_old_success_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = "oom_task"
            ckpt = "Motus_robotwin2"
            task_config = "demo_randomized_motus"
            result_dir = root / "eval_result" / task / "Motus" / task_config / ckpt / "old-run"
            result_dir.mkdir(parents=True)
            (result_dir / "_result.txt").write_text("1.0\n", encoding="utf-8")

            quant_root = root / "quantization"
            task_quant = quant_root / task
            task_quant.mkdir(parents=True)
            (task_quant / "status.json").write_text(
                json.dumps({"task_name": task, "status": "oom_waiting", "oom_detected": True}),
                encoding="utf-8",
            )

            args = SimpleNamespace(result_root=str(root), task_config=task_config)
            config = {"checkpoint_path": f"/tmp/{ckpt}"}
            log_dir = root / "logs"
            log_dir.mkdir()

            run_quantized_eval.summarize_results([task], config, args, log_dir, quant_root)

            result_txt = (root / "result.txt").read_text(encoding="utf-8")
            self.assertIn(f"{task}\toom_waiting\tNA", result_txt)
            self.assertIn("Scored tasks: 0", result_txt)

    def test_oom_waiting_status_does_not_count_stale_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = "oom_task"
            quant_root = root / "quantization"
            task_quant = quant_root / task
            task_quant.mkdir(parents=True)
            (task_quant / "status.json").write_text(
                json.dumps({"task_name": task, "status": "oom_waiting", "oom_detected": True}),
                encoding="utf-8",
            )
            (task_quant / "metrics.json").write_text(
                json.dumps(
                    {
                        "inference_call_count": 5,
                        "metrics": {
                            "cuda_latency_ms": {"total": 100.0},
                            "wall_time_ms": {"total": 200.0},
                            "actions_per_policy_call": {"total": 10.0},
                        },
                        "flops": {"totals": {"total": 123}},
                    }
                ),
                encoding="utf-8",
            )

            run_quantized_eval.summarize_quantization([task], quant_root)

            summary = json.loads((quant_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["aggregate"]["tasks_with_metrics"], 0)
            self.assertEqual(summary["aggregate"]["inference_call_count"], 0)

    def test_running_status_does_not_score_old_success_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = "running_task"
            ckpt = "Motus_robotwin2"
            task_config = "demo_randomized_motus"
            result_dir = root / "eval_result" / task / "Motus" / task_config / ckpt / "old-run"
            result_dir.mkdir(parents=True)
            (result_dir / "_result.txt").write_text("1.0\n", encoding="utf-8")

            quant_root = root / "quantization"
            task_quant = quant_root / task
            task_quant.mkdir(parents=True)
            (task_quant / "status.json").write_text(
                json.dumps({"task_name": task, "status": "running"}),
                encoding="utf-8",
            )

            args = SimpleNamespace(result_root=str(root), task_config=task_config)
            config = {"checkpoint_path": f"/tmp/{ckpt}"}
            log_dir = root / "logs"
            log_dir.mkdir()

            run_quantized_eval.summarize_results([task], config, args, log_dir, quant_root)

            result_txt = (root / "result.txt").read_text(encoding="utf-8")
            self.assertIn(f"{task}\trunning\tNA", result_txt)
            self.assertIn("Scored tasks: 0", result_txt)

    def test_failed_oom_status_is_retry_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "task.log"
            log_path.write_text("torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8")
            status = {"status": "failed", "log_path": str(log_path)}

            candidate = getattr(run_quantized_eval_resume, "is_oom_retry_candidate", None)
            self.assertIsNotNone(candidate)
            self.assertTrue(candidate("task", status, log_path))

    def test_non_oom_failed_status_is_not_retry_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "task.log"
            log_path.write_text("ValueError: unrelated failure\n", encoding="utf-8")
            status = {"status": "failed", "log_path": str(log_path)}

            candidate = getattr(run_quantized_eval_resume, "is_oom_retry_candidate", None)
            self.assertIsNotNone(candidate)
            self.assertFalse(candidate("task", status, log_path))


    def test_retry_oom_only_reads_old_logs_but_writes_fresh_retry_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_log_dir = root / "logs_quantized_demo_randomized_motus_20260712_160612"
            old_log_dir.mkdir()
            previous_state = {"log_dir": str(old_log_dir)}

            resolver = getattr(run_quantized_eval_resume, "resolve_log_dirs", None)
            self.assertIsNotNone(resolver)

            candidate_log_dir, run_log_dir = resolver(
                root,
                previous_state,
                retry_oom_only=True,
                stamp="20260713_164000",
            )

            self.assertEqual(candidate_log_dir, old_log_dir)
            self.assertEqual(run_log_dir, root / "logs_quantized_oom_retry_20260713_164000")
            self.assertNotEqual(run_log_dir, old_log_dir)




    def test_current_attempt_oom_detection_ignores_stale_oom_before_latest_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "task.log"
            log_path.write_text(
                "===== 2026-07-13 16:00:00 | task=task | gpu=1 =====\n"
                "torch.OutOfMemoryError: CUDA out of memory\n"
                "\n===== 2026-07-13 20:00:00 | task=task | gpu=0 =====\n"
                "`torch_dtype` is deprecated! Use `dtype` instead!\n",
                encoding="utf-8",
            )
            status = {"status": "running", "log_path": str(log_path)}

            self.assertFalse(run_quantized_eval_resume.status_attempt_has_oom(status, log_path, "task"))

    def test_current_attempt_oom_detection_uses_log_start_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "task.log"
            stale = "torch.OutOfMemoryError: CUDA out of memory\n"
            current = "===== 2026-07-13 20:00:00 | task=task | gpu=0 =====\n"
            log_path.write_text(stale + current + "torch.OutOfMemoryError: CUDA out of memory\n", encoding="utf-8")
            status = {"status": "running", "log_path": str(log_path), "log_start_offset": len(stale)}

            self.assertTrue(run_quantized_eval_resume.status_attempt_has_oom(status, log_path, "task"))


if __name__ == "__main__":
    unittest.main()
