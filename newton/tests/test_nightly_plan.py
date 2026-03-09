# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path


def _load_plan_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "nightly" / "plan.py"
    spec = importlib.util.spec_from_file_location("nightly_plan", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = _load_plan_module()


class TestNightlyPlan(unittest.TestCase):
    def test_load_plan_expands_env_backed_cache_paths(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})

        cache_env = loaded["defaults"]["cache_env"]
        self.assertEqual(cache_env["UV_CACHE_DIR"], "/tmp/plan-test-user/uv-cache")
        self.assertEqual(cache_env["UV_PROJECT_ENVIRONMENT"], "/tmp/plan-test-user/newton-venv")
        self.assertEqual(cache_env["WARP_CACHE_PATH"], "/tmp/plan-test-user/warp-cache")
        self.assertEqual(cache_env["NEWTON_CACHE_PATH"], "/tmp/plan-test-user/newton-cache")
        self.assertEqual(cache_env["CUDA_CACHE_PATH"], "/tmp/plan-test-user/cuda-compute-cache")

    def test_expand_plan_full_mode_is_deterministic(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")

        task_ids = [task["id"] for task in expanded["tasks"]]
        self.assertEqual(
            task_ids,
            [
                "g1_flat_sweep",
                "g1_flat_ablation",
                "g1_flat_renders",
                "h1_tabletop_sweep_early",
                "h1_tabletop_sweep_late",
                "h1_tabletop_sweep_extreme",
                "h1_tabletop_ablation",
                "h1_tabletop_renders",
            ],
        )

        g1_sweep = expanded["tasks"][0]
        self.assertEqual(g1_sweep["job_count"], 32)
        self.assertEqual(g1_sweep["jobs"][0]["id"], "g1_flat_sweep__0001")
        self.assertEqual(g1_sweep["jobs"][0]["solver"], "fpgs_tiled")
        self.assertEqual(g1_sweep["jobs"][0]["substeps"], 2)
        self.assertEqual(g1_sweep["jobs"][0]["num_worlds"], 1024)
        self.assertEqual(g1_sweep["jobs"][-1]["id"], "g1_flat_sweep__0032")
        self.assertEqual(g1_sweep["jobs"][-1]["solver"], "mujoco")
        self.assertEqual(g1_sweep["jobs"][-1]["substeps"], 4)
        self.assertEqual(g1_sweep["jobs"][-1]["num_worlds"], 131072)
        self.assertEqual(expanded["tasks"][2]["profile"], "rtx_pro_6000_render")
        self.assertTrue(expanded["tasks"][1]["jobs"][0]["nsys_profile"])
        self.assertTrue(expanded["tasks"][6]["jobs"][0]["nsys_profile"])
        h1_early = expanded["tasks"][3]
        h1_late = expanded["tasks"][4]
        h1_extreme = expanded["tasks"][5]
        self.assertEqual(h1_early["job_count"], 16)
        self.assertEqual(h1_early["jobs"][0]["id"], "h1_tabletop_sweep_early__0001")
        self.assertEqual(h1_early["jobs"][-1]["id"], "h1_tabletop_sweep_early__0016")
        self.assertEqual(h1_late["job_count"], 4)
        self.assertEqual(h1_late["jobs"][0]["num_worlds"], 16384)
        self.assertEqual(h1_extreme["job_count"], 4)
        self.assertEqual(h1_extreme["jobs"][0]["num_worlds"], 32768)
        self.assertEqual(expanded["tasks"][-1]["profile"], "rtx_pro_6000_render")

    def test_expand_plan_validation_mode_uses_validation_tasks(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="validation")

        task_ids = [task["id"] for task in expanded["tasks"]]
        self.assertEqual(task_ids, ["validation_g1_flat_sweep", "validation_g1_flat_render"])
        self.assertEqual(expanded["tasks"][0]["job_count"], 4)
        self.assertEqual(expanded["tasks"][1]["job_count"], 1)
        self.assertEqual(expanded["tasks"][0]["profile"], "rtx_5090")
        self.assertEqual(expanded["tasks"][1]["profile"], "rtx_pro_6000_render")

    def test_ablation_expansion_adds_mujoco_baseline(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")
        ablation_task = next(task for task in expanded["tasks"] if task["id"] == "h1_tabletop_ablation")

        self.assertEqual(ablation_task["ablation_sequence"], "streaming")
        self.assertEqual(len(ablation_task["jobs"]), len(plan.ABLATION_SEQUENCES["streaming"]) + 1)
        self.assertEqual(ablation_task["jobs"][-4]["ablation_label"], "fully matrix-free GS")
        self.assertEqual(ablation_task["jobs"][-4]["solver"], "feather_pgs")
        self.assertFalse(ablation_task["jobs"][-4]["double_buffer"])
        self.assertFalse(ablation_task["jobs"][-4]["pipeline_collide"])
        self.assertEqual(ablation_task["jobs"][-3]["ablation_label"], "+ double buffer")
        self.assertTrue(ablation_task["jobs"][-3]["double_buffer"])
        self.assertFalse(ablation_task["jobs"][-3]["pipeline_collide"])
        self.assertEqual(ablation_task["jobs"][-2]["ablation_label"], "+ pipeline collide")
        self.assertTrue(ablation_task["jobs"][-2]["double_buffer"])
        self.assertTrue(ablation_task["jobs"][-2]["pipeline_collide"])
        self.assertEqual(ablation_task["jobs"][-1]["solver"], "mujoco")
        self.assertEqual(ablation_task["jobs"][-1]["ablation_label"], "MuJoCo baseline")

    def test_g1_ablation_sequence_ends_with_pipeline_collide_before_mujoco(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")
        ablation_task = next(task for task in expanded["tasks"] if task["id"] == "g1_flat_ablation")

        self.assertEqual(ablation_task["jobs"][-3]["ablation_label"], "+ parallel streams")
        self.assertTrue(ablation_task["jobs"][-3]["use_parallel_streams"])
        self.assertFalse(ablation_task["jobs"][-3]["double_buffer"])
        self.assertFalse(ablation_task["jobs"][-3]["pipeline_collide"])
        self.assertEqual(ablation_task["jobs"][-2]["ablation_label"], "+ pipeline collide")
        self.assertFalse(ablation_task["jobs"][-2]["double_buffer"])
        self.assertTrue(ablation_task["jobs"][-2]["pipeline_collide"])
        self.assertEqual(ablation_task["jobs"][-1]["ablation_label"], "MuJoCo baseline")

    def test_invalid_list_field_is_rejected(self):
        invalid_plan_text = textwrap.dedent(
            """
            version: 1
            defaults:
              shared_state_dir: /tmp/state
              work_base_dir: /tmp/work
              cache_env:
                TMPDIR: /tmp
                UV_CACHE_DIR: /tmp/cache
                UV_PROJECT_ENVIRONMENT: /tmp/env
                WARP_CACHE_PATH: /tmp/warp
                NEWTON_CACHE_PATH: /tmp/newton
                CUDA_CACHE_PATH: /tmp/cuda
            hardware_profiles:
              test_gpu:
                label: test-gpu
            tasks:
              - id: invalid_task
                kind: benchmark
                profile: test_gpu
                series: invalid
                scenario: g1_flat
                solver: fpgs_tiled
                num_worlds: 256
                measure_frames: [8, 16]
            """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            plan_path = Path(tmp_dir) / "invalid.yaml"
            plan_path.write_text(invalid_plan_text, encoding="utf-8")
            loaded = plan.load_plan(plan_path)
            with self.assertRaises(plan.PlanValidationError):
                plan.validate_plan(loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
