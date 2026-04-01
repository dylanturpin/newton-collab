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

    def test_expand_plan_full_mode_expands_targets_deterministically(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")

        task_ids = [task["id"] for task in expanded["tasks"]]
        self.assertEqual(
            task_ids,
            [
                "g1_flat_sweep_rtx_a5000",
                "g1_flat_sweep_rtx_5090",
                "g1_flat_sweep_rtx_pro_6000_server",
                "g1_flat_sweep_b200",
                "g1_flat_ablation_rtx_a5000",
                "g1_flat_ablation_rtx_5090",
                "g1_flat_ablation_rtx_pro_6000_server",
                "g1_flat_ablation_b200",
                "g1_flat_renders",
                "h1_tabletop_sweep_early_rtx_a5000",
                "h1_tabletop_sweep_early_rtx_5090",
                "h1_tabletop_sweep_early_rtx_pro_6000_server",
                "h1_tabletop_sweep_early_b200",
                "h1_tabletop_sweep_late_rtx_pro_6000_server",
                "h1_tabletop_sweep_late_b200",
                "h1_tabletop_sweep_extreme_rtx_pro_6000_server",
                "h1_tabletop_sweep_extreme_b200",
                "h1_tabletop_ablation_rtx_a5000",
                "h1_tabletop_ablation_rtx_5090",
                "h1_tabletop_ablation_rtx_pro_6000_server",
                "h1_tabletop_ablation_b200",
                "h1_tabletop_renders",
            ],
        )

        g1_sweep_5090 = expanded["tasks"][1]
        self.assertEqual(g1_sweep_5090["job_count"], 28)
        self.assertEqual(g1_sweep_5090["jobs"][0]["id"], "g1_flat_sweep_rtx_5090__0001")
        self.assertEqual(g1_sweep_5090["jobs"][-1]["id"], "g1_flat_sweep_rtx_5090__0028")
        self.assertEqual(g1_sweep_5090["jobs"][-1]["num_worlds"], 65536)

        g1_sweep_6000 = expanded["tasks"][2]
        self.assertEqual(g1_sweep_6000["job_count"], 32)
        self.assertEqual(g1_sweep_6000["jobs"][-1]["num_worlds"], 131072)

        h1_late = expanded["tasks"][13]
        self.assertEqual(h1_late["job_count"], 4)
        self.assertEqual(h1_late["jobs"][0]["num_worlds"], 16384)

        h1_extreme = expanded["tasks"][15]
        self.assertEqual(h1_extreme["job_count"], 4)
        self.assertEqual(h1_extreme["jobs"][0]["num_worlds"], 32768)

        g1_render = expanded["tasks"][8]
        self.assertEqual(g1_render["profile"], "rtx_pro_6000_render")
        self.assertEqual(g1_render["job_count"], 4)

    def test_expand_plan_validation_mode_uses_validation_tasks(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="validation")

        task_ids = [task["id"] for task in expanded["tasks"]]
        self.assertEqual(task_ids, ["validation_g1_flat_sweep", "validation_g1_flat_render"])
        self.assertEqual(expanded["tasks"][0]["job_count"], 4)
        self.assertEqual(expanded["tasks"][1]["job_count"], 1)
        self.assertEqual(expanded["tasks"][0]["profile"], "rtx_5090")
        self.assertEqual(expanded["tasks"][1]["profile"], "rtx_pro_6000_render")

    def test_explicit_jobs_live_in_plan_and_carry_generic_metadata(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")
        ablation_task = next(
            task for task in expanded["tasks"] if task["id"] == "h1_tabletop_ablation_rtx_pro_6000_server"
        )

        self.assertEqual(ablation_task["job_style"], "explicit")
        self.assertEqual(ablation_task["job_count"], 10)
        self.assertEqual(ablation_task["tags"], ["h1_tabletop_ablation"])
        self.assertEqual(ablation_task["jobs"][7]["label"], "matrix-free")
        self.assertEqual(ablation_task["jobs"][7]["variant_id"], "matrix_free")
        self.assertEqual(ablation_task["jobs"][7]["step_index"], 7)
        self.assertFalse(ablation_task["jobs"][7]["double_buffer"])
        self.assertTrue(ablation_task["jobs"][8]["double_buffer"])
        self.assertTrue(ablation_task["jobs"][9]["solver"] == "mujoco")
        self.assertEqual(ablation_task["jobs"][9]["label"], "MuJoCo baseline")

    def test_target_overrides_apply_to_explicit_jobs(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full")
        ablation_task = next(task for task in expanded["tasks"] if task["id"] == "h1_tabletop_ablation_rtx_5090")

        self.assertTrue(all(job["num_worlds"] == 4096 for job in ablation_task["jobs"]))
        self.assertEqual(ablation_task["jobs"][0]["id"], "h1_tabletop_ablation_rtx_5090__baseline_loop")

    def test_can_select_expanded_concrete_task_id(self):
        loaded = plan.load_plan(plan.DEFAULT_PLAN_PATH, env={"USER": "plan-test-user"})
        expanded = plan.expand_plan(loaded, run_mode="full", selected_task_ids=["g1_flat_ablation_b200"])

        self.assertEqual([task["id"] for task in expanded["tasks"]], ["g1_flat_ablation_b200"])
        self.assertEqual(expanded["tasks"][0]["job_count"], 8)

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
                solver: fpgs_split
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
