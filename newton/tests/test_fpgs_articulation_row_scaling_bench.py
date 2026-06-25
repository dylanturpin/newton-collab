# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp


def test_fpgs_articulation_row_scaling_benchmark_smoke(tmp_path):
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for the FPGS production scaling benchmark")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_row_scaling.py"
    out_dir = tmp_path / "fpgs_articulation_row_scaling"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "smoke",
            "--no-plots",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert (out_dir / "results.csv").is_file()
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    paths = {row["path"] for row in results}
    assert paths == {"mf_immediate", "compact_tree"}
    assert not (
        paths
        & {
            "new_free_fused",
            "new_articulated_fused",
            "mf_local_plus_propagation",
            "deferred_cholesky",
            "compact_cholesky",
        }
    )
    assert "SolverFeatherPGS.step" in summary
    assert "private generated-kernel" not in summary

    free_rows = [row for row in results if row["case_kind"] == "free_free"]
    articulated_rows = [row for row in results if row["case_kind"] == "articulated_free"]
    assert free_rows
    assert articulated_rows

    free_compact = next(row for row in free_rows if row["path"] == "compact_tree")
    assert free_compact["compact_active"] is False
    assert free_compact["propagation_extra_mib"] == 0.0
    assert free_compact["dense_rows_total"] == 0
    assert free_compact["mf_rows_total"] > 0
    assert free_compact["compact_rows_total"] == 0

    articulated_baseline = next(row for row in articulated_rows if row["path"] == "mf_immediate")
    articulated_compact = next(row for row in articulated_rows if row["path"] == "compact_tree")
    assert articulated_baseline["dense_rows_total"] > 0
    assert articulated_baseline["compact_rows_total"] == 0
    assert articulated_compact["compact_active"] is True
    assert articulated_compact["propagation_extra_mib"] > 0.0
    assert articulated_compact["dense_rows_total"] == 0
    assert articulated_compact["compact_rows_total"] > 0

    for row in results:
        assert np.isfinite(row["ms_per_step"])
        assert row["ms_per_step"] > 0.0
        assert np.isfinite(row["row_solver_mib"])
        assert row["row_solver_mib"] >= 0.0
        assert np.isfinite(row["propagation_extra_mib"])
        assert row["propagation_extra_mib"] >= 0.0
        if row["case_kind"] == "free_free":
            assert row["joint_qd_rel_l2"] is None
            assert row["state_linf"] is None
        else:
            assert np.isfinite(row["joint_qd_rel_l2"])
            assert np.isfinite(row["state_linf"])
            if row["path"] == "compact_tree":
                assert row["joint_qd_rel_l2"] < 2.0e-2
                assert row["state_linf"] < 2.0e-2
