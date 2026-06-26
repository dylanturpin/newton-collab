# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
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
    assert "## free_free_dof" in summary
    assert "## articulated_contact_rows" in summary

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
                # Compact-tree uses deferred propagation, so high-contact smoke
                # rows are not expected to be bit-close to immediate D-wide GS
                # after only two iterations. Keep this as a gross sanity bound.
                assert row["joint_qd_rel_l2"] < 2.0e-1
                assert row["state_linf"] < 2.0e-1


def test_fpgs_articulation_operator_diagnostic_smoke(tmp_path):
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for the FPGS operator diagnostic")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_operator_diagnostic.py"
    out_dir = tmp_path / "fpgs_articulation_operator_diagnostic"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--links",
            "2",
            "--contacts-per-articulation",
            "1",
            "--max-sources",
            "3",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    results = json.loads((out_dir / "operator_diagnostic.json").read_text(encoding="utf-8"))
    assert results
    assert "row operator `J_world @ Y_world.T`" in summary
    assert "operator_match" in summary or "operator_mismatch" in summary
    for row in results:
        assert row["rows"] > 0
        assert row["D"] > 0
        assert row["sampled_sources"] > 0
        assert row["operator_status"] in {"operator_match", "operator_mismatch"}
        assert np.isfinite(row["rel_fro"])
        assert np.isfinite(row["abs_linf"])
        assert np.isfinite(row["qd_rel_l2"])
        assert np.isfinite(row["qd_abs_linf"])
        assert row["mass_condition_est"] is None or np.isfinite(row["mass_condition_est"])
        assert row["worst_columns"]


def test_fpgs_articulation_row_scaling_stress_case_coverage():
    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_row_scaling.py"
    spec = importlib.util.spec_from_file_location("fpgs_articulation_row_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    cases = module._cases_for_preset("stress")

    free_free = [case for case in cases if case.sweep == "free_free_dof"]
    articulated_contacts = [case for case in cases if case.sweep == "articulated_contact_rows"]
    chain_depth = [case for case in cases if case.sweep == "chain_depth"]
    articulation_count = [case for case in cases if case.sweep == "articulation_count"]

    assert [case.free_pairs for case in free_free] == [1, 2, 4, 8, 16, 21]
    assert all(case.case_kind == "free_free" for case in free_free)

    assert [(case.links, case.contacts_per_articulation) for case in articulated_contacts] == [
        (4, 4),
        (16, 8),
        (16, 16),
        (64, 8),
        (64, 32),
        (64, 64),
    ]
    assert all(case.case_kind == "articulated_free" for case in articulated_contacts)

    assert [case.links for case in chain_depth] == [2, 16, 32, 64]
    assert [case.contacts_per_articulation for case in chain_depth] == [2, 16, 32, 64]
    assert [case.articulations for case in articulation_count] == [1, 2, 16, 32, 64, 128]
