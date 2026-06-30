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
    if articulated_baseline["joint_dof_count"] >= 64:
        assert articulated_compact["row_solver_mib"] < articulated_baseline["row_solver_mib"]

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
                assert row["joint_qd_rel_l2"] < 1.0e-1
                assert row["state_linf"] < 1.0e-1


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
    assert "operator_match" in summary
    for row in results:
        assert row["rows"] > 0
        assert row["D"] > 0
        assert row["sampled_sources"] > 0
        assert row["operator_status"] == "operator_match"
        assert np.isfinite(row["rel_fro"])
        assert np.isfinite(row["abs_linf"])
        assert np.isfinite(row["qd_rel_l2"])
        assert np.isfinite(row["qd_abs_linf"])
        assert row["rel_fro"] <= 1.0e-4
        assert row["abs_linf"] <= 1.0e-4
        assert row["mass_condition_est"] is None or np.isfinite(row["mass_condition_est"])
        assert row["worst_columns"]


def test_fpgs_compact_debug_residuals_include_compact_rows():
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for compact PGS debug diagnostics")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_row_scaling.py"
    spec = importlib.util.spec_from_file_location("fpgs_articulation_row_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    case = module.BenchCase(
        "debug_compact",
        "L=4,boxes=1",
        "articulated_free",
        articulations=1,
        links=4,
        contacts_per_articulation=1,
    )
    model, initial, contacts = module._prepare_case_run(case, "cuda:0")
    solver = module.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_dense_response_mode="compact_tree",
        hinv_jt_kernel="par_row",
        pgs_iterations=2,
        enable_contact_friction=True,
        dense_max_constraints=16,
        mf_max_constraints=16,
        compact_max_constraints=16,
        compact_fast_body_map=True,
        compact_existing_row_phases="auto",
        compact_warp_propagation=True,
        pgs_warmstart=False,
        mf_warmstart=False,
        pgs_debug=True,
    )

    state_in = model.state()
    state_out = model.state()
    module._restore_state(model, state_in, initial)
    module._restore_state(model, state_out, initial)
    state_in.clear_forces()
    state_out.clear_forces()
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    wp.synchronize()

    assert int(np.sum(solver.compact_constraint_count.numpy())) > 0
    assert solver._pgs_convergence_log
    assert solver._pgs_ncp_residual_log
    convergence = solver._pgs_convergence_log[-1]
    ncp = solver._pgs_ncp_residual_log[-1]
    assert convergence.shape == (2, 4)
    assert ncp.shape == (2, model.world_count, 6)
    assert np.all(np.isfinite(convergence))
    assert np.all(np.isfinite(ncp))
    assert np.max(convergence[:, 0]) > 0.0


def test_fpgs_articulation_accuracy_probe_smoke(tmp_path):
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for the FPGS accuracy probe")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_accuracy_probe.py"
    out_dir = tmp_path / "fpgs_articulation_accuracy_probe"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--links",
            "4",
            "--contacts-per-articulation",
            "1",
            "--iterations",
            "1,2",
            "--reference-iterations",
            "4",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    results = json.loads((out_dir / "accuracy_probe.json").read_text(encoding="utf-8"))
    assert "Contacts are generated once" in summary
    assert {row["path"] for row in results} == {"mf_immediate", "compact_tree"}
    assert {row["pgs_iterations"] for row in results} == {1, 2}

    compact_rows = [row for row in results if row["path"] == "compact_tree"]
    assert compact_rows
    for row in results:
        assert np.isfinite(row["final_r_compl"])
        assert np.isfinite(row["self_ref_joint_qd_abs_linf"])
        assert np.isfinite(row["mf_ref_joint_qd_abs_linf"])
        assert row["self_ref_joint_qd_abs_linf"] >= 0.0
        assert row["mf_ref_joint_qd_abs_linf"] >= 0.0
    for row in compact_rows:
        assert row["dense_rows_total"] == 0
        assert row["compact_rows_total"] > 0
        assert np.isfinite(row["same_iter_vs_mf_joint_qd_abs_linf"])
        assert row["same_iter_vs_mf_joint_qd_abs_linf"] >= 0.0


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
