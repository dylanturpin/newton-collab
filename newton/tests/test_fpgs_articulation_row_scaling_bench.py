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

import newton
from newton.solvers import SolverFeatherPGS


def _build_d6_articulated_free_contact_model(device: str):
    builder = newton.ModelBuilder(gravity=0.0)
    cfg = newton.ModelBuilder.JointDofConfig

    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, hx=0.12, hy=0.08, hz=0.06)
    joint = builder.add_joint_d6(
        parent=-1,
        child=link,
        linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        angular_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint])

    cube = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.588), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(cube, hx=0.05, hy=0.05, hz=0.04)
    return builder.finalize(device=device)


def _step_d6_contact_case(model, contacts, mode: str) -> tuple[np.ndarray, tuple[int, int, int], int | None]:
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
    state_in.clear_forces()
    state_out.clear_forces()

    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=mode,
        hinv_jt_kernel="par_row",
        pgs_iterations=2,
        pgs_velocity_iterations=0,
        enable_contact_friction=True,
        dense_max_constraints=32,
        mf_max_constraints=32,
        pgs_warmstart=False,
        mf_warmstart=False,
    )
    solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)
    wp.synchronize()

    propagation_count = getattr(solver, "propagation_constraint_count", None)
    rows = (
        int(np.sum(solver.constraint_count.numpy())),
        int(np.sum(solver.mf_constraint_count.numpy())),
        int(np.sum(propagation_count.numpy())) if propagation_count is not None else 0,
    )
    return state_out.joint_qd.numpy().copy(), rows, getattr(solver, "_propagation_full_fused_size", None)


def test_fpgs_propagation_fused_supports_multi_dof_contact_rows():
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for propagation-fused multi-DOF contact rows")

    model = _build_d6_articulated_free_contact_model("cuda:0")
    assert model.joint_dof_dim.numpy().tolist() == [[3, 3], [3, 3]]

    initial = model.state()
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    contacts = model.contacts()
    model.collide(initial, contacts)
    wp.synchronize()
    assert int(contacts.rigid_contact_count.numpy()[0]) > 0

    qd_immediate, rows_immediate, fused_immediate = _step_d6_contact_case(model, contacts, "immediate")
    qd_propagation, rows_propagation, fused_propagation = _step_d6_contact_case(model, contacts, "propagation")
    qd_fused, rows_fused, fused_size = _step_d6_contact_case(model, contacts, "propagation-fused")

    assert rows_immediate[0] > 0
    assert rows_immediate[2] == 0
    assert fused_immediate is None

    assert rows_propagation[0] == 0
    assert rows_propagation[2] > 0
    assert fused_propagation is None

    assert rows_fused[0] == 0
    assert rows_fused[2] == rows_propagation[2]
    assert fused_size == 6

    np.testing.assert_allclose(qd_propagation, qd_immediate, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(qd_fused, qd_immediate, rtol=1.0e-5, atol=1.0e-5)


def test_fpgs_propagation_mixed_size_free_body_response():
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for propagation articulated/free contact rows")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_row_scaling.py"
    spec = importlib.util.spec_from_file_location("fpgs_articulation_row_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    case = module.BenchCase(
        "mixed_size_response",
        "L=4,boxes=1",
        "articulated_free",
        articulations=1,
        links=4,
        contacts_per_articulation=1,
    )
    model, initial, contacts = module._prepare_case_run(case, "cuda:0")

    state_in = model.state()
    state_out = model.state()
    module._restore_state(model, state_in, initial)
    module._restore_state(model, state_out, initial)
    state_in.clear_forces()
    state_out.clear_forces()

    solver = module.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response="propagation",
        hinv_jt_kernel="par_row",
        pgs_iterations=2,
        pgs_velocity_iterations=0,
        enable_contact_friction=True,
        dense_max_constraints=32,
        mf_max_constraints=16,
        pgs_warmstart=False,
        mf_warmstart=False,
    )
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
    wp.synchronize()

    row_count = int(solver.propagation_constraint_count.numpy()[0])
    assert row_count > 0

    body_to_art = solver.body_to_articulation.numpy()
    is_free = solver.is_free_rigid.numpy()
    body_b = solver.propagation_body_b.numpy()[0, :row_count]
    free_bodies = [int(body) for body in body_b if body >= 0 and is_free[body_to_art[int(body)]] != 0]
    assert free_bodies

    response = solver.propagation_body_response.numpy()
    mijt_b = solver.propagation_MiJt_b.numpy()[0, :row_count]
    assert max(float(np.linalg.norm(response[body])) for body in free_bodies) > 0.0
    assert float(np.linalg.norm(mijt_b)) > 0.0


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
    assert paths == {"mf_immediate", "propagation", "propagation-fused"}
    assert not (
        paths
        & {
            "new_free_fused",
            "new_articulated_fused",
            "mf_local_plus_propagation",
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

    free_propagation = next(row for row in free_rows if row["path"] == "propagation")
    assert free_propagation["propagation_active"] is False
    assert free_propagation["propagation_extra_mib"] == 0.0
    assert free_propagation["dense_rows_total"] == 0
    assert free_propagation["mf_rows_total"] > 0
    assert free_propagation["propagation_rows_total"] == 0

    articulated_baseline = next(row for row in articulated_rows if row["path"] == "mf_immediate")
    articulated_propagation = next(row for row in articulated_rows if row["path"] == "propagation")
    assert articulated_baseline["dense_rows_total"] > 0
    assert articulated_baseline["propagation_rows_total"] == 0
    assert articulated_propagation["propagation_active"] is True
    assert articulated_propagation["propagation_extra_mib"] > 0.0
    assert articulated_propagation["dense_rows_total"] == 0
    assert articulated_propagation["propagation_rows_total"] > 0
    if articulated_baseline["joint_dof_count"] >= 64:
        assert articulated_propagation["row_solver_mib"] < articulated_baseline["row_solver_mib"]

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
            if row["path"] == "propagation":
                # Propagation uses deferred articulation response, so high-contact smoke
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


def test_fpgs_propagation_debug_residuals_include_propagation_rows():
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        pytest.skip("CUDA is required for propagation PGS debug diagnostics")

    script = Path(__file__).parent / "benchmarks" / "fpgs_articulation_row_scaling.py"
    spec = importlib.util.spec_from_file_location("fpgs_articulation_row_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    case = module.BenchCase(
        "debug_propagation",
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
        articulated_contact_response="propagation",
        hinv_jt_kernel="par_row",
        pgs_iterations=2,
        enable_contact_friction=True,
        dense_max_constraints=16,
        mf_max_constraints=16,
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

    assert int(np.sum(solver.propagation_constraint_count.numpy())) > 0
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
    assert {row["path"] for row in results} == {"mf_immediate", "propagation"}
    assert {row["pgs_iterations"] for row in results} == {1, 2}

    propagation_rows = [row for row in results if row["path"] == "propagation"]
    assert propagation_rows
    for row in results:
        assert np.isfinite(row["final_r_compl"])
        assert np.isfinite(row["self_ref_joint_qd_abs_linf"])
        assert np.isfinite(row["mf_ref_joint_qd_abs_linf"])
        assert row["self_ref_joint_qd_abs_linf"] >= 0.0
        assert row["mf_ref_joint_qd_abs_linf"] >= 0.0
    for row in propagation_rows:
        assert row["dense_rows_total"] == 0
        assert row["propagation_rows_total"] > 0
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
