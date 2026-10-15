# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction patches, anchors, directions, and torsional and rolling friction of SolverFeatherPGS."""

import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.contact_angular_friction import angular_friction_gs_sources
from newton._src.solvers.feather_pgs.friction_patches import (
    _pose_motion,
    finish_patch_impulses,
    link_patch_rows,
    seed_patch_impulses,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION,
    PGS_CONSTRAINT_TYPE_FRICTION,
    compute_mf_effective_mass_and_rhs,
    compute_mf_rhs_bias,
    compute_propagation_effective_mass_and_rhs,
    compute_propagation_rhs_bias,
    compute_world_contact_bias,
    pgs_solve_loop,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel
from newton.solvers import SolverBase, SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import _patch_fixture, fixture, run_fixture, set_torsion_radius

_MU_JAW, _MU_BOX = 5.0, 0.5


_BOX_HALF, _JAW_HALF_T, _GAP0 = 0.02, 0.005, 0.001


def _build_v_jaws(tilt_deg: float, geometry: str = "box"):
    """Fixed base, two prismatic jaws (left driven, right slaved through a mimic row) pinching a
    free 0.1 kg box. Both jaw faces tilt toward +z by ``tilt_deg`` so the two normal rows'
    depenetration biases share a +z tangential component: the geometry that leaks tangential
    drift through velocity-only friction rows (a 2F-85 pad pair has ~4 deg of such tilt)."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    base = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="base")
    b.add_shape_box(
        base,
        hx=0.05,
        hy=0.05,
        hz=0.005,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, has_shape_collision=False),
    )
    root = b.add_joint_fixed(-1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    jaw_cfg = newton.ModelBuilder.ShapeConfig(density=2000.0, mu=_MU_JAW)
    x0 = _BOX_HALF + _JAW_HALF_T + _GAP0
    jaws, joints = [], []
    for name, sign in (("jaw_L", -1.0), ("jaw_R", 1.0)):
        body = b.add_link(xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.5), wp.quat_identity()), label=name)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(-sign * np.radians(tilt_deg)))
        b.add_shape_box(body, hx=_JAW_HALF_T, hy=0.02, hz=0.02, xform=wp.transform(wp.vec3(0.0), rot), cfg=jaw_cfg)
        driven = sign < 0
        joints.append(
            b.add_joint_prismatic(
                base,
                body,
                parent_xform=wp.transform(wp.vec3(sign * x0, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.X,
                limit_lower=-0.05,
                limit_upper=0.05,
                target_pos=0.01 if driven else 0.0,
                target_ke=100.0 if driven else 0.0,
                target_kd=10.0 if driven else 0.0,
            )
        )
        jaws.append(body)
    b.add_articulation([root, *joints], label="gripper")
    b.set_joint_mimic(joints[1], joints[0], coeffs=(0.0, -1.0))
    for dof in range(len(b.joint_effort_limit)):
        b.joint_effort_limit[dof] = 10.0
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="box")
    if geometry == "sphere":
        b.add_shape_sphere(
            box,
            radius=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX),
        )
    elif geometry == "capsule":
        b.add_shape_capsule(
            box,
            radius=_BOX_HALF,
            half_height=_BOX_HALF,
            xform=wp.transform(wp.vec3(0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)),
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.1 / (np.pi * _BOX_HALF**2 * 2 * _BOX_HALF + 4.0 / 3.0 * np.pi * _BOX_HALF**3), mu=_MU_BOX
            ),
        )
    else:
        b.add_shape_box(
            box,
            hx=_BOX_HALF,
            hy=_BOX_HALF,
            hz=_BOX_HALF,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (2 * _BOX_HALF) ** 3, mu=_MU_BOX),
        )
    return b.finalize(), jaws, box


_SQUEEZE_SOLVER = {
    "solve_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "iterations": 64,
    "velocity_iterations": 0,
    "baumgarte": 0.05,
    "contact_regularization": 0.01,
    "contact_friction_gap_max": 0.001,
    "contact_friction_position_iterations": 4,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "enable_bilateral_preelimination": True,
}


def _run_squeeze(
    tilt_deg: float, steps: int, dt: float = 0.005, matching: str = "latest", substeps: int = 1, **solver_kwargs
):
    """Return the box's z drift relative to the jaws (m, positive = up) and the solver."""
    model, jaws, box = _build_v_jaws(tilt_deg)
    kwargs = dict(_SQUEEZE_SOLVER)
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn", contact_matching=matching)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    rel = []
    for _ in range(steps):
        pipeline.collide(s0, contacts)  # once per environment step; substeps reuse the buffer
        for _sub in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt / substeps)
            s0, s1 = s1, s0
        bq = s0.body_q.numpy()
        rel.append(bq[box][2] - bq[jaws[0]][2])
    rel = np.asarray(rel)
    settle = int(0.5 / dt)  # skip the free fall before the pinch closes
    return rel[-1] - rel[settle], solver, s0


def _build_incline(theta_deg: float, mu: float):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.rigid_gap = 0.005
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.radians(theta_deg)))
    # static slab as the incline (shape on the world body)
    b.add_shape_box(
        -1,
        hx=2.0,
        hy=0.5,
        hz=0.05,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), rot),
        cfg=newton.ModelBuilder.ShapeConfig(mu=mu),
    )
    n = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))  # slab normal
    box = b.add_body(xform=wp.transform(n * (0.05 + 0.0002), rot), label="box")
    b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=mu))
    model = b.finalize()
    return model, box, rot


def _run_incline(steps: int, dt: float, **solver_kwargs):
    theta, mu = 30.0, 0.3
    model, box, _rot = _build_incline(theta, mu)
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=32, baumgarte=0.05, **solver_kwargs
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", contact_matching="latest")
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    v = s0.body_qd.numpy()[box][:3]  # Newton body_qd = [linear, angular]
    down_slope = np.array([np.cos(np.radians(theta)), 0.0, -np.sin(np.radians(theta))])  # slab tilted about +y by theta
    return float(np.dot(v, down_slope)), float(np.linalg.norm(v - np.dot(v, down_slope) * down_slope)), solver, contacts


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSFrictionAnchors(unittest.TestCase):
    def setUp(self):
        # The shared-anchor squeeze fixture intentionally combines contact_shared_anchor with patch friction.
        filters = warnings.catch_warnings()
        filters.__enter__()
        self.addCleanup(filters.__exit__, None, None, None)
        warnings.filterwarnings(
            "ignore", message=r"Patch friction selects its own friction locations", category=UserWarning
        )
        # The per-route row-builder check runs propagation routes with the default pre-elimination.
        warnings.filterwarnings(
            "ignore",
            message=r"SolverFeatherPGS: bilateral pre-elimination does not support propagation",
            category=UserWarning,
        )

    def test_explicit_opt_out_keeps_friction_rows_velocity_only(self):
        """Preserve velocity-only rows and omit anchor state with ``friction_anchor_beta=0``."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, **_SQUEEZE_SOLVER, friction_anchor_beta=0.0)
        self.assertFalse(solver._friction_anchors_enabled)
        self.assertFalse(hasattr(solver._friction_patches, "current"))
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")  # no matching needed
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _ in range(40):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 0.005)
            s0, s1 = s1, s0
        self.assertEqual(float(np.abs(solver._friction_patches.view.phi.numpy()).max()), 0.0)
        row_type = solver._row_type.numpy()
        friction = row_type == PGS_CONSTRAINT_TYPE_FRICTION
        self.assertGreater(int(friction.sum()), 0)
        self.assertEqual(float(np.abs(solver._phi.numpy()[friction]).max()), 0.0)
        self.assertEqual(float(np.abs(solver._row_beta.numpy()[friction]).max()), 0.0)

    def test_anchors_do_not_require_contact_matching(self):
        """Keep patch history owned by the solver, independently of collision matching."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, **_SQUEEZE_SOLVER, friction_anchor_beta=0.2)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, model.state(), model.control(), contacts, 0.005)

    def test_anchors_stop_tangential_drift_of_a_held_box(self):
        """Bound tangential drift in a tilted pinch while leaving flat-jaw behavior unaffected."""
        steps = int(3.0 / 0.005)
        drift_off, _, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.0)
        drift_on, solver_on, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.05)
        drift_on_02, _, _ = _run_squeeze(5.0, steps, friction_anchor_beta=0.2)
        self.assertGreater(
            abs(drift_off), 1.5e-4, f"tilted pinch should leak drift without anchors, got {drift_off:.2e} m"
        )
        self.assertLess(
            abs(drift_on),
            0.3 * abs(drift_off),
            f"anchored pinch (beta 0.05) drifted {drift_on:.2e} m vs {drift_off:.2e} m",
        )
        self.assertLess(abs(drift_on_02), 5.0e-5, f"anchored pinch (beta 0.2) drifted {drift_on_02:.2e} m")
        # the held contacts are anchored (not sliding) at the end of the hold
        valid = solver_on._friction_patches.current.valid.numpy()
        self.assertGreater(int(valid.sum()), 0)
        flat_off, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.0)
        # sticky matching replays body-local witness points; anchors must still hold
        drift_sticky, _, _ = _run_squeeze(5.0, steps, matching="sticky", friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sticky), 1.0e-4, f"anchored pinch (sticky matching) drifted {drift_sticky:.2e} m")
        # two solver substeps on one collide per environment step (Isaac Lab's loop): anchors
        # must map 1:1 on the reused buffer, not through the stale match index
        drift_sub, _, _ = _run_squeeze(5.0, steps, substeps=2, friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sub), 1.0e-4, f"anchored pinch with 2 substeps per collide drifted {drift_sub:.2e} m")
        flat_on, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.05)
        self.assertLess(abs(flat_on), 1.0e-4)
        self.assertLess(abs(flat_off), 1.0e-4)

    def test_anchors_hold_a_pinched_box_at_a_coarse_timestep(self):
        """Hold the tilted pinch at 60 Hz without re-anchoring below the Baumgarte equilibrium separation."""
        dt = 1.0 / 60.0
        drift, solver, _ = _run_squeeze(5.0, int(3.0 / dt), dt=dt, friction_anchor_beta=0.05)
        self.assertLess(abs(drift), 1.0e-3, f"anchored pinch at 60 Hz drifted {drift:.2e} m")
        sources = solver._friction_patches.current.source.numpy()
        self.assertGreater(int((sources >= 0).sum()), 0, "anchors were re-created instead of carried")

    def test_anchors_do_not_oppose_genuine_sliding(self):
        """Release saturated anchors on a 30-degree incline and preserve sliding acceleration."""
        dt, steps = 0.005, 200
        v_off, _lat_off, _, _ = _run_incline(steps, dt, friction_anchor_beta=0.0)
        v_on, lat_on, solver, contacts = _run_incline(steps, dt, friction_anchor_beta=0.2)
        g, th, mu = 9.81, np.radians(30.0), 0.3
        v_ref = g * (np.sin(th) - mu * np.cos(th)) * steps * dt
        self.assertGreater(v_off, 0.5 * v_ref)
        self.assertAlmostEqual(v_on, v_off, delta=0.05 * v_ref)
        self.assertLess(lat_on, 1.0e-3)
        # every loaded contact slid this step, so its anchor was dropped
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n > 0:
            self.assertEqual(int(solver._friction_patches.current.valid.numpy()[:n].sum()), 0)

    def test_row_builders_store_anchor_separation_per_route(self):
        """Store raw separation in dense rows and gain-scaled separation in the other routes."""
        beta = 0.3

        def check(model, solver_kwargs, matching="latest", steps=120):
            solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=beta, **solver_kwargs)
            pipeline = newton.CollisionPipeline(
                model, rigid_contact_max=256, broad_phase="nxn", contact_matching=matching
            )
            contacts = pipeline.contacts()
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            for _ in range(steps):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, 0.005)
                s0, s1 = s1, s0
            n = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(n, 0)
            fa = solver._friction_patches.view.phi.numpy()[:n]
            slot = solver._contact_slot.numpy()[:n]
            path = solver._contact_path.numpy()[:n]
            checked = {0: 0, 1: 0, 2: 0}
            for c in range(n):
                if slot[c] < 0 or path[c] < 0:
                    continue
                if path[c] == 0:
                    rt, ph, rb = solver._row_type.numpy()[0], solver._phi.numpy()[0], solver._row_beta.numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], fa[c], rtol=1.0e-6, atol=1.0e-9)
                    np.testing.assert_allclose(rb[slot[c] + 1 : slot[c] + 3], beta)
                else:
                    ph = (solver._mf_phi if path[c] == 1 else solver._propagation_phi).numpy()[0]
                    rt = (solver._mf_row_type if path[c] == 1 else solver._propagation_row_type).numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], beta * fa[c], rtol=1.0e-5, atol=1.0e-9)
                checked[int(path[c])] += 1
            return checked

        # dense route: articulated jaws vs free box under immediate response
        model, _, _ = _build_v_jaws(5.0)
        c_dense = check(model, _SQUEEZE_SOLVER)
        self.assertGreater(c_dense[0], 0)
        # propagation route: same scene, contacts routed to propagation rows
        model, _, _ = _build_v_jaws(5.0)
        prop_kwargs = dict(_SQUEEZE_SOLVER, articulated_contact_response="propagation", contact_regularization=0.0)
        c_prop = check(model, prop_kwargs)
        self.assertGreater(c_prop[2], 0)
        # matrix-free route: free box resting on a static slab
        model, _, _ = _build_incline(0.0, 0.5)
        c_mf = check(model, {"solve_mode": "matrix_free", "iterations": 16, "baumgarte": 0.05})
        self.assertGreater(c_mf[1], 0)

    def test_graph_capture_replays(self):
        """Capture and replay two steps with device-side anchor history under a CUDA graph."""
        model, jaws, box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.05, **_SQUEEZE_SOLVER)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn", contact_matching="latest")
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

        def step():
            nonlocal s0, s1
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 0.005)
            s0, s1 = s1, s0

        for _ in range(4):
            step()
        with wp.ScopedCapture(device=model.device) as capture:
            step()
            step()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        bq = s0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(bq)))
        self.assertLess(abs(bq[box][2] - bq[jaws[0]][2]), 0.01)


def _rhs_for_family(family: str, *, phi, row_beta, baumgarte, dt, bias_scale, device="cpu"):
    """RHS of one world with rows [CONTACT, FRICTION, FRICTION] for one RHS kernel family.

    ``phi`` holds what the row builders store: raw separation for dense rows (paired with
    ``row_beta``), gain-premultiplied separation for matrix-free and propagation rows.
    """
    contact, friction = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
    count = wp.array([3], dtype=wp.int32, device=device)
    row_type = wp.array([[contact, friction, friction]], dtype=wp.int32, device=device)
    phi_arr = wp.array([list(phi)], dtype=wp.float32, device=device)
    zeros3 = lambda: wp.zeros((1, 3), dtype=wp.float32, device=device)  # noqa: E731
    neg1 = wp.full((1, 3), -1, dtype=wp.int32, device=device)
    zJ = lambda: wp.zeros((1, 3, 6), dtype=wp.float32, device=device)  # noqa: E731
    inf = wp.array([float("inf")], dtype=wp.float32, device=device)
    rhs = zeros3()
    if family == "dense":
        wp.launch(
            compute_world_contact_bias,
            dim=1,
            inputs=[
                count,
                phi_arr,
                wp.array([list(row_beta)], dtype=wp.float32, device=device),
                row_type,
                zeros3(),
                dt,
                bias_scale,
                1.0,
                1.0,
                1.0,
            ],
            outputs=[rhs, zeros3()],
            device=device,
        )
    elif family == "mf_setup":
        wp.launch(
            compute_mf_effective_mass_and_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1,), dtype=wp.spatial_matrix, device=device),
                phi_arr,
                row_type,
                zeros3(),
                zeros3(),
                0,
                wp.array([-1], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                inf,
                1.0e-6,
                baumgarte,
                1.0,
                dt,
                1.0,
                0.5,
                3,
            ],
            outputs=[zeros3(), zJ(), zJ(), rhs, zeros3()],
            device=device,
        )
    elif family == "mf_velocity":
        wp.launch(
            compute_mf_rhs_bias,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1, 1), dtype=wp.int32, device=device),
                phi_arr,
                row_type,
                zeros3(),
                zeros3(),
                0,
                dt,
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                0.5,
                3,
            ],
            outputs=[rhs],
            device=device,
        )
    elif family == "propagation_setup":
        wp.launch(
            compute_propagation_effective_mass_and_rhs,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                wp.zeros((1, 6, 6), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.ones((1,), dtype=wp.int32, device=device),
                phi_arr,
                row_type,
                zeros3(),
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                inf,
                1.0e-6,
                baumgarte,
                1.0,
                dt,
                1.0,
                0.5,
                3,
            ],
            outputs=[zeros3(), zJ(), zJ(), rhs, zeros3(), zeros3()],
            device=device,
        )
    elif family == "propagation_velocity":
        wp.launch(
            compute_propagation_rhs_bias,
            dim=3,
            inputs=[
                count,
                neg1,
                neg1,
                zJ(),
                zJ(),
                phi_arr,
                row_type,
                zeros3(),
                dt,
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                3,
            ],
            outputs=[rhs],
            device=device,
        )
    else:
        raise ValueError(family)
    return rhs.numpy()[0]


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    return builder.finalize(device=device)


class TestFeatherPGSFrictionAnchorKernels(unittest.TestCase):
    """Device-agnostic checks of the anchor bookkeeping (run on CPU)."""

    def test_rhs_bias_matches_on_every_row_family_and_vanishes_in_velocity_pass(self):
        """Apply equal positional bias on all row families and drop it in velocity-only passes."""
        baumgarte, fa_beta, dt = 0.05, 0.3, 0.005
        phi_n, e0, e1 = -2.0e-3, 1.0e-4, -2.5e-4
        expect_pos = np.array([baumgarte * phi_n / dt, fa_beta * e0 / dt, fa_beta * e1 / dt], dtype=np.float32)
        dense = lambda s: _rhs_for_family(  # noqa: E731
            "dense",
            phi=[phi_n, e0, e1],
            row_beta=[baumgarte, fa_beta, fa_beta],
            baumgarte=baumgarte,
            dt=dt,
            bias_scale=s,
        )
        pre = [phi_n, fa_beta * e0, fa_beta * e1]
        np.testing.assert_allclose(dense(1.0), expect_pos, rtol=1.0e-5)
        np.testing.assert_allclose(dense(0.0), 0.0, atol=1.0e-9)
        for family in ("mf_setup", "propagation_setup"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, baumgarte=baumgarte, dt=dt, bias_scale=1.0)
            np.testing.assert_allclose(got, expect_pos, rtol=1.0e-5, err_msg=family)
        for family in ("mf_velocity", "propagation_velocity"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, baumgarte=baumgarte, dt=dt, bias_scale=None)
            np.testing.assert_allclose(got, 0.0, atol=1.0e-9, err_msg=family)

    def test_reset_clears_anchor_history_full_and_masked(self):
        """Drop carried anchors of selected worlds on reset, including without warm starting."""
        device = "cpu"
        model = _build_two_world_free_model(device)
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="split",
            friction_anchor_beta=0.2,
            articulation_constraint_max=4,
            free_body_constraint_max=4,
        )
        self.assertFalse(solver._contact_warmstart)
        n = solver._friction_patches.previous.valid.shape[0]
        worlds = np.arange(n, dtype=np.int32) % 2
        for mask, expect_cleared in (
            (None, (True, True)),
            ((True, False), (True, False)),
            ((False, True), (False, True)),
        ):
            solver._friction_patches.previous.valid.fill_(1)
            solver._friction_patches.previous_world.assign(worlds)
            wm = None if mask is None else wp.array(mask, dtype=wp.bool, device=device)
            solver.reset(model.state(), wm)
            valid = solver._friction_patches.previous.valid.numpy()
            for world, cleared in enumerate(expect_cleared):
                sel = valid[worlds == world]
                np.testing.assert_array_equal(sel, 0 if cleared else 1, err_msg=f"mask={mask} world={world}")


def _slide(device, mode, beta, mesh, angle, response=None, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_body if response is None else builder.add_link
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if response is not None:
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            parent_xform=wp.transform((0, 0, 0.025)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=axis)
                for axis in ((newton.Axis.X, newton.Axis.Z) if locked_tangent else newton.Axis)
            ],
        )
        builder.add_articulation([joint])
    if mesh:
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.1, 0.1, 0.025), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.025, cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode=mode,
        friction_anchor_beta=beta,
        iterations=64,
        cfm=0.0 if locked_tangent else 1.0e-6,
        free_body_constraint_max=128,
        articulation_constraint_max=32,
        row_watermark=True,
        articulated_contact_response=response or "immediate",
    )
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = np.array([np.cos(angle), np.sin(angle)])
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, :2] = direction * 1.2 * 0.3 * 9.81
    hz = 240
    for step in range(hz // 5 + hz // 2):
        state.clear_forces()
        if step >= hz // 5:
            state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / hz)
        state, next_state = next_state, state
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_row_watermarks()


class TestFeatherPGSFrictionDirection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_articulated_sliding_direction(self):
        """Preserve sliding direction through dense and every articulated propagation route."""
        for response in ("immediate", "propagation", "propagation_fused", "propagation_colored"):
            with self.subTest(response=response):
                velocity, direction, _ = _slide("cuda:0", "matrix_free", 0.2, False, np.pi / 4, response)
                np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_locked_tangent_preserves_other_friction_direction(self):
        """Retain friction along a slider when its first tangent has zero response."""
        for response in ("immediate", "propagation", "propagation_fused", "propagation_colored"):
            for beta in (0.0, 0.2):
                with self.subTest(response=response, beta=beta):
                    velocity, direction, watermarks = _slide(
                        "cuda:0", "matrix_free", beta, False, 0.0, response, locked_tangent=True
                    )
                    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                    for key, value in watermarks.items():
                        if "overflow" in key or "dropped" in key:
                            self.assertEqual(value, 0, key)

    def test_unequal_tangent_masses(self):
        """Solve sticking and sliding blocks with rotated, unequal tangent masses in one sweep."""
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        capacity = 32
        for device in devices:
            for angle, small_eigenvalue in (
                (0.0, 1.0),
                (0.37, 1.0),
                (1.1, 1.0),
                (0.37, 0.0),
                (0.0, 0.0),
                (0.0, 1.0e-4),
                (0.0, 1.0e-6),
                (0.0, 1.0e-8),
            ):
                for regime in ("sticking", "sliding", "fixed", "weak_sliding", "sliding_fixed"):
                    if regime == "weak_sliding" and small_eigenvalue == 0.0:
                        continue
                    sliding = regime in ("sliding", "weak_sliding", "sliding_fixed")
                    with self.subTest(device=device, angle=angle, small_eigenvalue=small_eigenvalue, regime=regime):
                        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                        tangent_mass = rotation @ np.diag([small_eigenvalue, 7.0]) @ rotation.T
                        direction = np.array([0.6, 0.8])
                        expected = -(0.3 if sliding else 0.1) * direction
                        if regime == "fixed":
                            expected = np.array([0.125, 0.0])
                        velocity = 2.0 * direction if sliding else np.zeros(2)
                        if regime == "weak_sliding":
                            # A tiny positive KKT multiplier still requires a boundary impulse.
                            # Absolute bisection resolution loses the weak direction here.
                            velocity = small_eigenvalue * direction
                        matrix = np.eye(capacity, dtype=np.float32)
                        matrix[1:3, 1:3] = tangent_mass
                        rhs = np.zeros((1, capacity), dtype=np.float32)
                        rhs[0, :3] = [-1.0, *(velocity - tangent_mass @ expected)]
                        if regime == "fixed":
                            # Power-of-two impulses make the stored row residual exactly zero.
                            rhs[0, 1:3] = -(matrix[1:3, 1:3] @ expected)
                        row_types = np.full((1, capacity), -1, dtype=np.int32)
                        row_types[0, :3] = [0, 2, 2]
                        parents = np.full((1, capacity), -1, dtype=np.int32)
                        parents[0, 1:3] = 0
                        initial = np.zeros((1, capacity), dtype=np.float32)
                        if regime in ("fixed", "sliding_fixed"):
                            initial[0, :3] = [1.0, *expected]
                        impulses = wp.array(initial, dtype=float, device=device)
                        args = [
                            wp.array([3], dtype=int, device=device),
                            wp.array(np.diag(matrix)[None], dtype=float, device=device),
                            wp.array(matrix[None], dtype=float, device=device),
                            wp.array(rhs, dtype=float, device=device),
                            impulses,
                            1,
                            1.0,
                            wp.array(row_types, dtype=int, device=device),
                            wp.array(parents, dtype=int, device=device),
                            wp.full((1, capacity), 0.3, dtype=float, device=device),
                            0,
                            0,
                        ]
                        if device == "cpu":
                            wp.launch(pgs_solve_loop, dim=1, inputs=[args[0], *args[1:]], device=device)
                        else:
                            kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(wp.get_device(device).arch))
                            wp.launch_tiled(kernel, dim=[1], inputs=args, block_dim=32, device=device)
                        actual = impulses.numpy()[0, :3]
                        if regime in ("fixed", "sliding_fixed"):
                            np.testing.assert_array_equal(actual, initial[0, :3])
                        elif small_eigenvalue == 0.0 and not sliding:
                            # A singular sticking block has multiple impulse solutions.
                            np.testing.assert_allclose(tangent_mass @ actual[1:] + rhs[0, 1:3], 0.0, atol=1.0e-6)
                            self.assertLessEqual(float(np.linalg.norm(actual[1:])), 0.300001)
                        else:
                            np.testing.assert_allclose(actual, [1.0, *expected], atol=1.0e-6)

    def test_sliding_direction(self):
        """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        modes = ("split", "matrix_free") if wp.is_cuda_available() else ("split",)
        for mode in modes:
            for beta in (0.0, 0.2):
                for mesh in (False, True):
                    for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                        with self.subTest(mode=mode, beta=beta, mesh=mesh, angle=angle):
                            velocity, direction, watermarks = _slide(device, mode, beta, mesh, angle)
                            expected = direction * (0.2 * 0.3 * 9.81) * 0.5
                            np.testing.assert_allclose(velocity, expected, atol=0.015, rtol=0.03)
                            for key, value in watermarks.items():
                                if "overflow" in key or "dropped" in key:
                                    self.assertEqual(value, 0, key)


@wp.kernel(enable_backward=False)
def _pose_motion_probe(
    before: wp.array[wp.transform],
    after: wp.array[wp.transform],
    point: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    i = wp.tid()
    result[i] = _pose_motion(after, before, i, point[i])


def _ground_box(device, **solver_kwargs):
    """Return a resting 1 kg box on the ground with patch friction enabled."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5))
    model = builder.finalize(device=device)
    kwargs = {
        "iterations": 32,
        "solve_mode": "matrix_free" if model.device.is_cuda else "split",
    }
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step(observables=None):
        s0, s1 = states
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 0.005, observables=observables)
        states.reverse()

    return model, solver, contacts, step


class TestFrictionPatchHistory(unittest.TestCase):
    def test_support_edge_centers_ignore_contact_order(self):
        """Keep the same edge centers when contact ordering changes at rest."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        points = np.array(
            [[x, y, 0.0] for x in np.linspace(-0.001, 0.001, 128) for y in (-0.05, 0.05)], dtype=np.float32
        )
        expected = np.array([[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]], dtype=np.float32)
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                order = np.random.default_rng(seed).permutation(len(points))
                _, _, _, patches = _patch_fixture(points[order], device=device)
                locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                locations = locations[np.argsort(locations[:, 1])]
                # Summation order perturbs the principal axis only by roundoff.
                np.testing.assert_allclose(locations, expected, rtol=0.0, atol=1.0e-12)

    def test_narrow_patch_preserves_footprint_reflection_symmetry(self):
        """Avoid a diagonal friction couple on a symmetric narrow contact footprint."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for angle in (0.0, 0.37, 1.1):
            axis = np.array([np.sin(angle), np.cos(angle), 0.0])
            across = np.array([np.cos(angle), -np.sin(angle), 0.0])
            points = [x * across + y * axis for x in (-0.001, 0.001) for y in (-0.025, 0.025)]
            with self.subTest(angle=angle):
                _, _, _, patches = _patch_fixture(points, device=device)
                active = patches.view.weight.numpy() > 0.0
                locations = patches.view.point_a.numpy()[active]
                self.assertEqual(len(locations), 2)
                np.testing.assert_allclose(locations @ across, 0.0, atol=1.0e-7)
                np.testing.assert_allclose(np.sort(locations @ axis), [-0.025, 0.025], atol=1.0e-7)

    def test_sheared_narrow_patch_keeps_anchors_at_its_support_edges(self):
        """Keep both anchors near the support-edge centers when a slight shear tilts the principal axis."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        # A faceted wheel's line contact: three points across each cap edge, one cap
        # offset slightly along the edges. The principal axis then deviates from the
        # caps' normal, and keeping only each edge's extreme point anchors opposite
        # corners, a diagonal couple that steers a symmetric wheel. Anchors on the
        # principal line through the centroid move only in proportion to the shear.
        across_offsets = (-0.0145, 0.0, 0.0145)
        for angle in (0.0, 0.37):
            axis = np.array([np.sin(angle), np.cos(angle), 0.0])
            across = np.array([np.cos(angle), -np.sin(angle), 0.0])
            for shear in (1.0e-6, 1.0e-5, -1.0e-4, 1.0e-3):
                points = [x * across - 0.025 * axis for x in across_offsets]
                points += [(x + shear) * across + 0.025 * axis for x in across_offsets]
                with self.subTest(angle=angle, shear=shear):
                    _, _, _, patches = _patch_fixture(points, device=device)
                    active = patches.view.weight.numpy() > 0.0
                    locations = patches.view.point_a.numpy()[active]
                    self.assertEqual(len(locations), 2)
                    locations = locations[np.argsort(locations @ axis)]
                    tolerance = abs(shear) + 1.0e-7
                    np.testing.assert_allclose(locations @ axis, [-0.025, 0.025], rtol=0.0, atol=tolerance)
                    np.testing.assert_allclose(locations @ across, [0.0, shear], rtol=0.0, atol=tolerance)

    def test_aligned_narrow_patch_anchors_span_the_full_footprint(self):
        """Anchor aligned narrow footprints at their axial extremes, including near-extreme samples."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        theta = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        footprints = {
            # Edge samples just inside each end must not pull the anchors inward.
            "intermediate_edge_samples": [[x, y, 0.0] for x in (-0.001, 0.001) for y in (-0.025, -0.023, 0.023, 0.025)],
            "sampled_ellipse": np.stack([0.019 * np.cos(theta), 0.025 * np.sin(theta), np.zeros_like(theta)], axis=1),
        }
        for name, points in footprints.items():
            with self.subTest(footprint=name):
                _, _, _, patches = _patch_fixture(np.asarray(points, dtype=np.float32), device=device)
                active = patches.view.weight.numpy() > 0.0
                locations = patches.view.point_a.numpy()[active]
                self.assertEqual(len(locations), 2)
                locations = locations[np.argsort(locations[:, 1])]
                np.testing.assert_allclose(locations[:, 0], 0.0, atol=1.0e-7)
                np.testing.assert_allclose(locations[:, 1], [-0.025, 0.025], rtol=0.0, atol=1.0e-6)

    def test_narrow_patch_anchors_keep_each_end_height(self):
        """Anchor each end of a narrow footprint at the height of its own support edge."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        # A faceted wheel bouncing on one cap touches at different heights at the
        # two ends. Level anchors there see opposite slip at the ends of a rolling
        # wheel, and the friction answers with a yaw kick.
        footprints = {
            "two_levels": ([[x, -0.025, 0.0] for x in (-0.0145, 0.0, 0.0145)] + [[0.0, 0.025, 0.001]], [0.0, 0.001]),
            "three_levels": ([[-0.05, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, 0.0, 0.003]], [0.0, 0.003]),
        }
        for device in devices:
            for name, (members, heights) in footprints.items():
                with self.subTest(device=device, footprint=name):
                    points = np.asarray(members, dtype=np.float32)
                    _, _, _, patches = _patch_fixture(points, device=device)
                    locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                    self.assertEqual(len(locations), 2)
                    axis = points[-1] - points[0]
                    locations = locations[np.argsort(locations @ axis)]
                    np.testing.assert_allclose(locations[:, 2], heights, rtol=0.0, atol=1.0e-7)

    def test_thin_triangle_anchor_span_varies_continuously(self):
        """Keep the anchor span continuous as a triangle thins into a collinear footprint."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        heights = np.linspace(1.0e-6, 4.0e-6, 41)
        for device in devices:
            with self.subTest(device=device):
                spans = []
                for height in heights:
                    points = np.array([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, height, 0.0]], dtype=np.float32)
                    _, _, _, patches = _patch_fixture(points, device=device)
                    locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                    self.assertEqual(len(locations), 2)
                    spans.append(np.ptp(locations[:, 0]))
                # A switch between two anchor constructions near collinearity would jump by tens of mm.
                self.assertLess(np.abs(np.diff(spans)).max(), 2.0e-3)

    def test_pose_increment_preserves_fixed_pivots_and_no_slip_rolling(self):
        """Distinguish rigid rotation from slip without querying the collision shape."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        pivot = wp.vec3(0.3, -0.2, 0.5)
        origin = wp.vec3(-0.1, 0.2, 0.3)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7)
        before = [wp.transform(origin, wp.quat_identity()), wp.transform_identity()]
        after = [
            wp.transform(pivot + wp.quat_rotate(rotation, origin - pivot), rotation),
            wp.transform(wp.vec3(0.002, 0, 0), wp.quat_identity()),
        ]
        points = [pivot, wp.vec3(0)]
        for rate in (60, 120, 240):
            advance = 1.0 / rate
            # The rigid integrator advances translation and normalizes its
            # first-order quaternion update. Its no-slip rolling pair has
            # this relative rotation; a material chord falsely reports slip.
            angle = float(2.0 * np.arctan(0.5 * 20.0 / rate))
            before.append(wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
            after.append(wp.transform(wp.vec3(advance, 0, 0.05), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle)))
            points.append(wp.vec3(advance, 0, 0))
        result = wp.zeros(len(points), dtype=wp.vec3, device=device)
        wp.launch(
            _pose_motion_probe,
            dim=len(points),
            inputs=[
                wp.array(before, dtype=wp.transform, device=device),
                wp.array(after, dtype=wp.transform, device=device),
                wp.array(points, dtype=wp.vec3, device=device),
                result,
            ],
            device=device,
        )
        values = result.numpy()
        np.testing.assert_allclose(values[0], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(values[1], [0.002, 0, 0], atol=1.0e-7)
        np.testing.assert_allclose(values[2:, :2], 0.0, atol=1.0e-7)

    def test_supported_normal_turn_preserves_friction_history(self):
        """Retain supported history during small relative rotations on either body."""
        for body in (0, 1):
            for axis, retained in ((wp.vec3(0, 1, 0), True), (wp.vec3(0, 0, 1), True)):
                with self.subTest(body=body, axis=axis):
                    model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
                    types = model.shape_type.numpy()
                    types[body] = int(newton.GeoType.SPHERE)
                    model.shape_type.assign(types)
                    patches.build(model, state, contacts)
                    patches.store(state)
                    patches.build(model, state, contacts)
                    self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
                    transforms = [wp.transform_identity(), wp.transform_identity()]
                    transforms[body] = wp.transform(wp.vec3(0), wp.quat_from_axis_angle(axis, 0.01))
                    state.body_q.assign(transforms)
                    patches.build(model, state, contacts)
                    self.assertEqual(int(patches.current.source.numpy()[0]) >= 0, retained)

    def test_normal_turn_transports_tangent_error_without_decay(self):
        """Preserve spring length during supported rocking without additional slip."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        patches.store(state)
        pivot = wp.vec3(0.0005, 0, 0)
        local_point = wp.vec3(-0.0005, 0, 0)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([local_point])
        contacts.rigid_contact_point1.assign([pivot])
        patches.build(model, state, contacts)
        for step in range(100):
            patches.store(state)
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.025 if step % 2 else -0.025)
            state.body_q.assign(
                [wp.transform(pivot - wp.quat_rotate(rotation, local_point), rotation), wp.transform_identity()]
            )
            patches.build(model, state, contacts)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_shared_pad_randomization_preserves_material_regions(self):
        """Pool one sampled pad material while keeping genuinely different coefficients separate."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]], shape0=[0, 2])
        for coefficient in np.random.default_rng(42).uniform(0.1, 1.0, 3):
            model.shape_material_mu.assign([coefficient, 0.5, coefficient])
            patches.update_geometry(model)
            patches.build(model, state, contacts)
            self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
            np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])
            patches.store(state)
        model.shape_material_mu.assign([0.2, 0.5, 0.8])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_pose_history_covers_bodies_beyond_contact_capacity(self):
        """Capture every body pose even when the contact buffer is smaller."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(9, 10, 9))
        transforms = state.body_q.numpy()
        transforms[10, 0] = 0.002
        state.body_q.assign(transforms)
        patches.build(model, state, contacts)
        patches.store(state)
        transforms[10, 0] = 0.003
        state.body_q.assign(transforms)
        patches.build(model, state, contacts)
        self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
        self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_rocking_face_retires_the_lifted_anchor(self):
        """Keep the supported edge's history and replace the opposite anchor after rocking."""
        points = [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]]
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        pivot = wp.vec3(-0.1, 0, 0)
        state.body_q.assign([wp.transform(pivot - wp.quat_rotate(rotation, pivot), rotation), wp.transform_identity()])
        contacts.rigid_contact_count.assign([2])
        edge = [points[0], points[2], points[0], points[2]]
        contacts.rigid_contact_point0.assign(edge)
        contacts.rigid_contact_point1.assign(edge)
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-7)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy()[active] >= 0), 1)

    def test_unloading_penetration_keeps_supported_anchors(self):
        """Retain history while decompression leaves the actual surfaces in contact."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_point0.assign([[-0.1, 0, -0.005], [0.1, 0, -0.005]])
        patches.build(model, state, contacts)
        patches.store(state)
        state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.003), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2)

    def test_support_witnesses_use_existing_contact_gap_limits(self):
        """Keep supported history within the contact envelope and respect tighter solver gates."""
        for limits, carried in (
            ({}, True),
            ({"contact_gap_limit": 0.0001}, False),
            ({"friction_gap_limit": 0.0001}, True),
        ):
            with self.subTest(limits=limits):
                model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
                model.shape_gap.fill_(0.0005)
                patches.store(state)
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0002), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0004], [0.1, 0, -0.0004]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2 if carried else 0)
                # A fresh sample on the supported region cannot retain history
                # whose old footprint has lifted beyond the shape gap envelope.
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0012), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0014], [0.1, 0, -0.0014]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 0)

    def test_geometry_edits_retire_only_affected_history(self):
        """Invalidate edited surfaces across carrier seams while preserving unrelated patches."""
        for field in ("shape_transform", "shape_scale", "shape_margin", "shape_source_ptr"):
            for static in (False, True):
                with self.subTest(field=field, static=static):
                    model, state, contacts, patches = _patch_fixture(
                        [[0, 0, 0], [0, 0, 0]],
                        shape0=[0, 3],
                        shape_bodies=(0, -1 if static else 1, 0, 2, 3),
                        materials=(0.5,) * 5,
                    )
                    contacts.rigid_contact_shape1.assign([1, 4])
                    patches.build(model, state, contacts)
                    patches.store(state)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [1, 1])
                    # Static geometry is identified by shape. A dynamic edit on
                    # shape 2 must also retire shape 0's anchor on the same body.
                    edited = 1 if static else 2
                    values = getattr(model, field).numpy()
                    if field == "shape_transform":
                        values[edited, 0] += 1
                    else:
                        values[edited] += 1
                    getattr(model, field).assign(values)
                    patches.update_geometry(model)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [0, 1])

    def test_geometry_updates_allow_contacts_without_a_shape(self):
        """Preserve implicit static contacts when an unrelated shape changes."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(0, 1, 1))
        contacts.rigid_contact_shape1.assign([-1])
        patches.build(model, state, contacts)
        patches.store(state)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)
        # Editing the last shape must not alias the contact's -1 sentinel.
        transforms = model.shape_transform.numpy()
        transforms[-1, 0] = 1.0
        model.shape_transform.assign(transforms)
        patches.update_geometry(model)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)

    def test_reused_contacts_keep_anchors_across_substeps(self):
        """Accumulate slip across substeps regardless of stale collision match indices."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_match_index = wp.array([1, 0], dtype=int, device=model.device)
        active = patches.current.valid.numpy() != 0
        patches.store(state)
        for substep in range(1, 4):
            displacement = substep * 0.001
            state.body_q.assign(
                [wp.transform(wp.vec3(displacement, 0, 0), wp.quat_identity()), wp.transform_identity()]
            )
            patches.build(model, state, contacts)
            active = patches.current.valid.numpy() != 0
            np.testing.assert_allclose(
                np.linalg.norm(patches.view.phi.numpy()[active], axis=1), displacement, atol=1.0e-7
            )
            self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))
            patches.store(state)

    def test_connected_convex_chain_forms_one_region(self):
        """Join a chain into one friction region even when the two end shapes do not overlap."""
        model, state, contacts, patches = _patch_fixture(
            [[-0.15, 0, 0], [0.15, 0, 0], [-0.05, 0, 0], [0.05, 0, 0]],
            shape0=[0, 4, 2, 3],
            shape_bodies=(0, 1, 0, 0, 0),
            materials=(0.5,) * 5,
        )
        model.shape_transform.assign(
            [wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()) for x in (-0.15, 0, -0.05, 0.05, 0.15)]
        )
        model.shape_collision_radius.assign([0.06, 0.3, 0.06, 0.06, 0.06])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:4], [0.5, 0.5, 0, 0])

    def test_disconnected_convex_shapes_form_separate_regions(self):
        """Keep two separated pads on one body from pooling friction through empty space."""
        model, state, contacts, patches = _patch_fixture([[-0.15, 0, 0], [0.15, 0, 0]], shape0=[0, 2])
        model.shape_transform.assign(
            [
                wp.transform(wp.vec3(-0.15, 0, 0), wp.quat_identity()),
                wp.transform_identity(),
                wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity()),
            ]
        )
        model.shape_collision_radius.assign([0.03, 0.2, 0.03])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_incomplete_frame_cannot_create_anchor_history(self):
        """Reject an overflowing contact frame before selecting or carrying anchors."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_count.assign([3])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.valid.numpy()), 0)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 0)

    def test_reversed_contact_orientation_uses_same_patch(self):
        """Give a body pair one identity regardless of collision shape ordering."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 0])
        contacts.rigid_contact_normal.assign([[0, 0, -1], [0, 0, 1]])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])

    def test_saturation_only_releases_an_anchor_with_sliding_motion(self):
        """Keep history under impending slip, and release it under motion against saturated friction."""
        _, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        zeros = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[0, 0, 0]], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.5, 0.5]], dtype=float, device="cpu")
        impulses = wp.array([[2, 1, 0]], dtype=float, device="cpu")
        velocity = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        args = [
            contacts.rigid_contact_count,
            patches.current,
            state.body_q,
            state.body_q,
            velocity,
            wp.zeros(2, dtype=wp.vec3, device="cpu"),
            zeros,
            zeros,
            zeros,
            lengths,
            0,
            parents,
            mu,
            impulses,
            0.005,
        ]
        impulses.zero_()
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 0, "unloaded speculative contacts must not accrue stiction")
        patches.current.valid.fill_(1)
        impulses.assign([[2, 1, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 1)
        velocity.assign([[0, -0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 0)

    def test_convex_shapes_share_one_body_patch(self):
        """Pool friction load across adjacent convex pieces with compatible materials."""
        _, _, _, patches = _patch_fixture(
            [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], shape0=[0, 2, 0, 2]
        )
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 2)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        self.assertAlmostEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_material_normal_and_disconnected_regions_stay_separate(self):
        """Keep friction from pooling across incompatible or spatially separate regions."""
        cases = (
            {"points": [[0, 0, 0], [0.01, 0, 0]], "shape0": [0, 2], "materials": (0.5, 0.5, 0.8)},
            {"points": [[0, 0, 0], [0.01, 0, 0]], "normals": [[0, 0, -1], [0, -1, 0]]},
            {"points": [[0, 0, 0], [0, 0, 0.05]]},
            {"points": [[0, 0, 0], [1, 0, 0]]},
        )
        for case in cases:
            with self.subTest(case=case):
                _, _, _, patches = _patch_fixture(**case)
                self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
                np.testing.assert_array_equal(patches.view.weight.numpy()[:2], [1, 1])

    def test_contact_churn_preserves_error_at_current_points(self):
        """Preserve a sticking region's history even when every contact sample changes."""
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        points = points[[3, 1, 0, 2]] * 0.98
        contacts.rigid_contact_point0.assign(points)
        contacts.rigid_contact_point1.assign(points)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        np.testing.assert_allclose(patches.view.point_a.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(patches.view.point_b.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(np.linalg.norm(patches.view.phi.numpy()[active], axis=1), 0.001, atol=1.0e-7)
        self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))

    def test_contact_migration_transports_twist_to_new_lever_arms(self):
        """Preserve a region's twist rather than copying error from old lever arms."""
        points = np.array([[-0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.01)
        state.body_q.assign([wp.transform(wp.vec3(0), rotation), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        patches.build(model, state, contacts)
        before = patches.view.phi.numpy()[:2].copy()
        self.assertGreater(np.linalg.norm(before), 0.001)
        patches.store(state)
        points *= 0.8
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        contacts.rigid_contact_point1.assign(points)
        patches.build(model, state, contacts)
        np.testing.assert_allclose(patches.view.phi.numpy()[:2], 0.8 * before, atol=1.0e-7)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())

    def test_coincident_contacts_use_one_anchor(self):
        """Reject duplicate witnesses that would create two coincident friction constraints."""
        _, _, _, patches = _patch_fixture([[0, 0, 0]] * 4)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 1)
        self.assertEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_warmstart_uses_patch_load_when_anchor_normal_is_unloaded(self):
        """Keep a cached tangent supported when other normals carry the load."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
        patches.current.tangent_impulse.assign([[0.8, 0, 0], [0, 0, 0], [0.8, 0, 0]] + [[0, 0, 0]] * 5)
        patches.store(state)
        patches.build(model, state, contacts)
        slots = wp.array([0, 3, 4], dtype=int, device="cpu")
        worlds = wp.zeros(3, dtype=int, device="cpu")
        paths = wp.zeros(3, dtype=int, device="cpu")
        lengths = wp.array([3, 1, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, -1, 4, 4]], dtype=int, device="cpu")
        mu = wp.array([[0.5] * 7], dtype=float, device="cpu")
        impulses = wp.array([[0, 0, 0, 4, 0, 0, 0]], dtype=float, device="cpu")
        wp.launch(
            link_patch_rows,
            dim=3,
            inputs=[contacts.rigid_contact_count, patches.view, worlds, slots, paths, lengths, 0, parents, mu],
            device="cpu",
        )
        wp.launch(
            seed_patch_impulses,
            dim=3,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                worlds,
                slots,
                paths,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
                wp.ones(1, dtype=float, device="cpu"),
            ],
            device="cpu",
        )
        result = impulses.numpy()[0]
        self.assertAlmostEqual(float(np.linalg.norm(result[1:3])), 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(result[5:7])), 0.8)

    def test_seed_bounds_matched_warmstart_without_patch_history(self):
        """Keep a contact-matched friction seed inside the patch cone on an anchor without patch history."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        self.assertEqual(int(patches.current.source.numpy()[0]), -1)
        slots = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25] + [0.0] * 5], dtype=float, device="cpu")
        # The anchor's cone radius is 0.25 * 1.0; the second seed is outside it.
        for seeded_tangent, expected_tangent in (((0.15, -0.1), (0.15, -0.1)), ((0.3, -0.2), (0.2080, -0.1387))):
            with self.subTest(seed=seeded_tangent):
                impulses = wp.array([[1.0, *seeded_tangent] + [0.0] * 5], dtype=float, device="cpu")
                wp.launch(
                    seed_patch_impulses,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        patches.current,
                        patches.previous,
                        state.body_q,
                        slots,
                        slots,
                        slots,
                        lengths,
                        0,
                        parents,
                        mu,
                        impulses,
                        1.0,
                        wp.ones(1, dtype=float, device="cpu"),
                    ],
                    device="cpu",
                )
                result = impulses.numpy()[0]
                self.assertEqual(float(result[0]), 1.0)
                np.testing.assert_allclose(result[1:3], expected_tangent, atol=1.0e-4)
                self.assertLessEqual(float(np.linalg.norm(result[1:3])), 0.25 + 1.0e-6)

    def test_anchors_without_rows_keep_history(self):
        """Keep a valid anchor and its cached impulse when its region gets no friction rows this step."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        cached = [[0.1, 0.2, 0.0]] + [[0.0, 0.0, 0.0]] * 7
        zeros = wp.zeros(1, dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.zeros((1, 8), dtype=float, device="cpu")
        impulses = wp.zeros((1, 8), dtype=float, device="cpu")
        qd = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        com = wp.zeros(2, dtype=wp.vec3, device="cpu")
        for slot, length, load, retained in ((-1, 3, 0.0, 1), (0, 1, 2.0, 1), (0, 1, 0.0, 0)):
            with self.subTest(slot=slot, slots_needed=length):
                impulses.assign([[load] + [0.0] * 7])
                patches.current.valid.assign([1] + [0] * 7)
                patches.current.tangent_impulse.assign(cached)
                wp.launch(
                    finish_patch_impulses,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        patches.current,
                        state.body_q,
                        state.body_q,
                        qd,
                        com,
                        zeros,
                        wp.array([slot], dtype=int, device="cpu"),
                        zeros,
                        wp.array([length], dtype=int, device="cpu"),
                        0,
                        parents,
                        mu,
                        impulses,
                        0.005,
                    ],
                    device="cpu",
                )
                self.assertEqual(int(patches.current.valid.numpy()[0]), retained)
                np.testing.assert_array_equal(
                    patches.current.tangent_impulse.numpy()[0], np.asarray(cached[0], dtype=np.float32)
                )

    def test_stationary_history_does_not_accumulate_roundoff(self):
        """Avoid accumulating world round-trip error while a transformed contact remains stationary."""
        model, state, contacts, patches = _patch_fixture([[0.02, -0.03, 0.0]])
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.5, 0.8)), 1.1)
        state.body_q.assign([wp.transform(wp.vec3(1.3, -0.7, 2.1), rotation), wp.transform_identity()])
        patches.build(model, state, contacts)
        for _ in range(100):
            patches.store(state)
            patches.build(model, state, contacts)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            np.testing.assert_array_equal(patches.view.phi.numpy()[0], [0.0, 0.0])

    def test_filtered_members_carry_history_without_starting_it(self):
        """Carry existing anchors through a friction-filtered step with zero weight, and never create new ones."""
        model, state, contacts, patches = _patch_fixture([[-0.05, 0, 0], [0.05, 0, 0]])
        patches.store(state)
        patches.build(model, state, contacts, friction_gap_limit=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 2)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
        patches.previous.valid.zero_()
        patches.build(model, state, contacts, friction_gap_limit=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 0)
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)


def _run_rolling(geometry, device, *, friction_anchor_beta=None, segments=64, hz=240, direction=1, deterministic=False):
    """Roll a generated wheel for one second and average speed over its final quarter."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
    if geometry == "sphere":
        builder.add_shape_sphere(body, radius=0.05)
    elif geometry in ("mesh", "convex_hull"):
        mesh = newton.Mesh.create_cylinder(radius=0.05, half_height=0.025, segments=segments)
        getattr(builder, "add_shape_" + geometry)(body, mesh=mesh)
    else:
        rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
        add_shape = getattr(builder, "add_shape_" + geometry)
        add_shape(body, radius=0.05, half_height=0.025, xform=wp.transform(wp.vec3(0), rotation))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=friction_anchor_beta)
    pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    s0.joint_qd.assign([direction, 0, 0, 0, 20 * direction, 0])
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    control = model.control()
    final_velocity = []
    for step in range(hz):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 1.0 / hz)
        s0, s1 = s1, s0
        if step >= 3 * hz // 4:
            # CPU arrays expose their buffer; copy before the states swap.
            final_velocity.append(s0.body_qd.numpy()[0].copy())
    pose, velocity = s0.body_q.numpy()[0], s0.body_qd.numpy()[0]
    return pose, velocity, np.mean(final_velocity, axis=0)


@unittest.skipUnless(wp.get_device().is_cuda, "the device flood runs on CUDA")
class TestFrictionPatchDeviceFlood(unittest.TestCase):
    def test_device_flood_matches_the_serial_flood(self):
        """Growing regions one warp per body pair gives the serial flood's patches exactly.

        Plates made of 3x3 box tiles stack with small tilts, so their contacts form
        regions that span convex seams and several regions per body pair. The same
        state and contacts are solved with the device flood on and off; body state,
        row impulses and every patch output must match bit for bit.
        """
        rng = np.random.default_rng(3)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.002
        builder.add_ground_plane()
        tile = 0.03
        for level in range(6):
            angle = rng.uniform(-0.08, 0.08, size=3)
            rotation = wp.quat_rpy(float(angle[0]), float(angle[1]), float(angle[2]))
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.012 + 0.024 * level), rotation))
            for ix in range(3):
                for iy in range(3):
                    builder.add_shape_box(
                        body,
                        xform=wp.transform(wp.vec3((ix - 1) * 2 * tile, (iy - 1) * 2 * tile, 0.0), wp.quat_identity()),
                        hx=tile,
                        hy=tile,
                        hz=0.01,
                    )
        model = builder.finalize()
        model.rigid_contact_max = 4096
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            iterations=8,
            free_body_constraint_max=4096,
        )
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=4096)
        contacts = pipeline.contacts()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        for _ in range(20):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        patches = solver._friction_patches

        def run(device_flood):
            patches.device_flood = device_flood
            a, b = model.state(), model.state()
            for field, value in start.items():
                getattr(a, field).assign(value)
            solver.reset(a, flags=0)
            for _ in range(3):
                solver.step(a, b, control, contacts, 1.0 / 240.0)
                a, b = b, a
            frame, view = patches.current, patches.view
            outputs = {"body_q": a.body_q, "body_qd": a.body_qd, "impulses": solver._propagation_impulses}
            for name in ("owner", "members", "valid", "source", "anchor_a", "anchor_b", "displacement"):
                outputs["frame." + name] = getattr(frame, name)
            for name in ("weight", "next_contact", "point_a", "point_b", "phi"):
                outputs["view." + name] = getattr(view, name)
            return {name: array.numpy().copy() for name, array in outputs.items()}

        serial = run(False)
        device = run(True)
        self.assertGreater(
            int(patches._flood_pair_count.numpy()[0]), 0, "no body pair was large enough for the device flood"
        )
        count = int(contacts.rigid_contact_count.numpy()[0])
        owner = device["frame.owner"][:count]
        regions, sizes = np.unique(owner[owner >= 0], return_counts=True)
        self.assertGreater(len(regions), 4, "too few friction regions to exercise the flood")
        self.assertGreater(int(sizes.max()), 4, "no region spans several convex pieces")
        for name, value in serial.items():
            np.testing.assert_array_equal(device[name], value, err_msg=f"{name} differs between the floods")


class TestFeatherPGSFrictionPatches(unittest.TestCase):
    def test_default_friction_preserves_free_rolling(self):
        """Match velocity-only free rolling without adding a persistent rearward friction force."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for geometry in ("sphere", "capsule", "cylinder", "mesh", "convex_hull"):
            results = []
            # Isolate displacement persistence from row reduction on faceted
            # wheels: a negligible positive gain keeps the same patch rows with
            # effectively no positional correction. The separate tessellation
            # matrix compares default patches against full point friction.
            reference_beta = 1.0e-8 if geometry in ("mesh", "convex_hull") else 0.0
            # At 240 Hz the 64-facet wheel advances almost one facet per step.
            # Resolve facet crossings for this strict persistence comparison;
            # the tessellation matrix below retains the coarse-timestep cases.
            hz = 1920 if geometry in ("mesh", "convex_hull") else 240
            for kwargs in ({"friction_anchor_beta": reference_beta}, {}):
                result = _run_rolling(geometry, device, hz=hz, deterministic=True, **kwargs)
                self.assertTrue(all(np.isfinite(value).all() for value in result))
                results.append(result)
            with self.subTest(geometry=geometry):
                reference, actual = results
                self.assertAlmostEqual(float(actual[0][0]), float(reference[0][0]), delta=0.01)
                # Patch rows on a faceted wheel wobble laterally (the point-friction wheel stays within
                # 0.3 mm); keep the lateral error inside the tessellation matrix's 3% travel bound.
                lateral_bound = 0.02 if geometry in ("mesh", "convex_hull") else 0.01
                self.assertLess(abs(float(actual[0][1] - reference[0][1])), lateral_bound)
                # Facet impacts lose energy even without positional correction;
                # compare averaged speed rather than individual impact phases.
                self.assertAlmostEqual(float(actual[2][0]), float(reference[2][0]), delta=0.02)
                self.assertAlmostEqual(float(actual[2][4]), float(reference[2][4]), delta=0.4)
                self.assertLess(abs(float(actual[2][1])), 0.01)
                if geometry not in ("mesh", "convex_hull"):
                    self.assertAlmostEqual(float(actual[1][0]), float(reference[1][0]), delta=0.01)
                    self.assertAlmostEqual(float(actual[1][4]), float(reference[1][4]), delta=0.2)
                    self.assertGreater(float(actual[1][0]), 0.9)
                    self.assertLess(abs(float(actual[1][0] - 0.05 * actual[1][4])), 0.01)

    def test_faceted_rolling_across_tessellation_direction_and_timestep(self):
        """Bound patch approximation error across coarse and fine faceted wheels."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for geometry in ("mesh", "convex_hull"):
            for segments in (32, 64, 128):
                for hz in (120, 240):
                    for direction in (-1, 1):
                        case = {"segments": segments, "hz": hz, "direction": direction}
                        reference = _run_rolling(geometry, device, friction_anchor_beta=0.0, **case)
                        actual = _run_rolling(geometry, device, **case)
                        with self.subTest(geometry=geometry, **case):
                            self.assertTrue(all(np.isfinite(value).all() for value in actual))
                            # A two-location friction wrench approximates the
                            # per-contact reference. Bound its one-second travel
                            # error to 3% and mean speed error to 6% of launch speed;
                            # include lateral motion so steering errors cannot hide.
                            # The 128-segment convex hull at 120 Hz lands within a
                            # few percent of these bounds and differs slightly across
                            # GPU architectures (5.1% speed error on an RTX 5080).
                            self.assertLess(np.max(np.abs(actual[0][:2] - reference[0][:2])), 0.03)
                            self.assertLess(abs(float(actual[2][0] - reference[2][0])), 0.06)
                            # Facet rocking can change the phase of lateral
                            # oscillations; bound their absolute speed as well
                            # as the lateral displacement checked above.
                            self.assertLess(abs(float(actual[2][1])), 0.05)
                            # Same 6% margin on the 20 rad/s launch spin.
                            self.assertLess(abs(float(actual[2][4] - reference[2][4])), 1.2)

    def test_curved_grasp_preserves_history_under_small_disturbances(self):
        """Bound held sphere/capsule drift despite repeated small relative rotations."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            for geometry in ("sphere", "capsule"):
                drifts = []
                for enabled in (False, True):
                    model, jaws, obj = _build_v_jaws(5.0, geometry)
                    kwargs = dict(_SQUEEZE_SOLVER)
                    kwargs.pop("contact_shared_anchor")
                    kwargs.pop("contact_friction_shared_anchor")
                    if not model.device.is_cuda:
                        kwargs.update(solve_mode="split", enable_bilateral_preelimination=False)
                    if not enabled:
                        kwargs["friction_anchor_beta"] = 0.0
                    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")
                    contacts = pipeline.contacts()
                    s0, s1 = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
                    relative_height = []
                    retained = []
                    for step in range(600):
                        if step >= 100:
                            # External pose disturbances exercise history through
                            # normal changes, even when stiction resists the torque.
                            q = s0.joint_q.numpy()
                            angle = 2.0e-4 * (np.sin(step * 0.07) - np.sin((step - 1) * 0.07))
                            q[-4:] = np.asarray(
                                wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle)) * wp.quat(*q[-4:])
                            )
                            s0.joint_q.assign(q)
                            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
                        pipeline.collide(s0, contacts)
                        s0.clear_forces()
                        force = s0.body_f.numpy()
                        force[obj, 4] = 2.0e-4 * np.sin(step * 0.07)
                        s0.body_f.assign(force)
                        solver.step(s0, s1, control, contacts, 0.005)
                        s0, s1 = s1, s0
                        pose = s0.body_q.numpy()
                        self.assertTrue(np.isfinite(pose).all())
                        relative_height.append(pose[obj, 2] - pose[jaws[0], 2])
                        if enabled and step >= 100:
                            retained.append(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0) >= 2)
                    drifts.append(float(np.ptp(relative_height[100:])))
                    if enabled:
                        with self.subTest(geometry=geometry):
                            self.assertGreater(np.mean(retained), 0.8)
                            self.assertLess(drifts[-1], 1.0e-4)
                with self.subTest(geometry=geometry):
                    # Require a strong anchor benefit while bounding anchored drift.
                    self.assertGreater(drifts[0], 5 * drifts[1] + 5.0e-5)
                    self.assertGreater(drifts[0], 20 * drifts[1] + 1.0e-5)

    def test_default_friction_keeps_a_box_stack_at_rest(self):
        """Settle an ordinary stack without grasp-specific settings or anchor opt-in."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for height in (0.05, 0.151, 0.252):
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, height), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0
        poses, velocities = s0.body_q.numpy(), s0.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all() and np.isfinite(velocities).all())
        np.testing.assert_allclose(poses[:, 2], [0.05, 0.15, 0.25], atol=0.005)
        self.assertLess(float(np.max(np.abs(poses[:, :2]))), 0.005)
        self.assertLess(float(np.max(np.linalg.norm(velocities[:, :3], axis=1))), 0.05)
        self.assertGreater(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 0)

    def test_default_patch_friction_and_explicit_opt_out(self):
        """Build persistent patches by default while retaining an explicit velocity-only opt-out."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for kwargs, enabled in (({}, True), ({"friction_anchor_beta": 0.0}, False)):
            with self.subTest(enabled=enabled):
                _, solver, _, step = _ground_box(device, **kwargs)
                for _ in range(3):
                    step()
                self.assertEqual(solver._friction_anchors_enabled, enabled)
                if enabled:
                    self.assertAlmostEqual(solver._friction_anchor_beta, 0.2)
                    self.assertEqual(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 2)
                else:
                    self.assertFalse(hasattr(solver._friction_patches, "current"))

    def test_rocking_cube_assigns_friction_only_to_supported_edge(self):
        """Retire the lifted face anchor before building rows for a cube resting on one edge."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.rigid_gap = 1.0e-4
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=1.0e-4)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        self.assertEqual(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 2)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        translation = wp.vec3(-0.1, 0, 0) - wp.quat_rotate(rotation, wp.vec3(-0.1, 0, -0.1))
        transform = np.array([*translation, *rotation], dtype=np.float32)
        s1.joint_q.assign(transform)
        newton.eval_fk(model, s1.joint_q, s1.joint_qd, s1)
        pipeline.collide(s1, contacts)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        active = patches.view.weight.numpy() > 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-6)

    def test_contact_churn_reports_linear_force_but_not_a_wrench(self):
        """Compare the linear export with solved rows under an off-centre load after churn."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        ground = builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, iterations=64)
        # A pipeline publishes the contact capacity; the contacts are written by hand and never collided.
        contacts = newton.CollisionPipeline(model, rigid_contact_max=4, soft_contact_max=0).contacts()
        contacts.rigid_contact_count.assign([4])
        contacts.rigid_contact_shape0.assign([box] * 4)
        contacts.rigid_contact_shape1.assign([ground] * 4)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 4)
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        s0, s1 = model.state(), model.state()
        dt = 0.005
        solver.step(s0, s1, model.control(), contacts, dt)
        points[:, :2] *= 0.8
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        # The x force and z torque correspond to an off-centre tangential load.
        s1.body_f.assign([[1, 0, 0, 0, 0, 0.05]])
        observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
        solver.step(s1, s0, model.control(), contacts, dt, observables=observables)
        self.assertGreater(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 0)
        count = int(solver._mf_constraint_count.numpy()[0])
        self.assertTrue(np.all(solver._contact_path.numpy()[:4] == 1))
        solved = np.einsum("ij,i->j", solver._mf_J_a.numpy()[0, :count], solver._mf_impulses.numpy()[0, :count]) / dt
        self.assertGreater(np.linalg.norm(solved[3:]), 0.01, "fixture must exercise a nonzero contact moment")
        reported = observables.contact_f.numpy().sum(axis=0)
        np.testing.assert_allclose(reported[:3], solved[:3], atol=1.0e-4)
        # Torque is explicitly unsupported by this export, including on the base
        # branch. Preserve that limitation visibly instead of asserting a wrench.
        np.testing.assert_array_equal(reported[3:], [0, 0, 0])

    def test_shape_translation_retires_anchors_outside_new_geometry(self):
        """Retire the old contact footprint after changing a shape's local transform."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

        transforms = model.shape_transform.numpy()
        transforms[box, 0] = 1.0
        model.shape_transform.assign(transforms)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        pipeline.collide(s1, contacts)
        patches.build(model, s1, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_array_equal(patches.current.source.numpy()[active], [-1, -1])
        local_x = patches.current.anchor_a.numpy()[active, 0]
        self.assertTrue(np.all(local_x >= 0.89), "friction still acts on the removed x=-0.1..0.1 footprint")

    def test_rejected_contact_history_keeps_correct_reset_world(self):
        """Keep world ownership valid when contact order changes and normal gates reject rows."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        template = newton.ModelBuilder()
        template.add_ground_plane()
        body = template.add_body()
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=0.001)
        contacts = newton.CollisionPipeline(model, rigid_contact_max=2).contacts()
        shape_body = model.shape_body.numpy()
        boxes = np.flatnonzero(shape_body >= 0)
        grounds = np.flatnonzero(shape_body < 0)
        contacts.rigid_contact_count.assign([2])
        contacts.rigid_contact_shape0.assign(boxes)
        contacts.rigid_contact_shape1.assign(grounds)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 2)
        contacts.rigid_contact_point0.zero_()
        contacts.rigid_contact_point1.zero_()
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertEqual(int(patches.previous.valid.numpy().sum()), 2)

        contacts.rigid_contact_shape0.assign(boxes[::-1].copy())
        contacts.rigid_contact_shape1.assign(grounds[::-1].copy())
        contacts.rigid_contact_point0.assign([[0, 0, 0.002]] * 2)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        self.assertTrue((solver._contact_slot.numpy()[:2] == -1).all())
        bodies = patches.current.body_a.numpy()[:2]
        worlds = model.body_world.numpy()[bodies]
        np.testing.assert_array_equal(patches.previous_world.numpy()[:2], worlds)
        solver.reset(s0, wp.array([True, False], dtype=bool, device=device))
        valid = patches.previous.valid.numpy()[:2]
        np.testing.assert_array_equal(valid, (worlds != 0).astype(np.int32))

    @unittest.skipUnless(wp.is_cuda_available(), "explicit point kernels require CUDA")
    def test_explicit_point_solver_defaults_remain_compatible(self):
        """Honor existing point-solver selections without requiring a new opt-out argument."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cuda:0")
        for kwargs in ({"solve_mode": "matrix_free", "friction_mode": "bisection"},):
            with self.subTest(kwargs=kwargs), self.assertWarnsRegex(UserWarning, "point-contact"):
                solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
                self.assertFalse(solver._friction_anchors_enabled)
                self.assertEqual(solver._friction_anchor_beta, 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix_free requires CUDA")
    def test_matrix_free_ignores_dense_pgs_kernel_for_friction(self):
        """pgs_kernel only selects the split-mode dense solve, so matrix_free keeps patch friction."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cuda:0")
        for kernel in ("tiled_contact", "streaming"):
            with self.subTest(pgs_kernel=kernel):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with kernel_overrides({"pgs_kernel": kernel}):
                        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free")
                    self.assertTrue(solver._friction_anchors_enabled)
                    self.assertAlmostEqual(solver._friction_anchor_beta, 0.2)
                    with kernel_overrides({"pgs_kernel": kernel}):
                        solver = newton.solvers.SolverFeatherPGS(
                            model, solve_mode="matrix_free", friction_anchor_beta=0.3
                        )
                    self.assertAlmostEqual(solver._friction_anchor_beta, 0.3)
                self.assertIsNone(solver._pgs_solve_streaming_kernel)
                self.assertIsNone(solver._pgs_solve_tiled_contact_kernel)

    def test_shared_point_flags_warn_with_patch_friction(self):
        """Explain when patch anchors override an explicitly requested shared friction point."""
        model = newton.ModelBuilder().finalize(device="cpu")
        for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
            with self.subTest(flag=flag), self.assertWarnsRegex(UserWarning, "friction_anchor_beta=0"):
                newton.solvers.SolverFeatherPGS(model, **{flag: True})

    def test_incompatible_point_solvers_are_rejected(self):
        """Reject coupled point solves that would silently consume a patch's shared normal load."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "Patch friction requires"):
            newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, friction_mode="bisection")
        # CPU resolves every native kernel selector to the scalar loop, so only CUDA rejects them.
        with kernel_overrides({"pgs_kernel": "tiled_contact"}):
            solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        self.assertEqual(solver._pgs_kernel, "loop")
        if wp.is_cuda_available():
            cuda_model = builder.finalize(device="cuda:0")
            for kernel in ("tiled_contact", "streaming"):
                with self.subTest(pgs_kernel=kernel), self.assertRaisesRegex(ValueError, "Patch friction requires"):
                    with kernel_overrides({"pgs_kernel": kernel}):
                        newton.solvers.SolverFeatherPGS(cuda_model, friction_anchor_beta=0.2)

    def test_anchor_selection_respects_contact_gap_filters(self):
        """Keep filtered extreme points from removing friction from a loaded middle contact."""
        for gate in ("contact_friction_gap_max", "contact_gap_gate"):
            with self.subTest(gate=gate), wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                ground = builder.add_ground_plane()
                body = builder.add_body()
                box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, **{gate: 0.001})
                contacts = newton.CollisionPipeline(model, rigid_contact_max=3).contacts()
                contacts.rigid_contact_count.assign([3])
                contacts.rigid_contact_shape0.assign([box] * 3)
                contacts.rigid_contact_shape1.assign([ground] * 3)
                contacts.rigid_contact_point0.assign([[-0.08, 0, 0.002], [0, 0, 0], [0.08, 0, 0.002]])
                contacts.rigid_contact_point1.assign([[-0.08, 0, 0], [0, 0, 0], [0.08, 0, 0]])
                contacts.rigid_contact_normal.assign([[0, 0, -1]] * 3)
                solver.step(model.state(), model.state(), model.control(), contacts, 0.005)
                rows = solver._mf_row_type.numpy()[0, : solver._mf_constraint_count.numpy()[0]]
                self.assertEqual(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION), 2)
                np.testing.assert_allclose(solver._friction_patches.view.weight.numpy()[:3], [0, 1, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_propagation_variants_preserve_a_warmstarted_patch(self):
        """Retain a sticking articulated grasp across cached, fused, and colored propagation."""
        for response in ("propagation", "propagation_fused", "propagation_colored"):
            with self.subTest(response=response), self.assertWarnsRegex(UserWarning, "Patch friction selects"):
                drift, solver, state = _run_squeeze(
                    5.0,
                    220,
                    friction_anchor_beta=0.2,
                    contact_warmstart=True,
                    contact_friction_position_iterations=-1,
                    contact_regularization=0.0,
                    articulated_contact_response=response,
                )
                self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                self.assertLess(abs(drift), 2.0e-4)
                self.assertGreater(np.count_nonzero(solver._friction_patches.current.valid.numpy()), 0)

    def test_shape_updates_refresh_geometry_and_keep_history(self):
        """Refresh broadphase bounds without discarding anchors on unchanged geometry."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        radius = patches.body_radius.numpy().copy()
        patches.previous.valid.fill_(1)
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(patches.body_radius.numpy(), 2 * radius)
        self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), patches.previous.valid.shape[0])

    def test_patch_resists_twist_and_releases_above_its_limit(self):
        """Resist a static yaw torque with separated anchors, but permit a larger torque to spin."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for torque in (0.2, 1.0):
            with self.subTest(torque=torque), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
                builder.add_shape_box(
                    body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5)
                )
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    friction_anchor_beta=0.2,
                    iterations=64,
                    solve_mode="matrix_free" if wp.is_cuda_available() else "split",
                )
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
                contacts = pipeline.contacts()
                s0, s1 = model.state(), model.state()
                control = model.control()
                for _ in range(100):
                    s0.body_f.assign([[0, 0, 0, 0, 0, torque]])
                    pipeline.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, 0.005)
                    s0, s1 = s1, s0
                angular_speed = abs(float(s0.body_qd.numpy()[0, 5]))
                self.assertTrue(np.isfinite(s0.body_q.numpy()).all())
                if torque == 0.2:
                    self.assertLess(angular_speed, 0.01)
                    self.assertLess(abs(float(s0.body_q.numpy()[0, 5])), 0.01)
                else:
                    self.assertGreater(angular_speed, 1.0)

    def test_pooled_coulomb_budget(self):
        """Share the loaded middle normal's budget across two unloaded anchor normals."""
        self._check_pooled_projection("cpu", native=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_native_pooled_coulomb_budget(self):
        """Use the complete patch's normal load in native CUDA projection."""
        self._check_pooled_projection("cuda:0", native=True)

    def _check_pooled_projection(self, device, native):
        capacity = 32  # Native row kernels use a full warp of storage.
        count = wp.array([7], dtype=int, device=device)
        diag = wp.ones((1, capacity), dtype=float, device=device)
        matrix = wp.array(np.eye(capacity, dtype=np.float32)[None], dtype=float, device=device)
        rhs = wp.array([[0, -100, 0, -4, 0, -100, 0] + [0] * 25], dtype=float, device=device)
        impulses = wp.zeros((1, capacity), dtype=float, device=device)
        rows = wp.array([[0, 2, 2, 0, 0, 2, 2] + [-1] * 25], dtype=int, device=device)
        parents = wp.array([[3, 0, 0, 4, 0, 4, 4] + [-1] * 25], dtype=int, device=device)
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25] + [0] * 25], dtype=float, device=device)
        args = [count, diag, matrix, rhs, impulses, 2, 1.0, rows, parents, mu, 0, 0]
        if native:
            kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(wp.get_device(device).arch))
            wp.launch_tiled(kernel, dim=[1], inputs=args, block_dim=32, device=device)
        else:
            wp.launch(pgs_solve_loop, dim=1, inputs=[count, *args[1:]], device=device)
        np.testing.assert_allclose(impulses.numpy()[0], [0, 1, 0, 4, 0, 1, 0] + [0] * 25, atol=1.0e-6)

    def test_planar_patch_reduces_friction_rows(self):
        """Retain a box face's four normals while using only two friction anchors."""
        self._check_planar_patch("cpu", "split")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_planar_patch_reduces_friction_rows(self):
        """Solve the same patch layout with native split and fused kernels."""
        for mode in ("split", "matrix_free"):
            with self.subTest(mode=mode):
                self._check_planar_patch("cuda:0", mode)

    def _check_planar_patch(self, device, mode):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize()
            solver = newton.solvers.SolverFeatherPGS(
                model, friction_anchor_beta=0.2, articulation_constraint_max=64, solve_mode=mode
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="latest")
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, model.state(), model.control(), contacts, 0.005)
            rows = np.concatenate(
                [
                    solver._row_type.numpy()[0, : solver._constraint_count.numpy()[0]],
                    solver._mf_row_type.numpy()[0, : solver._mf_constraint_count.numpy()[0]],
                ]
            )
            normals = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_CONTACT))
            friction = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION))
            self.assertEqual(normals, 4)
            self.assertEqual(friction, 4)

    def test_contact_forces_report_per_row_normal_load(self):
        """Report each contact's own normal force so a resting box's contact forces sum to its weight."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            model, solver, contacts, step = _ground_box(device)
            observables = solver.observables({newton.solvers.SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            for _ in range(100):
                step(observables)
            count = int(contacts.rigid_contact_count.numpy()[0])
            forces = observables.contact_f.numpy()[:count, :3]
        weight = float(model.body_mass.numpy()[0]) * 9.81
        self.assertGreaterEqual(count, 3)
        self.assertAlmostEqual(float(abs(forces[:, 2].sum())), weight, delta=0.05 * weight)
        self.assertLess(float(np.abs(forces[:, 2]).max()), weight)

    def test_gap_filtered_step_keeps_anchor_history(self):
        """Carry a region's anchors through a step in which the gap filter removes every friction row."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            _model, solver, _contacts, step = _ground_box(device, contact_friction_gap_max=0.001)
            patches = solver._friction_patches
            for _ in range(40):
                step()
            self.assertGreater(int((patches.current.source.numpy() >= 0).sum()), 0)
            solver.contact_friction_gap_max = -1.0
            step()
            self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
            self.assertGreater(int(patches.previous.valid.numpy().sum()), 0)
            solver.contact_friction_gap_max = 0.001
            step()
            weights = patches.view.weight.numpy()
            sources = patches.current.source.numpy()
            self.assertGreater(int(((weights > 0) & (sources >= 0)).sum()), 0)


# FeatherPGS and its patch kernels are exercised on CUDA, the solver's only device.
_DEVICE = "cuda:0"


@unittest.skipUnless(wp.is_cuda_available(), "SolverFeatherPGS requires CUDA")
class TestFeatherPGSFrictionPatchesUpstream(unittest.TestCase):
    def test_notified_friction_edits_reach_the_patch_builder(self):
        """Build patches from notified friction edits, in place or replaced, in eager and captured steps."""
        for replace in (False, True):
            for capture in (False, True):
                with self.subTest(replace=replace, capture=capture), wp.ScopedDevice(_DEVICE):
                    _check_notified_friction_edit(self, _DEVICE, replace, capture)


def _check_notified_friction_edit(test, device, replace, capture):
    """Edit friction after the anchors formed, notify, and require the next patch build to read it."""
    # A box held on a 0.3 rad incline by static friction carries patch anchors.
    builder = newton.ModelBuilder(gravity=(9.81 * np.sin(0.3), 0.0, -9.81 * np.cos(0.3)))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", iterations=32)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    def advance():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 0.005)
        state_0.assign(state_1)

    def patch_mu():
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        return solver._friction_patches.current.mu.numpy()[:count]

    for _ in range(20):
        advance()
    test.assertGreater(int(solver._friction_patches.previous.valid.numpy().sum()), 0)
    np.testing.assert_array_equal(patch_mu(), np.float32(0.5))
    if capture:
        with wp.ScopedCapture(device=device) as graph:
            advance()

    def run():
        if capture:
            wp.capture_launch(graph.graph)
        else:
            advance()

    mu = np.full(model.shape_count, 0.8, dtype=np.float32)
    if replace:
        model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
    else:
        model.shape_material_mu.assign(mu)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))
    # Like the contact rows, the builder sees friction edits only through a notification.
    model.shape_material_mu.fill_(0.3)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))


class TestPatchMaterialCompatibility(unittest.TestCase):
    def test_disabled_material_options_keep_default_patches(self):
        """Keep ordinary patch defaults when neither experimental material law is enabled."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, contact_compliance=False)
        self.assertEqual(solver._friction_anchor_beta, 0.2)
        self.assertTrue(solver._friction_anchors_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_torsion_opt_in_keeps_default_patches(self):
        """Resolve omitted patch gain to default patch friction alongside torsion, without a warning."""
        options = {"center_only": True, "contact_shared_anchor": False, "contact_friction_shared_anchor": False}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual, solver, *_ = fixture(0.01, friction_anchor_beta=None, **options)
        self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
        reference, explicit, *_ = fixture(0.01, friction_anchor_beta=0.2, **options)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertTrue(explicit._friction_anchors_enabled)
        self.assertAlmostEqual(solver._friction_anchor_beta, 0.2)
        self.assertGreater(solver._torsion_stats["rows"], 0)
        for name in reference:
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_compliance_opt_in_preserves_material_law(self):
        """Keep dense and MF material responses identical to explicit point friction."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                options = {"articulated": articulated, "enabled": True, "steps": 20, "friction_scale": 0.4}
                with self.assertWarnsRegex(UserWarning, "velocity-only point friction"):
                    actual, paths, solver, _ = run_fixture(**options, solver_options={"friction_anchor_beta": None})
                reference, reference_paths, _, _ = run_fixture(**options, solver_options={"friction_anchor_beta": 0.0})
                np.testing.assert_array_equal(actual, reference)
                self.assertEqual(paths, reference_paths)
                self.assertGreater(solver.compliance_contact_count, 0)
                self.assertFalse(solver._friction_anchors_enabled)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_explicit_patch_compliance_conflict_fails_before_step(self):
        """Reject incompatible explicit requests before any simulation state can advance."""
        with patch.object(SolverFeatherPGS, "step", side_effect=AssertionError("step must not run")):
            with self.assertRaisesRegex(ValueError, "contact_compliance.*friction_anchor_beta"):
                run_fixture(articulated=False, enabled=True, steps=1, solver_options={"friction_anchor_beta": 0.2})


GRAVITY = 9.81


DT = 0.00125


MU = 0.8


RADIUS = 0.02


MU_ROLLING = 0.01


SOLVER_OPTIONS = {
    "solve_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "iterations": 16,
    "friction_anchor_beta": 0.0,
    "articulation_constraint_max": 32,
    "free_body_constraint_max": 16,
    "double_buffer": False,
}


def ball(
    *,
    radius=RADIUS,
    mu=MU,
    mu_torsional=0.0,
    mu_rolling=MU_ROLLING,
    speed=1.0,
    rolling=True,
    spin=0.0,
    tilt=0.0,
    enable=True,
    cone="pyramidal",
    joint_friction=None,
    torsion_radius=None,
    **solver_overrides,
):
    """Build a unit-mass solid sphere on the ground, articulated by slide x/z and hinge y/z joints.

    The joint chain keeps the slides world-aligned, so the generalized velocities are the
    center speed (0), vertical speed (1), rolling rate (2) and spin rate (3).
    """
    b = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius)
    body = b.add_link(xform=pose, mass=1.0, inertia=inertia)
    identity = wp.transform_identity()
    joints = [
        b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0)),
        b.add_joint_prismatic(
            carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
        ),
        b.add_joint_revolute(
            carriers[1], carriers[2], parent_xform=identity, child_xform=identity, axis=(0.0, 1.0, 0.0)
        ),
        b.add_joint_revolute(carriers[2], body, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)),
    ]
    b.add_articulation(joints)
    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_shape_sphere(body, radius=radius, cfg=sphere)
    model = b.finalize(device="cuda:0")
    model.set_gravity((GRAVITY * math.sin(tilt), 0.0, -GRAVITY * math.cos(tilt)))
    state = model.state()
    state.joint_qd.assign(np.array([speed, 0.0, speed / radius if rolling else 0.0, spin], np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if joint_friction is not None:
        model.joint_friction.assign(np.asarray(joint_friction, np.float32))
    if torsion_radius is not None:
        set_torsion_radius(model, torsion_radius)
    options = dict(SOLVER_OPTIONS)
    options.update(solver_overrides)
    solver = SolverFeatherPGS(
        model, enable_torsional_rolling_friction=enable, torsional_rolling_friction_cone=cone, **options
    )
    solver._rigid_body_angular_damping.zero_()
    return Scene(model, state, solver)


def responsive_ball(*, radius, mu, mu_torsional, mu_rolling, velocity, cone, dt, **solver_overrides):
    """Build a unit-mass solid sphere on three world-aligned slides and a ball joint, so all five friction rows respond.

    ``velocity`` is the generalized velocity ``(vx, vy, vz, wx, wy, wz)``.
    """
    b = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    body = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius))
    identity = wp.transform_identity()
    joints = [b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0))]
    for parent, child, axis in ((0, 1, (0.0, 1.0, 0.0)), (1, 2, (0.0, 0.0, 1.0))):
        joints.append(
            b.add_joint_prismatic(
                carriers[parent], carriers[child], parent_xform=identity, child_xform=identity, axis=axis
            )
        )
    joints.append(b.add_joint_ball(carriers[2], body, parent_xform=identity, child_xform=identity))
    b.add_articulation(joints)
    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_shape_sphere(body, radius=radius, cfg=sphere)
    model = b.finalize(device="cuda:0")
    model.set_gravity((0.0, 0.0, -GRAVITY))
    state = model.state()
    state.joint_qd.assign(np.asarray(velocity, np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    options = dict(SOLVER_OPTIONS)
    options.update(solver_overrides)
    solver = SolverFeatherPGS(
        model, enable_torsional_rolling_friction=True, torsional_rolling_friction_cone=cone, **options
    )
    solver._rigid_body_angular_damping.zero_()
    return Scene(model, state, solver, dt)


def solved_block(scene):
    """Return the normal impulse, friction impulses, coefficients, and the friction block's Delassus and free velocity."""
    solver = scene.solver
    rows = int(solver._constraint_count.numpy()[0])
    impulse = solver._impulses.numpy()[0, :rows].astype(float)
    jacobian = solver._J_world.numpy()[0, :rows].astype(float)
    response = solver._Y_world.numpy()[0, :rows].astype(float)
    mu = solver._row_mu.numpy()[0, :rows].astype(float)
    free_velocity = solver._v_out.numpy().astype(float) - response.T @ impulse
    delassus = jacobian @ response.T
    # Velocity of the friction rows with the normal impulse applied and friction removed.
    velocity = delassus[1:, :1] @ impulse[:1] + (jacobian @ free_velocity)[1:]
    return impulse[0], impulse[1:], mu[1:], delassus[1:, 1:], velocity


BLOCK_GROUPS = ([0, 1], [2], [3, 4])


def cone_norm(normalized, cone):
    """The norm whose ball of radius ``lambda_n`` is the cone, over coefficient-normalized friction impulses."""
    if cone == "elliptic":
        return float(np.linalg.norm(normalized))
    return float(np.linalg.norm(normalized[:2]) + abs(normalized[2]) + np.linalg.norm(normalized[3:]))


def elliptic_kkt_minimum(delassus, velocity, mu, load):
    """Minimize ``0.5 l'Al + v'l`` over the elliptic cone by eigendecomposition and a bisected KKT multiplier."""
    active = (mu > 0.0) & (np.diag(delassus) > 1.0e-12 * np.diag(delassus).max())
    hessian = (mu[:, None] * delassus * mu[None, :])[np.ix_(active, active)]
    gradient = (mu * velocity)[active]
    eigenvalues, basis = np.linalg.eigh(hessian)
    rotated = basis.T @ gradient

    def minimizer(multiplier):
        return -basis @ (rotated / np.maximum(eigenvalues + multiplier, 1.0e-300))

    if eigenvalues.min() > 1.0e-12 * eigenvalues.max() and np.linalg.norm(minimizer(0.0)) <= load:
        best = minimizer(0.0)
    else:
        low, high = 0.0, np.linalg.norm(gradient) / load
        for _ in range(200):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if np.linalg.norm(minimizer(middle)) > load else (low, middle)
        best = minimizer(high)
    out = np.zeros(len(mu))
    out[active] = mu[active] * best
    return out


def pyramidal_gap(delassus, velocity, mu, load, impulse):
    """Duality gap of ``impulse`` on the block-L1 cone, which bounds its objective excess, relative to the objective."""
    gradient = delassus @ impulse + velocity
    scaled = mu * gradient
    gap = gradient @ impulse + load * max(np.linalg.norm(scaled[g]) for g in BLOCK_GROUPS)
    return gap / abs(0.5 * impulse @ delassus @ impulse + velocity @ impulse)


def certified_pyramidal_minimum(delassus, velocity, mu, load):
    """Minimize over the block-L1 cone in float64 by group-scaled accelerated projection until the gap certifies it."""
    active = (mu > 0.0) & (np.diag(delassus) > 1.0e-12 * np.diag(delassus).max())
    hessian = mu[:, None] * delassus * mu[None, :] * np.outer(active, active)
    gradient = mu * velocity * active
    scale = np.ones(len(mu))
    for g in BLOCK_GROUPS:
        curvature = hessian[g, g].max()
        scale[g] = math.sqrt(curvature) if curvature > 0.0 else 1.0
    hessian = hessian / np.outer(scale, scale)
    gradient = gradient / scale
    weights = np.array([1.0 / scale[g[0]] for g in BLOCK_GROUPS])
    step = 1.0 / np.linalg.eigvalsh(hessian).max()

    def project(u):
        norms = np.array([np.linalg.norm(u[g]) for g in BLOCK_GROUPS])
        if weights @ norms <= load:
            return u
        low, high = 0.0, (norms / weights).max()
        for _ in range(200):
            middle = 0.5 * (low + high)
            if weights @ np.maximum(norms - middle * weights, 0.0) > load:
                low = middle
            else:
                high = middle
        out = u.copy()
        for g, norm, weight in zip(BLOCK_GROUPS, norms, weights, strict=True):
            out[g] *= max(norm - high * weight, 0.0) / norm if norm > 0.0 else 0.0
        return out

    u = np.zeros(len(mu))
    z = u.copy()
    momentum = 1.0
    for iteration in range(1, 200001):
        following = project(z - step * (gradient + hessian @ z)) * active
        momentum = 1.0 if (z - following) @ (following - u) > 0.0 else momentum
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        z = following + (momentum - 1.0) / next_momentum * (following - u)
        u, momentum = following, next_momentum
        impulse = mu * u / scale
        if iteration % 200 == 0 and pyramidal_gap(delassus, velocity, mu, load, impulse) < 1.0e-10:
            return impulse
    raise AssertionError("The block-L1 reference did not certify")


_BLOCK_KERNELS = {}


# Single-lane shims for running the emitted CUDA block solve through Warp's CPU backend.
_CPU_SHIMS = """#define __device__
#define __noinline__
#define __forceinline__
#define __syncwarp(...)
#define __shfl_down_sync(mask, value, offset) 0.0f
#define __shfl_sync(mask, value, lane) (value)
#define fmaxf(a, b) ([](float x, float y) { return x > y || y != y ? x : y; }((a), (b)))
"""


def block_visits(device, cone, jacobian, response, normal_jacobian, normal_response, velocity, mu, load, visits):
    """Alternate the normal row (cfm 1e-6) and the emitted friction block on a batch of contacts, one warp each.

    Shapes are (n, 5, dofs), (n, 5, dofs), (n, dofs), (n, dofs), (n, dofs), (n, 5) and (n,) for the initial normal
    impulse; a zero normal Jacobian keeps that load. Returns the velocity after each normal update and each friction
    visit, (n, visits, 2, dofs), the loads (n, visits) and the friction impulses (n, visits, 5), cold-started.
    """
    count, _, dofs = np.shape(jacobian)
    cuda = wp.get_device(device).is_cuda
    key = (cone, dofs, cuda)
    if key not in _BLOCK_KERNELS:
        helpers = angular_friction_gs_sources(cone, 0.0, dofs)["helpers"]
        stride = 32 if cuda else 1
        if not cuda:
            helpers = _CPU_SHIMS + helpers.replace("d += 32", "d += 1")
        snippet = (
            (
                "#if defined(__CUDA_ARCH__)\n    int lane = static_cast<int>(threadIdx.x) & 31;\n"
                if cuda
                else "#if !defined(__CUDA_ARCH__)\n    int lane = 0;\n"
            )
            + helpers
            + f"""
    const unsigned MASK = 0xffffffffu;
    const float* J = jacobian.data + index * {5 * dofs};
    const float* Y = response.data + index * {5 * dofs};
    const float* Jn = normal_jacobian.data + index * {dofs};
    const float* Yn = normal_response.data + index * {dofs};
    float* v = velocity.data + index * {dofs};
    const float* mu = coefficients.data + index * 5;
    float lam[5] = {{0.f, 0.f, 0.f, 0.f, 0.f}}, rhs[5] = {{0.f, 0.f, 0.f, 0.f, 0.f}}, normal = loads.data[index];
    float p = 0.f;
    for (int d = lane; d < {dofs}; d += {stride}) p += Jn[d] * Yn[d];
    for (int offset = 16; offset > 0; offset >>= 1) p += __shfl_down_sync(MASK, p, offset);
    float denom = __shfl_sync(MASK, p, 0) + 1.0e-6f;
    int changed = 0;
    for (int visit = 0; visit < visits; ++visit) {{
        p = 0.f;
        for (int d = lane; d < {dofs}; d += {stride}) p += Jn[d] * v[d];
        for (int offset = 16; offset > 0; offset >>= 1) p += __shfl_down_sync(MASK, p, offset);
        float next = fmaxf(normal - __shfl_sync(MASK, p, 0) / denom, 0.0f);
        __syncwarp(MASK);
        for (int d = lane; d < {dofs}; d += {stride}) {{
            v[d] += Yn[d] * (next - normal);
            history.data[(index * visits + visit) * {2 * dofs} + d] = v[d];
        }}
        normal = next;
        __syncwarp(MASK);
        float old = lam[0];
        lam[0] = AngularFrictionBlock::solve(v, lam, mu, rhs, J, Y, lane, MASK, normal, 1.f, &changed);
        __syncwarp(MASK);
        for (int d = lane; d < {dofs}; d += {stride}) {{
            v[d] += Y[d] * (lam[0] - old);
            history.data[(index * visits + visit) * {2 * dofs} + {dofs} + d] = v[d];
        }}
        if (lane == 0) {{
            normals.data[index * visits + visit] = normal;
            for (int k = 0; k < 5; ++k) impulses.data[(index * visits + visit) * 5 + k] = lam[k];
        }}
        __syncwarp(MASK);
    }}
#endif
"""
        )

        @wp.func_native(snippet)
        def block(
            jacobian: wp.array[float],
            response: wp.array[float],
            normal_jacobian: wp.array[float],
            normal_response: wp.array[float],
            velocity: wp.array[float],
            coefficients: wp.array[float],
            loads: wp.array[float],
            visits: int,
            index: int,
            history: wp.array[float],
            normals: wp.array[float],
            impulses: wp.array[float],
        ): ...

        @wp.kernel(enable_backward=False, module="unique")
        def probe(
            jacobian: wp.array[float],
            response: wp.array[float],
            normal_jacobian: wp.array[float],
            normal_response: wp.array[float],
            velocity: wp.array[float],
            coefficients: wp.array[float],
            loads: wp.array[float],
            visits: int,
            lanes: int,
            history: wp.array[float],
            normals: wp.array[float],
            impulses: wp.array[float],
        ):
            block(
                jacobian,
                response,
                normal_jacobian,
                normal_response,
                velocity,
                coefficients,
                loads,
                visits,
                wp.tid() // lanes,
                history,
                normals,
                impulses,
            )

        _BLOCK_KERNELS[key] = probe
    inputs = [
        wp.array(np.asarray(a, np.float32).ravel(), dtype=float, device=device)
        for a in (jacobian, response, normal_jacobian, normal_response, velocity, mu, load)
    ]
    history = wp.zeros(count * visits * 2 * dofs, dtype=float, device=device)
    normals = wp.zeros(count * visits, dtype=float, device=device)
    impulses = wp.zeros(count * visits * 5, dtype=float, device=device)
    lanes = 32 if cuda else 1
    wp.launch(
        _BLOCK_KERNELS[key],
        dim=count * lanes,
        block_dim=lanes if cuda else 256,
        inputs=[*inputs, visits, lanes, history, normals, impulses],
        device=device,
    )
    return (
        history.numpy().astype(float).reshape(count, visits, 2, dofs),
        normals.numpy().astype(float).reshape(count, visits),
        impulses.numpy().astype(float).reshape(count, visits, 5),
    )


def block_impulses(cone, jacobian, velocity, mu, load):
    """Run the emitted friction block once on rows ``jacobian`` (5 x 3) with unit inverse mass and a fixed load."""
    jacobian = np.asarray(jacobian, float)[None]
    zero = np.zeros((1, jacobian.shape[2]))
    _, _, impulses = block_visits(
        "cuda:0", cone, jacobian, jacobian, zero, zero, np.asarray(velocity)[None], np.asarray(mu)[None], [load], 1
    )
    return impulses[0, 0]


def friction_visit_gains(cone, response, mu, mass, history, normals, impulses):
    """Kinetic energy after each friction visit minus that of its starting impulses moved into the visit's cone."""
    count, visits, _, _ = history.shape
    gains = np.zeros((count, visits))
    for i in range(count):
        previous = np.zeros(5)
        for visit in range(visits):
            start = project_cone(previous / mu[i], normals[i, visit], cone, BLOCK_GROUPS) * mu[i]
            before = history[i, visit, 0] + response[i].astype(float).T @ (start - previous)
            after = history[i, visit, 1]
            gains[i, visit] = 0.5 * after @ mass[i] @ after - 0.5 * before @ mass[i] @ before
            previous = impulses[i, visit]
    return gains


def rigid_contacts(seed, count, dofs):
    """Off-center contacts on a rigid body: about a ball joint's fixed pivot (3 DOFs) or free (6 DOFs).

    Principal inertias span up to 1e4, contact offsets 1e-3 to 1e-1 in random directions, normals are random, and
    torsional and rolling coefficients range over 1e-4 to 1e-1. The body approaches with one unit of normal impulse.
    Returns float32 rows, responses and velocities with the float64 coefficients and generalized mass.
    """
    rng = np.random.default_rng(seed)

    def rotation():
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        return q * np.sign(np.diag(r))

    out = {name: [] for name in ("J", "Y", "Jn", "Yn", "v", "mu", "mass")}
    for _ in range(count):
        principal = 10.0 ** rng.uniform(-4.0, 0.0, size=3)
        axes = rotation()
        offset = rotation()[:, 0] * 10.0 ** rng.uniform(-3.0, -1.0)
        normal = rotation()[:, 0]
        t0 = np.cross(normal, (1.0, 0.0, 0.0)) if abs(normal[0]) < 1.0 - 1.0e-6 else np.cross(normal, (0.0, 1.0, 0.0))
        t0 /= np.linalg.norm(t0)
        t1 = np.cross(normal, t0)
        rows = np.array([np.cross(offset, t0), np.cross(offset, t1), normal, t0, t1])
        normal_row = np.cross(offset, normal)
        inertia = axes @ np.diag(principal) @ axes.T
        inverse = axes @ np.diag(1.0 / principal) @ axes.T
        if dofs == 6:
            m = 10.0 ** rng.uniform(-1.0, 1.0)
            rows = np.hstack([np.array([t0, t1, np.zeros(3), np.zeros(3), np.zeros(3)]), rows])
            normal_row = np.concatenate([normal, normal_row])
            inertia = np.block([[m * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), inertia]])
            inverse = np.block([[np.eye(3) / m, np.zeros((3, 3))], [np.zeros((3, 3)), inverse]])
        J = rows.astype(np.float32)
        Y = J @ inverse.astype(np.float32)
        Jn = normal_row.astype(np.float32)
        Yn = Jn @ inverse.astype(np.float32)
        v = np.linalg.cholesky(inverse) @ rng.normal(size=dofs) * 10.0 ** rng.uniform(-4.0, 0.0)
        v = (v - Yn * (Jn @ v) / (Jn @ Yn)).astype(np.float32) - Yn
        sliding, spin, rolling = rng.uniform(0.2, 1.5), *(10.0 ** rng.uniform(-4.0, -1.0, size=2))
        for name, value in zip(
            out, (J, Y, Jn, Yn, v, [sliding, sliding, spin, rolling, rolling], inertia), strict=True
        ):
            out[name].append(value)
    return {name: np.array(value) for name, value in out.items()}


def devices():
    return ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]


def joint_axis_loads(model, state, wrench):
    """Project each revolute/prismatic joint's published wrench onto its axis, about the joint anchor."""
    body_q = state.body_q.numpy()
    body_com = model.body_com.numpy()
    loads = []
    for joint, joint_type in enumerate(model.joint_type.numpy()):
        if joint_type not in (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)):
            continue
        child = model.joint_child.numpy()[joint]
        X_wc = wp.transform(*body_q[child])
        X_wj = X_wc * wp.transform(*model.joint_X_c.numpy()[joint])
        axis = np.array(
            wp.quat_rotate(
                wp.transform_get_rotation(X_wj), wp.vec3(*model.joint_axis.numpy()[model.joint_qd_start.numpy()[joint]])
            )
        )
        force, torque = wrench[child, :3], wrench[child, 3:]
        if joint_type == int(newton.JointType.PRISMATIC):
            loads.append(float(np.dot(axis, force)))
        else:
            com = np.array(wp.transform_point(X_wc, wp.vec3(*body_com[child])))
            anchor = np.array(wp.transform_get_translation(X_wj))
            loads.append(float(np.dot(axis, torque + np.cross(com - anchor, force))))
    return np.array(loads)


class Scene:
    """Step one model with its own collision pipeline and record generalized velocities."""

    def __init__(self, model, state, solver, dt=DT):
        self.model = model
        self.dt = dt
        self.solver = solver
        self.state = state
        self.next = model.state()
        self.control = model.control()
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, broad_phase="explicit")
        self.contacts = self.pipeline.contacts()

    def step(self, observables=None):
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next, self.control, self.contacts, self.dt, observables=observables)
        self.state, self.next = self.next, self.state

    def run(self, steps):
        """Return the generalized velocity after each step, shape (steps, dofs)."""
        out = []
        for _ in range(steps):
            self.step()
            out.append(self.state.joint_qd.numpy().copy())
        return np.array(out)


def project_cone(y, load, cone, groups):
    """Euclidean projection onto the elliptic ball or the L1-of-group-norms ball of radius ``load``."""
    if cone == "elliptic":
        norm = np.linalg.norm(y)
        return y * (load / norm) if norm > load else y
    norms = np.array([np.linalg.norm(y[g]) for g in groups])
    if norms.sum() <= load:
        return y
    ordered = np.sort(norms)[::-1]
    theta = 0.0
    for k in range(len(ordered)):
        candidate = (ordered[: k + 1].sum() - load) / (k + 1)
        if ordered[k] - candidate >= 0.0:
            theta = candidate
    out = y.copy()
    for g, norm in zip(groups, norms, strict=True):
        out[g] = y[g] * (max(norm - theta, 0.0) / norm if norm > 0.0 else 0.0)
    return out


def cone_minimum(delassus, velocity, coefficients, load, cone, groups, iterations=3000):
    """Approximate the impulse minimizing ``0.5 l'Al + v'l`` over the cone by a fixed number of float64 iterations."""
    scale = np.asarray(coefficients, float)
    hessian = scale[:, None] * delassus * scale[None, :]
    gradient = scale * velocity
    lipschitz = np.abs(hessian).sum(1).max()
    y = np.zeros(len(scale))
    z = y.copy()
    momentum = 1.0
    for _ in range(iterations):
        step = project_cone(z - (gradient + hessian @ z) / lipschitz, load, cone, groups)
        step[scale == 0.0] = 0.0
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        z = step + (momentum - 1.0) / next_momentum * (step - y)
        y, momentum = step, next_momentum
    return scale * y


def reference_velocities(
    cone, steps, *, radius=RADIUS, mu=MU, mu_torsional=0.0, mu_rolling=MU_ROLLING, speed=1.0, rolling=True, spin=0.0
):
    """Velocities (center speed, rolling rate, spin rate) under a numerically solved joint cone each step, float64."""
    mass = 1.0 + 3.0e-4
    inertia = 0.4 * radius * radius
    inverse_mass = np.diag([1.0 / mass, 1.0 / (inertia + 1.0e-8), 1.0 / inertia])
    rows = np.array([[1.0, -radius, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # slip, spin, roll
    delassus = rows @ inverse_mass @ rows.T
    velocity = np.array([speed, speed / radius if rolling else 0.0, spin])
    out = []
    for _ in range(steps):
        impulse = cone_minimum(
            delassus, rows @ velocity, (mu, mu_torsional, mu_rolling), GRAVITY * DT, cone, [[0], [1], [2]]
        )
        velocity = velocity + inverse_mass @ rows.T @ impulse
        out.append(velocity.copy())
    return np.array(out)


def holding_slope(cone, radius=RADIUS, mu=MU, mu_rolling=MU_ROLLING):
    """Largest incline slope on which the cone holds the sphere at rest."""
    if cone == "pyramidal":
        return 1.0 / (1.0 / mu + radius / mu_rolling)
    return 1.0 / math.hypot(1.0 / mu, radius / mu_rolling)


@unittest.skipUnless(wp.is_cuda_available(), "Torsional/rolling friction requires CUDA")
class TestFeatherPGSRollingFriction(unittest.TestCase):
    def test_trajectories_match_the_reference_joint_cone(self):
        """Rolling, sliding and spinning starts follow a float64 reference that solves the cone per step."""
        cases = (
            {"radius": 0.02, "mu_rolling": 0.01},
            {"radius": 0.1, "mu_rolling": 0.002},
            {"radius": 0.02, "mu_rolling": 0.01, "rolling": False},
            {"radius": 0.1, "mu_torsional": 0.02, "mu_rolling": 0.0, "speed": 0.05, "rolling": False, "spin": 5.0},
        )
        for cone in ("pyramidal", "elliptic"):
            for case in cases:
                with self.subTest(cone=cone, **case):
                    qd = ball(cone=cone, **case).run(80)[:, [0, 2, 3]]
                    expected = reference_velocities(cone, 80, **case)
                    scale = np.abs(expected[0]) + np.array([1.0, 1.0, 1.0])
                    np.testing.assert_array_less(np.abs(qd - expected).max(0), 0.005 * scale)

    def test_rolling_stops_and_rests(self):
        qd = ball(speed=0.1).run(200)
        self.assertLess(abs(qd[-1, 0]), 1e-5)
        self.assertLess(abs(qd[-1, 2]), 1e-3)

    def test_incline_holds_below_and_rolls_above_threshold(self):
        for cone in ("pyramidal", "elliptic"):
            slope = holding_slope(cone)
            for factor, rolls in ((0.9, False), (1.1, True)):
                with self.subTest(cone=cone, factor=factor):
                    scene = ball(speed=0.0, tilt=math.atan(factor * slope), cone=cone)
                    qd = scene.run(400)
                    if rolls:
                        self.assertGreater(qd[-1, 0], 0.05)
                    else:
                        self.assertLess(abs(qd[-1, 0]), 1e-4)

    def test_spin_decays_at_the_torsional_bound(self):
        radius, mu_torsional = 0.1, 0.005
        for cone in ("pyramidal", "elliptic"):
            with self.subTest(cone=cone):
                scene = ball(radius=radius, mu_torsional=mu_torsional, mu_rolling=0.0, speed=0.0, spin=5.0, cone=cone)
                qd = scene.run(160)
                t = np.arange(1, 161) * DT
                rate = -np.polyfit(t[8:], qd[8:, 3], 1)[0]
                expected = mu_torsional * GRAVITY / (0.4 * radius * radius)
                self.assertAlmostEqual(rate, expected, delta=0.005 * expected)

    def test_joint_friction_and_torsional_friction_share_the_spin_dof(self):
        """Check that joint dry-friction rows and torsional rows on one DOF add their decelerations."""
        radius, mu_torsional, joint_friction = 0.1, 0.005, 0.002
        inertia = 0.4 * radius * radius
        for cone in ("pyramidal", "elliptic"):
            with self.subTest(cone=cone):
                scene = ball(
                    radius=radius,
                    mu_torsional=mu_torsional,
                    mu_rolling=0.0,
                    speed=0.0,
                    spin=5.0,
                    cone=cone,
                    joint_friction=[0.0, 0.0, 0.0, joint_friction],
                    enable_joint_friction=True,
                )
                qd = scene.run(160)
                t = np.arange(1, 161) * DT
                rate = -np.polyfit(t[8:], qd[8:, 3], 1)[0]
                expected = (mu_torsional * GRAVITY + joint_friction) / inertia
                self.assertAlmostEqual(rate, expected, delta=0.005 * expected)

    def test_creep_speed_softens_stiction_by_load_fraction(self):
        """Below the bound the coefficient times the angular rate settles at creep speed times the load fraction."""
        creep_speed, mu_torsional = 2.0e-3, 0.02
        for dof, coefficient in ((3, mu_torsional), (2, MU_ROLLING)):
            for fraction in (0.5, 0.9):
                with self.subTest(dof=dof, fraction=fraction):
                    scene = ball(
                        speed=0.0, mu_torsional=mu_torsional, torsional_rolling_friction_creep_speed=creep_speed
                    )
                    scene.run(40)
                    force = np.zeros(scene.model.joint_dof_count, np.float32)
                    force[dof] = fraction * coefficient * GRAVITY
                    scene.control.joint_f.assign(force)
                    qd = scene.run(200)
                    self.assertAlmostEqual(
                        qd[-1, dof], creep_speed * fraction / coefficient, delta=0.02 * creep_speed / coefficient
                    )
        rigid = ball(speed=0.0, mu_torsional=mu_torsional)
        rigid.run(40)
        force = np.zeros(rigid.model.joint_dof_count, np.float32)
        force[3] = 0.9 * mu_torsional * GRAVITY
        rigid.control.joint_f.assign(force)
        self.assertLess(abs(rigid.run(200)[-1, 3]), 1e-5)

    def test_competing_sliding_and_spin_share_the_cone(self):
        """A sliding, spinning sphere gives spin budget within tolerance of the reference objective, and more sweeps
        do not worsen it."""
        objectives = {}
        for cone in ("pyramidal", "elliptic"):
            for iterations in (16, 128):
                scene = ball(radius=0.1, speed=0.0, mu_torsional=0.02, mu_rolling=0.0, cone=cone, iterations=iterations)
                scene.run(40)
                scene.state.joint_qd.assign(np.array([0.05, 0.0, 0.0, 5.0], np.float32))
                newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
                scene.step()
                solver = scene.solver
                rows = int(solver._constraint_count.numpy()[0])
                impulse = solver._impulses.numpy()[0, :rows].astype(float)
                jacobian = solver._J_world.numpy()[0, :rows].astype(float)
                response = solver._Y_world.numpy()[0, :rows].astype(float)
                free_velocity = solver._v_out.numpy().astype(float) - response.T @ impulse
                delassus = jacobian @ response.T

                def objective(candidate, delassus=delassus, velocity=jacobian @ free_velocity):
                    return 0.5 * candidate @ delassus @ candidate + velocity @ candidate

                mu = solver._row_mu.numpy()[0, :rows].astype(float)
                velocity = delassus[1:, :1] @ impulse[:1] + (jacobian @ free_velocity)[1:]
                friction = cone_minimum(delassus[1:, 1:], velocity, mu[1:], impulse[0], cone, [[0, 1], [2], [3, 4]])
                best = np.concatenate([impulse[:1], friction])
                with self.subTest(cone=cone, iterations=iterations):
                    self.assertGreater(abs(impulse[3]), 0.1 * mu[3] * impulse[0])
                    self.assertLessEqual(objective(impulse), objective(best) + 1e-3 * abs(objective(best)))
                objectives[cone, iterations] = objective(impulse)
        for cone in ("pyramidal", "elliptic"):
            self.assertLessEqual(objectives[cone, 128], objectives[cone, 16] + 1e-9)

    def test_zero_and_tiny_normal_loads_keep_friction_inside_the_cone(self):
        """A separating contact applies no friction, and a barely loaded one stays inside its cone."""
        for cone in ("pyramidal", "elliptic"):
            for vertical in (1.0, -1.0e-9):
                with self.subTest(cone=cone, vertical=vertical):
                    scene = ball(radius=0.1, rolling=False, mu=1.0, mu_torsional=0.005, mu_rolling=0.0001, cone=cone)
                    scene.model.set_gravity((0.0, 0.0, 0.0))
                    scene.state.joint_qd.assign(np.array([1.0, vertical, 0.0, 0.0], np.float32))
                    newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
                    scene.step()
                    self.assertEqual(int(scene.solver._constraint_count.numpy()[0]), 6)
                    load, impulse, mu, _, _ = solved_block(scene)
                    if vertical > 0.0:
                        self.assertEqual(load, 0.0)
                        np.testing.assert_array_equal(impulse, 0.0)
                        self.assertEqual(scene.state.joint_qd.numpy()[0], 1.0)
                    else:
                        self.assertGreater(load, 0.0)
                        self.assertLess(load, 1.0e-6)
                    self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))

    def test_small_coefficients_reach_the_block_optimum(self):
        """Small spin and rolling coefficients get the optimal block impulse at modest sweep counts."""
        builders = {
            "planar": lambda cone, iterations: ball(
                radius=0.1,
                speed=0.001,
                mu=1.0,
                mu_torsional=0.005,
                mu_rolling=0.0001,
                cone=cone,
                iterations=iterations,
            ),
            "responsive": lambda cone, iterations: responsive_ball(
                radius=0.1,
                mu=1.0,
                mu_torsional=0.005,
                mu_rolling=0.0001,
                velocity=(0.001, 0.0, 0.0, 0.0, 0.01, 0.0),
                cone=cone,
                dt=1.0 / 240.0,
                iterations=iterations,
            ),
        }
        for name, build in builders.items():
            for cone in ("elliptic", "pyramidal"):
                for iterations in (16, 128):
                    with self.subTest(scene=name, cone=cone, iterations=iterations):
                        scene = build(cone, iterations)
                        scene.step()
                        load, impulse, mu, delassus, velocity = solved_block(scene)
                        self.assertGreater(load, 0.0)
                        self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))
                        if cone == "elliptic":
                            expected = elliptic_kkt_minimum(delassus, velocity, mu, load)
                        else:
                            expected = certified_pyramidal_minimum(delassus, velocity, mu, load)
                        np.testing.assert_array_less(np.abs(impulse - expected), 1.0e-3 * mu * load + 1.0e-12)
                        rolling = 3 + int(np.abs(expected[3:]).argmax())
                        self.assertAlmostEqual(impulse[rolling] / expected[rolling], 1.0, delta=1.0e-3)

    def test_dependent_sliding_rows_keep_spin_friction(self):
        """Two tangent rows driven by one diagonal slide are dependent; the independent spin row still gets its bound."""
        for cone in ("elliptic", "pyramidal"):
            for iterations in (16, 128):
                with self.subTest(cone=cone, iterations=iterations):
                    b = newton.ModelBuilder()
                    material = newton.ModelBuilder.ShapeConfig(mu=1.0, mu_torsional=1.0e-4, mu_rolling=0.0)
                    b.add_ground_plane(cfg=material)
                    pose = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
                    identity = wp.transform_identity()
                    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
                    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(2)]
                    body = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                    diagonal = (2.0**-0.5, 2.0**-0.5, 0.0)
                    joints = [
                        b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=diagonal),
                        b.add_joint_prismatic(
                            carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
                        ),
                        b.add_joint_revolute(
                            carriers[1], body, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
                        ),
                    ]
                    b.add_articulation(joints)
                    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, mu_torsional=1.0e-4, mu_rolling=0.0)
                    b.add_shape_sphere(body, radius=0.1, cfg=sphere)
                    model = b.finalize(device="cuda:0")
                    state = model.state()
                    state.joint_qd.assign(np.array([0.0, 0.0, 1.0e-5], np.float32))
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    options = dict(SOLVER_OPTIONS, iterations=iterations)
                    solver = SolverFeatherPGS(
                        model, enable_torsional_rolling_friction=True, torsional_rolling_friction_cone=cone, **options
                    )
                    scene = Scene(model, state, solver, dt=1.0 / 240.0)
                    scene.step()
                    load, impulse, mu, delassus, velocity = solved_block(scene)
                    np.testing.assert_allclose(delassus[0, :2], delassus[1, :2], rtol=1.0e-5)
                    # The spin row is decoupled: it stops the spin or saturates at its bound.
                    expected = -min(abs(velocity[2]) / delassus[2, 2], mu[2] * load)
                    self.assertAlmostEqual(impulse[2] / expected, 1.0, delta=1.0e-3)
                    normalized = np.divide(impulse, mu, out=np.zeros_like(impulse), where=mu > 0.0)
                    self.assertLessEqual(cone_norm(normalized, cone), load * (1.0 + 1.0e-5))

    def test_large_dependency_coefficients_stay_finite(self):
        """Weak tangents made dependent by a strong row keep the independent rolling row exact, never NaN."""
        mu = [0.5, 0.5, 0.01, 1.0e-4, 1.0e-4]
        for cone in ("elliptic", "pyramidal"):
            for scale in (1.0e-4, 1.0e-5, 2.0e-6, 1.0e-6, 1.0e-7):
                for speed in (1.0e-5, 1.0e-9):
                    with self.subTest(cone=cone, scale=scale, speed=speed):
                        jacobian = [[scale, 0, 0], [0, scale, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]
                        impulse = block_impulses(cone, jacobian, [0.0, 0.0, speed], mu, 1.0)
                        self.assertTrue(np.isfinite(impulse).all())
                        # The rolling row is independent and well inside its bound, so it stops the rotation.
                        self.assertAlmostEqual(impulse[3] / -speed, 1.0, delta=1.0e-3)

    def test_dependent_blocks_stay_finite_feasible_and_dissipative(self):
        """Rank-deficient blocks with extreme row scales give finite impulses inside the cone that remove energy."""
        rng = np.random.default_rng(3)
        for cone in ("elliptic", "pyramidal"):
            for trial in range(60):
                with self.subTest(cone=cone, trial=trial):
                    # Three generalized velocities make any five rows dependent.
                    jacobian = rng.normal(size=(5, 3)) * 10.0 ** rng.uniform(-7.0, 0.0, size=(5, 1))
                    if trial % 3 == 1:
                        jacobian[1] = jacobian[0] * 10.0 ** rng.uniform(-3.0, 3.0)
                    elif trial % 3 == 2:
                        # Two weak tangents, just above the activity cutoff, that a strong spin row combines with large
                        # coefficients, and an independent rolling row.
                        weak = 10.0 ** rng.uniform(-5.8, -5.5)
                        u, w = rng.normal(size=(2, 2))
                        jacobian[:] = 0.0
                        jacobian[0, :2], jacobian[1, :2] = weak * u, weak * w
                        jacobian[2, :2] = u + w
                        jacobian[3, 2] = rng.uniform(0.5, 2.0)
                    velocity = rng.normal(size=3) * 10.0 ** rng.uniform(-6.0, 0.0)
                    if trial % 3 == 2:
                        velocity[:2] = 0.0
                    mu = np.concatenate([[rng.uniform(0.2, 1.5)] * 2, 10.0 ** rng.uniform(-5.0, -1.0, size=3)])
                    mu[4] = mu[3]
                    if trial % 3 == 2:
                        # Normalized dependency coefficients mu_spin / (mu_slide * weak) of 1e3 to 1e5.
                        mu[:2] = rng.uniform(0.2, 0.6)
                        mu[2], mu[3:] = 10.0 ** rng.uniform(-2.0, -1.5), 10.0 ** rng.uniform(-4.5, -3.5)
                    load = 10.0 ** rng.uniform(-4.0, 0.0)
                    impulse = block_impulses(cone, jacobian, velocity, mu, load)
                    self.assertTrue(np.isfinite(impulse).all())
                    self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))
                    delassus = jacobian @ jacobian.T
                    rate = jacobian @ velocity
                    objective = 0.5 * impulse @ delassus @ impulse + rate @ impulse
                    self.assertLessEqual(objective, 1.0e-6 * abs(rate @ rate))

    def test_disabled_and_zero_coefficients_match_the_baseline_bitwise(self):
        baseline = ball(enable=False).run(60)
        self.assertAlmostEqual(baseline[-1, 0], 1.0, delta=1e-4)
        zero = ball(mu_torsional=0.0, mu_rolling=0.0).run(60)
        np.testing.assert_array_equal(zero, ball(mu_torsional=0.0, mu_rolling=0.0, enable=False).run(60))
        np.testing.assert_array_equal(baseline, ball(enable=False, mu_rolling=0.5).run(60))

    def test_contact_force_reports_the_sliding_pair(self):
        """CONTACT_F reports the normal and sliding force; the angular couples stay out of it."""
        scene = ball(speed=1.0, rolling=False)
        observables = scene.solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=scene.contacts)
        scene.step(observables)
        # The sliding pair carries the angular row type, so the readers must accept it.
        row_type = scene.solver._row_type.numpy()[0, 1:6]
        np.testing.assert_array_equal(row_type, PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION)
        count = int(scene.contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 1)
        contact_f = observables.contact_f.numpy()
        wrench = contact_f[0]
        self.assertAlmostEqual(abs(float(wrench[2])), GRAVITY * 1.0003, delta=0.01)
        self.assertAlmostEqual(float(np.linalg.norm(wrench[:2])), MU * GRAVITY * 1.0003, delta=0.02)
        np.testing.assert_array_equal(wrench[3:], 0.0)
        np.testing.assert_array_equal(contact_f[count:], 0.0)

    def test_body_parent_f_includes_angular_rows(self):
        """Each joint's published wrench projects onto its axis as the applied joint torque.

        Spin and roll torques on the sphere's hinges are resisted by the torsional and rolling rows; the projection
        holds only when those rows' solved torques are part of the wrench.
        """
        applied = np.array([0.0, 0.0, 0.02, 0.05], dtype=np.float32)
        for enable in (False, True):
            with self.subTest(enable=enable):
                scene = ball(enable=enable, mu_torsional=0.02, speed=0.0, rolling=False)
                scene.control.joint_f.assign(applied)
                observables = scene.solver.observables({SolverBase.ObservableKind.BODY_PARENT_F})
                for _ in range(5):
                    state_in = scene.state
                    scene.step(observables)
                loads = joint_axis_loads(scene.model, state_in, observables.body_parent_f.numpy())
                np.testing.assert_allclose(loads, applied, atol=1.0e-4)

    def test_coefficient_edits_reach_captured_graphs_at_notify(self):
        """Assigned or replaced torsional/rolling arrays take effect at the notify, in a graph captured before the edit."""
        cases = {
            "shape_material_mu_torsional": ({"speed": 0.0, "rolling": False, "spin": 20.0, "mu_rolling": 0.0}, 3, 0.02),
            "shape_material_mu_rolling": ({"mu_rolling": 0.0}, 0, MU_ROLLING),
        }
        for field, (start, dof, value) in cases.items():
            for replace in (False, True):
                with self.subTest(field=field, replace=replace):
                    runs = {}
                    for mode in ("control", "eager", "graph"):
                        scene = ball(**start)
                        graph = None
                        if mode == "graph":
                            with wp.ScopedCapture(device=scene.model.device) as capture:
                                scene.step()
                                scene.step()
                            graph = capture.graph

                        def advance(pairs, scene=scene, graph=graph, dof=dof):
                            out = []
                            for _ in range(pairs):
                                if graph is None:
                                    scene.step()
                                    scene.step()
                                else:
                                    wp.capture_launch(graph)
                                out.append(scene.state.joint_qd.numpy()[dof])
                            return np.array(out)

                        before = advance(5)
                        if mode != "control":
                            values = np.full(scene.model.shape_count, value, np.float32)
                            if replace:
                                setattr(scene.model, field, wp.array(values, device=scene.model.device))
                            else:
                                getattr(scene.model, field).assign(values)
                            scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                        runs[mode] = np.concatenate([before, advance(20)])
                    np.testing.assert_array_equal(runs["graph"], runs["eager"])
                    self.assertLess(abs(runs["eager"][-1]), abs(runs["control"][-1]) - 0.02 * abs(runs["control"][4]))

    def test_zero_coefficients_allocate_no_rows(self):
        scene = ball(mu_torsional=0.0, mu_rolling=0.0)
        scene.step()
        self.assertEqual(int(scene.solver._constraint_count.numpy()[0]), 3)
        scene = ball()
        scene.step()
        self.assertEqual(int(scene.solver._constraint_count.numpy()[0]), 6)

    def test_runtime_coefficient_edits_and_array_replacement(self):
        scene = ball(mu_rolling=0.0)
        qd = scene.run(20)
        self.assertAlmostEqual(qd[-1, 0], 1.0, delta=1e-4)
        shapes = scene.model.shape_count
        scene.model.shape_material_mu_rolling.assign(np.full(shapes, MU_ROLLING, np.float32))
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        start = scene.state.joint_qd.numpy()
        expected = reference_velocities("pyramidal", 60, speed=float(start[0]))
        np.testing.assert_allclose(scene.run(60)[:, 0], expected[:, 0], atol=2e-3)
        scene.model.shape_material_mu_rolling = wp.zeros(shapes, dtype=wp.float32, device=scene.model.device)
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        # Sliding friction removes the rolling-friction slip, then the sphere rolls freely.
        qd = scene.run(30)
        self.assertAlmostEqual(qd[-1, 0], qd[-11, 0], delta=1e-5)
        self.assertAlmostEqual(qd[-1, 0], qd[-1, 2] * RADIUS, delta=1e-4)

    def test_graph_replay_matches_eager(self):
        eager = ball().run(40)
        scene = ball()
        scene.step()
        scene = ball()
        with wp.ScopedCapture(device=scene.model.device) as capture:
            scene.step()
            scene.step()
        replay = []
        for _ in range(20):
            wp.capture_launch(capture.graph)
            replay.append(scene.state.joint_qd.numpy().copy())
        np.testing.assert_array_equal(np.array(replay), eager[1::2])

    def test_reset_keeps_no_angular_history(self):
        fresh = ball().run(30)
        scene = ball()
        start_q = scene.state.joint_q.numpy().copy()
        start_qd = scene.state.joint_qd.numpy().copy()
        scene.run(30)
        scene.state.joint_q.assign(start_q)
        scene.state.joint_qd.assign(start_qd)
        newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
        scene.solver.reset(scene.state)
        np.testing.assert_array_equal(scene.run(30), fresh)

    def test_row_capacity_overflow_is_reported(self):
        scene = ball(articulation_constraint_max=5, warn_constraint_overflow=False)
        scene.step()
        self.assertEqual(int(scene.solver._constraint_count.numpy()[0]), 0)
        self.assertTrue(bool(scene.solver.constraint_overflow.numpy()[0]))

    def test_unsupported_routes_are_rejected(self):
        rejected = (
            {"solve_mode": "split"},
            {"articulated_contact_response": "propagation"},
            {"friction_anchor_beta": 0.2},
            {"contact_warmstart": True},
            {"velocity_iterations": 2},
            {"friction_mode": "bisection"},
            {"row_schedule": "contact_then_internal"},
            {"torsion_radius": 0.01},
            {"enable_sleeping": True},
            {"contact_compliance": True},
        )
        for overrides in rejected:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                ball(**overrides)
        with self.assertRaises(ValueError):
            ball(cone="cubic")
        with self.assertRaises(ValueError):
            ball(torsional_rolling_friction_creep_speed=-1.0)

    def test_free_rigid_route_coefficients_are_rejected(self):
        """Free spheres on a world-static floor or a floor link fixed to the world both take the matrix-free route."""
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                b = newton.ModelBuilder()
                floor_material = newton.ModelBuilder.ShapeConfig(mu=MU, mu_torsional=0.02, mu_rolling=0.01, density=0.0)
                pose = wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity())
                if fixed:
                    floor = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                    b.add_articulation([b.add_joint_fixed(-1, floor, parent_xform=pose)])
                else:
                    floor = -1
                floor_shape = b.add_shape_box(
                    floor, xform=None if fixed else pose, hx=2.0, hy=2.0, hz=0.05, cfg=floor_material
                )
                body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
                sphere = newton.ModelBuilder.ShapeConfig(mu=MU, mu_torsional=0.0, mu_rolling=0.0)
                b.add_shape_sphere(body, radius=0.1, cfg=sphere)
                model = b.finalize(device="cuda:0")
                with self.assertRaises(ValueError):
                    SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
                model.shape_material_mu_torsional.zero_()
                model.shape_material_mu_rolling.zero_()
                solver = SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
                rolling = model.shape_material_mu_rolling.numpy()
                rolling[floor_shape] = 0.01
                model.shape_material_mu_rolling.assign(rolling)
                with self.assertRaises(ValueError):
                    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


class TestFeatherPGSAngularFrictionBlock(unittest.TestCase):
    """Run the emitted friction block on the CPU, and on CUDA when available, against the kinetic energy it changes."""

    def test_off_center_ball_joint_contact_stops_without_gaining_energy(self):
        """An off-center contact on a ball-jointed body, whose stopping impulse lies inside both cones, loses its
        energy, and no friction visit ends above its starting impulses moved into the cone."""
        principal = np.array([0.0009087130361371299, 0.1, 0.10090871303613713])
        axes = np.array(
            [
                [-0.29702703703288535, -0.1784369493775102, -0.9380486098109803],
                [0.9020758170058639, -0.3745632638563419, -0.2143865241625219],
                [-0.3131040715874967, -0.9098695601382277, 0.27221907333836765],
            ]
        )
        rx, ry, rz = 0.005242330836890978, 0.0041898143292664084, 0.0018544005835015218
        mu = np.array([0.8, 0.8, 0.0005484633425010389, 0.07270315887818246, 0.07270315887818246])
        seed = np.array([-0.00022417971789122183, 0.001579519377004289, -0.0001815279526704816])
        inverse = (axes @ np.diag(1.0 / principal) @ axes.T).astype(np.float32)
        jacobian = np.array([[0, rz, -ry], [-rz, 0, rx], [0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float32)
        response = jacobian @ inverse
        normal_jacobian = np.array([ry, -rx, 0], np.float32)
        normal_response = normal_jacobian @ inverse
        velocity = (axes @ np.diag(principal**-0.5) @ seed).astype(np.float32)
        velocity -= normal_response * (normal_jacobian @ velocity) / (normal_jacobian @ normal_response)
        velocity -= normal_response
        mass = axes @ np.diag(principal) @ axes.T
        for device in devices():
            for cone in ("elliptic", "pyramidal"):
                with self.subTest(device=device, cone=cone):
                    history, normals, impulses = block_visits(
                        device,
                        cone,
                        jacobian[None],
                        response[None],
                        normal_jacobian[None],
                        normal_response[None],
                        velocity[None],
                        mu[None],
                        [0.0],
                        128,
                    )
                    start = 0.5 * history[0, 0, 0] @ mass @ history[0, 0, 0]
                    gains = friction_visit_gains(cone, response[None], mu[None], mass[None], history, normals, impulses)
                    self.assertLessEqual(gains.max(), 1.0e-6 * start)
                    for visit in (0, 15, 127):
                        self.assertLess(0.5 * history[0, visit, 1] @ mass @ history[0, visit, 1], 1.0e-6 * start)

    def test_random_rigid_contacts_never_gain_energy(self):
        """No friction visit raises kinetic energy above that of its starting impulses moved into the cone."""
        for device in devices():
            for dofs in (3, 6):
                contacts = rigid_contacts(seed=dofs, count=200, dofs=dofs)
                for cone in ("elliptic", "pyramidal"):
                    with self.subTest(device=device, dofs=dofs, cone=cone):
                        history, normals, impulses = block_visits(
                            device,
                            cone,
                            contacts["J"],
                            contacts["Y"],
                            contacts["Jn"],
                            contacts["Yn"],
                            contacts["v"],
                            contacts["mu"],
                            np.zeros(len(contacts["v"])),
                            16,
                        )
                        self.assertTrue(np.isfinite(history).all())
                        start = 0.5 * np.einsum("ci,cij,cj->c", history[:, 0, 0], contacts["mass"], history[:, 0, 0])
                        gains = friction_visit_gains(
                            cone, contacts["Y"], contacts["mu"], contacts["mass"], history, normals, impulses
                        )
                        # Float32 velocities carry relative energy error of about eps times the inertia ratio, 1e4.
                        np.testing.assert_array_less(gains.max(axis=1), 1.0e-4 * start)


if __name__ == "__main__":
    unittest.main()
