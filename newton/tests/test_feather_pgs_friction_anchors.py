"""Tests for FeatherPGS positional friction anchors (``friction_anchor_beta``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    compute_mf_effective_mass_and_rhs,
    compute_mf_rhs_bias,
    compute_propagation_effective_mass_and_rhs,
    compute_propagation_rhs_bias,
    compute_world_contact_bias,
    mark_sliding_friction_anchors,
    update_friction_anchors,
)

_MU_JAW, _MU_BOX = 5.0, 0.5
_BOX_HALF, _JAW_HALF_T, _GAP0 = 0.02, 0.005, 0.001


def _build_v_jaws(tilt_deg: float):
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
    b.add_constraint_mimic(joints[1], joints[0], coef0=0.0, coef1=-1.0)
    for dof in range(len(b.joint_effort_limit)):
        b.joint_effort_limit[dof] = 10.0
    box = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), label="box")
    b.add_shape_box(
        box,
        hx=_BOX_HALF,
        hy=_BOX_HALF,
        hz=_BOX_HALF,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.1 / (2 * _BOX_HALF) ** 3, mu=_MU_BOX),
    )
    return b.finalize(), jaws, box


_SQUEEZE_SOLVER = {
    "pgs_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 64,
    "pgs_velocity_iterations": 0,
    "pgs_beta": 0.05,
    "pgs_contact_regularization": 0.01,
    "contact_friction_gap_threshold": 0.001,
    "contact_friction_position_iterations": 4,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "enable_bilateral_preelimination": True,
}


def _run_squeeze(tilt_deg: float, steps: int, dt: float = 0.005, matching: str = "latest", **solver_kwargs):
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
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
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
        model, pgs_mode="matrix_free", pgs_iterations=32, pgs_beta=0.05, **solver_kwargs
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
    def test_default_off_keeps_friction_rows_velocity_only(self):
        """With the default ``friction_anchor_beta=0`` no anchor state exists and every friction
        row keeps ``phi = 0`` / ``beta = 0``: the legacy row layout is untouched."""
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, **_SQUEEZE_SOLVER)
        self.assertFalse(solver._friction_anchors_enabled)
        self.assertIsNone(solver._fa_anchor_a)
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
        self.assertEqual(float(np.abs(solver._fa_phi.numpy()).max()), 0.0)
        row_type = solver.row_type.numpy()
        friction = row_type == PGS_CONSTRAINT_TYPE_FRICTION
        self.assertGreater(int(friction.sum()), 0)
        self.assertEqual(float(np.abs(solver.phi.numpy()[friction]).max()), 0.0)
        self.assertEqual(float(np.abs(solver.row_beta.numpy()[friction]).max()), 0.0)

    def test_requires_contact_matching(self):
        model, _jaws, _box = _build_v_jaws(5.0)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.05, **_SQUEEZE_SOLVER)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        pipeline.collide(s0, contacts)
        with self.assertRaises(NotImplementedError):
            solver.step(s0, s1, control, contacts, 0.005)

    def test_anchors_stop_tangential_drift_of_a_held_box(self):
        """A V-tilted pinch leaks tangential drift through velocity-only friction rows; positional
        anchors bound it. Flat jaws (no leak) must stay unaffected."""
        steps = int(3.0 / 0.005)
        drift_off, _, _ = _run_squeeze(5.0, steps)
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
        valid = solver_on._fa_valid.numpy()
        self.assertGreater(int(valid.sum()), 0)
        flat_off, _, _ = _run_squeeze(0.0, steps)
        # sticky matching replays body-local witness points; anchors must still hold
        drift_sticky, _, _ = _run_squeeze(5.0, steps, matching="sticky", friction_anchor_beta=0.2)
        self.assertLess(abs(drift_sticky), 1.0e-4, f"anchored pinch (sticky matching) drifted {drift_sticky:.2e} m")
        flat_on, _, _ = _run_squeeze(0.0, steps, friction_anchor_beta=0.05)
        self.assertLess(abs(flat_on), 1.0e-4)
        self.assertLess(abs(flat_off), 1.0e-4)

    def test_anchors_do_not_oppose_genuine_sliding(self):
        """A box on a 30 deg incline with mu=0.3 (< tan 30) slides at g (sin - mu cos); anchors
        must reset on the saturated cone and leave the sliding speed unchanged."""
        dt, steps = 0.005, 200
        v_off, _lat_off, _, _ = _run_incline(steps, dt)
        v_on, lat_on, solver, contacts = _run_incline(steps, dt, friction_anchor_beta=0.2)
        g, th, mu = 9.81, np.radians(30.0), 0.3
        v_ref = g * (np.sin(th) - mu * np.cos(th)) * steps * dt
        self.assertGreater(v_off, 0.5 * v_ref)
        self.assertAlmostEqual(v_on, v_off, delta=0.05 * v_ref)
        self.assertLess(lat_on, 1.0e-3)
        # every loaded contact slid this step, so its anchor was dropped
        n = int(contacts.rigid_contact_count.numpy()[0])
        if n > 0:
            self.assertEqual(int(solver._fa_valid.numpy()[:n].sum()), 0)

    def test_row_builders_store_anchor_separation_per_route(self):
        """Dense rows: ``phi`` = raw separation, ``row_beta`` = gain. Matrix-free and propagation
        rows: ``phi`` = gain * separation. Checked against the per-contact anchor state on the
        dense (immediate), matrix-free (free body) and propagation routes."""
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
            fa = solver._fa_phi.numpy()[:n]
            slot = solver.contact_slot.numpy()[:n]
            path = solver.contact_path.numpy()[:n]
            checked = {0: 0, 1: 0, 2: 0}
            for c in range(n):
                if slot[c] < 0 or path[c] < 0:
                    continue
                if path[c] == 0:
                    rt, ph, rb = solver.row_type.numpy()[0], solver.phi.numpy()[0], solver.row_beta.numpy()[0]
                    if rt[slot[c] + 1] != PGS_CONSTRAINT_TYPE_FRICTION:
                        continue
                    np.testing.assert_allclose(ph[slot[c] + 1 : slot[c] + 3], fa[c], rtol=1.0e-6, atol=1.0e-9)
                    np.testing.assert_allclose(rb[slot[c] + 1 : slot[c] + 3], beta)
                else:
                    ph = (solver.mf_phi if path[c] == 1 else solver.propagation_phi).numpy()[0]
                    rt = (solver.mf_row_type if path[c] == 1 else solver.propagation_row_type).numpy()[0]
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
        prop_kwargs = dict(_SQUEEZE_SOLVER, articulated_contact_response="propagation", pgs_contact_regularization=0.0)
        c_prop = check(model, prop_kwargs)
        self.assertGreater(c_prop[2], 0)
        # matrix-free route: free box resting on a static slab
        model, _, _ = _build_incline(0.0, 0.5)
        c_mf = check(model, {"pgs_mode": "matrix_free", "pgs_iterations": 16, "pgs_beta": 0.05})
        self.assertGreater(c_mf[1], 0)

    def test_graph_capture_replays(self):
        """Anchor carry is device-side only: two steps capture and replay under a CUDA graph."""
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
            solver.seed_double_buffer_events()
            step()
            step()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        bq = s0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(bq)))
        self.assertLess(abs(bq[box][2] - bq[jaws[0]][2]), 0.01)


def _launch_anchor_update(
    device,
    *,
    point0,
    point1,
    match_index,
    prev,
    shape_body=(0, 1),
    body_q=None,
    normal=(0.0, 0.0, -1.0),
    reset_distance=0.005,
):
    """Drive ``update_friction_anchors`` on ``len(point0)`` contacts between shape 0 (body 0) and
    shape 1 (body 1); returns (anchor_a, anchor_b, valid, phi) arrays."""
    n = len(point0)
    if body_q is None:
        body_q = [wp.transform_identity(), wp.transform_identity()]
    outs = (
        wp.zeros(n, dtype=wp.vec3, device=device),
        wp.zeros(n, dtype=wp.vec3, device=device),
        wp.zeros(n, dtype=wp.int32, device=device),
        wp.zeros(n, dtype=wp.vec2, device=device),
    )
    prev_a, prev_b, prev_valid = prev
    wp.launch(
        update_friction_anchors,
        dim=n,
        inputs=[
            wp.array([n], dtype=wp.int32, device=device),
            wp.array([wp.vec3(*p) for p in point0], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(*p) for p in point1], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(*normal)] * n, dtype=wp.vec3, device=device),  # A-to-B; solver uses -normal
            wp.zeros(n, dtype=wp.int32, device=device),
            wp.ones(n, dtype=wp.int32, device=device),
            wp.zeros(n, dtype=wp.float32, device=device),
            wp.zeros(n, dtype=wp.float32, device=device),
            wp.array(list(match_index), dtype=wp.int32, device=device),
            wp.array(list(shape_body), dtype=wp.int32, device=device),
            wp.array(body_q, dtype=wp.transform, device=device),
            prev_a,
            prev_b,
            prev_valid,
            float(reset_distance),
        ],
        outputs=list(outs),
        device=device,
    )
    return outs


def _anchor_pairs(device, offsets):
    """Previous-frame anchor state: pair ``i`` is body-local ``(offset_i, 0, 0)`` on A and the origin
    on B (both bodies at identity), i.e. a tangential separation of ``offset_i`` along x."""
    a = wp.array([wp.vec3(o, 0.0, 0.0) for o in offsets], dtype=wp.vec3, device=device)
    b = wp.array([wp.vec3(0.0, 0.0, -1.0e-3)] * len(offsets), dtype=wp.vec3, device=device)
    valid = wp.ones(len(offsets), dtype=wp.int32, device=device)
    return a, b, valid


def _launch_mark_sliding(device, *, slots, paths, impulses, row_type, row_parent, row_mu, valid):
    """Drive ``mark_sliding_friction_anchors`` on dense-path contacts in one world."""
    n = len(slots)
    dummy_f = wp.zeros((1, 1), dtype=wp.float32, device=device)
    dummy_i = wp.zeros((1, 1), dtype=wp.int32, device=device)
    dummy_c = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        mark_sliding_friction_anchors,
        dim=n,
        inputs=[
            wp.array([n], dtype=wp.int32, device=device),
            wp.zeros(n, dtype=wp.int32, device=device),
            wp.array(list(slots), dtype=wp.int32, device=device),
            wp.array(list(paths), dtype=wp.int32, device=device),
            wp.array([impulses], dtype=wp.float32, device=device),
            dummy_f,
            dummy_f,
            wp.array([len(impulses)], dtype=wp.int32, device=device),
            dummy_c,
            dummy_c,
            wp.array([row_type], dtype=wp.int32, device=device),
            wp.array([row_parent], dtype=wp.int32, device=device),
            wp.array([row_mu], dtype=wp.float32, device=device),
            dummy_i,
            dummy_i,
            dummy_f,
            dummy_i,
            dummy_i,
            dummy_f,
            0.98,
        ],
        outputs=[valid],
        device=device,
    )
    return valid.numpy()


def _rhs_for_family(family: str, *, phi, row_beta, pgs_beta, dt, bias_scale, device="cpu"):
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
                3,
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
                pgs_beta,
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
                inf,
                pgs_beta,
                dt,
                bias_scale,
                1.0,
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                0,
                0,
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
                phi_arr,
                row_type,
                zeros3(),
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                inf,
                1.0e-6,
                pgs_beta,
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
                inf,
                pgs_beta,
                dt,
                bias_scale,
                1.0,
                wp.zeros((1, 6), dtype=wp.float32, device=device),
                0,
                0,
                3,
            ],
            outputs=[rhs],
            device=device,
        )
    else:
        raise ValueError(family)
    return rhs.numpy()[0]


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=0.0)
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])
    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(template, 2)
    return builder.finalize(device=device)


class TestFeatherPGSFrictionAnchorKernels(unittest.TestCase):
    """Device-agnostic checks of the anchor bookkeeping (run on CPU)."""

    def test_reanchor_is_tangentially_aligned_for_replayed_witness_points(self):
        """Sticky matching replays body-local witness points, so after a slip the pair carries the
        old tangential offset. Re-anchoring must not save that offset: a re-anchored contact held
        stationary reports zero tangential separation on the next step."""
        device = "cpu"
        # witness points 0.1 mm apart tangentially (x) and 1 mm apart along the normal (z)
        point0, point1 = [(1.0e-4, 0.0, 0.0)], [(0.0, 0.0, -1.0e-3)]
        empty = (
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),  # previous anchor invalidated
        )
        a, b, valid, phi = _launch_anchor_update(device, point0=point0, point1=point1, match_index=[0], prev=empty)
        self.assertEqual(int(valid.numpy()[0]), 1)
        np.testing.assert_allclose(phi.numpy()[0], 0.0, atol=1.0e-9)
        # the stored pair is aligned along the normal: no tangential separation
        sep = a.numpy()[0] - b.numpy()[0]
        np.testing.assert_allclose(sep[:2], 0.0, atol=1.0e-9)
        # next step, same replayed points, matched to the pair just stored -> still zero
        _, _, valid2, phi2 = _launch_anchor_update(
            device, point0=point0, point1=point1, match_index=[0], prev=(a, b, valid)
        )
        self.assertEqual(int(valid2.numpy()[0]), 1)
        np.testing.assert_allclose(phi2.numpy()[0], 0.0, atol=1.0e-9)
        # a genuine 0.2 mm tangential move of body A after anchoring is reported
        moved = [wp.transform(wp.vec3(2.0e-4, 0.0, 0.0), wp.quat_identity()), wp.transform_identity()]
        _, _, _, phi3 = _launch_anchor_update(
            device, point0=point0, point1=point1, match_index=[0], prev=(a, b, valid), body_q=moved
        )
        self.assertAlmostEqual(float(np.linalg.norm(phi3.numpy()[0])), 2.0e-4, places=9)

    def test_identity_reordering_lost_match_and_invalidated_anchor(self):
        """Anchors follow ``rigid_contact_match_index`` (previous *sorted* index), not the row
        position; an unmatched or invalidated contact re-anchors with zero separation."""
        device = "cpu"
        aligned = [(0.0, 0.0, 0.0)] * 3, [(0.0, 0.0, -1.0e-3)] * 3
        prev_a, prev_b, prev_valid = _anchor_pairs(device, [1.0e-4, 3.0e-4, 2.0e-4])
        prev_valid.assign(np.array([1, 1, 0], dtype=np.int32))  # pair 2 was invalidated (slid)
        # current contacts 0, 1, 2 match previous 1, 0, 2
        _, _, valid, phi = _launch_anchor_update(
            device, point0=aligned[0], point1=aligned[1], match_index=[1, 0, 2], prev=(prev_a, prev_b, prev_valid)
        )
        np.testing.assert_array_equal(valid.numpy(), [1, 1, 1])
        got = phi.numpy()
        # (the row tangent basis for a +z normal is t0 = +y, t1 = -x; compare magnitudes)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), [3.0e-4, 1.0e-4, 0.0], atol=1.0e-9)
        # lost identity: re-anchor with zero separation
        _, _, valid_lost, phi_lost = _launch_anchor_update(
            device, point0=aligned[0], point1=aligned[1], match_index=[-1, -1, -1], prev=(prev_a, prev_b, prev_valid)
        )
        np.testing.assert_array_equal(valid_lost.numpy(), [1, 1, 1])
        np.testing.assert_allclose(phi_lost.numpy(), 0.0, atol=1.0e-9)

    def test_reset_distance_threshold_reanchors_large_separations(self):
        device = "cpu"
        aligned = [(0.0, 0.0, 0.0)] * 2, [(0.0, 0.0, -1.0e-3)] * 2
        prev = _anchor_pairs(device, [4.0e-3, 6.0e-3])  # below / above the 5 mm default
        _, _, valid, phi = _launch_anchor_update(
            device, point0=aligned[0], point1=aligned[1], match_index=[0, 1], prev=prev, reset_distance=0.005
        )
        np.testing.assert_array_equal(valid.numpy(), [1, 1])
        self.assertAlmostEqual(float(np.linalg.norm(phi.numpy()[0])), 4.0e-3, places=9)  # carried
        self.assertEqual(float(np.abs(phi.numpy()[1]).max()), 0.0)  # re-anchored

    def test_mark_sliding_drops_saturated_and_rejected_keeps_unloaded(self):
        """Only a loaded contact at its Coulomb cone loses its anchor; unloaded, frictionless and
        normal-only contacts keep it, a contact without rows drops it."""
        device = "cpu"
        c, f = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
        mu = 0.5
        # contact 0: saturated (|lam_t| = 0.99 mu lam_n)  -> drop
        # contact 1: loaded, well inside the cone           -> keep
        # contact 2: friction rows present, no normal load   -> keep
        # contact 3: rejected (slot -1)                      -> drop
        # contact 4: normal row only (no friction rows)      -> keep
        row_type = [c, f, f, c, f, f, c, f, f, c, c, c, c]
        row_parent = [-1, 0, 0, -1, 3, 3, -1, 6, 6, -1, -1, -1, -1]
        row_mu = [mu] * 13
        impulses = [1.0, 0.99 * mu, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        valid = wp.ones(5, dtype=wp.int32, device=device)
        got = _launch_mark_sliding(
            device,
            slots=[0, 3, 6, -1, 9],
            paths=[0, 0, 0, -1, 0],
            impulses=impulses,
            row_type=row_type,
            row_parent=row_parent,
            row_mu=row_mu,
            valid=valid,
        )
        np.testing.assert_array_equal(got, [0, 1, 1, 0, 1])

    def test_rhs_bias_matches_on_every_row_family_and_vanishes_in_velocity_pass(self):
        """Friction rows carry ``friction_anchor_beta * separation / dt`` on the dense, matrix-free
        and propagation routes alike, and the velocity-only pass (``bias_scale = 0``) drops it.
        Dense rows store the raw separation with ``row_beta``; matrix-free and propagation rows
        store the gain-premultiplied separation, so both spellings are exercised."""
        pgs_beta, fa_beta, dt = 0.05, 0.3, 0.005
        phi_n, e0, e1 = -2.0e-3, 1.0e-4, -2.5e-4
        expect_pos = np.array([pgs_beta * phi_n / dt, fa_beta * e0 / dt, fa_beta * e1 / dt], dtype=np.float32)
        dense = lambda s: _rhs_for_family(  # noqa: E731
            "dense", phi=[phi_n, e0, e1], row_beta=[pgs_beta, fa_beta, fa_beta], pgs_beta=pgs_beta, dt=dt, bias_scale=s
        )
        pre = [phi_n, fa_beta * e0, fa_beta * e1]
        np.testing.assert_allclose(dense(1.0), expect_pos, rtol=1.0e-5)
        np.testing.assert_allclose(dense(0.0), 0.0, atol=1.0e-9)
        for family in ("mf_setup", "propagation_setup"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=1.0)
            np.testing.assert_allclose(got, expect_pos, rtol=1.0e-5, err_msg=family)
        for family in ("mf_velocity", "propagation_velocity"):
            got = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=1.0)
            np.testing.assert_allclose(got, expect_pos, rtol=1.0e-5, err_msg=family)
            got0 = _rhs_for_family(family, phi=pre, row_beta=None, pgs_beta=pgs_beta, dt=dt, bias_scale=0.0)
            np.testing.assert_allclose(got0, 0.0, atol=1.0e-9, err_msg=family)

    def test_reset_clears_anchor_history_full_and_masked(self):
        """``reset()`` drops carried anchors of the selected worlds even with warm start disabled."""
        device = "cpu"
        model = _build_two_world_free_model(device)
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_mode="split", friction_anchor_beta=0.2, dense_max_constraints=4, mf_max_constraints=4
        )
        self.assertFalse(solver.pgs_warmstart)
        n = solver._fa_prev_valid.shape[0]
        worlds = np.arange(n, dtype=np.int32) % 2
        for mask, expect_cleared in (
            (None, (True, True)),
            ((True, False), (True, False)),
            ((False, True), (False, True)),
        ):
            solver._fa_prev_valid.fill_(1)
            solver._fa_prev_world.assign(worlds)
            wm = None if mask is None else wp.array(mask, dtype=wp.bool, device=device)
            solver.reset(model.state(), wm)
            valid = solver._fa_prev_valid.numpy()
            for world, cleared in enumerate(expect_cleared):
                sel = valid[worlds == world]
                np.testing.assert_array_equal(sel, 0 if cleared else 1, err_msg=f"mask={mask} world={world}")


if __name__ == "__main__":
    unittest.main()
