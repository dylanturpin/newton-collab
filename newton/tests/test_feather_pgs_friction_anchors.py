"""Tests for FeatherPGS positional friction anchors (``friction_anchor_beta``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_FRICTION, update_friction_anchors

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


def _launch_anchor_update(device, *, point0, point1, match_index, prev, shape_body=(0, 1), body_q=None):
    """Drive ``update_friction_anchors`` on one contact; returns (anchor_a, anchor_b, valid, phi) arrays."""
    if body_q is None:
        body_q = [wp.transform_identity(), wp.transform_identity()]
    n = 1
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
            wp.array([wp.vec3(*point0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(*point1)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device),  # A-to-B; solver normal = +z
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros(n, dtype=wp.float32, device=device),
            wp.zeros(n, dtype=wp.float32, device=device),
            wp.array([match_index], dtype=wp.int32, device=device),
            wp.array(list(shape_body), dtype=wp.int32, device=device),
            wp.array(body_q, dtype=wp.transform, device=device),
            prev_a,
            prev_b,
            prev_valid,
            0.005,
        ],
        outputs=list(outs),
        device=device,
    )
    return outs


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
        point0, point1 = (1.0e-4, 0.0, 0.0), (0.0, 0.0, -1.0e-3)
        empty = (
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=wp.vec3, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),  # previous anchor invalidated
        )
        a, b, valid, phi = _launch_anchor_update(device, point0=point0, point1=point1, match_index=0, prev=empty)
        self.assertEqual(int(valid.numpy()[0]), 1)
        np.testing.assert_allclose(phi.numpy()[0], 0.0, atol=1.0e-9)
        # the stored pair is aligned along the normal: no tangential separation
        sep = a.numpy()[0] - b.numpy()[0]
        np.testing.assert_allclose(sep[:2], 0.0, atol=1.0e-9)
        # next step, same replayed points, matched to the pair just stored -> still zero
        _, _, valid2, phi2 = _launch_anchor_update(
            device, point0=point0, point1=point1, match_index=0, prev=(a, b, valid)
        )
        self.assertEqual(int(valid2.numpy()[0]), 1)
        np.testing.assert_allclose(phi2.numpy()[0], 0.0, atol=1.0e-9)
        # a genuine 0.2 mm tangential move of body A after anchoring is reported
        moved = [wp.transform(wp.vec3(2.0e-4, 0.0, 0.0), wp.quat_identity()), wp.transform_identity()]
        _, _, _, phi3 = _launch_anchor_update(
            device, point0=point0, point1=point1, match_index=0, prev=(a, b, valid), body_q=moved
        )
        self.assertAlmostEqual(float(np.linalg.norm(phi3.numpy()[0])), 2.0e-4, places=9)

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
