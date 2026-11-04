# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverSteel friction and restitution combine modes (``friction_combine``, ``restitution_combine``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.kernels import combine_shape_coefficients, mixed_contact_restitution
from newton._src.solvers.steel.solver_steel import _MATERIAL_COMBINE_MODES
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import steel_mode_options

COMBINE_MODES = ("average", "min", "multiply", "max")
RESPONSE_MODES = ("immediate", "propagation", "colored")
SLIDE_DT = 5.0e-3
SLIDE_STEPS = 40
SLIDE_SPEED = 3.0
SLIDE_HALF = 0.1
BOUNCE_DT = 1.0 / 240.0
BOUNCE_SPEED = 3.0
BOUNCE_RADIUS = 0.1


def _combined(mode, a, b):
    """Return the float32 reference combination of two coefficients."""
    a, b = np.float32(a), np.float32(b)
    return {"average": (a + b) / np.float32(2.0), "min": min(a, b), "multiply": a * b, "max": max(a, b)}[mode]


def _inertia(value):
    return wp.mat33(*(np.eye(3) * value).astype(np.float32).ravel())


def _body(builder, scene, xform, inertia):
    """Add a free body, or the root of a FREE-rooted two-link articulation."""
    if scene == "free":
        return builder.add_body(xform=xform, mass=1.0, inertia=inertia, lock_inertia=True)
    body = builder.add_link(xform=xform, mass=1.0, inertia=inertia, lock_inertia=True)
    child = builder.add_link(xform=xform, mass=0.25, inertia=_inertia(0.001), lock_inertia=True)
    builder.add_articulation([builder.add_joint_free(child=body), builder.add_joint_fixed(parent=body, child=child)])
    return body


def _run(model, solver, steps, dt, initial_qd, edit=None):
    """Step from ``initial_qd`` on the first DOFs and return the body twists after every step."""
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[: len(initial_qd)] = initial_qd
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    twists = []
    for step in range(steps):
        if edit is not None:
            edit(step)
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, dt)
        state_in, state_out = state_out, state_in
        twists.append(state_in.body_qd.numpy().copy())
    return twists


def _slide_box(device, *, scene, response, friction_combine, box_mu=0.8, ground_mu=0.4, edit_box_mu=None):
    """Slide a box across the ground and return its x speed after every step."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=ground_mu, restitution=0.0, gap=0.01))
    xform = wp.transform(wp.vec3(0.0, 0.0, SLIDE_HALF), wp.quat_identity())
    body = _body(builder, scene, xform, _inertia(2.0 / 3.0 * SLIDE_HALF * SLIDE_HALF))
    box_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=box_mu, restitution=0.0, gap=0.01)
    box = builder.add_shape_box(body, hx=SLIDE_HALF, hy=SLIDE_HALF, hz=SLIDE_HALF, cfg=box_cfg)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, iterations=32, **steel_mode_options(response), friction_combine=friction_combine)

    def edit(step):
        if edit_box_mu is not None and step == edit_box_mu[0]:
            mu = model.shape_material_mu.numpy()
            mu[box] = edit_box_mu[1]
            model.shape_material_mu.assign(mu)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    twists = _run(model, solver, SLIDE_STEPS, SLIDE_DT, [SLIDE_SPEED], edit)
    return np.array([twist[body][0] for twist in twists])


def _bounce(device, *, scene, response, restitution_combine, sphere_e=0.8, ground_e=0.4):
    """Drop a sphere onto the ground at ``BOUNCE_SPEED`` and return its peak rebound speed."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=ground_e))
    xform = wp.transform(wp.vec3(0.0, 0.0, BOUNCE_RADIUS + 0.5 * BOUNCE_SPEED * BOUNCE_DT), wp.quat_identity())
    body = _body(builder, scene, xform, _inertia(0.4 * BOUNCE_RADIUS * BOUNCE_RADIUS))
    sphere_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, restitution=sphere_e)
    builder.add_shape_sphere(body, radius=BOUNCE_RADIUS, cfg=sphere_cfg)
    model = builder.finalize(device=device)
    solver = SolverSteel(
        model,
        iterations=16,
        cfm=0.0,
        restitution_velocity_threshold=0.0,
        **steel_mode_options(response),
        restitution_combine=restitution_combine,
    )
    solver._rigid_body_angular_damping.zero_()
    twists = _run(model, solver, 4, BOUNCE_DT, [0.0, 0.0, -BOUNCE_SPEED])
    return max(float(twist[body][2]) for twist in twists)


def _assert_deceleration(test, speeds, mu, label, first=5, last=None):
    """Assert the box decelerates at mu * g between two steps while it still slides."""
    last = len(speeds) - 1 if last is None else last
    test.assertGreater(speeds[last], 0.1, f"{label}: the box stopped, so the slide no longer measures mu")
    deceleration = (speeds[first] - speeds[last]) / ((last - first) * SLIDE_DT)
    test.assertAlmostEqual(deceleration, mu * 9.81, delta=0.02 * mu * 9.81, msg=f"{label}: {deceleration:.4f} m/s^2")


@wp.kernel
def _combine_probe(
    shape_a: wp.array[int],
    shape_b: wp.array[int],
    coefficients: wp.array[float],
    combine: int,
    mixed: wp.array[float],
    restitution: wp.array[float],
):
    i = wp.tid()
    mixed[i] = combine_shape_coefficients(shape_a[i], shape_b[i], coefficients, combine)
    restitution[i] = mixed_contact_restitution(shape_a[i], shape_b[i], coefficients, combine)


class TestSteelMaterialCombine(unittest.TestCase):
    def test_combine_helpers_match_float32_references(self):
        """Combine pairs exactly per mode, keep a lone shape's value, and sanitize restitution first."""
        values = np.array([0.8, 0.4, 0.3, 1.7, np.nan, -0.5], dtype=np.float32)
        pairs = [(0, 1), (1, 0), (2, 3), (0, -1), (-1, 2), (-1, -1), (4, 0), (5, 3)]
        coefficients = wp.array(values, dtype=wp.float32, device="cpu")
        shape_a = wp.array([a for a, _ in pairs], dtype=wp.int32, device="cpu")
        shape_b = wp.array([b for _, b in pairs], dtype=wp.int32, device="cpu")
        for mode, mode_id in _MATERIAL_COMBINE_MODES.items():
            with self.subTest(mode=mode):
                mixed = wp.zeros(len(pairs), dtype=wp.float32, device="cpu")
                restitution = wp.zeros(len(pairs), dtype=wp.float32, device="cpu")
                wp.launch(
                    _combine_probe,
                    dim=len(pairs),
                    inputs=[shape_a, shape_b, coefficients, mode_id],
                    outputs=[mixed, restitution],
                    device="cpu",
                )
                for i, (a, b) in enumerate(pairs):
                    raw = [values[k] for k in (a, b) if k >= 0]
                    clean = [np.float32(np.clip(v, 0.0, 1.0)) if np.isfinite(v) else np.float32(0.0) for v in raw]
                    expected_mu = np.float32(0.0) if not raw else raw[0] if len(raw) == 1 else _combined(mode, *raw)
                    expected_e = (
                        np.float32(0.0) if not clean else clean[0] if len(clean) == 1 else _combined(mode, *clean)
                    )
                    if all(np.isfinite(raw)):  # Unsanitized non-finite mu is undefined input.
                        np.testing.assert_array_equal(mixed.numpy()[i], expected_mu, err_msg=f"pair {a},{b}")
                    np.testing.assert_array_equal(restitution.numpy()[i], expected_e, err_msg=f"pair {a},{b}")

    def test_rejects_unknown_combine_mode_and_stays_read_only(self):
        """Reject an unknown mode at construction and keep both options construction-only."""
        model = newton.ModelBuilder().finalize(device="cpu")
        for name in ("friction_combine", "restitution_combine"):
            with self.subTest(option=name):
                with self.assertRaisesRegex(ValueError, f"{name} must be one of"):
                    SolverSteel(model, **{name: "geometric"})
                solver = SolverSteel(model, **{name: "max"})
                self.assertEqual(getattr(solver, name), "max")
                with self.assertRaises(AttributeError):
                    setattr(solver, name, "min")


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestSteelMaterialCombineRoutes(unittest.TestCase):
    def test_friction_combine_modes_on_every_route(self):
        """Decelerate a sliding box at each mode's combined coefficient on the free and articulated routes."""
        for mode in COMBINE_MODES:
            mu = float(_combined(mode, 0.8, 0.4))
            for scene in ("free", "articulated"):
                for response in RESPONSE_MODES if scene == "articulated" else ("immediate",):
                    label = f"{mode}/{scene}/{response}"
                    with self.subTest(case=label):
                        speeds = _slide_box("cuda:0", scene=scene, response=response, friction_combine=mode)
                        _assert_deceleration(self, speeds, mu, label)

    def test_friction_combine_reads_notified_coefficient_edits(self):
        """Combine an in-place, notified shape friction edit under a non-average mode."""
        for scene, response in (("free", "immediate"), ("articulated", "immediate"), ("articulated", "propagation")):
            label = f"{scene}/{response}"
            with self.subTest(case=label):
                speeds = _slide_box(
                    "cuda:0", scene=scene, response=response, friction_combine="min", edit_box_mu=(20, 0.2)
                )
                _assert_deceleration(self, speeds, 0.4, f"{label}, before the edit", first=5, last=19)
                _assert_deceleration(self, speeds, 0.2, f"{label}, after the edit", first=22, last=39)

    def test_restitution_combine_modes_on_every_route(self):
        """Rebound at each mode's combined coefficient on the free and articulated routes."""
        for mode in COMBINE_MODES:
            expected = float(_combined(mode, 0.8, 0.4)) * BOUNCE_SPEED
            for scene in ("free", "articulated"):
                for response in RESPONSE_MODES if scene == "articulated" else ("immediate",):
                    label = f"{mode}/{scene}/{response}"
                    with self.subTest(case=label):
                        rebound = _bounce("cuda:0", scene=scene, response=response, restitution_combine=mode)
                        self.assertAlmostEqual(rebound, expected, delta=0.01 * BOUNCE_SPEED, msg=label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
