# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Predictor/integrator agreement for a spinning, translating free base.

``jcalc_integrate`` adds the ``omega x v`` transport term to the free root's
linear coordinate (see :mod:`newton.tests.test_feather_pgs_free_base_momentum`
for why that term must exist).  Every other place that converts between the
root's velocity coordinate and its acceleration must use the same convention:

* the velocity predictor ``v_hat = qd + qdd * dt`` feeds contact, friction,
  and velocity-limit rows, so without the term those rows are built against a
  COM velocity the integrator never realizes -- off by ``dt * (omega x v)``;
* the post-solve conversion ``qdd = (v_out - qd) / dt`` decides what the
  integrator receives, so without the inverse term the realized velocity is
  ``v_out + dt * (omega x v)``, not the velocity the solver actually chose.

Contact-free runs cannot see the mismatch (the two conversions cancel), which
is how the momentum suite stayed green over it.  Both tests here have exact
ground truths that hold only when the conventions agree.

The anchored-spinner test pins the related free-joint lever defect: the
integrator's anchor-to-COM lever assumed the joint's child anchor sits at the
child body's origin, ignoring a non-identity ``child_xform``.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import integrate_generalized_joints
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 200.0
STEPS = 80
RADIUS = 0.1
# Spin transverse to the slide so omega x v points along +/-z, straight into
# the contact normal; spin parallel to the slide cannot excite the term.
OMEGA_Y = 20.0
SLIDE_VX = 2.0
# (pgs_mode, articulated_contact_response, pgs_velocity_iterations): the split
# velocity post-pass converts velocity to acceleration at two more patched
# sites, and each propagation response variant is a separately generated
# contact path
SLIDER_CONFIGS = [
    ("split", "immediate", 0),
    ("dense", "immediate", 0),
]
SLIDER_CONFIGS_CUDA = [
    ("matrix_free", "immediate", 0),
    ("matrix_free", "immediate", 2),
    ("matrix_free", "propagation", 0),
    ("matrix_free", "propagation-fused", 0),
    ("matrix_free", "propagation-colored", 0),
]


def _build_slider(device):
    """A frictionless sphere resting on a frictionless ground plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
    )
    builder.add_shape_sphere(body, radius=RADIUS)
    free = builder.add_joint_free(body)
    builder.add_articulation([free])
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _slide_heights(model, omega_y, pgs_mode, response="immediate", velocity_iterations=0):
    """Slide the sphere at ``SLIDE_VX`` while spinning at ``omega_y``.

    Returns per-step COM heights [m] and the peak live contact count, so the
    caller can prove the comparison exercised actual contact rows rather than
    two contact-free (and therefore trivially matching) runs.
    """
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, omega_y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(
        model,
        angular_damping=0.0,
        pgs_mode=pgs_mode,
        articulated_contact_response=response,
        pgs_velocity_iterations=velocity_iterations,
    )
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()
    heights = []
    peak_contacts = 0
    for _ in range(STEPS):
        pipeline.collide(state_0, contacts)
        peak_contacts = max(peak_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
    return np.asarray(heights), peak_contacts


def _world_com(model, state):
    """World position of the root body's centre of mass [m]."""
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)[0]
    q = state.body_q.numpy().astype(np.float64).reshape(-1, 7)[0]
    rot = wp.quat(*(float(c) for c in q[3:7]))
    return np.asarray(q[0:3]) + np.asarray(wp.quat_rotate(rot, wp.vec3(*com_local)))


def _com_positions(model, state_0, state_1, steps):
    """Step the model contact-free; return per-step world COM positions [m]."""
    solver = SolverFeatherPGS(model, angular_damping=0.0)
    control = model.control()
    positions = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        positions.append(_world_com(model, state_0))
    return np.asarray(positions)


class TestFeatherPgsFreeRootPredictor(unittest.TestCase):
    """Constraint rows must see the COM velocity the integrator realizes."""

    def test_frictionless_slide_is_spin_invariant(self):
        """Spinning a frictionless sliding sphere must not change its height trajectory.

        Without friction the spin exerts no force, so the height trajectory is
        exactly that of the unspun sphere.  A predictor missing the
        ``omega x v`` term feeds the normal rows a phantom vertical velocity of
        ``dt * omega_y * vx`` (0.2 m/s here): the sphere's height diverges by
        millimetres within the run, in every ``pgs_mode``.  With the conventions
        agreeing the trajectories match to solver precision.
        """
        model = _build_slider(wp.get_device())
        configs = list(SLIDER_CONFIGS)
        if wp.get_device().is_cuda:
            configs += SLIDER_CONFIGS_CUDA
        for pgs_mode, response, vel_iters in configs:
            with self.subTest(pgs_mode=pgs_mode, response=response, velocity_iterations=vel_iters):
                still, contacts_still = _slide_heights(model, 0.0, pgs_mode, response, vel_iters)
                spinning, contacts_spin = _slide_heights(model, OMEGA_Y, pgs_mode, response, vel_iters)
                # the comparison is vacuous unless both runs really ride on
                # active contact rows at the rest height
                self.assertGreater(min(contacts_still, contacts_spin), 0, "no contacts were generated")
                for label, h in (("still", still), ("spinning", spinning)):
                    self.assertLess(float(np.abs(h - RADIUS).max()), 2e-3, f"{label} run left the ground support band")
                divergence = float(np.abs(spinning - still).max())
                self.assertLess(divergence, 1e-4, f"spin changed a frictionless slide by {divergence} m")

    def _anchored_spinner_drift(self, child_xform, com, omega=(0.0, 0.0, 10.0)):
        """Max world COM drift [m] of a force-free spinner with an offset joint anchor."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=com,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014),
        )
        free = builder.add_joint_free(body, child_xform=child_xform)
        builder.add_articulation([free])
        model = builder.finalize(device=wp.get_device())

        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[3:6] = omega
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        start = _world_com(model, state_0)

        positions = _com_positions(model, state_0, state_1, STEPS)
        return float(np.linalg.norm(positions - start[None, :], axis=1).max())

    def test_anchored_spinner_com_stays_put(self):
        """A force-free spinning body with an offset joint anchor keeps its COM in place.

        The free joint's coordinate tracks the child ANCHOR frame, offset from
        the body's COM by ``child_xform``.  With zero COM velocity and no
        forces, the COM must stay put and the anchor must orbit it.  A lever
        built from ``body_com`` alone pins the anchor instead, so the COM
        orbits at the anchor-offset radius -- a 0.2 m excursion within half a
        turn, from nothing; advancing the anchor with the linearized lever
        velocity still leaked ``omega * r * dt`` = 5 mm of truncation.  The
        anchor is now reconstructed from the directly integrated COM
        (``p_new = x_com_new - R_new * r_ac``, as SolverFeatherstone does), so
        a force-free COM holds to roundoff at ANY step size and the bound can
        sit far below both failure modes.  The second configuration adds a
        rotated anchor and a non-collinear COM offset, so both transform
        compositions in the lever are exercised, with a tumbling-capable
        triaxial inertia.
        """
        configs = [
            ("plain-offset", wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)),
            (
                "rotated-anchor-offset-com",
                wp.transform(wp.vec3(0.1, 0.02, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)),
                wp.vec3(0.03, -0.05, 0.02),
            ),
        ]
        for label, child_xform, com in configs:
            with self.subTest(config=label):
                drift = self._anchored_spinner_drift(child_xform, com)
                self.assertLess(drift, 1e-4, f"COM of a force-free spinner drifted {drift} m")

    def test_integrator_transport_identities(self):
        """Pin the integrator's root transport and descendant pass-through directly.

        A single launch of ``integrate_generalized_joints`` with zero
        ``joint_qdd`` must realize the two velocity identities: the ROOT free
        joint's linear coordinate gains exactly ``omega x v * dt`` (the
        transport term the predictor and the qdd conversion mirror), and a
        DESCENDANT free joint's linear coordinate passes through unchanged --
        its coordinate is a relative twist in the parent anchor frame where
        the root rule is not derived, so the branch is deliberately
        component-wise.  (End-to-end descendant momentum is NOT asserted
        anywhere: a force-free nested free body's world COM velocity drifts
        through the dynamics stages independently of integration -- a
        pre-existing gap outside this fix's scope.)
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        )
        child = builder.add_link(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        )
        root = builder.add_joint_free(parent)
        nested = builder.add_joint_free(child, parent=parent)
        builder.add_articulation([root, nested])
        model = builder.finalize(device=wp.get_device())

        state = model.state()
        joint_qd = state.joint_qd.numpy()
        joint_qd[0:3] = (0.4, 0.0, 0.0)
        joint_qd[3:6] = (0.0, 0.0, 3.0)
        joint_qd[6:9] = (0.4, 0.0, 0.0)
        joint_qd[9:12] = (0.0, 0.0, 3.0)
        state.joint_qd.assign(joint_qd)

        device = wp.get_device()
        q_new = wp.zeros_like(state.joint_q)
        qd_new = wp.zeros_like(state.joint_qd)
        wp.launch(
            integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                wp.zeros(model.joint_count, dtype=wp.int32, device=device),
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state.joint_q,
                state.joint_qd,
                wp.zeros_like(state.joint_qd),  # joint_qdd = 0
                DT,
                0.0,
            ],
            outputs=[q_new, qd_new],
            device=device,
        )
        out = qd_new.numpy()
        transport = np.cross((0.0, 0.0, 3.0), (0.4, 0.0, 0.0)) * DT
        np.testing.assert_allclose(out[0:3], np.array((0.4, 0.0, 0.0)) + transport, rtol=1e-6)
        np.testing.assert_allclose(out[3:6], (0.0, 0.0, 3.0), atol=1e-7)
        np.testing.assert_allclose(out[6:9], (0.4, 0.0, 0.0), atol=1e-7, err_msg="descendant gained transport")
        np.testing.assert_allclose(out[9:12], (0.0, 0.0, 3.0), atol=1e-7)

    def test_captured_step_matches_uncaptured(self):
        """Replaying a captured collide+step reproduces the uncaptured trajectory.

        The transport corrections are fixed-size device launches guarded by a
        host-side flag, so they must record and replay exactly; a divergence
        here means part of the correction ran host-side.  Skipped on CPU
        devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        heights_ref, _ = _slide_heights(_build_slider(device), OMEGA_Y, "matrix_free")

        model = _build_slider(device)
        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
        joint_qd[3:6] = (0.0, OMEGA_Y, 0.0)
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_mode="matrix_free")
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        def one_step():
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            # copy back instead of swapping so the captured pointers are stable
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)

        one_step()  # warm-up, matches the reference run's first step
        with wp.ScopedCapture(device) as capture:
            one_step()  # capture RECORDS this step without executing it
        heights = []
        for _ in range(STEPS - 1):
            wp.capture_launch(capture.graph)
            heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
        np.testing.assert_allclose(heights, heights_ref[1:], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
