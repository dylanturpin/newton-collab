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
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 200.0
STEPS = 80
RADIUS = 0.1
# Spin transverse to the slide so omega x v points along +/-z, straight into
# the contact normal; spin parallel to the slide cannot excite the term.
OMEGA_Y = 20.0
SLIDE_VX = 2.0
PGS_MODES = ("split", "dense", "matrix_free")


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


def _slide_heights(model, omega_y, pgs_mode):
    """Slide the sphere at ``SLIDE_VX`` while spinning at ``omega_y``; return per-step COM heights [m]."""
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, omega_y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model, angular_damping=0.0, pgs_mode=pgs_mode)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()
    heights = []
    for _ in range(STEPS):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
    return np.asarray(heights)


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
        for pgs_mode in PGS_MODES:
            with self.subTest(pgs_mode=pgs_mode):
                still = _slide_heights(model, 0.0, pgs_mode)
                spinning = _slide_heights(model, OMEGA_Y, pgs_mode)
                divergence = float(np.abs(spinning - still).max())
                self.assertLess(divergence, 1e-4, f"spin changed a frictionless slide by {divergence} m")


if __name__ == "__main__":
    unittest.main()
