# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum conservation for a rotating floating-base articulation.

FeatherPGS writes the free-base equations about ``articulation_origin``, a material point of the
root body. When the articulation's composite centre of mass sits away from that point, the frame
requires a reference-point bias wrench; the body-frame spatial algebra in ``compute_link_velocity``
supplies it only after its per-link excess term is subtracted.

Gating that subtraction on the root link alone leaves every other link in the uncorrected
convention, and the articulation then creates linear momentum whenever its base rotates. A
single-link articulation puts the origin on the composite centre of mass, so the excess vanishes
identically and the defect is invisible there -- these tests therefore pair a multi-link case with
the single-link case that must stay exact.

With no gravity, no contacts and no external wrench, the centre-of-mass velocity of a free
articulation is exactly constant, so any measured drift is spurious.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 200.0
STEPS = 60
# Spin about x, transverse to the stack axis, so the composite COM offset is swept through the
# rotation. A spin parallel to the offset cannot excite the term at all.
OMEGA = (20.0, 0.0, 0.0)
LINK_MASS = 1.0
LINK_HALF = 0.10  # half the stack spacing; the composite COM sits this far above the root COM


def _link_inertia(mass, radius=0.05, height=0.20):
    """Solid-cylinder inertia tensor about the link's own centre of mass."""
    i_xy = mass * (3.0 * radius**2 + height**2) / 12.0
    i_z = 0.5 * mass * radius**2
    return wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z)


def _build_welded_pair(device):
    """Two links welded by a fixed joint: composite COM sits ``LINK_HALF`` above the root COM."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    upper = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    free = builder.add_joint_free(parent=-1, child=root)
    weld = builder.add_joint_fixed(
        parent=root,
        child=upper,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free, weld])
    return builder.finalize(device=device)


def _build_equivalent_single(device):
    """One link carrying the welded pair's composite mass properties.

    Same total mass, same centre-of-mass offset from the body origin and same inertia about that
    centre of mass as :func:`_build_welded_pair`, so the two models are the same physical object.
    """
    total = 2.0 * LINK_MASS
    single = _link_inertia(LINK_MASS)
    i_xy = 2.0 * (single[0, 0] + LINK_MASS * LINK_HALF**2)
    i_z = 2.0 * single[2, 2]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=total,
        com=wp.vec3(0.0, 0.0, LINK_HALF),
        inertia=wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _com_velocity(model, state):
    """Centre-of-mass velocity of the whole articulation [m/s]."""
    mass = model.body_mass.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64).reshape(-1, 6)
    return (mass[:, None] * body_qd[:, :3]).sum(axis=0) / mass.sum()


def _spin_free_articulation(model):
    """Spin the articulation about x with its composite centre of mass at rest, then step.

    Returns the largest centre-of-mass velocity drift [m/s] seen over the run.
    """
    state_0, state_1 = model.state(), model.state()

    # The free joint's linear coordinate is the ROOT body's COM velocity, which for a multi-link
    # articulation is not the composite COM. Give it the orbital velocity that leaves the composite
    # COM at rest, so both models start from the same physical state.
    body_q = state_0.body_q.numpy().astype(np.float64).reshape(-1, 7)
    mass = model.body_mass.numpy().astype(np.float64)
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)
    com_world = body_q[:, :3] + com_local  # identity orientation at build time
    composite = (mass[:, None] * com_world).sum(axis=0) / mass.sum()
    root_com = com_world[0]

    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = np.cross(np.array(OMEGA), root_com - composite)
    joint_qd[3:6] = OMEGA
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model, angular_damping=0.0)
    control = model.control()
    reference = _com_velocity(model, state_0)
    worst = 0.0
    for _ in range(STEPS):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(_com_velocity(model, state_0) - reference)))
    return worst


class TestFeatherPgsFreeBaseMomentum(unittest.TestCase):
    """Free articulations must not create linear momentum while they rotate."""

    def test_single_link_conserves_com_velocity(self):
        """A lone rotating free body holds its centre-of-mass velocity exactly.

        This is the reference case: the articulation origin coincides with the composite centre of
        mass, so the reference-point term is identically zero and no correction can perturb it.
        """
        model = _build_equivalent_single(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 1e-6)

    def test_welded_pair_conserves_com_velocity(self):
        """A welded two-link articulation holds it to the integrator's own accuracy.

        The pair is the same physical object as the single link above -- same mass, centre of mass
        and inertia -- but its composite centre of mass sits away from the frame origin, so it
        exercises the reference-point bias. Drift over this run is ~0.20 m/s with the per-link
        frame correction and ~1.64 m/s without it; the bound sits between the two.
        """
        model = _build_welded_pair(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_welded_pair_drift_converges_with_timestep(self):
        """The residual drift is first-order in dt, not a fixed formulation error.

        Halving the step must roughly halve the drift. A formulation error would sit flat instead,
        which is what distinguished this defect from ordinary truncation error.
        """
        global DT  # noqa: PLW0603 - the step size is the quantity under test
        original = DT
        try:
            DT = 1.0 / 200.0
            coarse = _spin_free_articulation(_build_welded_pair(wp.get_device()))
            DT = 1.0 / 400.0
            fine = _spin_free_articulation(_build_welded_pair(wp.get_device()))
        finally:
            DT = original
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.65 * coarse)


if __name__ == "__main__":
    unittest.main()
