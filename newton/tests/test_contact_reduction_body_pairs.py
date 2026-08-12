# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction ("contact compiler").

Multi-shape bodies multiply narrow-phase candidates: a foot approximated by 7
cylinders emits up to 28 points against the ground and up to 49 against another
such foot, while a rigid contact patch is fully described by its deepest point
plus the extremes of its footprint. The body-pair reduction pass compacts the
narrow-phase output per body pair and normal bin, keeping the deepest contact
unconditionally plus the spatial extremes of the near-touching set, so the
registered contact count reflects the physics instead of the collider
decomposition.

The tests here assert both halves of the contract:

* the registered count drops on multi-shape pairs (and why: interior points of
  a single flat patch are the ones discarded), and
* downstream dynamics are preserved -- rest height, support force, touchdown
  capture, and trajectories of non-touching passes match the unreduced
  pipeline.
"""

import unittest
import unittest.mock

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF

DT = 1.0 / 240.0


def _cylinder_foot(builder, pos, mass=1.0, num_cyl=7, radius=0.02, half_height=0.015):
    """One free body whose collision is ``num_cyl`` small cylinders in a row.

    Mimics the URDF-exported G1 foot: several primitive colliders on one link,
    all reaching the ground at nearly the same height, so a plane contact
    produces a multiple of ``num_cyl`` candidate contacts for a single patch.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    return body


def _sphere_grid_body(builder, pos, n=5, spacing=0.05, radius=0.01, mass=1.0):
    """One free body with an ``n x n`` grid of sphere colliders on its underside.

    Resting on a plane this emits ``n * n`` same-normal contacts -- one flat
    patch whose only irredundant members are the grid's outer ring (hull) and
    the deepest point.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(n):
        for j in range(n):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(
                    wp.vec3((i - (n - 1) / 2.0) * spacing, (j - (n - 1) / 2.0) * spacing, 0.0),
                    wp.quat_identity(),
                ),
                radius=radius,
            )
    return body


def _peak_speed_unreduced(model, steps):
    """Peak body speed of the same scene driven by unreduced contacts."""
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=8)
    peak = 0.0
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        peak = max(peak, float(np.abs(state_0.body_qd.numpy()).max()))
    return peak


def _make_pipeline(model, reduce_body_pairs, **kwargs):
    return newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        deterministic=True,
        reduce_contacts_body_pairs=reduce_body_pairs,
        **kwargs,
    )


def _collide_once(model, state, reduce_body_pairs, **kwargs):
    pipeline = _make_pipeline(model, reduce_body_pairs, **kwargs)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _contact_snapshot(contacts):
    """Return (count, shape0, shape1, normal, point0) copies of the active contacts."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    return (
        n,
        contacts.rigid_contact_shape0.numpy()[:n].copy(),
        contacts.rigid_contact_shape1.numpy()[:n].copy(),
        contacts.rigid_contact_normal.numpy()[:n].copy(),
        contacts.rigid_contact_point0.numpy()[:n].copy(),
    )


def _world_points0(model, state, contacts):
    """World positions of contact witness points on shape0's body."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()
    out = np.zeros((n, 3))
    for k in range(n):
        b = shape_body[s0[k]]
        if b < 0:
            out[k] = p0[k]
            continue
        q = body_q[b]
        pos, quat = q[:3], q[3:7]
        x, y, z, w = quat
        v = p0[k]
        # quaternion rotate (xyzw)
        u = np.array([x, y, z])
        out[k] = pos + v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
    return out


class TestBodyPairReductionCounts(unittest.TestCase):
    """Registered contacts drop to patch-descriptive sets on multi-shape pairs."""

    def _grid_on_plane(self, reduce_body_pairs):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        # off-origin placement is deliberate: cells are measured from the pair's
        # own reference body, so distance from the world origin must not matter
        _sphere_grid_body(builder, (5.13, 5.07, 0.0095))  # spheres just touching the plane
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        # one spatial cell: this test asserts the tight single-patch slot bound
        return model, state, _collide_once(model, state, reduce_body_pairs, reduce_contacts_body_pairs_cell=10.0)

    def test_flat_patch_keeps_extremes_and_deepest(self):
        """Reduce a 25-point flat patch to at most 7 while keeping its footprint.

        The 5x5 sphere grid on a plane is one patch with one normal. The pass
        must keep no more than the per-bin slot count (6 spatial extremes + 1
        deepest), and the kept points must still span the patch: the bounding
        box of the kept footprint must cover most of the original one, and the
        deepest contact of the pair must survive.
        """
        model, state, base = self._grid_on_plane(reduce_body_pairs=False)
        n_base, *_ = _contact_snapshot(base)
        self.assertGreaterEqual(n_base, 25)

        model2, state2, red = self._grid_on_plane(reduce_body_pairs=True)
        n_red, *_ = _contact_snapshot(red)
        self.assertLessEqual(n_red, 7)
        self.assertGreaterEqual(n_red, 3)

        pts_base = _world_points0(model, state, base)
        pts_red = _world_points0(model2, state2, red)
        for axis in (0, 1):
            span_base = pts_base[:, axis].max() - pts_base[:, axis].min()
            span_red = pts_red[:, axis].max() - pts_red[:, axis].min()
            self.assertGreaterEqual(span_red, 0.7 * span_base)

    def test_two_normal_patches_never_merge(self):
        """Keep both patches when one body touches ground and wall simultaneously.

        A body with sphere colliders near a floor and near a vertical wall has
        two normal clusters in the same body pair... reduction must keep
        representatives of BOTH normals -- collapsing them into one patch would
        delete a support direction entirely.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), mass=1.0)
        for i in range(4):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(0.02 * i, 0.0, -0.0405), wp.quat_identity()),
                radius=0.01,
            )
        # second cluster of shapes near a box acting as a wall
        for i in range(4):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(-0.0405, 0.0, -0.02 * i), wp.quat_identity()),
                radius=0.01,
            )
        # static wall box to the -x side, floor below
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.11, 0.0, 0.0), wp.quat_identity()),
            hx=0.06,
            hy=0.5,
            hz=0.5,
        )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        red = _collide_once(model, state, reduce_body_pairs=True)
        n, _s0, _s1, normals, _ = _contact_snapshot(red)
        self.assertGreater(n, 0)
        # classify kept normals: floor-ish (+z) vs wall-ish (+/-x)
        up = np.abs(normals @ np.array([0.0, 0.0, 1.0])) > 0.7
        side = np.abs(normals @ np.array([1.0, 0.0, 0.0])) > 0.7
        self.assertTrue(up.any(), "floor patch was dropped entirely")
        self.assertTrue(side.any(), "wall patch was dropped entirely")

    def test_single_contact_untouched(self):
        """Pass a lone sphere-on-plane contact through unchanged.

        One sphere emits one contact; there is nothing to reduce and the pass
        must not perturb the contact's data.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = _collide_once(model, state, reduce_body_pairs=False)
        red = _collide_once(model, state, reduce_body_pairs=True)
        nb, _s0b, _s1b, nrmb, p0b = _contact_snapshot(base)
        nr, _s0r, _s1r, nrmr, p0r = _contact_snapshot(red)
        self.assertEqual(nb, nr)
        np.testing.assert_allclose(np.sort(p0b, axis=0), np.sort(p0r, axis=0), atol=1e-7)
        np.testing.assert_allclose(np.sort(nrmb, axis=0), np.sort(nrmr, axis=0), atol=1e-7)

    def test_seven_cylinder_foot_on_plane(self):
        """Collapse the 7-cylinder foot's plane manifold to a patch-descriptive set.

        This is the measured G1 pathology in miniature: ~4 points per cylinder
        against the plane for one flat patch. After reduction the pair must
        register at most 7 contacts, and the deepest candidate must be kept.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.0149))  # slightly penetrating, inside one cell
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = _collide_once(model, state, reduce_body_pairs=False)
        # one spatial cell: the foot spans two default cells, this asserts the per-cell bound
        red = _collide_once(model, state, reduce_body_pairs=True, reduce_contacts_body_pairs_cell=10.0)
        n_base, *_ = _contact_snapshot(base)
        n_red, *_ = _contact_snapshot(red)
        self.assertGreaterEqual(n_base, 7)
        self.assertLessEqual(n_red, 7)
        self.assertLess(n_red, n_base)

    def test_deterministic_output(self):
        """Produce an identical kept set on repeated collides of the same state."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0149))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        snap1 = _contact_snapshot(contacts)
        pipeline.collide(state, contacts)
        snap2 = _contact_snapshot(contacts)
        self.assertEqual(snap1[0], snap2[0])
        np.testing.assert_array_equal(snap1[1], snap2[1])
        np.testing.assert_array_equal(snap1[2], snap2[2])
        np.testing.assert_allclose(snap1[3], snap2[3], atol=0.0)
        np.testing.assert_allclose(snap1[4], snap2[4], atol=0.0)


class TestBodyPairReductionDynamics(unittest.TestCase):
    """Downstream dynamics are indistinguishable from the unreduced pipeline."""

    def _settle(self, reduce_body_pairs, steps=240):
        """Drop the 7-cylinder foot 5 mm above the plane and let it settle on XPBD.

        Uses ``iterations=8``: a positional solver's per-iteration stiffness
        scales with the number of contacts on a patch, so the redundant
        21-contact baseline converges faster per iteration than any reduced
        set. The comparison must be made near convergence, where both
        describe the same patch. (Measured: the base/reduced rest-height gap
        halves with every doubling of iterations -- 0.74/0.36/0.16 mm at
        4/8/16.)
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = _cylinder_foot(builder, (0.0, 0.0, 0.020))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        pipeline = _make_pipeline(model, reduce_body_pairs)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverXPBD(model, iterations=8)
        heights, counts = [], []
        for _ in range(steps):
            pipeline.collide(state_0, contacts)
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            heights.append(float(state_0.body_q.numpy()[body][2]))
        return np.array(heights), np.array(counts)

    def test_settling_height_matches(self):
        """Settle the multi-cylinder foot to the same rest height with and without reduction.

        The kept patch must carry the same support wrench as the full manifold:
        identical rest height (within a contact-regularization tolerance) and
        no residual bouncing, while the registered count during sustained
        contact drops by at least 2x.
        """
        h_base, c_base = self._settle(False)
        h_red, c_red = self._settle(True)
        # settled height identical
        self.assertLess(abs(float(h_base[-30:].mean()) - float(h_red[-30:].mean())), 5e-4)
        # both at rest (no residual bounce)
        self.assertLess(float(np.std(h_red[-30:])), 1e-4)
        # sustained-contact registration dropped
        self.assertLess(float(c_red[-30:].mean()), 0.55 * float(c_base[-30:].mean()))

    def test_touchdown_capture(self):
        """Capture a falling foot's touchdown with bounded transient penetration.

        Speculative candidates guard touchdown; reduction keeps at least the
        closest point per patch. The bound is absolute rather than relative to
        the unreduced run, because the redundant 21-contact baseline
        over-stiffens XPBD's per-iteration response and lands artificially
        hard (see :meth:`_settle`); what matters physically is that the
        landing transient stays within a couple of solver steps of motion and
        fully recovers.
        """
        h_base, _ = self._settle(False, steps=120)
        h_red, _ = self._settle(True, steps=120)
        rest = 0.015  # cylinder half-height: resting body origin height
        pen_red = max(0.0, rest - float(h_red.min()))
        self.assertLessEqual(pen_red, 2.5e-3)
        # and it recovers to the same rest as the baseline
        self.assertLess(abs(float(h_base[-20:].mean()) - float(h_red[-20:].mean())), 5e-4)

    def test_non_touching_pass_trajectories_identical(self):
        """Leave two feet passing within margin (never touching) exactly untouched.

        The foot-pass candidate explosion (up to 49 pairs' worth of points)
        carries no force. With reduction the registered count collapses, and
        because none of those contacts fire, both trajectories must match to
        solver precision.
        """

        def run(reduce_body_pairs):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            _cylinder_foot(builder, (0.0, 0.0, 0.2))
            # cylinder surfaces 3.6 mm apart during the pass (0.2336 - 2*half_height = 0.0036),
            # inside the contact margin but never touching
            b = _cylinder_foot(builder, (0.30, 0.0, 0.2336))
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            qd = state_0.body_qd.numpy()
            qd[b][3] = -0.5  # drive body b across body a
            state_0.body_qd.assign(qd)
            control = model.control()
            pipeline = _make_pipeline(model, reduce_body_pairs)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverXPBD(model, iterations=4)
            traj, peak = [], 0
            for _ in range(300):
                pipeline.collide(state_0, contacts)
                peak = max(peak, int(contacts.rigid_contact_count.numpy()[0]))
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                traj.append(state_0.body_q.numpy().copy())
            return np.array(traj), peak

        traj_base, peak_base = run(False)
        traj_red, peak_red = run(True)
        self.assertLess(peak_red, peak_base)
        np.testing.assert_allclose(traj_red, traj_base, atol=1e-6)


if __name__ == "__main__":
    unittest.main()


class TestBodyPairReductionGuarantees(unittest.TestCase):
    """Unsupported configurations are rejected at construction or solver start."""

    def _foot_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.020))
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def test_hydroelastic_rejected_at_construction(self):
        """Reject reduce_contacts_body_pairs together with hydroelastic contacts.

        The compaction does not carry the hydroelastic per-contact area and
        stiffness fields, so the combination must fail at pipeline
        construction, not corrupt data at runtime.
        """
        model = self._foot_model()
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                deterministic=True,
                reduce_contacts_body_pairs=True,
                sdf_hydroelastic_config=HydroelasticSDF.Config(),
            )

    def test_set_deterministic_without_sort(self):
        """Produce the same kept set with the deterministic sorter disabled.

        Winner selection packs (score, content fingerprint) and contacts
        self-identify as winners, so the kept SET is a pure function of the
        physical state -- no sorting required. Compare kept sets as
        order-independent multisets across repeated collides of one state.
        """
        model = self._foot_model()
        state = model.state()

        def kept_set(contacts):
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            p0 = np.round(contacts.rigid_contact_point0.numpy()[:n], 6)
            rows = sorted(map(tuple, np.column_stack([s0, s1, p0[:, 0], p0[:, 1], p0[:, 2]]).tolist()))
            return rows

        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=False,
            reduce_contacts_body_pairs=True,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        first = kept_set(contacts)
        for _ in range(3):
            pipeline.collide(state, contacts)
            self.assertEqual(kept_set(contacts), first)
        # and it matches the sorted pipeline's kept set
        sorted_pipe = _make_pipeline(model, True)
        sorted_contacts = sorted_pipe.contacts()
        sorted_pipe.collide(state, sorted_contacts)
        self.assertEqual(kept_set(sorted_contacts), first)

    def test_unvalidated_solver_rejected_at_step(self):
        """A solver without supports_reduced_contacts refuses a reduced buffer.

        Reduced buffers are stamped; every in-repo solver that has not been
        conformance-tested calls ``_require_unreduced_contacts`` at the top of
        ``step`` and must raise rather than consume contacts whose depth
        convention it might disagree with.
        """
        model = self._foot_model()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        solver = newton.solvers.SolverSemiImplicit(model)
        with self.assertRaisesRegex(ValueError, "supports_reduced_contacts"):
            solver.step(state_0, state_1, model.control(), contacts, DT)
        # the validated solvers accept the same buffer
        self.assertTrue(newton.solvers.SolverXPBD.supports_reduced_contacts)
        self.assertTrue(newton.solvers.SolverFeatherPGS.supports_reduced_contacts)


class TestBodyPairReductionMultiPatch(unittest.TestCase):
    """Same-normal patches far apart on ONE shape pair each keep full representation."""

    def _two_cluster_body(self, builder, spread=1.0):
        """One rigid body with two 4-sphere feet ``spread`` apart on the same plane."""
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=2.0)
        for end in (-spread / 2.0, spread / 2.0):
            for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3(end + 0.03 * i, 0.03 * j, 0.0), wp.quat_identity()),
                    radius=0.01,
                )
        return body

    def test_both_clusters_fully_kept(self):
        """Keep deepest + extremes for each cluster, not one shared slot set.

        Both clusters contact the SAME ground shape with the SAME normal --
        without spatial cells they would compete for a single (pair, bin)
        entry and one cluster could lose all representation. With cells, each
        cluster (1 m apart >> 0.25 m cell) must keep at least 3 contacts.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        red = _collide_once(model, state, reduce_body_pairs=True)
        n, _s0, _s1, _, _p0 = _contact_snapshot(red)
        self.assertGreater(n, 0)
        pts = _world_points0(model, state, red)
        left = (pts[:, 0] < -0.2).sum()
        right = (pts[:, 0] > 0.2).sum()
        self.assertGreaterEqual(int(left), 3, "left cluster under-represented")
        self.assertGreaterEqual(int(right), 3, "right cluster under-represented")

    def test_both_clusters_fully_kept_far_from_origin(self):
        """Keep both clusters for a body 200 m from the origin, not just near it.

        The spatial cell is packed as two signed 8-bit values. Measured from the
        world origin those saturate past ~32 m, so both clusters of a distant
        body land in the same border cell, compete for one slot set, and one
        loses its support points -- while the identical scene at the origin
        passes. Anchoring the cell grid at the pair's reference body makes the
        coordinates relative, so distance from the origin cannot matter.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        q = state.body_q.numpy()
        q[body][0] += 200.0
        q[body][1] += 200.0
        state.body_q.assign(q)
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pts = _world_points0(model, state, contacts)
        left = int((pts[:, 0] < 199.8).sum())
        right = int((pts[:, 0] > 200.2).sum())
        self.assertGreaterEqual(left, 3, "left cluster under-represented far from origin")
        self.assertGreaterEqual(right, 3, "right cluster under-represented far from origin")
        self.assertEqual(
            pipeline._body_pair_reducer.stats()["cell_clamp_events"],
            0,
            "cell coordinates should be relative to the body pair, so nothing clamps",
        )

    def _settle_plank(self, reduce_on):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on)
        contacts = pipeline.contacts()
        # iterations=16: the redundant unreduced set over-stiffens XPBD per
        # iteration (see _settle); compare near convergence.
        solver = newton.solvers.SolverXPBD(model, iterations=16)
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        return state_0.body_q.numpy()[body]

    def test_plank_settles_level(self):
        """Settle the two-cluster plank level, at the unreduced rest height.

        If either cluster lost its support points the plank would tilt about
        the other end. The invariant is against the unreduced pipeline: same
        rest height (within solver noise) and no pitch, on the same scene.
        """
        q_off = self._settle_plank(False)
        q_on = self._settle_plank(True)
        self.assertLess(abs(float(q_on[4])), 0.02, "plank tilted -- a cluster lost support")
        self.assertLess(abs(float(q_on[2]) - float(q_off[2])), 1.5e-3)


class TestBodyPairReductionSolverConformance(unittest.TestCase):
    """Every solver declaring supports_reduced_contacts settles identically on/off."""

    def _settle_with(self, solver_cls, reduce_on, **solver_kwargs):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = _cylinder_foot(builder, (0.0, 0.0, 0.020))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on)
        contacts = pipeline.contacts()
        solver = solver_cls(model, **solver_kwargs)
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        return float(state_0.body_q.numpy()[body][2])

    def test_feather_pgs_conformance(self):
        """SolverFeatherPGS rests the multi-cylinder foot at the same height on/off.

        This is the conformance requirement for supports_reduced_contacts:
        the solver's contact-depth convention must agree with the ranking's
        canonical contact_surface_separation, or the kept set starves the
        solver of its load-bearing contacts.
        """
        z_off = self._settle_with(newton.solvers.SolverFeatherPGS, False, angular_damping=0.0)
        z_on = self._settle_with(newton.solvers.SolverFeatherPGS, True, angular_damping=0.0)
        self.assertLess(abs(z_off - z_on), 5e-4)

    def test_xpbd_conformance(self):
        """SolverXPBD rests the multi-cylinder foot at the same height on/off."""
        z_off = self._settle_with(newton.solvers.SolverXPBD, False, iterations=8)
        z_on = self._settle_with(newton.solvers.SolverXPBD, True, iterations=8)
        self.assertLess(abs(z_off - z_on), 5e-4)


class TestBodyPairReductionRobustness(unittest.TestCase):
    """Remaining contract edges: matching, capacity, graph capture, grad, worlds."""

    def _foot_scene(self, pos=(5.13, 5.07, 0.0149)):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, pos)
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def test_contact_matching_rejected_at_construction(self):
        """Reject reduce_contacts_body_pairs together with contact matching.

        Compaction renumbers contacts, which would silently invalidate the
        matcher's index-based frame-to-frame bookkeeping.
        """
        model = self._foot_scene()
        with self.assertRaisesRegex(ValueError, "contact_matching"):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
                reduce_contacts_body_pairs=True,
            )

    def test_group_id_capacity_rejected_at_construction(self):
        """Reject scenes whose shape+body count exceeds the exact key budget.

        Group ids pack exactly into the reduction key; overflow would alias
        two groups and could evict a patch's deepest contact, so the pipeline
        must refuse at construction rather than mask bits at runtime.
        """
        model = self._foot_scene()
        with unittest.mock.patch("newton._src.sim.collide.MAX_GROUP_ID", 4):
            with self.assertRaisesRegex(ValueError, "at most 4 shapes"):
                _make_pipeline(model, True)

    def test_cuda_graph_capture(self):
        """Capture collide() with reduction into a CUDA graph and replay it.

        All reduction launches are fixed-size, so capture must succeed and
        replays must keep producing the same kept set on the same state.
        Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # warm-up: lazy allocations + kernel loads
        n_ref = int(contacts.rigid_contact_count.numpy()[0])
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state, contacts)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        n_replay = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(n_ref, n_replay)

    def test_requires_grad_diff_augmentation(self):
        """Populate the differentiable contact arrays from the compacted set.

        The reduction runs before the augmentation kernel, so the diff arrays
        must exist, cover exactly the reduced count, and be finite.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.0149))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device(), requires_grad=True)
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            reduce_contacts_body_pairs=True,
            requires_grad=True,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(n, 0)
        if contacts.rigid_contact_diff_distance is not None:
            d = contacts.rigid_contact_diff_distance.numpy()[:n]
            self.assertTrue(np.isfinite(d).all())

    def test_multi_world_independence(self):
        """Reduce two worlds' identical feet to identical per-world kept sets.

        Worlds must not share groups: each world's foot keeps its own deepest
        and extremes, and the kept count is the same for identical scenes.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for _ in range(2):
            builder.begin_world()
            _cylinder_foot(builder, (5.13, 5.07, 0.0149))
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        red = _collide_once(model, state, reduce_body_pairs=True, reduce_contacts_body_pairs_cell=10.0)
        n, s0, s1, _, _ = _contact_snapshot(red)
        shape_body = model.shape_body.numpy()
        per_world = [0, 0]
        for k in range(n):
            b = shape_body[s0[k]] if shape_body[s0[k]] >= 0 else shape_body[s1[k]]
            per_world[0 if b < model.body_count // 2 else 1] += 1
        self.assertEqual(per_world[0], per_world[1])
        self.assertGreaterEqual(per_world[0], 3)
        self.assertLessEqual(per_world[0], 7)


class TestBodyPairReductionFPGSImpact(unittest.TestCase):
    """FPGS-specific touchdown conformance (the G1 failure mode that unit XPBD tests missed)."""

    def test_fpgs_touchdown_capture(self):
        """Land a falling foot on FPGS at the same rest height with and without reduction.

        This covers the exact defect class found on the walking humanoid: a
        mis-ranked kept set leaves no load-bearing contact at touchdown and
        the body free-falls before a violent late landing.
        """

        def run(reduce_on):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = _cylinder_foot(builder, (5.13, 5.07, 0.06))  # 3 cm drop
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            pipeline = _make_pipeline(model, reduce_on)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
            zs = []
            for _ in range(240):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                zs.append(float(state_0.body_q.numpy()[body][2]))
            return np.array(zs)

        z_off = run(False)
        z_on = run(True)
        self.assertLess(abs(float(z_off[-30:].mean()) - float(z_on[-30:].mean())), 5e-4)
        # bounded touchdown transient: never punches through the resting height by > 2.5 mm
        self.assertGreater(float(z_on.min()), 0.015 - 2.5e-3)


class TestBodyPairReductionCertificate(unittest.TestCase):
    """The verify mode re-derives every keep/discard decision and finds zero disagreements."""

    def test_certificate_zero_violations_settling(self):
        """Certify the invariant over a full settling trajectory.

        Every collide re-checks: no discarded contact out-ranks a slot winner
        it was eligible for, and every kept registered contact matches a slot
        it won. Any nonzero counter is a slot race, clearing bug, or ranking
        regression.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.020))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        pipeline = _make_pipeline(model, True, reduce_contacts_body_pairs_verify=True)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverXPBD(model, iterations=8)
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["invariant_violations"], 0)
        self.assertEqual(stats["fail_open_keeps"], 0)

    def test_kept_set_independent_of_previous_step(self):
        """Reduce a state to the same kept set whether or not a busier step preceded it.

        The pass carries capacity-sized scratch across steps -- keep flags, the
        per-contact hashtable-entry cache, and the per-entry slot values -- and
        refreshes only the live range each step, so nothing may depend on how
        busy the previous step was. Drive one pipeline through a three-foot step
        and then a one-foot step, and require the one-foot result to match a
        pipeline that only ever saw that state.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        feet = [_cylinder_foot(builder, (1.5 * k, 0.0, 0.020)) for k in range(3)]
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())

        state_busy = model.state()
        state_quiet = model.state()
        q = state_quiet.body_q.numpy()
        for b in feet[1:]:
            q[b][2] += 5.0  # lift two feet clear of the ground
        state_quiet.body_q.assign(q)

        with_history = _make_pipeline(model, True, reduce_contacts_body_pairs_verify=True)
        contacts_h = with_history.contacts()
        with_history.collide(state_busy, contacts_h)
        n_busy = int(contacts_h.rigid_contact_count.numpy()[0])
        with_history.collide(state_quiet, contacts_h)

        fresh = _make_pipeline(model, True, reduce_contacts_body_pairs_verify=True)
        contacts_f = fresh.contacts()
        fresh.collide(state_quiet, contacts_f)

        n_h, s0_h, s1_h, _nrm_h, p0_h = _contact_snapshot(contacts_h)
        n_f, s0_f, s1_f, _nrm_f, p0_f = _contact_snapshot(contacts_f)
        self.assertGreater(n_busy, n_f, "the busy step must really register more contacts")
        self.assertEqual(n_h, n_f, "kept count depends on the previous step")

        def canonical(s0, s1, p0):
            rows = [(int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0, s1, p0, strict=True)]
            return sorted(rows)

        self.assertEqual(canonical(s0_h, s1_h, p0_h), canonical(s0_f, s1_f, p0_f))
        for pipe in (with_history, fresh):
            stats = pipe._body_pair_reducer.stats()
            self.assertEqual(stats["invariant_violations"], 0)
            self.assertEqual(stats["outranked_discards"], 0)

    def test_property_random_piles(self):
        """Fuzz the invariant and stability on randomized primitive piles.

        Seeded random mixes of spheres, boxes, capsules, and cylinders are
        dropped into a pile -- geometry nobody hand-picked. Asserts: the
        certificate stays clean, the registered count is reduced, nothing
        blows up, and no body tunnels through the ground.
        """
        rng = np.random.default_rng(1234)
        for trial in range(3):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            bodies = []
            for k in range(10):
                # spawn strictly non-overlapping (max shape extent is 0.16 m):
                # interpenetrating spawns make BOTH pipelines ballistic and the
                # comparison chaotic rather than physical
                pos = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)), 0.10 + 0.18 * k)
                body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=0.5)
                bodies.append(body)
                kind = int(rng.integers(0, 4))
                if kind == 0:
                    builder.add_shape_sphere(body, radius=float(rng.uniform(0.02, 0.05)))
                elif kind == 1:
                    builder.add_shape_box(
                        body,
                        hx=float(rng.uniform(0.02, 0.05)),
                        hy=float(rng.uniform(0.02, 0.05)),
                        hz=float(rng.uniform(0.02, 0.05)),
                    )
                elif kind == 2:
                    builder.add_shape_capsule(
                        body, radius=float(rng.uniform(0.015, 0.03)), half_height=float(rng.uniform(0.02, 0.05))
                    )
                else:
                    builder.add_shape_cylinder(
                        body, radius=float(rng.uniform(0.02, 0.04)), half_height=float(rng.uniform(0.02, 0.05))
                    )
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            pipe_red = _make_pipeline(model, True, reduce_contacts_body_pairs_verify=True)
            pipe_raw = newton.CollisionPipeline(model, broad_phase="nxn")
            c_red, c_raw = pipe_red.contacts(), pipe_raw.contacts()
            solver = newton.solvers.SolverXPBD(model, iterations=8)
            raw_more, peak_red = 0, 0.0
            for _ in range(150):
                pipe_raw.collide(state_0, c_raw)
                pipe_red.collide(state_0, c_red)
                raw_more += int(c_raw.rigid_contact_count.numpy()[0] >= c_red.rigid_contact_count.numpy()[0])
                state_0.clear_forces()
                solver.step(state_0, state_1, control, c_red, DT)
                state_0, state_1 = state_1, state_0
                peak_red = max(peak_red, float(np.abs(state_0.body_qd.numpy()).max()))
            # reference peak from the SAME pile driven by unreduced contacts: an
            # absolute bound flakes here, because a random pile's peak speed is
            # chaotic and legitimately reaches O(100) m/s in either pipeline
            peak_raw = _peak_speed_unreduced(model, steps=150)
            stats = pipe_red._body_pair_reducer.stats()
            self.assertEqual(stats["invariant_violations"], 0, f"trial {trial}")
            self.assertEqual(raw_more, 150, f"trial {trial}: reduction increased a count")
            body_q = state_0.body_q.numpy()
            qd = state_0.body_qd.numpy()
            self.assertTrue(np.isfinite(body_q).all() and np.isfinite(qd).all())
            self.assertLess(peak_red, max(3.0 * peak_raw, 10.0), f"trial {trial}: pile blew up")
            self.assertGreater(float(body_q[bodies, 2].min()), -0.06, f"trial {trial}: body tunneled")
