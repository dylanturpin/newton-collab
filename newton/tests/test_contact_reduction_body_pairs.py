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
from newton._src.geometry.contact_reduction_body_pairs import _BP_FACE_NORMALS_DATA, _up_axis_rotation
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF
from newton._src.sim.contacts import Contacts

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


def _free_jointed_foot(builder, pos, num_cyl=7, radius=0.02, half_height=0.015):
    """A cylinder-row foot attached to the world by an explicit free joint.

    :class:`newton.solvers.SolverFeatherPGS` does not simulate unjointed
    massive bodies as floating bodies (they stay frozen), so every FPGS
    dynamics test must build its bodies this way or it validates nothing.
    """
    link = builder.add_link(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            link,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    joint = builder.add_joint_free(parent=-1, child=link)
    builder.add_articulation([joint])
    return link


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

    def _settle(self, build_fn, make_solver, reduce_on, steps=240):
        """Drop, settle, and return (final z, fell, peak contact count).

        The fell/contact guards make the comparison non-vacuous: a body a
        solver silently refuses to simulate (e.g. an unjointed massive body on
        FeatherPGS) neither falls nor proves anything by matching heights.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = build_fn(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on)
        contacts = pipeline.contacts()
        solver = make_solver(model)
        z0 = float(state_0.body_q.numpy()[body][2])
        peak_contacts = 0
        for k in range(steps):
            pipeline.collide(state_0, contacts)
            if k % 20 == 0:
                peak_contacts = max(peak_contacts, int(contacts.rigid_contact_count.numpy()[0]))
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        z_end = float(state_0.body_q.numpy()[body][2])
        return z_end, z_end < z0 - 0.005, peak_contacts

    def _compare(self, build_fn, make_solver, tol=5e-4):
        z_off, fell_off, contacts_off = self._settle(build_fn, make_solver, False)
        z_on, fell_on, contacts_on = self._settle(build_fn, make_solver, True)
        self.assertTrue(fell_off and fell_on, "body did not fall: the solver is not simulating it")
        self.assertGreater(contacts_off, 0)
        self.assertGreater(contacts_on, 0)
        self.assertLess(abs(z_off - z_on), tol)

    def test_feather_pgs_conformance(self):
        """SolverFeatherPGS rests a free-jointed foot at the same height on/off.

        This is the conformance requirement for supports_reduced_contacts:
        the solver's contact-depth convention must agree with the ranking's
        canonical contact_surface_separation, or the kept set starves the
        solver of its load-bearing contacts. The foot is attached by an
        explicit free joint -- FeatherPGS does not simulate unjointed massive
        bodies, so the previous add_body version compared two frozen bodies.
        """
        self._compare(
            lambda b: _free_jointed_foot(b, (0.0, 0.0, 0.05)),
            lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0),
        )

    def test_feather_pgs_dynamic_dynamic_conformance(self):
        """Settle a free-jointed foot dropped onto another free-jointed foot.

        Covers articulated dynamic-dynamic contact: both the falling body and
        its support are simulated bodies, so the reduced set must preserve the
        body-body patch as well as the body-ground patches.
        """

        def build(b):
            _free_jointed_foot(b, (0.0, 0.0, 0.0175))
            return _free_jointed_foot(b, (0.0, 0.005, 0.09))

        self._compare(build, lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0), tol=2e-3)

    def test_feather_pgs_contact_mode_conformance(self):
        """Settle identically on/off in every FeatherPGS contact mode.

        supports_reduced_contacts is class-wide, so the evidence must cover
        the dense and matrix-free contact paths, not only the default split
        mode the other tests exercise (matrix-free additionally runs generated
        native CUDA).
        """
        if not wp.get_device().is_cuda:
            self.skipTest("matrix_free requires a CUDA device")
        # dense runs 16 iterations: at the default 8 the UNREDUCED redundant
        # set does not converge (settles 5 mm high; split and the reduced set
        # both land at the true height) -- redundant near-parallel rows hurt
        # dense-PGS conditioning, the same effect as the stack-collapse case.
        # The comparison must be against a converged baseline.
        for mode, iters in (("dense", 16), ("matrix_free", 8)):
            with self.subTest(pgs_mode=mode):
                self._compare(
                    lambda b: _free_jointed_foot(b, (0.0, 0.0, 0.05)),
                    lambda m, mode=mode, iters=iters: newton.solvers.SolverFeatherPGS(
                        m, angular_damping=0.0, pgs_mode=mode, pgs_iterations=iters
                    ),
                )

    def test_mixed_separation_tilted_patch(self):
        """Settle a tilted patch whose contacts mix penetrating and speculative.

        The adversarial topology for solvers that ignore non-penetrating
        contacts (XPBD): the spatial-extreme slots can be won by speculative
        endpoints while interior points penetrate, so the retained PENETRATING
        support could collapse toward the single deepest point. Measured, the
        per-frame migration of the deepest slot keeps the dynamics equivalent;
        this pins that behavior for both solver families.
        """
        pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.06)

        def build(b):
            body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.028), pitch), mass=1.0)
            for i in range(7):
                b.add_shape_cylinder(
                    body,
                    xform=wp.transform(wp.vec3((i - 3.0) * 0.044, 0.0, 0.0), wp.quat_identity()),
                    radius=0.02,
                    half_height=0.015,
                )
            return body

        def build_jointed(b):
            link = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.028), pitch))
            for i in range(7):
                b.add_shape_cylinder(
                    link,
                    xform=wp.transform(wp.vec3((i - 3.0) * 0.044, 0.0, 0.0), wp.quat_identity()),
                    radius=0.02,
                    half_height=0.015,
                )
            b.add_articulation([b.add_joint_free(parent=-1, child=link)])
            return link

        self._compare(build, lambda m: newton.solvers.SolverXPBD(m, iterations=8), tol=1e-3)
        self._compare(build_jointed, lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0), tol=1e-3)

    def test_xpbd_conformance(self):
        """SolverXPBD rests the multi-cylinder foot at the same height on/off."""
        self._compare(
            lambda b: _cylinder_foot(b, (0.0, 0.0, 0.05)),
            lambda m: newton.solvers.SolverXPBD(m, iterations=8),
        )


class TestBodyPairReductionGrouping(unittest.TestCase):
    """Contacts may only compete when merging them cannot change the physics."""

    def _pad_body(self, mu_pad, mu_ring):
        """One body: a 5x5 low-mu sphere grid with one center pad sphere of mu_pad.

        The pad sits at the patch's interior, the exact position body-level
        grouping discards.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        ring_cfg = newton.ModelBuilder.ShapeConfig(mu=mu_ring)
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=mu_pad)
        for i in range(5):
            for j in range(5):
                if i == 2 and j == 2:
                    continue
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3((i - 2.0) * 0.04, (j - 2.0) * 0.04, 0.0), wp.quat_identity()),
                    radius=0.01,
                    cfg=ring_cfg,
                )
        builder.add_shape_sphere(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.01, cfg=pad_cfg
        )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        contacts = _collide_once(model, state, True, reduce_contacts_body_pairs_cell=10.0)
        n, s0, s1, _n2, _p0 = _contact_snapshot(contacts)
        mu_arr = model.shape_material_mu.numpy()
        sb = model.shape_body.numpy()
        kept_mus = {round(float(mu_arr[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        return n, kept_mus

    def test_heterogeneous_materials_never_compete(self):
        """Keep a high-friction center pad that body-level grouping would delete.

        Solvers read material laws from the surviving shape ids. A center pad
        is an interior point of the merged patch -- exactly what the reduction
        discards -- so merging it with the surrounding low-friction colliders
        deletes the high-friction law entirely. With material-equivalence
        classes the pad is its own group and must survive. Identical materials
        must still merge into one group (the point of body-level grouping).
        """
        _n_hetero, mus_hetero = self._pad_body(1.0, 0.2)
        self.assertIn(1.0, mus_hetero, "the high-friction pad's contact was deleted")
        n_homo, _mus_homo = self._pad_body(0.2, 0.2)
        self.assertLessEqual(n_homo, 7, "identical materials must merge into one group")

    def test_material_mutation_requires_refresh_and_then_splits(self):
        """Split reduction groups after a runtime material mutation + refresh.

        Group ids are construction-time snapshots; mutating one collider's
        friction afterwards must not silently keep it competing in its old
        class once the caller refreshes the grouping.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.2)
        for i in range(5):
            for j in range(5):
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3((i - 2.0) * 0.04, (j - 2.0) * 0.04, 0.0), wp.quat_identity()),
                    radius=0.01,
                    cfg=cfg,
                )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True, reduce_contacts_body_pairs_cell=10.0)
        contacts = pipeline.contacts()

        # mutate the CENTER collider (index 12: interior point of the patch)
        mu = model.shape_material_mu.numpy()
        mu[12] = 1.0
        model.shape_material_mu.assign(mu)

        pipeline.collide(state, contacts)  # stale classes: center still merged
        _n, s0, s1, _n1, _p0 = _contact_snapshot(contacts)
        sb = model.shape_body.numpy()
        kept_mus_stale = {round(float(mu[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        self.assertNotIn(1.0, kept_mus_stale, "scene no longer discards the interior point: rebuild it")

        pipeline.refresh_body_pair_reduction_groups()
        pipeline.collide(state, contacts)
        _n2b, s0, s1, _n2, _p0 = _contact_snapshot(contacts)
        kept_mus = {round(float(mu[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        self.assertIn(1.0, kept_mus, "refreshed classes did not separate the mutated collider")

    def test_swapped_endpoints_share_one_group(self):
        """Merge one physical patch whose contacts arrive with both endpoint orders.

        The narrow phase orders each shape pair by geometry type, so a flat
        interface between two bodies with interleaved sphere/box colliders
        produces contacts in BOTH directions for the same body pair. Without a
        canonical group normal they bin as two opposite-normal groups and keep
        two slot sets for one patch.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        base = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=0.0)
        builder.add_shape_box(base, hx=0.3, hy=0.1, hz=0.1)
        top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2195), wp.quat_identity()), mass=1.0)
        for i in range(4):
            x = (i - 1.5) * 0.12
            if i % 2 == 0:
                builder.add_shape_sphere(top, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), radius=0.02)
            else:
                builder.add_shape_box(
                    top, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), hx=0.02, hy=0.02, hz=0.02
                )
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True, reduce_contacts_body_pairs_cell=10.0)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n, _s0, _s1, _nrm, _p0 = _contact_snapshot(contacts)
        self.assertGreater(n, 0)
        # the raw narrow-phase output must actually contain both orientations,
        # or this test is vacuous (geometry-type dispatch produces the swap)
        base_pipe = newton.CollisionPipeline(model, broad_phase="nxn")
        base_contacts = base_pipe.contacts()
        base_pipe.collide(state, base_contacts)
        nb = int(base_contacts.rigid_contact_count.numpy()[0])
        nz = base_contacts.rigid_contact_normal.numpy()[:nb, 2]
        self.assertGreater(int((nz > 0).sum()), 0)
        self.assertGreater(int((nz < 0).sum()), 0, "scene no longer produces swapped endpoint orders")
        # one flat patch between one body pair must occupy exactly ONE group;
        # without the canonical normal the two orientations bin separately
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["max_hashtable_entries"], 1, "endpoint order leaked into grouping")
        self.assertLessEqual(n, 7)

    def test_up_axis_rotation_protects_ground_normals(self):
        """Give the ground direction its wide bin margin in X-, Y-, and Z-up models.

        The bin table is oriented for +Z; other up axes must be rotated into
        that frame or their ground normals land near the table's narrow
        horizontal margins and curved-terrain grouping churns again.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        for up_axis in (0, 1, 2):
            rot = np.array(_up_axis_rotation(up_axis), dtype=np.float64)
            self.assertTrue(np.allclose(rot @ rot.T, np.eye(3)), "rotation must be orthonormal")
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)
            up = np.zeros(3)
            up[up_axis] = 1.0
            for sign in (1.0, -1.0):
                mapped = rot @ (sign * up)
                dots = np.sort(faces @ mapped)[::-1]
                self.assertGreater(dots[0] - dots[1], 0.2, f"ground margin lost for up_axis={up_axis}")


class TestBodyPairReductionSafety(unittest.TestCase):
    """Resource exhaustion must fail open deterministically, never corrupt memory."""

    def test_narrowphase_overflow_fails_open(self):
        """Deliver the raw over-capacity counter untouched when the input overflows.

        Newton's narrow phase reserves contact indices before checking capacity,
        so the raw counter legitimately exceeds ``rigid_contact_max`` on
        overflow and is NOT a safe array bound. The reduction must detect that
        on device, do nothing that frame, and leave the counter and contact
        prefix exactly as an unreduced pipeline would deliver them.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _sphere_grid_body(builder, (0.0, 0.0, 0.0095))  # 25 same-plane contacts
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        counts = {}
        for reduce_on in (False, True):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                rigid_contact_max=8,  # deliberately far below the ~25 candidates
                reduce_contacts_body_pairs=reduce_on,
            )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)  # must not crash or corrupt memory
            counts[reduce_on] = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(counts[False], 8, "scene must actually overflow the buffer")
        self.assertEqual(counts[True], counts[False], "overflow frame must pass the raw counter through")

    def test_hashtable_saturation_keeps_whole_frame(self):
        """Keep the entire unreduced set when the group table budget is exceeded.

        Per-contact fail-open would let CUDA scheduling decide which groups get
        the last table entries -- a scheduling-dependent kept set. Exceeding
        the budget must instead fall back for the whole frame: every contact
        kept, which is deterministic, and counted in telemetry.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rng = np.random.default_rng(9)
        n_bodies = 1200  # > the 1024 minimum table capacity, one group each
        for k in range(n_bodies):
            x, y = (k % 40) * 0.5, (k // 40) * 0.5
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, 0.0095 + float(rng.uniform(0, 0.001))), wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base = int(contacts_base.rigid_contact_count.numpy()[0])
        self.assertGreater(n_base, 1024)

        # a factor small enough to hit the 1024-entry floor: 1200 groups cannot fit
        pipeline = _make_pipeline(model, True, reduce_contacts_body_pairs_hashtable_factor=1e-6)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red = int(contacts.rigid_contact_count.numpy()[0])
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(n_red, n_base, "saturated frame must keep the whole unreduced set")
        self.assertGreaterEqual(stats["fallback_frames"], 1)

    def test_mismatched_contacts_buffer_rejected(self):
        """Refuse an external Contacts buffer whose capacity differs from the pipeline's.

        The reducer's caches and launch bounds are sized to the pipeline's
        capacity; a larger external buffer would let the narrow phase write
        beyond them.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        oversized = Contacts(
            rigid_contact_max=pipeline.rigid_contact_max * 2, soft_contact_max=0, device=wp.get_device()
        )
        with self.assertRaisesRegex(ValueError, "rigid_contact_max"):
            pipeline.collide(state, oversized)

    def test_invalid_configuration_rejected(self):
        """Reject non-finite or non-positive reduction configuration at construction."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        for kwargs in (
            {"reduce_contacts_body_pairs_cell": 0.0},
            {"reduce_contacts_body_pairs_cell": float("inf")},
            {"reduce_contacts_body_pairs_cell": float("nan")},
            {"reduce_contacts_body_pairs_hysteresis": float("inf")},
            {"reduce_contacts_body_pairs_hysteresis": -1.0},
            {"reduce_contacts_body_pairs_hashtable_factor": 0.0},
        ):
            with self.assertRaises(ValueError, msg=f"accepted invalid {kwargs}"):
                _make_pipeline(model, True, **kwargs)

    def test_identity_frames_leave_buffer_untouched(self):
        """Detect no-benefit frames and skip compaction without changing results.

        Single-collider bodies produce one contact per group; reduction cannot
        remove anything, so the fast path must deliver the identical buffer an
        unreduced pipeline would, and count the frame in telemetry.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rng = np.random.default_rng(4)
        for k in range(24):
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3((k % 6) * 0.5, (k // 6) * 0.5, 0.0095 + float(rng.uniform(0, 0.001))),
                    wp.quat_identity(),
                ),
                mass=1.0,
            )
            builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base, s0_b, s1_b, _n1, p0_b = _contact_snapshot(contacts_base)

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red, s0_r, s1_r, _n2, p0_r = _contact_snapshot(contacts)

        self.assertEqual(n_red, n_base, "identity frame must keep every contact")
        rows_b = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_b, s1_b, p0_b, strict=True)
        )
        rows_r = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_r, s1_r, p0_r, strict=True)
        )
        self.assertEqual(rows_r, rows_b)
        self.assertGreaterEqual(pipeline.body_pair_reduction_stats()["identity_frames"], 1)

    def test_noop_frames_skip_compaction_for_small_groups(self):
        """Detect no-op frames whose groups are larger than one contact.

        A plain box on the ground is ONE group of four contacts that all fit
        in the slots: nothing is discarded, so the copy-back must be skipped
        (tier-2 detection after the keep scan) and the buffer delivered
        exactly as the unreduced pipeline would.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0495), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base, _s0, _s1, _n1, p0_b = _contact_snapshot(contacts_base)
        self.assertGreater(n_base, 1, "a box should rest on several contacts")

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red, _s2, _s3, _n2, p0_r = _contact_snapshot(contacts)
        self.assertEqual(n_red, n_base)
        self.assertEqual(
            sorted(tuple(round(float(v), 6) for v in q) for q in p0_r),
            sorted(tuple(round(float(v), 6) for v in q) for q in p0_b),
        )
        self.assertGreaterEqual(pipeline.body_pair_reduction_stats()["identity_frames"], 1)

    def test_property_schema_switch_is_safe(self):
        """Reduce a property-enabled buffer after a property-less one safely.

        The first reduce installs zero-length material placeholders in the
        gather scratch; a later same-capacity buffer WITH per-contact material
        arrays must re-provision them or the gather writes out of bounds.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        plain = pipeline.contacts()
        pipeline.collide(state, plain)  # installs the property-less scratch

        rich = Contacts(
            rigid_contact_max=pipeline.rigid_contact_max,
            soft_contact_max=0,
            device=wp.get_device(),
            per_contact_shape_properties=True,
        )
        pipeline.collide(state, rich)  # must not write into zero-length arrays
        n = int(rich.rigid_contact_count.numpy()[0])
        self.assertGreater(n, 0)
        self.assertTrue(np.isfinite(rich.rigid_contact_friction.numpy()[:n]).all())

    def test_clear_resets_reduced_provenance(self):
        """Reset the reduced marker when the buffer is cleared.

        A cleared buffer holds no contacts at all; keeping the stale marker
        would make unsupported solvers reject an empty buffer.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        self.assertTrue(contacts.rigid_contacts_reduced)
        contacts.clear()
        self.assertFalse(contacts.rigid_contacts_reduced)

    def test_malformed_reset_masks_rejected(self):
        """Reject reset masks with the wrong length, rank, or dtype.

        A short mask would silently reset only a prefix of worlds.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for _w in range(2):
            builder.begin_world()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        pipeline = _make_pipeline(model, True)
        for bad in (
            np.array([1], dtype=np.int32),  # short
            np.array([1, 0, 1], dtype=np.int32),  # long
            np.array([[1], [0]], dtype=np.int32),  # wrong rank
        ):
            with self.assertRaises(ValueError, msg=f"accepted malformed mask {bad!r}"):
                pipeline.reset_body_pair_reduction_history(bad)

    def test_reduced_marker_tracks_pipeline_mode(self):
        """Assign buffer provenance from the pipeline mode on every collide.

        A sticky marker would make an unsupported solver reject a buffer that
        was later refilled by an ordinary pipeline.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        reduced = _make_pipeline(model, True)
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = reduced.contacts()
        self.assertFalse(contacts.rigid_contacts_reduced)
        reduced.collide(state, contacts)
        self.assertTrue(contacts.rigid_contacts_reduced)
        plain.collide(state, contacts)
        self.assertFalse(contacts.rigid_contacts_reduced, "marker must clear when an ordinary pipeline refills")


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
        """Reject scenes whose reduction-group count exceeds the exact key budget.

        Group ids pack exactly into the reduction key; overflow would alias
        two groups and could evict a patch's deepest contact, so the pipeline
        must refuse at construction rather than mask bits at runtime. Groups
        are material-equivalence classes: five bodies with distinct friction
        coefficients plus the ground are six groups against a patched budget
        of five (ids are 0-based, so a patched MAX_GROUP_ID of 4 admits five).
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for k in range(5):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.2 * k, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(mu=0.1 * (k + 1)))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        with unittest.mock.patch("newton._src.sim.collide.MAX_GROUP_ID", 4):
            with self.assertRaisesRegex(ValueError, "at most 5 reduction groups"):
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
            body = _free_jointed_foot(builder, (5.13, 5.07, 0.06))  # 3 cm drop
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
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
        self.assertLess(float(z_off.min()), 0.04, "foot never fell: the solver is not simulating it")
        self.assertLess(abs(float(z_off[-30:].mean()) - float(z_on[-30:].mean())), 5e-4)
        # bounded touchdown transient: never punches through the resting height by > 2.5 mm
        self.assertGreater(float(z_on.min()), 0.015 - 2.5e-3)


class TestBodyPairReductionGroupAssignment(unittest.TestCase):
    """Group assignment must be stable at world axes and under rigid translation."""

    def test_bin_margins_at_world_axes(self):
        """Keep every world axis well inside a bin, especially the ground normal.

        The shared icosahedron table stores the solid Y-up, which places +Z on a
        face boundary to within 7e-8 of dot product: any curved or mesh surface
        in a Z-up world then flickers its patch normals between two bins step to
        step, and the kept set churns (measured as a permanent rocking limit
        cycle). The body-pair table is rotated so a face CENTER points at +/-Z;
        this pins that property plus a sane margin at the horizontal axes.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(faces, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6), "face normals must be unit length")

        def margin(d):
            dots = np.sort(faces @ np.asarray(d, dtype=np.float64))[::-1]
            return dots[0] - dots[1]

        for axis in ((0, 0, 1), (0, 0, -1)):
            self.assertGreater(margin(axis), 0.2, f"ground-normal margin too small at {axis}")
        for axis in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            self.assertGreater(margin(axis), 0.05, f"horizontal margin too small at {axis}")

    def test_bin_stable_under_ground_normal_wobble(self):
        """Assign every normal within 15 degrees of +Z to the same single bin.

        Emulates curved/mesh terrain: patch normals wobble around world-up. If
        any of them crosses a bin boundary, contacts of one patch regroup
        between steps and the kept set flickers.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        rng = np.random.default_rng(5)
        tilt = np.radians(rng.uniform(0.0, 15.0, size=512))
        yaw = rng.uniform(0.0, 2.0 * np.pi, size=512)
        normals = np.stack(
            [np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), np.cos(tilt)],
            axis=1,
        )
        bins = np.argmax(normals @ faces.T, axis=1)
        self.assertEqual(len(np.unique(bins)), 1, "normals near +Z must all share one bin")

    def test_kept_set_invariant_under_translation(self):
        """Keep the identical relative contact set after translating the scene.

        Symmetric collider layouts tie in the scan directions constantly (a row
        of foot cylinders shares a coordinate exactly). The old tie-break hashed
        contact content, and a static shape's witness point is world-space, so
        pure translation flipped winners: a foot sliding in a straight line
        churned its kept set. The geometric tie-break is measured in the
        pair-anchored face frame and cannot see translation.
        """
        offsets = ((0.0, 0.0), (0.37, 0.53), (1.24, -0.86))
        kept_sets = []
        for ox, oy in offsets:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = _cylinder_foot(builder, (ox, oy, 0.0175))
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state = model.state()
            contacts = _collide_once(model, state, reduce_body_pairs=True)
            n, s0, s1, _nrm, _p0 = _contact_snapshot(contacts)
            self.assertGreater(n, 0)
            # canonicalize in the BODY frame: witness points on the static plane
            # are stored world-space and legitimately move with the scene, so
            # compare positions relative to the foot instead.
            pts = _world_points0(model, state, contacts) - state.body_q.numpy()[body][:3]
            rows = sorted(
                (int(a), int(b), *(round(float(v), 5) for v in p)) for a, b, p in zip(s0, s1, pts, strict=True)
            )
            kept_sets.append(rows)
        self.assertEqual(kept_sets[0], kept_sets[1], "kept set changed under translation")
        self.assertEqual(kept_sets[0], kept_sets[2], "kept set changed under translation")


class TestBodyPairReductionHysteresis(unittest.TestCase):
    """Temporal hysteresis: kept-set continuity without sticky wrong winners."""

    def _curved_rock(self, hysteresis, steps=400):
        """Sphere-grid plate rocking on a large-radius static sphere.

        Curvature makes the plate's contact scores near-degenerate and
        continuously varying -- the regime where memoryless winner selection
        churns the kept set every step and the plate never settles. Returns the
        mean |angular velocity| over the final quarter of the run.
        """
        r_ground = 20.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_shape_sphere(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, -r_ground), wp.quat_identity()), radius=r_ground
        )
        body = _sphere_grid_body(builder, (0.0, 0.0, 0.0095), n=4, spacing=0.05)
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        pipeline = _make_pipeline(model, True, reduce_contacts_body_pairs_hysteresis=hysteresis)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverXPBD(model, iterations=8)
        tail = []
        for k in range(steps):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            if k >= steps * 3 // 4:
                tail.append(float(np.abs(state_0.body_qd.numpy()[body][:3]).max()))
        return float(np.mean(tail))

    def test_curved_support_settles(self):
        """Settle a plate on curved support instead of sustaining a limit cycle.

        With hysteresis off, near-degenerate winner handoffs re-excite the
        contact set each step and the plate rocks forever; the incumbent bias
        removes the churn. The off-case is measured in the same test so the
        assertion is relative, not an absolute magic number.
        """
        residual_off = self._curved_rock(hysteresis=0.0)
        residual_on = self._curved_rock(hysteresis=0.001)
        self.assertLess(
            residual_on, 0.5 * residual_off, f"hysteresis did not calm the limit cycle: {residual_on} vs {residual_off}"
        )
        self.assertLess(residual_on, 0.02, f"plate still not at rest: {residual_on} rad/s")

    def test_challenger_beyond_margin_wins_immediately(self):
        """Dethrone an incumbent depth winner as soon as a clearly deeper one exists.

        Hysteresis must only stop near-tie handoffs. A three-cylinder foot
        whose MIDDLE cylinder sits 2 mm lower is kept only through the depth
        slot (both end cylinders own every spatial extreme). Establish
        incumbency, then pitch the body so an end cylinder becomes deeper by
        several times the 1 mm margin while every contact's in-plane position
        moves less than the 0.5 mm identity quantum -- incumbency stays attached
        AND must lose: the middle contact has to vanish from the kept set on
        the very next collide.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0195), wp.quat_identity()), mass=1.0)
        for i in range(3):
            z_local = -0.002 if i == 1 else 0.0
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3((i - 1.0) * 0.05, 0.0, z_local), wp.quat_identity()),
                radius=0.02,
                half_height=0.015,
            )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # establish winners
        pipeline.collide(state, contacts)  # winners now incumbents
        pts = _world_points0(model, state, contacts)
        mid = np.abs(pts[:, 0]) < 0.02
        self.assertTrue(bool(mid.any()), "middle cylinder should be kept as the depth winner")

        # pitch about y: the -x end cylinder gains ~5 mm of depth over the
        # middle (>> 1 mm hysteresis), while in-plane positions shift by
        # 0.05 * (1 - cos 0.1) ~ 0.25 mm (< the 0.5 mm identity quantum)
        q = state.body_q.numpy()
        pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.1)
        q[0][3:7] = [pitch[0], pitch[1], pitch[2], pitch[3]]
        state.body_q.assign(q)
        pipeline.collide(state, contacts)
        pts = _world_points0(model, state, contacts)
        mid = np.abs(pts[:, 0]) < 0.02
        self.assertFalse(bool(mid.any()), "beaten incumbent depth winner was kept anyway")

    def test_reset_history_matches_fresh_pipeline(self):
        """Produce a fresh pipeline's kept set on the first collide after a reset.

        Hysteresis history is trajectory state; episode resets and teleports
        must be able to sever it so no incumbency bonus crosses the boundary.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # establish history
        pipeline.reset_body_pair_reduction_history()
        pipeline.collide(state, contacts)
        n_reset, s0_r, s1_r, _n1, p0_r = _contact_snapshot(contacts)

        fresh = _make_pipeline(model, True)
        contacts_f = fresh.contacts()
        fresh.collide(state, contacts_f)
        n_f, s0_f, s1_f, _n2, p0_f = _contact_snapshot(contacts_f)

        self.assertEqual(n_reset, n_f)
        rows_r = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_r, s1_r, p0_r, strict=True)
        )
        rows_f = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_f, s1_f, p0_f, strict=True)
        )
        self.assertEqual(rows_r, rows_f, "reset did not sever hysteresis history")

    def test_new_contacts_buffer_resets_history(self):
        """Sever hysteresis history automatically when a different buffer is supplied.

        A different Contacts instance means a different stream of states;
        winners recorded for the previous buffer must not bias the new one.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts_a = pipeline.contacts()
        pipeline.collide(state, contacts_a)
        contacts_b = pipeline.contacts()
        pipeline.collide(state, contacts_b)  # must not inherit buffer A's winners
        red = pipeline._body_pair_reducer
        masks = red.contact_incumbent.numpy()[: int(contacts_b.rigid_contact_count.numpy()[0])]
        self.assertTrue((masks == 0).all(), "incumbency leaked across Contacts buffers")

    def test_reducer_launches_stay_off_the_tape(self):
        """Record no reducer bookkeeping launches on an active autodiff tape.

        Every reducer kernel has backward disabled; recording them would only
        bloat the tape. This guards the full reduce() path including the
        hashtable maintenance kernels.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # warm up allocations outside the tape
        red = pipeline._body_pair_reducer
        tape = wp.Tape()
        with tape:
            red.reduce(model, state, contacts)
        self.assertEqual(len(tape.launches), 0, "reducer bookkeeping was recorded on the tape")

    def test_per_world_reset_severs_only_masked_worlds(self):
        """Reset one world's hysteresis history and leave the other's intact.

        Vectorized RL teleports individual environments mid-rollout; a global
        reset would sever every environment's history for one env's reset.
        The masked form advances only the selected worlds' generations: their
        contacts stop matching the snapshot while other worlds keep their
        incumbency bits.
        """
        # one sphere per world: identity frames skip compaction, so the
        # incumbency-mask cache (pre-compaction indexed) aligns with the
        # delivered contacts and worlds can be attributed exactly
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for w in range(2):
            builder.begin_world()
            body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * w, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        self.assertGreaterEqual(model.world_count, 2)
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)  # winners are now incumbents in both worlds

        def masks_by_world():
            red = pipeline._body_pair_reducer
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            sw = model.shape_world.numpy()
            masks = red.contact_incumbent.numpy()[:n]
            out = {0: [], 1: []}
            for k in range(n):
                w = max(int(sw[s0[k]]), int(sw[s1[k]]))
                out[w].append(int(masks[k]))
            return out

        before = masks_by_world()
        self.assertTrue(any(m != 0 for m in before[0]) and any(m != 0 for m in before[1]))

        mask = wp.array(np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=wp.get_device())
        pipeline.reset_body_pair_reduction_history(mask)
        pipeline.collide(state, contacts)
        after = masks_by_world()
        self.assertTrue(all(m == 0 for m in after[0]), "reset world kept incumbency")
        self.assertTrue(any(m != 0 for m in after[1]), "unreset world lost incumbency")

    def test_disabled_hysteresis_is_history_independent(self):
        """Reduce identically with hysteresis=0 regardless of what ran before.

        The zero setting must restore the exact memoryless behavior: the same
        state reduced by a pipeline that saw a different previous step and by a
        fresh pipeline must produce identical kept sets.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_a = model.state()
        state_b = model.state()
        q = state_b.body_q.numpy()
        q[body][0] += 0.01  # a slightly different "previous" step
        state_b.body_q.assign(q)

        with_history = _make_pipeline(model, True, reduce_contacts_body_pairs_hysteresis=0.0)
        contacts_h = with_history.contacts()
        with_history.collide(state_b, contacts_h)
        with_history.collide(state_a, contacts_h)

        fresh = _make_pipeline(model, True, reduce_contacts_body_pairs_hysteresis=0.0)
        contacts_f = fresh.contacts()
        fresh.collide(state_a, contacts_f)

        n_h, s0_h, s1_h, _n1, p0_h = _contact_snapshot(contacts_h)
        n_f, s0_f, s1_f, _n2, p0_f = _contact_snapshot(contacts_f)
        self.assertEqual(n_h, n_f)
        rows_h = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_h, s1_h, p0_h, strict=True)
        )
        rows_f = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_f, s1_f, p0_f, strict=True)
        )
        self.assertEqual(rows_h, rows_f)


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


if __name__ == "__main__":
    unittest.main()
