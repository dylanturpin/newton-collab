# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact warm start of SolverFeatherPGS."""

import inspect
import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    gather_dense_warmstart,
    gather_mf_warmstart,
    gather_propagation_warmstart,
    prepare_world_impulses,
)
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_selected_cuda_test_devices,
    get_test_devices,
)


def _build_press():
    """Build a 1-DOF prismatic press: an articulated box driven down onto the ground.

    The drive target sits below the contact height, so after touchdown the press
    stalls against the ground under a steady drive force — a persistent dense
    contact family (articulated vs. static routes to the dense path) whose
    converged impulse is constant per step. Exactly the regime warm starting
    carries impulses across, and the regime where the pre-fix carry diverged.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        target_ke=2.0e3,
        target_kd=50.0,
    )
    builder.add_articulation([joint], label="press")
    return builder.finalize()


def _run_press(
    steps: int,
    warm_kwargs: dict,
    contact_matching: str | None = "sticky",
    solve_mode: str = "matrix_free",
    articulated_contact_response: str = "immediate",
    propagation_cached_response: bool = True,
):
    """Drive the press to a stall and record per-step contact-impulse sums and speeds.

    Returns (impulse_sum_per_step, |joint_qd|_per_step, final_state).
    """
    model = _build_press()
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode=solve_mode,
        articulated_contact_response=articulated_contact_response,
        propagation_cached_response=propagation_cached_response,
        iterations=8,
        baumgarte=0.1,
        **warm_kwargs,
    )
    pipeline_kwargs = {} if contact_matching is None else {"contact_matching": contact_matching}
    pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = -0.25  # 5 cm below touchdown: sustained press after the stall
    control.joint_target_q.assign(targets)

    impulse_sums = np.zeros(steps)
    speeds = np.zeros(steps)
    for i in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        if articulated_contact_response.startswith("propagation"):
            counts = solver._propagation_constraint_count.numpy()
            rows = solver._propagation_row_type.numpy()[0, : counts[0]]
            lam = solver._propagation_impulses.numpy()[0, : counts[0]]
        else:
            counts = solver._constraint_count.numpy()
            rows = solver._row_type.numpy()[0, : counts[0]]
            lam = solver._impulses.numpy()[0, : counts[0]]
        impulse_sums[i] = np.abs(lam[rows == PGS_CONSTRAINT_TYPE_CONTACT]).sum()
        speeds[i] = np.abs(state_0.joint_qd.numpy()[0])
    return impulse_sums, speeds, state_0


def _matched_history(device):
    """Gather arguments for generation 1 of stream 1, matched against the solved generation 0."""
    return (
        wp.zeros(1, dtype=wp.int32, device=device),  # match generation
        wp.ones(1, dtype=wp.int32, device=device),  # contact generation
        1,  # contact stream
        wp.zeros(1, dtype=wp.int32, device=device),  # solved generation
        wp.ones(1, dtype=wp.int32, device=device),  # solved stream
    )


def _gather_dense_rows(
    *,
    prev_impulses,
    prev_types,
    prev_parents,
    prev_slots,
    current_types,
    current_parents,
    current_slots,
    match_indices,
    count,
    decay=1.0,
    dt_scale=1.0,
):
    """Launch the dense identity gather on CPU and return its row buffer."""
    max_c = len(current_types)
    n = len(current_slots)
    impulses = wp.zeros((1, max_c), dtype=wp.float32, device="cpu")
    wp.launch(
        gather_dense_warmstart,
        dim=n,
        inputs=[
            wp.array([n], dtype=wp.int32, device="cpu"),
            wp.array([0] * n, dtype=wp.int32, device="cpu"),
            wp.array(current_slots, dtype=wp.int32, device="cpu"),
            wp.array([0] * n, dtype=wp.int32, device="cpu"),
            wp.array(match_indices, dtype=wp.int32, device="cpu"),
            *_matched_history("cpu"),
            wp.array(prev_slots, dtype=wp.int32, device="cpu"),
            wp.array([prev_impulses], dtype=wp.float32, device="cpu"),
            wp.array([prev_types], dtype=wp.int32, device="cpu"),
            wp.array([prev_parents], dtype=wp.int32, device="cpu"),
            wp.array([count], dtype=wp.int32, device="cpu"),
            wp.array([current_types], dtype=wp.int32, device="cpu"),
            wp.array([current_parents], dtype=wp.int32, device="cpu"),
            wp.array([[0.0, 0.0, 1.0]] * n, dtype=wp.vec3, device="cpu"),
            wp.array([[0.0, 0.0, 1.0]] * n, dtype=wp.vec3, device="cpu"),
            wp.full((1, max_c), 100.0, dtype=wp.float32, device="cpu"),
            decay,
            wp.array([dt_scale], dtype=wp.float32, device="cpu"),
            max_c,
        ],
        outputs=[impulses],
        device="cpu",
    )
    return impulses.numpy()[0]


class TestFeatherPGSIdentityWarmstartKernel(unittest.TestCase):
    def test_noncontact_dense_cache_is_cold_initialized(self):
        """The dense initializer preserves no row without an identity contract."""
        impulses = wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=wp.float32, device="cpu")
        wp.launch(
            prepare_world_impulses,
            dim=1,
            inputs=[
                wp.array([4], dtype=wp.int32, device="cpu"),
                4,
                1,
            ],
            outputs=[impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.zeros(4, dtype=np.float32))

    def test_two_contact_friction_span_transitions_do_not_cross_seed(self):
        """A neighboring contact never owns the source or destination tangent row."""
        c, f, dead = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, -1

        # A: 3 -> 1 rows, B: 1 -> 3 rows. A's previous second tangent lands at
        # B's current first tangent by raw offset and must not be written there.
        got = _gather_dense_rows(
            prev_impulses=[11.0, 22.0, 33.0, 55.0],
            prev_types=[c, f, f, c],
            prev_parents=[dead, 0, 0, dead],
            prev_slots=[0, 3],
            current_types=[c, c, f, f],
            current_parents=[dead, dead, 1, 1],
            current_slots=[0, 1],
            match_indices=[0, 1],
            count=4,
        )
        np.testing.assert_array_equal(got, np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32))

        # A: 1 -> 3 rows, B: 3 -> 1 rows. A's second destination tangent
        # overlaps B's previous first tangent and must stay cold.
        got = _gather_dense_rows(
            prev_impulses=[11.0, 55.0, 66.0, 77.0],
            prev_types=[c, c, f, f],
            prev_parents=[dead, dead, 1, 1],
            prev_slots=[0, 1],
            current_types=[c, f, f, c],
            current_parents=[dead, 0, 0, dead],
            current_slots=[0, 3],
            match_indices=[0, 1],
            count=4,
        )
        np.testing.assert_array_equal(got, np.array([11.0, 0.0, 0.0, 55.0], dtype=np.float32))

    def test_slot_churn_uses_identity_and_scales_dt(self):
        """Contact-order and slot churn follows match identity, not row index."""
        c, dead = PGS_CONSTRAINT_TYPE_CONTACT, -1
        got = _gather_dense_rows(
            prev_impulses=[55.0, 0.0, 0.0, 0.0, 11.0],
            prev_types=[c, dead, dead, dead, c],
            prev_parents=[dead] * 5,
            prev_slots=[4, 0],  # previous sorted contacts A, B
            current_types=[dead, c, dead, dead, c],
            current_parents=[dead] * 5,
            current_slots=[1, 4],  # current sorted contacts B, A
            match_indices=[1, 0],
            count=5,
            decay=0.5,
            dt_scale=4.0,
        )
        np.testing.assert_array_equal(got, np.array([0.0, 110.0, 0.0, 0.0, 22.0], dtype=np.float32))

    def test_mf_and_propagation_share_friction_ownership_rule(self):
        """Every routed family rejects A's old tangent from B's new span."""
        c, f, dead = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, -1
        prev_impulses = wp.array([[11.0, 22.0, 33.0, 55.0]], dtype=wp.float32, device="cpu")
        prev_types = wp.array([[c, f, f, c]], dtype=wp.int32, device="cpu")
        prev_parents = wp.array([[dead, 0, 0, dead]], dtype=wp.int32, device="cpu")
        current_types = wp.array([[c, c, f, f]], dtype=wp.int32, device="cpu")
        current_parents = wp.array([[dead, dead, 1, 1]], dtype=wp.int32, device="cpu")
        contact_count = wp.array([2], dtype=wp.int32, device="cpu")
        contact_slots = wp.array([0, 1], dtype=wp.int32, device="cpu")
        contact_world = wp.array([0, 0], dtype=wp.int32, device="cpu")
        match_indices = wp.array([0, 1], dtype=wp.int32, device="cpu")
        prev_slots = wp.array([0, 3], dtype=wp.int32, device="cpu")
        constraint_count = wp.array([4], dtype=wp.int32, device="cpu")

        mf_impulses = wp.zeros((1, 4), dtype=wp.float32, device="cpu")
        wp.launch(
            gather_mf_warmstart,
            dim=2,
            inputs=[
                contact_count,
                wp.array([1, 1], dtype=wp.int32, device="cpu"),
                contact_slots,
                contact_world,
                match_indices,
                *_matched_history("cpu"),
                prev_slots,
                prev_impulses,
                prev_types,
                prev_parents,
                constraint_count,
                current_types,
                current_parents,
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.full((1, 4), 100.0, dtype=wp.float32, device="cpu"),
                1.0,
                wp.ones(1, dtype=wp.float32, device="cpu"),
                4,
            ],
            outputs=[mf_impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(mf_impulses.numpy()[0], np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32))

        propagation_impulses = wp.zeros((1, 4), dtype=wp.float32, device="cpu")
        wp.launch(
            gather_propagation_warmstart,
            dim=2,
            inputs=[
                contact_count,
                wp.array([2, 2], dtype=wp.int32, device="cpu"),
                contact_slots,
                contact_world,
                match_indices,
                *_matched_history("cpu"),
                prev_slots,
                prev_impulses,
                prev_types,
                prev_parents,
                constraint_count,
                current_types,
                current_parents,
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.full((1, 4), 100.0, dtype=wp.float32, device="cpu"),
                1.0,
                wp.ones(1, dtype=wp.float32, device="cpu"),
                4,
            ],
            outputs=[propagation_impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(
            propagation_impulses.numpy()[0], np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_current_slot_is_bounded_by_constraint_count(self):
        got = _gather_dense_rows(
            prev_impulses=[9.0, 0.0, 0.0, 0.0, 0.0],
            prev_types=[PGS_CONSTRAINT_TYPE_CONTACT, -1, -1, -1, -1],
            prev_parents=[-1] * 5,
            prev_slots=[0],
            current_types=[-1, -1, -1, -1, PGS_CONSTRAINT_TYPE_CONTACT],
            current_parents=[-1] * 5,
            current_slots=[4],
            match_indices=[0],
            count=4,
        )
        np.testing.assert_array_equal(got, np.zeros(5, dtype=np.float32))

    def test_constructor_layout_and_decay_validation(self):
        parameters = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
        start = parameters.index("contact_warmstart")
        self.assertEqual(
            parameters[start : start + 2],
            ("contact_warmstart", "solve_mode"),
        )
        self.assertGreater(parameters.index("contact_warmstart_decay"), parameters.index("contact_gap_gate"))

        model = newton.ModelBuilder().finalize(device="cpu")
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, contact_warmstart_decay=value)

    def test_contacts_none_is_valid(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="split", contact_warmstart=True)
        solver.step(model.state(), model.state(), model.control(), None, 1.0 / 60.0)

    def test_split_mode_holds_the_press(self):
        """The identity carry works on the CPU-supported dense-row route."""
        impulses, speeds, _ = _run_press(120, {"contact_warmstart": True}, solve_mode="split")
        self.assertGreater(float(impulses[-1]), 0.0)
        self.assertLess(float(np.ptp(impulses[-10:])), 1.0e-5)
        self.assertLess(float(speeds[-10:].max()), 1.0e-4)


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSIdentityWarmstart(unittest.TestCase):
    def test_real_contact_insertion_moves_slots_without_cross_seeding(self):
        """A newly inserted lower-key contact shifts a persistent contact's real MF slot."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
        shape_a = builder.add_shape_sphere(body_a, radius=0.1)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.099), wp.quat_identity()))
        shape_b = builder.add_shape_sphere(body_b, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=8,
            contact_warmstart=True,
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        pipeline.collide(state_0, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        old_slot = int(solver._ws_prev_slot_sorted.numpy()[0])
        old_impulse = float(solver._ws_prev_mf_impulses.numpy()[0, old_slot])
        self.assertGreater(old_impulse, 0.0)

        # Bring the lower shape-id sphere into contact. Sorting inserts it
        # before B, so B moves to a new solver slot while its match points to
        # the previous frame's sole contact.
        q = state_1.body_q.numpy()
        q[body_a][2] = 0.099
        state_1.body_q.assign(q)
        pipeline.collide(state_1, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 2)
        match = contacts.rigid_contact_match_index.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        b_indices = np.flatnonzero((shape0 == shape_b) | (shape1 == shape_b))
        a_indices = np.flatnonzero((shape0 == shape_a) | (shape1 == shape_a))
        self.assertEqual(len(b_indices), 1)
        self.assertEqual(len(a_indices), 1)
        b_contact = int(b_indices[0])
        a_contact = int(a_indices[0])
        self.assertEqual(int(match[b_contact]), 0)
        self.assertLess(int(match[a_contact]), 0)

        solver._iterations = 0
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
        slots = solver._contact_slot.numpy()[:count]
        b_slot = int(slots[b_contact])
        a_slot = int(slots[a_contact])
        self.assertNotEqual(b_slot, old_slot, "test did not produce actual solver-slot churn")
        impulses = solver._mf_impulses.numpy()[0]
        self.assertAlmostEqual(float(impulses[b_slot]), old_impulse, delta=1.0e-6)
        self.assertEqual(float(impulses[a_slot]), 0.0)

    def test_identity_warmstart_holds_static_press(self):
        """Identity warm start keeps a stalled press at the cold equilibrium.

        Regression net for the warm-start velocity accounting: carried impulses
        are installed into the starting velocity exactly once. A missing install
        accumulates the ledger (``lambda_{n+1} = lambda_n + lambda*``, the
        historical divergence); a duplicated install halves it (measured 0.5x
        cold when an earlier identity path added a second fold on top of
        ``_stage6_apply_impulses_world``). The identity-matched path must also
        stay quiet and match the cold ledger while seeding by contact identity
        rather than raw slot index.
        """
        steps = 240
        stall = slice(120, None)  # well past touchdown + transient

        lam_cold, _speed_cold, _ = _run_press(steps, {})
        lam_warm, speed_warm, state = _run_press(steps, {"contact_warmstart": True})

        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

        cold_end = lam_cold[-10:].mean()
        warm_end = lam_warm[-10:].mean()

        # The solver installs carried impulses into the starting velocity
        # (_stage6_apply_impulses_world under contact_warmstart), so warm and cold
        # must share an equilibrium. A duplicated velocity install shows
        # up here as a halved impulse ledger (measured 0.5x when the matched
        # path folded the seed a second time); a missing install shows up as an
        # accumulating ledger (lambda_{n+1} = lambda_n + lambda*).
        warm_growth = lam_warm[-10:].mean() / max(lam_warm[stall][:10].mean(), 1e-12)
        self.assertLess(warm_growth, 1.25, f"warm-start impulse grew x{warm_growth:.2f} at a stall")
        self.assertGreater(warm_end, 0.7 * cold_end, f"warm impulse ledger {warm_end:.3f} below cold {cold_end:.3f}")
        self.assertLess(warm_end, 1.4 * cold_end, f"warm impulse ledger {warm_end:.3f} above cold {cold_end:.3f}")
        self.assertLess(
            speed_warm[stall].max(),
            0.02,
            f"press not quiet under warm start (peak |qd| {speed_warm[stall].max():.3f} m/s)",
        )

    def test_identity_warmstart_matches_cold_equilibrium(self):
        """The matched warm start converges to the cold solve's stall pose, not a new one.

        Warm starting changes the sweep's starting point, never the physics: at
        the stall both runs must agree on the joint position to sub-millimetre.
        """
        _, _, state_cold = _run_press(240, {})
        _, _, state_warm = _run_press(240, {"contact_warmstart": True})
        q_cold = float(state_cold.joint_q.numpy()[0])
        q_warm = float(state_warm.joint_q.numpy()[0])
        self.assertAlmostEqual(q_warm, q_cold, delta=1.0e-3)

    def test_identity_warmstart_requires_contact_matching(self):
        """Stepping warm start with unmatched contacts raises.

        The identity gather needs ``Contacts.rigid_contact_match_index``; a
        pipeline built without ``contact_matching`` must fail loudly rather than
        silently reusing impulses by raw slot index.
        """
        with self.assertRaises(NotImplementedError):
            _run_press(3, {"contact_warmstart": True}, contact_matching=None)

    def test_single_flag_enables_dense_and_mf_carry(self):
        """contact_warmstart=True is the single all-contact warm-start mode."""
        model = _build_press()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", contact_warmstart=True, iterations=4)
        self.assertTrue(solver._contact_warmstart)
        self.assertTrue(solver._mf_warmstart_enabled)

    def test_propagation_routes_hold_static_press(self):
        """Serial, colored, and cached propagation install carried impulses once."""
        cases = (
            ("propagation", False),
            ("propagation_colored", False),
            ("propagation_colored", True),
            ("propagation_fused", False),
        )
        for response, cached in cases:
            with self.subTest(response=response, cached=cached):
                cold, _, state_cold = _run_press(
                    160,
                    {},
                    articulated_contact_response=response,
                    propagation_cached_response=cached,
                )
                warm, speed, state_warm = _run_press(
                    160,
                    {"contact_warmstart": True},
                    articulated_contact_response=response,
                    propagation_cached_response=cached,
                )
                self.assertGreater(float(warm[-1]), 0.0)
                self.assertLess(float(np.ptp(warm[-10:])), 1.0e-5)
                self.assertLess(float(speed[-10:].max()), 1.0e-4)
                self.assertAlmostEqual(float(warm[-10:].mean()), float(cold[-10:].mean()), delta=2.0e-3)
                self.assertAlmostEqual(
                    float(state_warm.joint_q.numpy()[0]),
                    float(state_cold.joint_q.numpy()[0]),
                    delta=1.0e-3,
                )


def _insertion_after_solved_single_contact(articulated: bool):
    """Solve sphere B alone on the ground, then lower sphere A so its contact sorts before B's.

    ``articulated`` mounts each sphere on a vertical prismatic joint, so the contacts are
    dense rows; otherwise the spheres are free bodies on the matrix-free rows.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    shapes = []
    for x, z in ((-0.5, 1.0), (0.5, 0.099)):
        xform = wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        shapes.append(builder.add_shape_sphere(body, radius=0.1))
    builder.add_ground_plane()
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model, iterations=8, contact_warmstart=True)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
    if articulated:
        prev_slot, prev_impulses = solver._ws_prev_dense_slot_sorted, solver._ws_prev_dense_impulses
    else:
        prev_slot, prev_impulses = solver._ws_prev_slot_sorted, solver._ws_prev_mf_impulses
    carried = float(prev_impulses.numpy()[0, int(prev_slot.numpy()[0])])
    if articulated:
        # Sphere A's joint coordinate is its height offset from the 1 m start.
        q = state_1.joint_q.numpy()
        q[0] = 0.099 - 1.0
        state_1.joint_q.assign(q)
        newton.eval_fk(model, state_1.joint_q, state_1.joint_qd, state_1)
    else:
        # Keep A's body pose and free-joint coordinates consistent.
        q = state_1.body_q.numpy()
        q[0][2] = 0.099
        state_1.body_q.assign(q)
        q = state_1.joint_q.numpy()
        q[2] = 0.099
        state_1.joint_q.assign(q)
    return model, pipeline, contacts, solver, state_1, state_0, tuple(shapes), carried


def _seeded_sphere_impulses(solver, contacts, shapes, model, articulated, state_in=None, state_out=None):
    """Return the seeded normal impulses of A's and B's contacts.

    With states, first step without a sweep, so the solved impulses are the seeds;
    otherwise the caller has already stepped (for example by a graph replay).
    """
    if state_in is not None:
        solver._iterations = 0
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    slots = solver._contact_slot.numpy()[:count]
    impulses = (solver._impulses if articulated else solver._mf_impulses).numpy()[0]
    seeds = []
    for shape in shapes:
        index = np.flatnonzero((shape0 == shape) | (shape1 == shape))
        seeds.append(float(impulses[int(slots[int(index[0])])]))
    return seeds


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSWarmstartProvenance(unittest.TestCase):
    """Warm start reads match indices only when they refer to the contact set it solved."""

    def test_one_collision_pass_carries_the_persistent_contact(self):
        """The reference sequence: B keeps its impulse at its new slot and new A starts cold."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                self.assertGreater(carried, 0.0)
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertAlmostEqual(seed_b, carried, delta=1.0e-6)

    def test_interleaved_contact_buffers_start_cold(self):
        """A pass into the solved buffer after a pass into another buffer starts cold.

        The pipeline matches against its last pass, whichever buffer it wrote, so the
        second pass's match indices refer to the other buffer's contacts. Read as indices
        into the solved set, they would seed the new contact A with B's impulse.
        """
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                self.assertGreater(carried, 0.0)
                pipeline.collide(state_in, pipeline.contacts())
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_interleaved_contact_buffers_start_cold_under_graph_replay(self):
        """Alternating captured collision passes keep the interleaved-buffer cold start."""
        device = wp.get_device()
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                other = pipeline.contacts()
                solver._iterations = 0
                graphs = {}
                with wp.ScopedCapture(device) as capture:
                    pipeline.collide(state_in, contacts)
                graphs["collide"] = capture.graph
                with wp.ScopedCapture(device) as capture:
                    pipeline.collide(state_in, other)
                graphs["collide_other"] = capture.graph
                with wp.ScopedCapture(device) as capture:
                    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
                graphs["step"] = capture.graph

                # One pass straight into the solved buffer carries B's impulse.
                wp.capture_launch(graphs["collide"])
                wp.capture_launch(graphs["step"])
                seed_a, seed_b = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertEqual(seed_a, 0.0)
                self.assertAlmostEqual(seed_b, carried, delta=1.0e-6)

                wp.capture_launch(graphs["collide_other"])
                wp.capture_launch(graphs["collide"])
                wp.capture_launch(graphs["step"])
                seed_a, seed_b = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_skipped_collision_pass_starts_cold(self):
        """Two collision passes between solves match against an unsolved contact set."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                pipeline.collide(state_in, contacts)
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_replaced_contact_buffer_starts_cold(self):
        """A fresh buffer has no solved history of its own, even at the same generation."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, _contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                replacement = pipeline.contacts()
                pipeline.collide(state_in, replacement)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, replacement, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_substep_on_the_same_contacts_reuses_their_own_impulses(self):
        """A second step on an unchanged contact set seeds each contact from itself.

        Its match indices refer to the set before the last collision pass, where B had
        another index; reading them would seed B from A's slot.
        """
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
                solved = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertGreater(solved[1], 0.0)
                seeded = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated, state_in, state_out)
                for got, want in zip(seeded, solved, strict=True):
                    self.assertAlmostEqual(got, want, delta=1.0e-6 * max(1.0, abs(want)))


def _resting_box_rows(device, articulated: bool, solve_mode: str = "matrix_free", **solver_kwargs):
    """Settle a box on the ground with warm start; return the model, pipeline, contacts, solver and states.

    ``articulated`` mounts the box on a vertical prismatic joint, so its contacts are dense
    rows; otherwise it is a free body on the free-body rows.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    if articulated:
        body = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=xform)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest", deterministic=True)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode=solve_mode, contact_warmstart=True, iterations=12, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(60):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return model, pipeline, contacts, solver, state_0, state_1, control


def test_graph_replay_rescales_history_once_after_a_timestep_change(test, device):
    """Rescale carried impulses by the step ratio once in captured steps, as in eager steps.

    The previous step lives on the device. A ratio fixed at capture would rescale the
    history again on every replay: after settling at 1/120 s, replays at 1/240 s would
    seed 0.5, 0.25 and 0.125 of the settled load instead of 0.5 each time.
    """
    for articulated in (True, False):
        ratios = {}
        for captured in (False, True):
            _model, pipeline, contacts, solver, state_0, state_1, control = _resting_box_rows(device, articulated)
            impulses, row_type = (
                (solver._impulses, solver._row_type) if articulated else (solver._mf_impulses, solver._mf_row_type)
            )

            def normal_load(impulses=impulses, row_type=row_type):
                return float(impulses.numpy()[row_type.numpy() == PGS_CONSTRAINT_TYPE_CONTACT].sum())

            settled = normal_load()
            test.assertGreater(settled, 0.0)
            # Without a sweep the solved impulses are the seeds.
            solver._iterations = 0
            pipeline.collide(state_0, contacts)
            if captured:
                with wp.ScopedCapture(device) as capture:
                    solver.step(state_0, state_1, control, contacts, 0.5 / 120.0)
            values = []
            for _ in range(3):
                if captured:
                    wp.capture_launch(capture.graph)
                else:
                    solver.step(state_0, state_1, control, contacts, 0.5 / 120.0)
                values.append(normal_load() / settled)
            ratios[captured] = values
        with test.subTest(articulated=articulated):
            np.testing.assert_allclose(ratios[False], [0.5, 0.5, 0.5], rtol=1.0e-5)
            np.testing.assert_allclose(ratios[True], ratios[False], rtol=1.0e-5)


def test_edited_warmstart_decay_reaches_every_row_family(test, device):
    """Seed dense and free-body rows with the edited decay, as a solver constructed with it reports."""
    # Matrix-free mode runs on CUDA only.
    solve_modes = ("split", "matrix_free") if wp.get_device(device).is_cuda else ("split",)
    for solve_mode, articulated, decay in itertools.product(solve_modes, (True, False), (0.0, 0.25)):
        label = f"{solve_mode} {'articulated' if articulated else 'free'} decay={decay}"
        with test.subTest(label):
            model, pipeline, contacts, solver, state_0, state_1, control = _resting_box_rows(
                device, articulated, solve_mode
            )
            fresh = newton.solvers.SolverFeatherPGS(
                model, solve_mode=solve_mode, contact_warmstart=True, contact_warmstart_decay=decay
            )
            impulses, row_type = (
                (solver._impulses, solver._row_type) if articulated else (solver._mf_impulses, solver._mf_row_type)
            )

            def normal_load(impulses=impulses, row_type=row_type):
                return float(impulses.numpy()[row_type.numpy() == PGS_CONSTRAINT_TYPE_CONTACT].sum())

            settled = normal_load()
            test.assertGreater(settled, 0.0)
            solver.contact_warmstart_decay = decay
            test.assertEqual(
                [(route, d) for route, *_rows, d in solver._patch_row_arrays()],
                [(route, d) for route, *_rows, d in fresh._patch_row_arrays()],
            )
            # Without a sweep the solved impulses are the seeds.
            solver._iterations = 0
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
            test.assertAlmostEqual(normal_load() / settled, decay, delta=1.0e-5, msg=label)


def test_captured_graph_keeps_its_warmstart_decay_until_recaptured(test, device):
    """Replay a graph with its capture-time decay; a recapture after the edit seeds with the new one."""
    for articulated in (True, False):
        with test.subTest(articulated=articulated):
            _model, pipeline, contacts, solver, state_0, state_1, control = _resting_box_rows(device, articulated)
            impulses, row_type = (
                (solver._impulses, solver._row_type) if articulated else (solver._mf_impulses, solver._mf_row_type)
            )

            def normal_load(impulses=impulses, row_type=row_type):
                return float(impulses.numpy()[row_type.numpy() == PGS_CONSTRAINT_TYPE_CONTACT].sum())

            settled = normal_load()
            test.assertGreater(settled, 0.0)
            # Without a sweep the solved impulses are the seeds.
            solver._iterations = 0
            pipeline.collide(state_0, contacts)
            with wp.ScopedCapture(device) as before_edit:
                solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
            solver.contact_warmstart_decay = 0.25
            with wp.ScopedCapture(device) as after_edit:
                solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
            wp.capture_launch(before_edit.graph)
            test.assertAlmostEqual(normal_load() / settled, 1.0, delta=1.0e-5)
            wp.capture_launch(after_edit.graph)
            test.assertAlmostEqual(normal_load() / settled, 0.25, delta=1.0e-5)


class TestFeatherPGSWarmstartReplay(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSWarmstartReplay,
    "test_edited_warmstart_decay_reaches_every_row_family",
    test_edited_warmstart_decay_reaches_every_row_family,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSWarmstartReplay,
    "test_captured_graph_keeps_its_warmstart_decay_until_recaptured",
    test_captured_graph_keeps_its_warmstart_decay_until_recaptured,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSWarmstartReplay,
    "test_graph_replay_rescales_history_once_after_a_timestep_change",
    test_graph_replay_rescales_history_once_after_a_timestep_change,
    devices=get_cuda_test_devices(),
)


def test_substeps_reuse_contacts_with_their_own_history(test, device):
    """Seed each contact from its own last solve across solver substeps on one contact set.

    Match indices refer to the contact set before the last collision pass, while the
    history is saved every solver step. Inserting and deleting a contact moves the
    persistent contact's index, so reading the match index on a substep would seed it
    from another contact or from nothing.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
    shape_a = builder.add_shape_sphere(body_a, radius=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()))
    shape_b = builder.add_shape_sphere(body_b, radius=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", iterations=8, contact_warmstart=True)
    states = [model.state(), model.state()]
    control = model.control()
    dt = 1.0 / 240.0

    def step(iterations):
        solver._iterations = iterations
        solver.step(states[0], states[1], control, contacts, dt)
        states.reverse()

    def contact_impulses():
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        slots = solver._contact_slot.numpy()[:count]
        impulses = solver._mf_impulses.numpy()[0]
        result = {}
        for name, shape in (("a", shape_a), ("b", shape_b)):
            index = np.flatnonzero((shape0 == shape) | (shape1 == shape))
            if len(index):
                result[name] = float(impulses[int(slots[int(index[0])])])
        return result

    def check_substeps(phase):
        # Solve a substep, then seed the next one without a sweep: its impulses are the seeds.
        for substep in range(2):
            step(8)
            solved = contact_impulses()
            step(0)
            seeded = contact_impulses()
            with test.subTest(phase=phase, substep=substep):
                test.assertEqual(seeded.keys(), solved.keys())
                test.assertGreater(solved["b"], 0.0)
                for name, value in solved.items():
                    test.assertAlmostEqual(seeded[name], value, delta=1.0e-6 * max(1.0, abs(value)))

    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    check_substeps("B alone")

    # Insert the lower shape-id sphere: sorting puts its contact before B's.
    q = states[0].body_q.numpy()
    q[body_a][2] = 0.1
    states[0].body_q.assign(q)
    qd = states[0].body_qd.numpy()
    qd[body_a] = 0.0
    states[0].body_qd.assign(qd)
    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
    test.assertGreaterEqual(int(contacts.rigid_contact_match_index.numpy()[1]), 0)
    check_substeps("insertion")

    # Delete A's contact again: B moves back to index 0.
    q = states[0].body_q.numpy()
    q[body_a][2] = 1.0
    states[0].body_q.assign(q)
    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    test.assertEqual(int(contacts.rigid_contact_match_index.numpy()[0]), 1)
    check_substeps("deletion")


def test_both_row_families_seed_impulses_scaled_by_the_step_ratio(test, device):
    """Seed the carried normal impulses of dense and free-body rows, scaled by ``dt / dt_previous``.

    With no sweep (``iterations = 0``) the solved impulses are exactly the seeds.
    """
    for articulated in (True, False):
        with test.subTest(articulated=articulated):
            _model, pipeline, contacts, solver, state_0, state_1, control = _resting_box_rows(device, articulated)
            if articulated:
                impulses, row_type, count = solver._impulses, solver._row_type, solver._constraint_count
            else:
                impulses, row_type, count = solver._mf_impulses, solver._mf_row_type, solver._mf_constraint_count
            n = int(count.numpy()[0])
            normal = row_type.numpy()[0, :n] == PGS_CONSTRAINT_TYPE_CONTACT
            previous = impulses.numpy()[0, :n][normal]
            test.assertGreater(float(previous.sum()), 0.0)

            solver._iterations = 0
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 0.5 / 120.0)
            test.assertEqual(int(count.numpy()[0]), n)
            seeded = impulses.numpy()[0, :n][row_type.numpy()[0, :n] == PGS_CONSTRAINT_TYPE_CONTACT]
            np.testing.assert_allclose(seeded, 0.5 * previous, rtol=1.0e-6, atol=1.0e-9)


class TestFeatherPGSIdentityWarmstartUpstream(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _fn in (
    test_both_row_families_seed_impulses_scaled_by_the_step_ratio,
    test_substeps_reuse_contacts_with_their_own_history,
):
    add_function_test(TestFeatherPGSIdentityWarmstartUpstream, _fn.__name__, _fn, devices=devices)


def _stack(device, restitution=0.0, drop=False):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=restitution)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=restitution))
    for k in range(3):
        z = 0.0505 + 0.101 * k + (0.15 if drop and k == 2 else 0.0)
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, z), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=8, velocity_iterations=2, contact_warmstart=True
    )
    return model, pipeline, solver


def _normal_impulse_total(solver):
    imp = solver._mf_impulses.numpy()[0]
    rt = solver._mf_row_type.numpy()[0]
    return float(imp[rt == PGS_CONSTRAINT_TYPE_CONTACT].sum())


def test_warm_carry_kernel_scales_rows_exactly(test: unittest.TestCase, device):
    """The gather kernel scales matched normal and friction rows before any
    solver iteration can conceal a missing or incorrect carry scale."""
    with wp.ScopedDevice(device):
        max_c = 8
        prev_impulses = np.zeros((1, max_c), dtype=np.float32)
        prev_impulses[0, 4:7] = (1.25, -0.5, 0.75)

        prev_types = np.full((1, max_c), -1, dtype=np.int32)
        prev_types[0, 4] = PGS_CONSTRAINT_TYPE_CONTACT
        prev_types[0, 5:7] = PGS_CONSTRAINT_TYPE_FRICTION
        prev_parents = np.full((1, max_c), -1, dtype=np.int32)
        prev_parents[0, 5:7] = 4

        current_types = np.full((1, max_c), -1, dtype=np.int32)
        current_types[0, 2] = PGS_CONSTRAINT_TYPE_CONTACT
        current_types[0, 3:5] = PGS_CONSTRAINT_TYPE_FRICTION
        current_parents = np.full((1, max_c), -1, dtype=np.int32)
        current_parents[0, 3:5] = 2

        impulses = wp.zeros((1, max_c), dtype=wp.float32)
        wp.launch(
            gather_mf_warmstart,
            dim=1,
            inputs=[
                wp.array([1], dtype=wp.int32),  # contact_count
                wp.array([1], dtype=wp.int32),  # MF contact_path
                wp.array([2], dtype=wp.int32),  # current base slot
                wp.array([0], dtype=wp.int32),  # contact_world
                wp.array([0], dtype=wp.int32),  # current -> previous contact
                wp.zeros(1, dtype=wp.int32),  # matched against the solved generation 0
                wp.ones(1, dtype=wp.int32),  # contact generation 1
                1,  # contact stream
                wp.zeros(1, dtype=wp.int32),  # solved generation 0
                wp.ones(1, dtype=wp.int32),  # solved stream
                wp.array([4], dtype=wp.int32),  # previous base slot
                wp.array(prev_impulses, dtype=wp.float32),
                wp.array(prev_types, dtype=wp.int32),
                wp.array(prev_parents, dtype=wp.int32),
                wp.array([5], dtype=wp.int32),
                wp.array(current_types, dtype=wp.int32),
                wp.array(current_parents, dtype=wp.int32),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.array([[0.0, 0.0, 1.0]], dtype=wp.vec3),
                wp.full((1, max_c), 100.0, dtype=wp.float32),
                0.75,
                wp.array([4.0], dtype=wp.float32),
                max_c,
            ],
            outputs=[impulses],
            device=device,
        )

        carried = impulses.numpy()[0]
        np.testing.assert_allclose(carried[2:5], 3.0 * prev_impulses[0, 4:7], rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(carried[[0, 1, 5, 6, 7]], np.zeros(5, dtype=np.float32))


def test_warm_carry_fixed_dt_identity(test: unittest.TestCase, device):
    """At fixed dt the carry ratio is exactly 1: a settled stack's converged
    contact impulses are a fixed point (steady step to step), and the stack
    holds its height."""
    with wp.ScopedDevice(device):
        model, pipeline, solver = _stack(device)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        dt = 1.0 / 60.0
        totals = []
        zs = []
        for k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt)
            s0, s1 = s1, s0
            if k >= 100:
                totals.append(_normal_impulse_total(solver))
                zs.append(float(s0.body_q.numpy()[2][2]))
        # per-step totals carry ~2-3% GS convergence jitter at 8 iterations;
        # a carry-scale defect (any ratio != 1 applied every frame) compounds
        # geometrically and blows far past this band
        test.assertLess(
            (max(totals) - min(totals)) / max(totals), 0.15, f"impulse totals unstable at fixed dt: {totals[-5:]}"
        )
        test.assertAlmostEqual(
            float(np.mean(totals[:10])),
            float(np.mean(totals[10:])),
            delta=0.02 * float(np.mean(totals)),
            msg="impulse mean drifted at fixed dt",
        )
        # steadiness, not absolute height: measured hover noise at this
        # budget is 0.2-0.4 mm and per-step impulse jitter up to ~7%; a
        # carry-scale defect compounds every frame and runs far past both
        test.assertLess(max(zs) - min(zs), 1.0e-3, f"stack height drifting at fixed dt ({zs[-3:]})")


def test_warm_carry_scales_with_dt(test: unittest.TestCase, device):
    """Across a step-size change the carried impulses rescale by exactly
    dt_new/dt_old: a stack settled at dt/4 keeps its height through a switch
    to dt, and the converged contact impulses scale by ~4x (support impulse
    is force x dt)."""
    with wp.ScopedDevice(device):
        model, pipeline, solver = _stack(device)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        dt_small, dt_big = 1.0 / 240.0, 1.0 / 60.0
        for _k in range(240):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt_small)
            s0, s1 = s1, s0
        i_small = _normal_impulse_total(solver)
        z_before = float(s0.body_q.numpy()[2][2])
        zs = []
        for _k in range(60):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, dt_big)
            s0, s1 = s1, s0
            zs.append(float(s0.body_q.numpy()[2][2]))
        i_big = _normal_impulse_total(solver)
        ratio = i_big / i_small
        test.assertAlmostEqual(ratio, dt_big / dt_small, delta=0.6, msg=f"impulse ratio {ratio:.2f}, want ~4")
        test.assertLess(abs(zs[0] - z_before), 1.0e-3, "stack popped on the timestep switch")
        test.assertLess(abs(zs[-1] - z_before), 2.0e-3, f"stack drifted after the switch (dz={zs[-1] - z_before:.4f})")


def test_warm_carry_no_kick_after_dt_change_on_impact(test: unittest.TestCase, device):
    """Impact impulses are not proportional to dt. Check both timestep
    directions, especially small-to-large where the carry scales a cached
    impact impulse upward, and reject any resulting energy gain."""
    with wp.ScopedDevice(device):
        for dt_before, dt_after in (
            (1.0 / 240.0, 1.0 / 60.0),
            (1.0 / 60.0, 1.0 / 240.0),
        ):
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=0.8)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=0.8))
            drop_h = 0.25
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, drop_h), wp.quat_identity()))
            builder.add_shape_sphere(b, radius=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=32,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model, solve_mode="matrix_free", iterations=8, velocity_iterations=2, contact_warmstart=True
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            dt = dt_before
            impacted = False
            v_in = 0.0
            max_up = 0.0
            max_z = 0.0
            for _k in range(240):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, dt)
                s0, s1 = s1, s0
                vz = float(s0.body_qd.numpy()[b][2])
                if not impacted and vz > 0.0 and v_in < -0.5:
                    impacted = True
                    dt = dt_after
                if not impacted:
                    v_in = min(v_in, vz)
                else:
                    max_up = max(max_up, vz)
                    max_z = max(max_z, float(s0.body_q.numpy()[b][2]))

            transition = f"{dt_before:.6f} -> {dt_after:.6f}"
            test.assertTrue(impacted, f"ball never bounced ({transition})")
            test.assertLess(
                max_up,
                0.8 * abs(v_in) * 1.10,
                f"rebound {max_up:.2f} exceeds e*v_in ({0.8 * abs(v_in):.2f}) after dt {transition}",
            )
            test.assertLess(max_z, drop_h, f"ball gained energy after dt {transition}")


class TestFeatherPGSWarmstartDt(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_kernel_scales_rows_exactly",
    test_warm_carry_kernel_scales_rows_exactly,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_fixed_dt_identity",
    test_warm_carry_fixed_dt_identity,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_scales_with_dt",
    test_warm_carry_scales_with_dt,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSWarmstartDt,
    "test_warm_carry_no_kick_after_dt_change_on_impact",
    test_warm_carry_no_kick_after_dt_change_on_impact,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
