# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check CUDA graph capture and replay of experimental contact compliance."""

import gc
import os
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton.tests.test_feather_pgs_contact_compliance import make_fixture


def _materials(contacts, stiffness):
    contacts.rigid_contact_stiffness.fill_(stiffness)
    contacts.rigid_contact_damping.fill_(20.0)
    contacts.rigid_contact_friction.fill_(1.0)


def _run_free_spheres(*, dense_rows, graph, steps=20):
    """Four free spheres in one world: their matrix-free slots exceed a one-row dense capacity."""
    device = os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        for i in range(4):
            body = builder.add_body(xform=wp.transform((i * 0.3, 0.0, 0.049), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        model.rigid_contact_max = 32
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(32, dtype=float))
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            contact_compliance=True,
            friction_anchor_beta=0.0,
            enable_restitution=False,
            dense_max_constraints=dense_rows,
            mf_max_constraints=32,
            pgs_iterations=8,
        )
        states = (model.state(), model.state())
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        control = model.control()

        def two_steps():
            for source, target in (states, states[::-1]):
                source.clear_forces()
                pipeline.collide(source, contacts)
                _materials(contacts, 3000.0)
                solver.step(source, target, control, contacts, 0.005)

        trace = []
        two_steps()
        trace.append(states[0].body_q.numpy().copy())
        slots = solver.contact_slot.numpy()[: int(contacts.rigid_contact_count.numpy()[0])].copy()
        captured = None
        if graph:
            with wp.ScopedCapture() as capture:
                solver.seed_double_buffer_events()
                two_steps()
            captured = capture.graph
        for _ in range(steps - 1):
            if graph:
                wp.capture_launch(captured)
            else:
                two_steps()
            solver.validate_contact_compliance()
            trace.append(states[0].body_q.numpy().copy())
        return solver, np.asarray(trace), slots


class _Episode:
    """Two fixed state buffers stepped A->B->A, so one captured graph covers two steps."""

    def __init__(self, *, stiffness=3000.0, dt=0.005, **fixture_options):
        self.fixture = make_fixture(**fixture_options)
        self.model = self.fixture.model
        self.stiffness = stiffness
        self.dt = dt
        self.states = (self.model.state(), self.model.state())
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        self.control = self.model.control()

    def two_steps(self):
        fixture = self.fixture
        for source, target in (self.states, self.states[::-1]):
            source.clear_forces()
            fixture.pipeline.collide(source, fixture.contacts)
            _materials(fixture.contacts, self.stiffness)
            fixture.solver.step(source, target, self.control, fixture.contacts, self.dt)

    def snapshot(self):
        state = self.states[0]
        return {name: getattr(state, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}

    def run(self, pairs, *, graph=False, between=None):
        """Advance ``2 * pairs`` steps eagerly or by replaying one captured two-step graph."""
        trace = []
        captured = None
        with wp.ScopedDevice(self.fixture.device):
            if graph:
                # Kernel modules load outside capture.
                self.two_steps()
                trace.append(self.snapshot())
                with wp.ScopedCapture() as capture:
                    self.fixture.solver.seed_double_buffer_events()
                    self.two_steps()
                captured = capture.graph
            for pair in range(len(trace), pairs):
                if between is not None:
                    between(self, pair)
                if graph:
                    wp.capture_launch(captured)
                    self.fixture.solver.validate_contact_compliance()
                else:
                    self.two_steps()
                trace.append(self.snapshot())
        return trace


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceCapture(unittest.TestCase):
    """Replay must reproduce eager compliant steps and keep their error contract."""

    def assert_traces_equal(self, eager, replay):
        self.assertEqual(len(eager), len(replay))
        for step, (a, b) in enumerate(zip(eager, replay, strict=True)):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} at pair {step}")

    def test_replay_matches_eager(self):
        """Dense and free-body routes, one and four worlds, with contacts appearing during replay."""
        for articulated in (True, False):
            for world_count in (1, 4):
                with self.subTest(articulated=articulated, worlds=world_count):
                    # Starting 2 cm above the gap gives zero contacts for the first steps.
                    options = {"articulated": articulated, "enabled": True, "world_count": world_count, "height": 0.08}
                    eager = _Episode(**options).run(60)
                    replay_episode = _Episode(**options)
                    replay = replay_episode.run(60, graph=True)
                    self.assert_traces_equal(eager, replay)
                    self.assertGreater(replay_episode.fixture.solver.compliance_contact_count, 0)
                    target = 0.05 - 0.3 * 9.81 / 3000.0
                    z = replay[-1]["body_q"][replay_episode.fixture.bodies, 2]
                    self.assertLess(np.max(np.abs(z - target)), 5e-4)

    def test_replay_zero_stiffness_matches_hard_contacts(self):
        """Compliance ON with zero stiffness stays the hard law under replay."""
        for articulated in (True, False):
            with self.subTest(articulated=articulated):
                hard = _Episode(articulated=articulated, enabled=False, stiffness=0.0).run(30, graph=True)
                noop_episode = _Episode(articulated=articulated, enabled=True, stiffness=0.0)
                noop = noop_episode.run(30, graph=True)
                self.assert_traces_equal(hard, noop)
                self.assertEqual(noop_episode.fixture.solver.compliance_contact_count, 0)

    def test_reset_between_replays_matches_eager(self):
        """A solver reset between replays acts as it does between eager steps."""

        def reset(episode, pair):
            if pair == 10:
                episode.fixture.solver.reset(episode.states[0])

        eager = _Episode(articulated=True, enabled=True).run(20, between=reset)
        replay = _Episode(articulated=True, enabled=True).run(20, graph=True, between=reset)
        self.assert_traces_equal(eager, replay)

    def test_replay_latches_input_overflow(self):
        """Replay never raises by itself; validation reports an overflowing step and then clears."""
        episode = _Episode(articulated=True, enabled=True)
        fixture = episode.fixture
        solver, contacts = fixture.solver, fixture.contacts
        with wp.ScopedDevice(fixture.device):
            state, output = episode.states
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            solver.step(state, output, episode.control, contacts, episode.dt)
            with wp.ScopedCapture() as capture:
                solver.seed_double_buffer_events()
                solver.step(state, output, episode.control, contacts, episode.dt)
            contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            wp.capture_launch(capture.graph)
            with self.assertRaisesRegex(RuntimeError, "overflowing contact"):
                solver.validate_contact_compliance()
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            wp.capture_launch(capture.graph)
            solver.validate_contact_compliance()
            self.assertGreater(solver.compliance_contact_count, 0)

    def test_replay_latches_row_capacity_loss(self):
        """A contact that first appears during replay and cannot get rows invalidates that replay."""
        for articulated in (True, False):
            with self.subTest(articulated=articulated):
                episode = _Episode(
                    articulated=articulated,
                    enabled=True,
                    height=0.08,
                    solver_options={
                        "dense_max_constraints": 1,
                        "mf_max_constraints": 1,
                        "warn_constraint_overflow": False,
                    },
                )
                fixture = episode.fixture
                with wp.ScopedDevice(fixture.device):
                    # Airborne: no contacts, so the warm-up and capture are valid.
                    episode.two_steps()
                    fixture.solver.validate_contact_compliance()
                    with wp.ScopedCapture() as capture:
                        fixture.solver.seed_double_buffer_events()
                        episode.two_steps()
                    with self.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
                        for _ in range(40):
                            wp.capture_launch(capture.graph)
                            fixture.solver.validate_contact_compliance()

    def test_unequal_row_capacities(self):
        """Free-body rows beyond the dense capacity prepare as with a large dense capacity, eager and replayed."""
        traces, counts = {}, {}
        for dense_rows in (1, 32):
            for graph in (False, True):
                solver, trace, slots = _run_free_spheres(dense_rows=dense_rows, graph=graph)
                counts[dense_rows, graph] = solver.compliance_contact_count
                if dense_rows == 1:
                    self.assertGreaterEqual(int(slots.max()), dense_rows)
                traces[dense_rows, graph] = trace
        self.assertGreaterEqual(counts[32, False], 4)
        for key, trace in traces.items():
            with self.subTest(dense_rows=key[0], graph=key[1]):
                self.assertEqual(counts[key], counts[32, False])
                np.testing.assert_array_equal(trace, traces[32, False])

    def test_validate_rejects_capture(self):
        """Status readback is refused inside a capture."""
        episode = _Episode(articulated=True, enabled=True)
        with wp.ScopedDevice(episode.fixture.device):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                with wp.ScopedCapture():
                    episode.fixture.solver.validate_contact_compliance()


def _reducer(fixture):
    """Body-pair reducer writing the fixture's Contacts buffer."""
    return newton.CollisionPipeline(
        fixture.model,
        broad_phase="nxn",
        deterministic=True,
        rigid_contact_max=fixture.contacts.rigid_contact_max,
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
    )


def _contact_rows(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    return (
        count,
        contacts.rigid_contact_normal.numpy()[:count].copy(),
        contacts.rigid_contact_point0.numpy()[:count].copy(),
    )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceContactsLease(unittest.TestCase):
    """A captured compliant step must keep body-pair reduction off its Contacts buffer."""

    def assert_reducer_rejected(self, episode, reducer):
        """Eager and captured reducer writes both fail before touching the buffer."""
        contacts = episode.fixture.contacts
        state = episode.states[0]
        before = _contact_rows(contacts)
        with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
            reducer.collide(state, contacts)
        capture = wp.ScopedCapture()
        capture.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
                reducer.collide(state, contacts)
        finally:
            capture.__exit__(None, None, None)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
        after = _contact_rows(contacts)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[2], after[2])

    def test_captured_step_blocks_reducer_and_replays(self):
        """Consumer first: reducer writes are refused while the replayed graph still matches eager."""
        options = {"articulated": True, "enabled": True}
        eager = _Episode(**options).run(12)
        episode = _Episode(**options)
        reducer = _reducer(episode.fixture)
        attempted = []

        def write_reduced(episode, pair):
            if pair in (3, 7):
                self.assert_reducer_rejected(episode, reducer)
                attempted.append(pair)

        replay = episode.run(12, graph=True, between=write_reduced)
        self.assertEqual(attempted, [3, 7])
        self.assertEqual(len(eager), len(replay))
        for pair, (a, b) in enumerate(zip(eager, replay, strict=True)):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} at pair {pair}")
        self.assertGreater(episode.fixture.solver.compliance_contact_count, 0)

    def test_reader_lease_lives_with_graph(self):
        """Lifetime: the lease outlives the capture object and ends with the last graph reference."""
        episode = _Episode(articulated=False, enabled=True)
        fixture = episode.fixture
        reducer = _reducer(fixture)
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            with wp.ScopedCapture() as capture:
                fixture.solver.seed_double_buffer_events()
                episode.two_steps()
            graph = capture.graph
            graph_ref = weakref.ref(graph)
            del capture
            gc.collect()
            wp.capture_launch(graph)
            fixture.solver.validate_contact_compliance()
            self.assert_reducer_rejected(episode, reducer)
            del graph
            gc.collect()
            self.assertIsNone(graph_ref())
            state, output = episode.states
            reducer.collide(state, fixture.contacts)
            self.assertTrue(fixture.contacts.rigid_contacts_body_pair_reduced)
            _materials(fixture.contacts, 3000.0)
            # Already-reduced eager input still rejects with the configuration named.
            with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                fixture.solver.step(state, output, episode.control, fixture.contacts, episode.dt)

    def test_live_reducer_graph_rejects_step(self):
        """Producer first: a live reducer graph rejects compliant steps after an ordinary refill."""
        episode = _Episode(articulated=True, enabled=True)
        fixture = episode.fixture
        contacts = fixture.contacts
        reducer = _reducer(fixture)
        state, output = episode.states
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            reducer.collide(state, contacts)
            with wp.ScopedCapture() as reducer_capture:
                reducer.collide(state, contacts)
            wp.capture_launch(reducer_capture.graph)
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced_capture)
            body_q = output.body_q.numpy().copy()
            with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            np.testing.assert_array_equal(output.body_q.numpy(), body_q)
            solver_capture = wp.ScopedCapture()
            solver_capture.__enter__()
            try:
                with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                    fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            finally:
                solver_capture.__exit__(None, None, None)
            del reducer_capture, solver_capture
            gc.collect()
            reducer.release_body_pair_reduction_capture()
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            fixture.solver.validate_contact_compliance()

    def test_noncompliant_capture_keeps_reduction(self):
        """Control: without compliance, FeatherPGS steps reduced contacts and its graph leaves reduction available."""
        episode = _Episode(articulated=True, enabled=False)
        fixture = episode.fixture
        contacts = fixture.contacts
        reducer = _reducer(fixture)
        state, output = episode.states
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            reducer.collide(state, contacts)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
            fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            self.assertTrue(np.isfinite(output.body_q.numpy()).all())
            with wp.ScopedCapture() as capture:
                fixture.solver.seed_double_buffer_events()
                episode.two_steps()
            wp.capture_launch(capture.graph)
            reducer.collide(state, contacts)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
            wp.capture_launch(capture.graph)
            self.assertTrue(np.isfinite(episode.states[0].body_q.numpy()).all())


if __name__ == "__main__":
    unittest.main()
