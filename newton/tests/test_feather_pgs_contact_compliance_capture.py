# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check CUDA graph capture and replay of experimental contact compliance."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_feather_pgs_contact_compliance import make_fixture


def _materials(contacts, stiffness):
    contacts.rigid_contact_stiffness.fill_(stiffness)
    contacts.rigid_contact_damping.fill_(20.0)
    contacts.rigid_contact_friction.fill_(1.0)


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

    def test_validate_rejects_capture(self):
        """Status readback is refused inside a capture."""
        episode = _Episode(articulated=True, enabled=True)
        with wp.ScopedDevice(episode.fixture.device):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                with wp.ScopedCapture():
                    episode.fixture.solver.validate_contact_compliance()


if __name__ == "__main__":
    unittest.main()
