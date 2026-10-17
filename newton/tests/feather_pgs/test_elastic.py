# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic links in SolverFeatherPGS: modal dynamics, frame coupling, contact and determinism."""

import hashlib
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.elastic_links import elastic_contact_jacobian
from newton.examples.basic._reduced_elastic import beam_render_sample_points, box_surface_mesh, cantilever_tip_mode
from newton.solvers import SolverFeatherPGS, SolverVBD

BEAM_LENGTH = 0.6
HUB_OFFSET = 0.05


@unittest.skipUnless(wp.is_cuda_available(), "SolverFeatherPGS elastic coupling tests run on CUDA")
class TestFeatherPGSElastic(unittest.TestCase):
    def test_modal_oscillator_matches_implicit_euler(self):
        m, k, c, q0, dt = 0.5, 200.0, 0.3, 0.02, 1.0 / 240.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link_elastic(
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
            mode_count=1,
            mode_mass=[m],
            mode_stiffness=[k],
            mode_damping=[c],
            mode_q=[q0],
            linear_axes=[],
            angular_axes=[],
        )
        builder.add_articulation([builder.body_elastic_joint[body]])
        model = builder.finalize()
        states = _run(model, SolverFeatherPGS(model), 240, dt)
        index = int(model.elastic_mode_q_start.numpy()[0])
        q, v = q0, 0.0
        for state in states:
            a = (-k * q - (c + dt * k) * v) / (m + dt * c + dt * dt * k)
            v += dt * a
            q += dt * v
            self.assertAlmostEqual(float(state.joint_q.numpy()[index]), q, delta=1.0e-6)

    def test_coupled_hub_frequency_matches_lagrangian(self):
        """A free hub carrying a clamped beam raises the in-plane mode's frequency through the H coupling."""
        dt = 1.0 / 3840.0
        model, state, info = _hub_beam(steel=True, omega=0.0, eta=(0.0, 0.02))
        eta = _mode_history(model, _run(model, SolverFeatherPGS(model), 1536, dt, state), 1)
        measured = _frequency(eta, dt)
        self.assertAlmostEqual(measured, info["coupled_frequency"], delta=2.0e-3 * info["coupled_frequency"])
        self.assertGreater(measured, 1.5 * info["uncoupled_frequency"])

    def test_coupled_hub_frequency_matches_vbd(self):
        dt = 1.0 / 3840.0
        frequencies = []
        for steel in (True, False):
            model, state, _info = _hub_beam(steel=steel, omega=0.0, eta=(0.0, 0.02))
            solver = SolverFeatherPGS(model) if steel else SolverVBD(model, iterations=32, rigid_compliant_alm=False)
            frequencies.append(_frequency(_mode_history(model, _run(model, solver, 1536, dt, state), 1), dt))
        self.assertAlmostEqual(frequencies[0], frequencies[1], delta=0.03 * frequencies[0])

    def test_spinning_hub_conserves_axial_angular_momentum(self):
        dt = 1.0 / 240.0
        model, state, info = _hub_beam(steel=True, omega=3.0, eta=(0.0, 0.02))
        momentum = [_cloud_momentum(model, s, info)[1][2] for s in _run(model, SolverFeatherPGS(model), 240, dt, state)]
        initial = _cloud_momentum(model, state, info)[1][2]
        self.assertLess(np.max(np.abs(np.asarray(momentum) - initial)), 2.0e-3 * abs(initial))

    def test_free_body_linear_momentum_converges_with_dt(self):
        errors = []
        for dt in (1.0 / 240.0, 1.0 / 960.0):
            model, state, info = _free_beam()
            p0 = _cloud_momentum(model, state, info)[0]
            final = _run(model, SolverFeatherPGS(model), round(0.5 / dt), dt, state)[-1]
            errors.append(np.linalg.norm(_cloud_momentum(model, final, info)[0] - p0) / np.linalg.norm(p0))
        self.assertLess(errors[0], 1.0e-2)
        self.assertLess(errors[1], 0.4 * errors[0])

    def test_contact_compresses_pad_to_static_solution(self):
        mass, stiffness, g = 0.4, 2000.0, 9.81
        model, _pad = _pad_on_ground(mass, stiffness, g)
        state_0, state_1 = model.state(), model.state()
        contacts = model.contacts()
        solver = SolverFeatherPGS(model)
        for _ in range(480):
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        solver.check_constraint_capacity()
        eta = float(state_0.joint_q.numpy()[int(model.elastic_mode_q_start.numpy()[0])])
        self.assertAlmostEqual(eta, -mass * g / stiffness, delta=0.01 * mass * g / stiffness)

    def test_repeat_runs_are_bitwise_identical(self):
        digests = []
        for _ in range(2):
            model, state, _info = _hub_beam(steel=True, omega=2.0, eta=(0.02, 0.01), gravity=(0.0, 0.0, -9.81))
            solver = SolverFeatherPGS(model, deterministic=True)
            digest = hashlib.sha256()
            for s in _run(model, solver, 120, 1.0 / 240.0, state):
                digest.update(s.joint_q.numpy().tobytes())
                digest.update(s.joint_qd.numpy().tobytes())
            digests.append(digest.hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_inverse_kinematics_keeps_modal_coordinates(self):
        for model, state, _info in (_hub_beam(steel=True, omega=1.5, eta=(0.01, -0.02)), _free_beam()):
            q, qd = wp.clone(state.joint_q), wp.clone(state.joint_qd)
            newton.eval_ik(model, state, q, qd)
            np.testing.assert_allclose(q.numpy(), state.joint_q.numpy(), atol=1.0e-6)
            np.testing.assert_allclose(qd.numpy(), state.joint_qd.numpy(), atol=1.0e-5)

    def test_elastic_shapes_size_dense_row_capacity(self):
        model, _pad = _pad_on_ground(0.4, 2000.0, 9.81)
        solver = SolverFeatherPGS(model)
        expected = min(32 + 3 * int(model.elastic_shape_vertex_count.numpy().sum()), 128)
        self.assertEqual(solver.articulation_constraint_max, expected)

    def test_frame_origin_invariance_under_driven_base_rotation(self):
        """Clamp-origin and mid-span-origin frames of the same beam on a rocking base respond identically."""
        model, base_joint, targets = _rocking_base_beams()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        solver = SolverFeatherPGS(model)
        q_index = int(model.joint_q_start.numpy()[base_joint])
        qd_index = int(model.joint_qd_start.numpy()[base_joint])
        mode = model.elastic_mode_q_start.numpy()
        target_q, target_qd = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
        dt, peak, error = 1.0 / 240.0, 0.0, 0.0
        for i in range(240):
            target_q[q_index], target_qd[qd_index] = targets((i + 1) * dt)
            control.joint_target_q.assign(target_q)
            control.joint_target_qd.assign(target_qd)
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            q = state_0.joint_q.numpy()
            peak = max(peak, abs(float(q[mode[0]])))
            error = max(error, abs(float(q[mode[0]] - q[mode[1]])))
        self.assertGreater(peak, 0.05)
        self.assertLess(error, 1.0e-5 * peak)

    def test_eager_step_before_first_replay_matches_eager_twin(self):
        """Step eagerly between capturing an elastic contact step and its first replay, and match an eager twin."""
        runs = []
        for _ in range(2):
            model, _pad = _pad_on_ground(0.4, 2000.0, 9.81)
            runs.append((model, SolverFeatherPGS(model), model.contacts(), [model.state(), model.state()]))

        def step(run):
            model, solver, contacts, states = run
            for i in (0, 1):
                model.collide(states[i], contacts)
                solver.step(states[i], states[1 - i], model.control(), contacts, 1.0 / 240.0)

        with wp.ScopedCapture() as capture:
            step(runs[0])
        step(runs[0])
        step(runs[1])
        wp.synchronize()
        for _ in range(3):
            wp.capture_launch(capture.graph)
            step(runs[1])
        captured, eager = (run[3][0].joint_q.numpy() for run in runs)
        self.assertTrue(np.all(np.isfinite(captured)))
        np.testing.assert_allclose(captured, eager, rtol=1.0e-5, atol=1.0e-6)

    def test_contact_jacobian_respects_normal_only_last_slot(self):
        """A normal-only contact in the last dense row gets its modal column and no tangent rows."""
        model, _pad = _pad_on_ground(0.4, 2000.0, 9.81)
        device = model.device
        dofs = 7
        J = wp.zeros((1, 2, dofs), dtype=float, device=device)
        normal = wp.array([[0.0, 0.0, -1.0]], dtype=wp.vec3, device=device)
        bottom_sample = int(np.argmin(model.elastic_shape_vertex_local.numpy()[:, 2]))
        one = lambda value: wp.array([value], dtype=wp.int32, device=device)  # noqa: E731
        wp.launch(
            elastic_contact_jacobian,
            dim=1,
            inputs=[
                one(1),
                normal,
                one(-1),
                one(bottom_sample),
                one(0),
                one(1),
                one(1),
                one(1),
                one(-1),
                one(0),
                one(0),
                dofs,
                wp.array([dofs], dtype=wp.int32, device=device),
                one(0),
                one(0),
                wp.array([-1, 0], dtype=wp.int32, device=device),
                model.body_elastic_index,
                model.elastic_mode_qd_start,
                model.elastic_mode_count,
                model.elastic_max_mode_count,
                model.elastic_shape_vertex_phi,
                model.body_q,
            ],
            outputs=[J],
            device=device,
        )
        rows = J.numpy()[0]
        # Side 1 carries sign -1; n = +z and phi_z = -1 at the bottom face.
        self.assertAlmostEqual(float(rows[1, 6]), 1.0, places=5)
        np.testing.assert_array_equal(rows[0], 0.0)
        np.testing.assert_array_equal(rows[1, :6], 0.0)

    def test_rejects_unsupported_elastic_frames(self):
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        axes = [newton.Axis.X, newton.Axis.Y, newton.Axis.Z]
        with self.assertRaises(ValueError):
            builder.add_joint_elastic(body, 1, linear_axes=axes, angular_axes=axes)
        model, _state, _info = _hub_beam(steel=True, omega=0.0, eta=(0.0, 0.0))
        with self.assertRaises(NotImplementedError):
            SolverVBD(model)

    def test_rejects_bodies_attached_to_an_elastic_link(self):
        model, _state, _info = _hub_beam(steel=True, omega=0.0, eta=(0.0, 0.0), tip_body=True)
        with self.assertRaises(NotImplementedError):
            SolverFeatherPGS(model)


def _run(model, solver, steps, dt, state=None):
    state_0 = model.state() if state is None else state
    state_1 = model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        history.append(_snapshot(state_0))
    return history


class _Snapshot:
    def __init__(self, state):
        self.joint_q = wp.clone(state.joint_q)
        self.joint_qd = wp.clone(state.joint_qd)
        self.body_q = wp.clone(state.body_q)
        self.body_qd = wp.clone(state.body_qd)


def _snapshot(state):
    return _Snapshot(state)


def _beam_basis():
    n = (24, 3, 3)
    half = (0.5 * BEAM_LENGTH, 0.02, 0.015)
    axes = [(np.arange(c) + 0.5) / c * 2.0 * h - h for c, h in zip(n, half, strict=True)]
    points = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1).astype(np.float32)
    s = np.clip(points[:, 0] + 0.5 * BEAM_LENGTH, 0.0, BEAM_LENGTH)
    w = s * s * (3.0 * BEAM_LENGTH - s) / (2.0 * BEAM_LENGTH**3)
    slope = 3.0 * s * (2.0 * BEAM_LENGTH - s) / (2.0 * BEAM_LENGTH**3)
    phi = np.zeros((points.shape[0], 2, 3), dtype=np.float32)
    phi[:, 0, 0], phi[:, 0, 2] = -points[:, 2] * slope, w
    phi[:, 1, 0], phi[:, 1, 1] = -points[:, 1] * slope, w
    sample_mass = np.full(points.shape[0], 0.5 / points.shape[0], dtype=np.float32)
    basis = newton.ModalBasis(
        sample_points=points, sample_phi=phi, sample_mass=sample_mass, mode_stiffness=[80.0, 120.0]
    )
    d = points.astype(np.float64)
    inertia = np.einsum(
        "s,sij->ij", sample_mass, (d * d).sum(1)[:, None, None] * np.eye(3) - d[:, :, None] * d[:, None, :]
    )
    return basis, points, phi, sample_mass, inertia


def _hub_beam(steel, omega, eta, gravity=(0.0, 0.0, 0.0), tip_body=False):
    """World --revolute z--> hub --clamp--> elastic beam along +x."""
    basis, points, phi, sample_mass, inertia = _beam_basis()
    builder = newton.ModelBuilder(gravity=gravity)
    hub_inertia = np.eye(3) * 0.02
    hub = builder.add_link(
        xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(hub_inertia)
    )
    beam_args = {
        "xform": wp.transform((HUB_OFFSET + 0.5 * BEAM_LENGTH, 0.0, 1.0), wp.quat_identity()),
        "mass": float(sample_mass.sum()),
        "inertia": wp.mat33(inertia.astype(np.float32)),
        "modal_basis": basis,
        "mode_q": list(eta),
    }
    revolute = {"axis": (0.0, 0.0, 1.0), "parent_xform": wp.transform((0.0, 0.0, 1.0), wp.quat_identity())}
    clamp_parent = wp.transform((HUB_OFFSET, 0.0, 0.0), wp.quat_identity())
    clamp_child = wp.transform((-0.5 * BEAM_LENGTH, 0.0, 0.0), wp.quat_identity())
    if steel:
        j_rev = builder.add_joint_revolute(-1, hub, **revolute)
        beam = builder.add_link_elastic(
            **beam_args, parent=hub, parent_xform=clamp_parent, child_xform=clamp_child, linear_axes=[], angular_axes=[]
        )
        joints = [j_rev, builder.body_elastic_joint[beam]]
    else:
        beam = builder.add_link_elastic(**beam_args)
        j_rev = builder.add_joint_revolute(-1, hub, **revolute)
        joints = [j_rev, builder.add_joint_fixed(hub, beam, parent_xform=clamp_parent, child_xform=clamp_child)]
    if tip_body:
        tip = builder.add_link(xform=wp.transform((HUB_OFFSET + BEAM_LENGTH, 0.0, 1.0), wp.quat_identity()), mass=0.1)
        joints.append(
            builder.add_joint_fixed(
                beam, tip, parent_xform=wp.transform((0.5 * BEAM_LENGTH, 0.0, 0.0), wp.quat_identity())
            )
        )
    builder.add_articulation(joints)
    builder.color()
    model = builder.finalize()
    state = model.state()
    qd = state.joint_qd.numpy()
    qd[model.joint_qd_start.numpy()[j_rev]] = omega
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if not steel:
        owner = builder.body_elastic_joint[beam]
        qd = state.joint_qd.numpy()
        start = model.joint_qd_start.numpy()[owner]
        qd[start : start + 6] = state.body_qd.numpy()[beam]
        state.joint_qd.assign(qd)

    radius = points.astype(np.float64) + np.array([HUB_OFFSET + 0.5 * BEAM_LENGTH, 0.0, 0.0])
    total_inertia = hub_inertia[2, 2] + np.sum(sample_mass * (radius[:, 0] ** 2 + radius[:, 1] ** 2))
    coupling = np.sum(sample_mass * np.cross(radius, phi[:, 1, :])[:, 2])
    modal_mass = np.sum(sample_mass * np.sum(phi[:, 1, :] ** 2, axis=1))
    info = {
        "hub": hub,
        "beam": beam,
        "points": points,
        "phi": phi,
        "sample_mass": sample_mass,
        "hub_mass": 1.0,
        "hub_inertia": hub_inertia,
        "coupled_frequency": np.sqrt(120.0 * total_inertia / (modal_mass * total_inertia - coupling**2)) / (2 * np.pi),
        "uncoupled_frequency": np.sqrt(120.0 / modal_mass) / (2 * np.pi),
    }
    return model, state, info


def _rocking_base_beams():
    """#4111's base-rotation scene: one beam framed at its clamp, one at mid-span, on a driven revolute base."""
    length, amplitude, frequency = 0.8, 0.25, 1.5
    centered = beam_render_sample_points(length, 0.04, 0.03, extra_points=((-0.5 * length, 0, 0), (0.5 * length, 0, 0)))
    phi = cantilever_tip_mode(centered, length).reshape((-1, 1, 3))
    sample_mass = np.full(centered.shape[0], 1.0 / centered.shape[0], dtype=np.float32)
    modal_mass = float(np.sum(sample_mass * np.sum(phi[:, 0] ** 2, axis=1)))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(
        xform=wp.transform((0.0, 0.0, 0.8), wp.quat_identity()), mass=50.0, inertia=wp.mat33(np.eye(3) * 5.0)
    )
    base_joint = builder.add_joint_revolute(
        -1,
        base,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 0.8), wp.quat_identity()),
        target_ke=1.0e6,
        target_kd=2.0e4,
    )
    joints = [base_joint]
    for origin_x, y in ((0.0, -0.22), (0.5 * length, 0.22)):
        basis = newton.ModalBasis(
            sample_points=centered + np.array([0.5 * length - origin_x, 0.0, 0.0], dtype=np.float32),
            sample_phi=phi,
            sample_mass=sample_mass,
            mode_stiffness=[60.0],
            mode_damping=[0.3 * np.sqrt(60.0 * modal_mass)],
        )
        beam = builder.add_link_elastic(
            xform=wp.transform((origin_x, y, 0.8), wp.quat_identity()),
            com=wp.vec3(0.5 * length - origin_x, 0.0, 0.0),
            mass=0.2,
            inertia=wp.mat33(np.eye(3) * 0.01),
            modal_basis=basis,
            parent=base,
            parent_xform=wp.transform((0.0, y, 0.0), wp.quat_identity()),
            child_xform=wp.transform((-origin_x, 0.0, 0.0), wp.quat_identity()),
            linear_axes=[],
            angular_axes=[],
        )
        joints.append(builder.body_elastic_joint[beam])
    builder.add_articulation(joints)

    def targets(t):
        w = 2.0 * np.pi * frequency
        return amplitude * np.sin(w * t), amplitude * w * np.cos(w * t)

    return builder.finalize(), base_joint, targets


def _free_beam():
    basis, points, phi, sample_mass, inertia = _beam_basis()
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    beam = builder.add_body_elastic(
        xform=wp.transform((0.0, 0.0, 1.0), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.3)),
        mass=float(sample_mass.sum()),
        inertia=wp.mat33(inertia.astype(np.float32)),
        modal_basis=basis,
        mode_qd=[0.4, -0.3],
    )
    model = builder.finalize()
    state = model.state()
    qd = state.joint_qd.numpy()
    start = model.joint_qd_start.numpy()[builder.body_elastic_joint[beam]]
    qd[start : start + 6] = [0.1, 0.0, 0.0, 0.5, 1.5, -0.7]
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, {"hub": None, "beam": beam, "points": points, "phi": phi, "sample_mass": sample_mass}


def _pad_on_ground(mass, stiffness, g):
    """Free pad whose single mode stretches it along z; contact at the bottom face projects through phi_z = -1.

    The render-mesh vertices are basis samples, so contact sees the exact mode values.
    """
    hx, hy, hz = 0.05, 0.05, 0.02
    axes = [(np.arange(c) + 0.5) / c * 2.0 * h - h for c, h in zip((8, 8, 4), (hx, hy, hz), strict=True)]
    volume = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
    surface, _ = box_surface_mesh(2.0 * hx, hy, hz)
    points = np.vstack([volume, surface]).astype(np.float32)
    phi = np.zeros((points.shape[0], 1, 3), dtype=np.float32)
    phi[:, 0, 0] = -0.3 * points[:, 0] / hz
    phi[:, 0, 1] = -0.3 * points[:, 1] / hz
    phi[:, 0, 2] = points[:, 2] / hz
    sample_mass = np.concatenate([np.full(len(volume), mass / len(volume)), np.zeros(len(surface))]).astype(np.float32)
    modal_mass = float(np.sum(sample_mass * np.sum(phi[:, 0] ** 2, axis=1)))
    basis = newton.ModalBasis(
        sample_points=points,
        sample_phi=phi,
        sample_mass=sample_mass,
        mode_stiffness=[stiffness],
        mode_damping=[np.sqrt(stiffness * modal_mass)],
    )
    inertia = np.diag([mass * (hy**2 + hz**2) / 3, mass * (hx**2 + hz**2) / 3, mass * (hx**2 + hy**2) / 3])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -g))
    # Frictionless: friction would resist the mode's Poisson spread of the bottom face.
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, gap=0.002)
    builder.add_ground_plane(cfg=cfg)
    pad = builder.add_body_elastic(
        xform=wp.transform((0.0, 0.0, hz + 0.001), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(inertia.astype(np.float32)),
        modal_basis=basis,
    )
    builder.add_shape_box(pad, hx=hx, hy=hy, hz=hz, cfg=cfg)
    return builder.finalize(), pad


def _mode_history(model, history, mode):
    start = int(model.elastic_mode_q_start.numpy()[0])
    return np.array([float(s.joint_q.numpy()[start + mode]) for s in history])


def _frequency(signal, dt):
    crossings = np.flatnonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0))
    exact = crossings - signal[crossings] / (signal[crossings + 1] - signal[crossings])
    return 1.0 / (np.mean(np.diff(exact)) * dt)


def _cloud_momentum(model, state, info):
    """Linear and angular momentum of the rigid hub plus the deformed, mass-weighted modal samples."""
    body_q, body_qd = state.body_q.numpy(), state.body_qd.numpy()
    start = int(model.elastic_mode_q_start.numpy()[0])
    dstart = int(model.elastic_mode_qd_start.numpy()[0])
    eta = state.joint_q.numpy()[start : start + 2]
    eta_dot = state.joint_qd.numpy()[dstart : dstart + 2]
    linear, angular = np.zeros(3), np.zeros(3)
    if info["hub"] is not None:
        hub = info["hub"]
        rotation = _rotation(body_q[hub][3:])
        linear += info["hub_mass"] * body_qd[hub][:3]
        angular += info["hub_mass"] * np.cross(body_q[hub][:3], body_qd[hub][:3])
        angular += rotation @ info["hub_inertia"] @ rotation.T @ body_qd[hub][3:]
    beam = info["beam"]
    rotation = _rotation(body_q[beam][3:])
    omega = body_qd[beam][3:]
    origin_velocity = body_qd[beam][:3] - np.cross(omega, rotation @ model.body_com.numpy()[beam])
    local = info["points"] + np.einsum("smc,m->sc", info["phi"], eta)
    x = body_q[beam][:3] + local @ rotation.T
    v = (
        origin_velocity
        + np.cross(omega, local @ rotation.T)
        + np.einsum("smc,m->sc", info["phi"], eta_dot) @ rotation.T
    )
    mass = info["sample_mass"][:, None]
    return linear + np.sum(mass * v, axis=0), angular + np.sum(mass * np.cross(x, v), axis=0)


def _rotation(quat):
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
