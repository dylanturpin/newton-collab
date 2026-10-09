# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS as the rigid source of experimental proxy coupling with VBD."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import CouplingInterface, SolverCoupled, SolverCoupledADMM, SolverCoupledProxy
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 480.0


def _fpgs(view, **kwargs):
    return SolverFeatherPGS(view, pgs_mode="matrix_free", **kwargs)


def _build_box_cloth(
    device,
    *,
    box_masses=(5.0,),
    box_z=1.3,
    box_joint_qd=None,
    cloth_z=1.0,
    pinned=True,
    gravity=-9.81,
    ground=False,
    box_xy=(0.0, 0.0),
    cloth=True,
):
    """Build one free box above one cloth sheet per world; returns the model and per-world box ids."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    builder.default_shape_cfg.ke = 2.0e4
    if ground:
        builder.add_ground_plane()
    boxes = []
    for mass in box_masses:
        builder.begin_world()
        box = builder.add_body(xform=wp.transform((box_xy[0], box_xy[1], box_z), wp.quat_identity()))
        cfg = newton.ModelBuilder.ShapeConfig(density=mass / 0.2**3, mu=0.8)
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        if box_joint_qd is not None:
            start = builder.joint_qd_start[-1]
            builder.joint_qd[start : start + 6] = list(box_joint_qd)
        boxes.append(box)
        if not cloth:
            builder.end_world()
            continue
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, cloth_z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=pinned,
            fix_right=pinned,
            dim_x=16,
            dim_y=16,
            cell_x=1.0 / 16.0,
            cell_y=1.0 / 16.0,
            mass=0.1,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=1.0e1,
            edge_ke=0.01,
            particle_radius=0.01,
        )
        builder.end_world()
    if cloth:
        builder.color()
    model = builder.finalize(device=device)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_mu = 0.5
    return model, boxes


def _coupled(model, boxes, rigid_factory=_fpgs, *, iterations=1, mode="lagged", in_place=False):
    rigid_bodies = list(range(model.body_count))
    return SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name="rigid",
                solver=rigid_factory,
                bodies=rigid_bodies,
                joints=list(range(model.joint_count)),
                in_place=in_place,
            ),
            SolverCoupledProxy.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=10, rigid_compliant_alm=True),
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="rigid",
                    destination="vbd",
                    bodies=rigid_bodies,
                    mode=mode,
                    collision_pipeline=newton.CollisionPipeline,
                    collide_interval=1,
                )
            ],
            iterations=iterations,
        ),
    )


class _Rollout:
    """Owns a solver, a double-buffered state and the outer contact buffer."""

    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        if isinstance(solver, SolverCoupledProxy):
            solver.prepare_contacts(self.contacts)

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def run(self, steps):
        body_q = []
        for _ in range(steps):
            self.step()
            body_q.append(self.state_0.body_q.numpy().copy())
        return np.asarray(body_q)


def test_no_contact_matches_standalone(test, device):
    """A box far from the cloth follows standalone FeatherPGS exactly: no feedback, no double gravity."""
    for iterations in (1, 3):
        model, _ = _build_box_cloth(device, box_z=3.0, ground=True)
        standalone_model, _ = _build_box_cloth(device, box_z=3.0, ground=True, cloth=False)
        coupled = _Rollout(model, _coupled(model, None, iterations=iterations))
        reference = _Rollout(standalone_model, _fpgs(standalone_model))
        coupled_q = coupled.run(60)
        reference_q = reference.run(60)
        np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-6)
        fall = reference_q[0, 0, 2] - reference_q[-1, 0, 2]
        expected = 0.5 * 9.81 * ((60 * DT) ** 2 - DT**2)
        test.assertAlmostEqual(fall, expected, delta=0.02 * expected)
        feedback = coupled.solver._proxy_mappings[0].coupling_forces.numpy()
        test.assertEqual(float(np.abs(feedback).max()), 0.0)


def test_in_place_restart_refreshes_kinematics(test, device):
    """Restarting an in-place source re-derives kinematics instead of reusing the previous solve's."""
    kwargs = {"box_z": 3.0, "box_joint_qd": (0.3, 0.0, 0.0, 0.0, 2.0, 1.0)}
    reference_model, _ = _build_box_cloth(device, cloth=False, **kwargs)
    reference_q = _Rollout(reference_model, _fpgs(reference_model)).run(60)
    model, _ = _build_box_cloth(device, **kwargs)
    coupled_q = _Rollout(model, _coupled(model, None, iterations=3, mode="staggered", in_place=True)).run(60)
    np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-5)


def test_gravity_acceleration_omits_disabled_bodies(test, device):
    """The reported gravity acceleration matches what FeatherPGS applies per body."""
    builder = newton.ModelBuilder()
    for disable_gravity in (False, True):
        body = builder.add_body(disable_gravity=disable_gravity)
        builder.add_shape_sphere(body, radius=0.1)
    model = builder.finalize(device=device)
    acceleration = wp.zeros(model.body_count, dtype=wp.vec3, device=device)
    _fpgs(model).coupling_eval_gravity_acceleration(acceleration, None)
    np.testing.assert_allclose(acceleration.numpy(), [[0.0, 0.0, -9.81], [0.0, 0.0, 0.0]], atol=1.0e-6)


def test_iteration_restart_preserves_patch_history(test, device):
    """Repeated proxy solves of one step leave persistent patch friction identical to standalone."""
    # A spinning box lands on the FeatherPGS-owned ground far from the cloth; landing creates new patches.
    kwargs = {"box_z": 0.13, "box_xy": (3.0, 0.0), "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0), "ground": True}
    reference_model, _ = _build_box_cloth(device, cloth=False, **kwargs)
    reference_q = _Rollout(reference_model, _fpgs(reference_model)).run(120)
    test.assertGreater(np.linalg.norm(reference_q[-1, 0, :2] - reference_q[0, 0, :2]), 0.01)
    for iterations in (1, 3):
        model, _ = _build_box_cloth(device, **kwargs)
        coupled_q = _Rollout(model, _coupled(model, None, iterations=iterations)).run(120)
        np.testing.assert_allclose(coupled_q, reference_q, atol=1.0e-5, err_msg=f"iterations={iterations}")


def test_contact_feedback_is_consumed_exactly(test, device):
    """In gravity-free contact, each step's box impulse equals the previous VBD harvest, force and torque."""
    # Off-center impact so the harvested wrench carries torque.
    model, boxes = _build_box_cloth(
        device, box_z=1.13, box_xy=(0.15, 0.05), box_joint_qd=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0), pinned=False, gravity=0.0
    )
    rollout = _Rollout(model, _coupled(model, boxes))
    mapping = rollout.solver._proxy_mappings[0]
    mass = float(model.body_mass.numpy()[0])
    inertia = model.body_inertia.numpy()[0]
    max_force = 0.0
    previous_wrench = np.zeros(6)
    for _ in range(150):
        qd_before = rollout.state_0.body_qd.numpy()[0].copy()
        q_before = rollout.state_0.body_q.numpy()[0].copy()
        rollout.step()
        qd_after = rollout.state_0.body_qd.numpy()[0]
        # Lagged mode: this step's source solve consumed the previous harvest.
        np.testing.assert_allclose(qd_after[:3] - qd_before[:3], previous_wrench[:3] * DT / mass, atol=1.0e-4)
        rotation = np.array(wp.quat_to_matrix(wp.quat(*q_before[3:7]))).reshape(3, 3)
        world_inertia = rotation @ inertia @ rotation.T
        np.testing.assert_allclose(
            qd_after[3:] - qd_before[3:], np.linalg.solve(world_inertia, previous_wrench[3:]) * DT, atol=2.0e-3
        )
        previous_wrench = mapping.coupling_forces.numpy()[0].astype(np.float64)
        max_force = max(max_force, float(np.linalg.norm(previous_wrench[:3])))
    test.assertGreater(max_force, 1.0, "the box never pressed into the cloth")
    # Reciprocal reaction: the box slowed down and the free cloth was pushed along -z.
    test.assertGreater(float(rollout.state_0.body_qd.numpy()[0][2]), -0.95)
    test.assertLess(float(rollout.state_0.particle_qd.numpy()[:, 2].mean()), -0.05)


def test_box_rests_on_cloth_like_mujoco(test, device):
    """A box settles on a pinned cloth; FeatherPGS and MuJoCo sources agree within 2 mm."""
    rest = {}
    for name, factory in (
        ("fpgs", _fpgs),
        ("mjc", lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=200)),
    ):
        model, boxes = _build_box_cloth(device)
        rollout = _Rollout(model, _coupled(model, boxes, factory))
        rollout.run(480)
        rest[name] = (
            float(rollout.state_0.body_q.numpy()[0][2]),
            float(rollout.state_0.particle_q.numpy()[:, 2].min()),
        )
    box_z, cloth_z = rest["fpgs"]
    test.assertLess(cloth_z, 0.97, "the cloth did not deform")
    test.assertGreater(box_z, cloth_z + 0.09, "the box sank through the cloth")
    test.assertAlmostEqual(box_z, rest["mjc"][0], delta=2.0e-3)
    test.assertAlmostEqual(cloth_z, rest["mjc"][1], delta=2.0e-3)


def test_worlds_are_isolated(test, device):
    """Two worlds with unequal masses match their single-world rollouts."""
    model, boxes = _build_box_cloth(device, box_masses=(2.0, 10.0))
    both = _Rollout(model, _coupled(model, boxes)).run(240)
    test.assertGreater(
        abs(both[-1, boxes[0], 2] - both[-1, boxes[1], 2]), 2.0e-2, "unequal masses should sag differently"
    )
    for world, mass in enumerate((2.0, 10.0)):
        single_model, _ = _build_box_cloth(device, box_masses=(mass,))
        single = _Rollout(single_model, _coupled(single_model, None)).run(240)
        # Contact ordering differs between layouts; leakage would show at the 2 cm inter-world scale.
        np.testing.assert_allclose(both[:, boxes[world], :3], single[:, 0, :3], atol=2.0e-3)


def test_masked_reset(test, device):
    """Coupled and standalone resets accept the SolverBase world-mask layout."""
    model, boxes = _build_box_cloth(device, box_masses=(5.0, 5.0))
    rollout = _Rollout(model, _coupled(model, boxes))
    rollout.run(10)
    mask = wp.array([True, False, False], dtype=wp.bool, device=device)
    rollout.solver.reset(rollout.state_0, world_mask=mask)
    rollout.run(10)
    test.assertTrue(np.all(np.isfinite(rollout.state_0.body_q.numpy())))

    standalone_model, _ = _build_box_cloth(device, box_masses=(5.0, 5.0))
    solver = _fpgs(standalone_model)
    state = standalone_model.state()
    solver.reset(state, world_mask=mask)
    solver.reset(state, world_mask=wp.array([True, False], dtype=wp.bool, device=device))
    with test.assertRaises(ValueError):
        solver.reset(state, world_mask=wp.array([True], dtype=wp.bool, device=device))


def test_graph_capture_matches_eager(test, device):
    """Repeated graph replay of the coupled step reproduces eager stepping, including after a reset."""
    results = []
    for use_graph in (False, True):
        model, boxes = _build_box_cloth(device, box_masses=(5.0, 8.0))
        rollout = _Rollout(model, _coupled(model, boxes, iterations=2))
        graph = None
        if use_graph:
            # Both buffers must swap back to the captured layout, so capture two steps.
            with wp.ScopedCapture(device) as capture:
                rollout.step()
                rollout.step()
            graph = capture.graph
        trajectory = []
        for _ in range(60):
            if graph is None:
                rollout.step()
                rollout.step()
            else:
                wp.capture_launch(graph)
            trajectory.append(rollout.state_0.body_q.numpy().copy())
        rollout.solver.reset(rollout.state_0)
        for _ in range(10):
            if graph is None:
                rollout.step()
                rollout.step()
            else:
                wp.capture_launch(graph)
            trajectory.append(rollout.state_0.body_q.numpy().copy())
        results.append(np.asarray(trajectory))
    np.testing.assert_allclose(results[1][:60], results[0][:60], atol=1.0e-5)
    # Host-side reset bookkeeping makes post-reset replay differ at the 1e-4 level for any source solver.
    np.testing.assert_allclose(results[1][60:], results[0][60:], atol=1.0e-3)


def test_unsupported_options_raise(test, device):
    """Options that are not validated in coupled use fail explicitly."""
    for kwargs in (
        {"pgs_warmstart": True},
        {"enable_sleeping": True},
    ):
        model, boxes = _build_box_cloth(device)
        with test.assertRaises(NotImplementedError):
            _Rollout(model, _coupled(model, boxes, lambda v, kwargs=kwargs: _fpgs(v, **kwargs))).step()


def test_articulated_effective_mass(test, device):
    """A hinged link reports the translational mobility of its COM, not its free mass."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    mass, half_length, armature = 2.0, 0.25, 0.01
    link = builder.add_link()
    builder.add_shape_box(
        link,
        hx=half_length,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=mass / (8 * half_length * 0.05**2)),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform((-half_length, 0.0, 0.0), wp.quat_identity()),
        armature=armature,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = _fpgs(model)
    kind = wp.array([int(CouplingInterface.EndpointKind.BODY)] * 2, dtype=int, device=device)
    index = wp.array([link, link], dtype=int, device=device)
    local_pos = wp.array([wp.vec3(0.0), wp.vec3(half_length, 0.0, 0.0)], dtype=wp.vec3, device=device)
    out_mass = wp.zeros(2, dtype=float, device=device)
    out_inertia = wp.zeros(2, dtype=wp.mat33, device=device)
    solver.coupling_eval_effective_mass_block(kind, index, local_pos, out_mass, out_inertia)

    inertia_yy = float(model.body_inertia.numpy()[link][1, 1])
    joint_inertia = inertia_yy + mass * half_length**2 + armature
    # Only one direction is mobile, so the axis-mean inverse weight is r^2 / (3 H).
    for row, lever in enumerate((half_length, 2.0 * half_length)):
        test.assertAlmostEqual(float(out_mass.numpy()[row]), 3.0 * joint_inertia / lever**2, delta=1.0e-3)
    scale = (np.trace(np.linalg.inv(model.body_inertia.numpy()[link])) / 3.0) / (1.0 / (3.0 * joint_inertia))
    np.testing.assert_allclose(out_inertia.numpy()[0], model.body_inertia.numpy()[link] * scale, rtol=1.0e-4)


def test_offset_contact_turns_hinge(test, device):
    """A soft block landing off the hinge axis turns the FeatherPGS joint, more so with a longer lever."""
    rotation = {}
    for lever in (0.08, 0.32):
        builder = newton.ModelBuilder()
        link = builder.add_link(disable_gravity=True)
        builder.add_shape_box(link, hx=0.2, hy=0.1, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=1.0 / 0.0032))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform((-0.2, 0.0, 0.0), wp.quat_identity()),
            target_ke=20.0,
            target_kd=1.0,
        )
        builder.add_articulation([joint])
        builder.add_soft_grid(
            pos=wp.vec3(lever - 0.05, -0.05, 1.04),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1.0e3,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1.0e-2,
            particle_radius=0.01,
        )
        builder.color()
        model = builder.finalize(device=device)
        model.soft_contact_ke = 1.0e4
        rollout = _Rollout(model, _coupled(model, [link]))
        peak = 0.0
        for _ in range(240):
            rollout.step()
            peak = max(peak, float(rollout.state_0.joint_q.numpy()[0]))
        rotation[lever] = peak
    # Pushing down on +x rotates the link positively about +y.
    test.assertGreater(rotation[0.08], 0.0)
    test.assertGreater(rotation[0.32], 2.0 * rotation[0.08])


def _build_box_rod(device, *, box_z=0.15, box_xy=(0.0, 0.0), box_joint_qd=None, rod=True):
    """Build a free FeatherPGS box above two VBD cables lying on the ground; returns model, box and rod ids."""
    builder = newton.ModelBuilder()
    SolverVBD.register_custom_attributes(builder)
    builder.add_ground_plane()
    box = builder.add_body(xform=wp.transform((box_xy[0], box_xy[1], box_z), wp.quat_identity()))
    # The box overhangs both cables so it cannot drop between them.
    builder.add_shape_box(box, hx=0.05, hy=0.08, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=2.0 / 0.0016))
    if box_joint_qd is not None:
        builder.joint_qd[0:6] = list(box_joint_qd)
    rod_bodies, rod_joints = [], []
    # Two parallel cables support the box stably.
    for y in (-0.04, 0.04) if rod else ():
        cable = newton.Rod.create_straight(
            start=wp.vec3(-0.2, y, 0.012),
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=0.4,
            segment_count=8,
            twist_total=0.0,
            radius=0.012,
        )
        bodies, joints = builder.add_rod(
            rod=cable,
            body_frame_origin="start",
            cfg=newton.ModelBuilder.ShapeConfig(density=1400.0, ke=5.0e4, kd=1.0e1, mu=0.9, margin=0.001, gap=0.002),
            stretch_stiffness=2.0e5,
            stretch_damping=2.0e-2,
            bend_stiffness=0.08,
            bend_damping=1.6e-3,
        )
        rod_bodies += list(bodies)
        rod_joints += list(joints)
    builder.color()
    return builder.finalize(device=device), box, rod_bodies, rod_joints


def _admm(model, box, rod_bodies, rod_joints, rigid_factory=_fpgs, *, iterations=3, gamma=0.001):
    model.rigid_contact_max = max(model.rigid_contact_max or 0, 4096)
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="rigid",
                solver=rigid_factory,
                bodies=[box],
                joints=[j for j in range(model.joint_count) if j not in rod_joints],
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(
                    model=v, iterations=8, rigid_compliant_alm=True, rigid_contact_history=False
                ),
                bodies=rod_bodies,
                joints=rod_joints,
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=iterations,
            rho=200.0,
            gamma=gamma,
            baumgarte=0.5,
            contact_pairs=[SolverCoupledADMM.ContactPair(source="rigid", destination="vbd")],
        ),
    )


def test_admm_no_contact_matches_standalone(test, device):
    """ADMM restarts leave a box that never touches the cable identical to standalone FeatherPGS."""
    kwargs = {"box_z": 0.08, "box_xy": (3.0, 0.0), "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0)}
    reference_model, _, _, _ = _build_box_rod(device, rod=False, **kwargs)
    reference_q = _Rollout(reference_model, _fpgs(reference_model)).run(120)
    for iterations in (1, 3):
        model, box, rod_bodies, rod_joints = _build_box_rod(device, **kwargs)
        rollout = _Rollout(model, _admm(model, box, rod_bodies, rod_joints, iterations=iterations, gamma=0.0))
        coupled_q = rollout.run(120)
        np.testing.assert_allclose(
            coupled_q[:, box], reference_q[:, 0], atol=1.0e-5, err_msg=f"iterations={iterations}"
        )


def test_admm_box_rests_on_cable(test, device):
    """A box dropped on two free VBD cables is supported by them through ADMM contact rows."""
    model, box, rod_bodies, rod_joints = _build_box_rod(device)
    rollout = _Rollout(model, _admm(model, box, rod_bodies, rod_joints))
    rod_z0 = rollout.state_0.body_q.numpy()[rod_bodies, 2].copy()
    rollout.run(360)
    body_q = rollout.state_0.body_q.numpy()
    # Box bottom on the cable top: cable diameter plus box half height, within contact compliance.
    test.assertAlmostEqual(float(body_q[box, 2]), 0.024 + 0.05, delta=4.0e-3)
    # Free cables let the supported box rock; this bounds total rotation to about 28 degrees.
    test.assertGreater(abs(float(body_q[box, 6])), 0.97, "the box rotated off its supports")
    test.assertGreater(float(body_q[rod_bodies, 2].min()), 0.0, "the cable was pushed through the ground")
    test.assertLess(float(np.abs(body_q[rod_bodies, 2] - rod_z0).max()), 0.01)
    test.assertTrue(np.all(np.isfinite(body_q)))


def _history_snapshot(solver):
    patches = solver._friction_patches
    return {
        "q": patches.previous_q.numpy().copy(),
        "world": patches.previous_world.numpy().copy(),
        "valid": patches.previous.valid.numpy().copy(),
        "displacement": patches.previous.displacement.numpy().copy(),
    }


def test_reset_preserves_unselected_patch_history(test, device):
    """All-false and global-only coupled resets are no-ops; a single-world reset keeps the other world's history."""
    kwargs = {
        "box_masses": (5.0, 5.0),
        "box_z": 0.13,
        "box_xy": (3.0, 0.0),
        "box_joint_qd": (1.0, 0.3, -0.5, 0.0, 0.0, 3.0),
        "ground": True,
    }
    for mask_values in ((False, False, False), (False, False, True), (False, True, False)):
        model, boxes = _build_box_cloth(device, **kwargs)
        rollout = _Rollout(model, _coupled(model, boxes, iterations=2))
        rollout.run(60)
        solver = rollout.solver.solver("rigid")
        before = _history_snapshot(solver)
        test.assertGreater(int(np.count_nonzero(before["valid"])), 0, "no carried patches to protect")
        rollout.solver.reset(rollout.state_0, wp.array(mask_values, dtype=wp.bool, device=device))
        after = _history_snapshot(solver)
        kept = before["world"] != 1 if mask_values[1] else np.ones_like(before["world"], dtype=bool)
        test.assertTrue(np.any(kept & (before["valid"] != 0)))
        np.testing.assert_array_equal(after["valid"][kept], before["valid"][kept], err_msg=str(mask_values))
        np.testing.assert_array_equal(after["displacement"][kept], before["displacement"][kept])
        np.testing.assert_array_equal(after["q"][boxes[0]], before["q"][boxes[0]])
        if mask_values[1]:
            test.assertEqual(int(np.count_nonzero(after["valid"][before["world"] == 1])), 0)
        else:
            np.testing.assert_array_equal(after["q"], before["q"])


def test_effective_mass_ignores_loop_closures(test, device):
    """Disabled loop-closing joints leave the tree-only effective mass unchanged, also before a later articulation."""
    results = {}
    for links in (1, 2):
        for closure in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for _articulation in range(2):
                bodies, joints = [], []
                for index in range(links):
                    body = builder.add_link()
                    builder.add_shape_box(
                        body, hx=0.25, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=400.0)
                    )
                    joints.append(
                        builder.add_joint_revolute(
                            -1 if index == 0 else bodies[-1],
                            body,
                            axis=wp.vec3(0.0, 1.0, 0.0),
                            parent_xform=wp.transform((0.0 if index == 0 else 0.25, 0.0, 0.0), wp.quat_identity()),
                            child_xform=wp.transform((-0.25, 0.0, 0.0), wp.quat_identity()),
                            armature=0.01,
                        )
                    )
                    bodies.append(body)
                builder.add_articulation(joints)
                if closure:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        builder.add_joint_ball(-1, bodies[0], enabled=False)
            model = builder.finalize(device=device)
            solver = _fpgs(model)
            last = [links - 1, 2 * links - 1]
            kind = wp.array([int(CouplingInterface.EndpointKind.BODY)] * 2, dtype=int, device=device)
            mass = wp.zeros(2, dtype=float, device=device)
            solver.coupling_eval_effective_mass(
                kind, wp.array(last, dtype=int, device=device), wp.zeros(2, dtype=wp.vec3, device=device), mass
            )
            results[(links, closure)] = mass.numpy()
        np.testing.assert_allclose(results[(links, True)], results[(links, False)], rtol=1.0e-5)
        np.testing.assert_allclose(results[(links, False)][0], results[(links, False)][1], rtol=1.0e-5)
    test.assertAlmostEqual(float(results[(1, False)][0]), 8.56, delta=1.0e-3)


def test_unsupported_roles_raise(test, device):
    """FeatherPGS rejects owning particles and acting as a proxy destination."""
    model, _ = _build_box_cloth(device)
    with test.assertRaises(NotImplementedError):
        solver = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="rigid",
                    solver=_fpgs,
                    bodies=list(range(model.body_count)),
                    joints=list(range(model.joint_count)),
                    particles=list(range(model.particle_count)),
                ),
            ],
        )
        _Rollout(model, solver).step()

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    rigid_box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(rigid_box, hx=0.1, hy=0.1, hz=0.1)
    soft_box = builder.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(soft_box, hx=0.1, hy=0.1, hz=0.1)
    builder.color()
    model = builder.finalize(device=device)
    with test.assertRaises(NotImplementedError):
        solver = SolverCoupledProxy(
            model=model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=4, rigid_compliant_alm=True),
                    bodies=[soft_box],
                    joints=[1],
                ),
                SolverCoupledProxy.Entry(name="rigid", solver=_fpgs, bodies=[rigid_box], joints=[0]),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[SolverCoupledProxy.Proxy(source="vbd", destination="rigid", bodies=[soft_box])]
            ),
        )
        _Rollout(model, solver).step()


class TestCoupledSolverFeatherPGS(unittest.TestCase):
    pass


for _name, _func in (
    ("test_no_contact_matches_standalone", test_no_contact_matches_standalone),
    ("test_iteration_restart_preserves_patch_history", test_iteration_restart_preserves_patch_history),
    ("test_in_place_restart_refreshes_kinematics", test_in_place_restart_refreshes_kinematics),
    ("test_gravity_acceleration_omits_disabled_bodies", test_gravity_acceleration_omits_disabled_bodies),
    ("test_contact_feedback_is_consumed_exactly", test_contact_feedback_is_consumed_exactly),
    ("test_box_rests_on_cloth_like_mujoco", test_box_rests_on_cloth_like_mujoco),
    ("test_worlds_are_isolated", test_worlds_are_isolated),
    ("test_masked_reset", test_masked_reset),
    ("test_graph_capture_matches_eager", test_graph_capture_matches_eager),
    ("test_unsupported_options_raise", test_unsupported_options_raise),
    ("test_articulated_effective_mass", test_articulated_effective_mass),
    ("test_offset_contact_turns_hinge", test_offset_contact_turns_hinge),
    ("test_reset_preserves_unselected_patch_history", test_reset_preserves_unselected_patch_history),
    ("test_effective_mass_ignores_loop_closures", test_effective_mass_ignores_loop_closures),
    ("test_unsupported_roles_raise", test_unsupported_roles_raise),
    ("test_admm_no_contact_matches_standalone", test_admm_no_contact_matches_standalone),
    ("test_admm_box_rests_on_cable", test_admm_box_rests_on_cable),
):
    add_function_test(TestCoupledSolverFeatherPGS, _name, _func, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
