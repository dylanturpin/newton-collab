# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mesh and inertia switching at an episode reset with SolverSteel."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import (
    add_function_test,
    configure_sdf_for_collision_shapes,
    get_selected_cuda_test_devices,
    get_test_devices,
)

DT = 1.0 / 120.0
DROP_HEIGHT = 0.3
NUT_HALF_HEIGHT = 0.05
ROD_HALF_HEIGHT = 0.10
NUT_DENSITY = 1000.0
ROD_DENSITY = 2500.0
SETTLE_STEPS = 240
PUSH_STEPS = 20
COAST_STEPS = 40
PUSH_FORCE = 30.0
POSE_ATOL = 2.0e-3

# World roles: world 0 keeps its nut, world 1 switches from the nut to the rod at the reset,
# world 2 carries the rod from construction and stands in for a fresh scene built with it.
NUT_WORLD, SWITCH_WORLD, ROD_WORLD = 0, 1, 2
WORLD_COUNT = 3

INERTIAL_ARRAYS = ("body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia")


def _world_builder(scene, active, resource, density):
    """One world: a support, a free body carrying ``active`` and the hidden ``resource`` variant."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if scene == "plane":
        builder.add_ground_plane()
    else:
        table = newton.Mesh.create_box(0.5, 0.5, 0.05)
        builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()), mesh=table)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, DROP_HEIGHT), wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=active, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    # The variant is finalized as a static shape of the same world so its collision companions
    # (baked AABB, edges, SDF) exist; collision_group=0 keeps it out of every pair list, which a
    # flag cleared after finalize would not, since the explicit broad phase takes its pairs there.
    builder.add_shape_mesh(
        -1, mesh=resource, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=False, collision_group=0)
    )
    return builder


def _build_model(scene, device, *, switch_mesh=True):
    """Three worlds: nut, nut (switched at the reset) and the reference variant."""
    nut = newton.Mesh.create_box(0.05, 0.05, NUT_HALF_HEIGHT)
    rod = newton.Mesh.create_box(0.04, 0.04, ROD_HALF_HEIGHT)
    reference = rod if switch_mesh else nut
    root = newton.ModelBuilder(up_axis=newton.Axis.Z)
    root.add_world(_world_builder(scene, nut, rod, NUT_DENSITY))
    root.add_world(_world_builder(scene, nut, rod, NUT_DENSITY))
    root.add_world(_world_builder(scene, reference, nut, ROD_DENSITY))
    if scene == "sdf":
        configure_sdf_for_collision_shapes(root)
    return root.finalize(device=device)


class _Sim:
    """A model with its solver, pipeline, states and the device-side selection arrays of the set path."""

    def __init__(self, model, *, solver_kwargs=None, pipeline_kwargs=None):
        self.model = model
        self.device = model.device
        # Deterministic contact order makes the switched world and the built-in reference agree exactly.
        self.solver = SolverSteel(model, deterministic=True, **(solver_kwargs or {}))
        self.pipeline = newton.CollisionPipeline(model, deterministic=True, **(pipeline_kwargs or {}))
        self.contacts = self.pipeline.contacts()
        self.state_0, self.state_1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self.graph = None

        body_world = model.body_world.numpy()
        shape_world, shape_body = model.shape_world.numpy(), model.shape_body.numpy()
        shape_group = model.shape_collision_group.numpy()
        self.body = [int(np.flatnonzero(body_world == w)[0]) for w in range(WORLD_COUNT)]
        self.active_shape = [
            int(np.flatnonzero((shape_world == w) & (shape_body == self.body[w]))[0]) for w in range(WORLD_COUNT)
        ]
        self.resource_shape = [
            int(np.flatnonzero((shape_world == w) & (shape_body == -1) & (shape_group == 0))[0])
            for w in range(WORLD_COUNT)
        ]
        # Fixed-size per-world selections, -1 for worlds that keep their shape, so the set path replays from a graph.
        sources = np.full(WORLD_COUNT, -1, dtype=np.int32)
        targets = np.full(WORLD_COUNT, -1, dtype=np.int32)
        sources[SWITCH_WORLD] = self.resource_shape[SWITCH_WORLD]
        targets[SWITCH_WORLD] = self.active_shape[SWITCH_WORLD]
        self.switch_sources = wp.array(sources, dtype=wp.int32, device=self.device)
        self.switch_targets = wp.array(targets, dtype=wp.int32, device=self.device)
        reset = np.zeros(WORLD_COUNT, dtype=bool)
        reset[[SWITCH_WORLD, ROD_WORLD]] = True
        self.reset_mask = wp.array(reset, dtype=wp.bool, device=self.device)
        self.pipeline_reset_mask = wp.array(np.append(reset, False), dtype=wp.bool, device=self.device)

    def step(self):
        self.pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, DT)
        self.state_0.assign(self.state_1)

    def run(self, steps):
        for _ in range(steps):
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self.step()

    def capture_step(self):
        self.step()  # load the modules outside the capture
        with wp.ScopedCapture(self.device) as capture:
            self.step()
        self.graph = capture.graph

    def body_q(self):
        return self.state_0.body_q.numpy().copy()

    def teleport_reset_worlds(self):
        """Put the reset worlds' bodies back at the drop height, at rest, on the host."""
        model = self.model
        joint_q, joint_qd = self.state_0.joint_q.numpy(), self.state_0.joint_qd.numpy()
        initial_q = model.joint_q.numpy()
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        joint_world = model.joint_world.numpy()
        for joint in np.flatnonzero(np.isin(joint_world, (SWITCH_WORLD, ROD_WORLD))):
            joint_q[q_start[joint] : q_start[joint + 1]] = initial_q[q_start[joint] : q_start[joint + 1]]
            joint_qd[qd_start[joint] : qd_start[joint + 1]] = 0.0
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def switch(self, *, mesh=True):
        """The set path: copy the variant's geometry and inertia into world 1, then notify and reset.

        Every call is a device operation so the whole sequence can be captured in a CUDA graph.
        """
        model = self.model
        flags = ModelFlags.BODY_INERTIAL_PROPERTIES
        if mesh:
            model.copy_shape_geometry(self.switch_sources, self.switch_targets)
            flags |= ModelFlags.SHAPE_PROPERTIES
        source, target = self.body[ROD_WORLD], self.body[SWITCH_WORLD]
        for name in INERTIAL_ARRAYS:
            array = getattr(model, name)
            wp.copy(array, array, dest_offset=target, src_offset=source, count=1)
        self.solver.notify_model_changed(flags, world_mask=self.reset_mask)
        self.reset()

    def reset(self):
        """Reset worlds 1 and 2 in the solver and the pipeline, as the set path does after its copies."""
        self.solver.reset(self.state_0, self.reset_mask)
        self.pipeline.reset_contact_matching(self.pipeline_reset_mask)

    def push(self, worlds, steps):
        force = np.zeros((self.model.body_count, 6), dtype=np.float32)
        for world in worlds:
            force[self.body[world], 0] = PUSH_FORCE
        for _ in range(steps):
            self.state_0.body_f.assign(force)
            self.run(1)
        self.state_0.body_f.zero_()


def _support_scene(device):
    """Mesh-on-mesh contacts on CUDA; the CPU variants rest the meshes on the ground plane."""
    return "table" if wp.get_device(device).is_cuda else "plane"


def _settle_reset_and_switch(scene, device, *, switch_mesh=True, solver_kwargs=None, pipeline_kwargs=None):
    """Settle two identical scenes, reset worlds 1 and 2 in both, and switch world 1 in the first only.

    The second scene is the baseline: its world 0 shows what the untouched world should do, and its
    world 1 keeps the nut the switch replaced.
    """
    sims = []
    for _ in range(2):
        model = _build_model(scene, device, switch_mesh=switch_mesh)
        sims.append(_Sim(model, solver_kwargs=solver_kwargs, pipeline_kwargs=pipeline_kwargs))
    sim, baseline = sims
    for s in sims:
        s.run(SETTLE_STEPS)
        s.teleport_reset_worlds()
    sim.switch(mesh=switch_mesh)
    baseline.reset()
    for s in sims:
        s.run(SETTLE_STEPS)
    return sim, baseline


def _assert_untouched_world_unchanged(sim, baseline):
    """World 0 was never reset: its trajectory is the one it has without a switch next door."""
    np.testing.assert_allclose(
        sim.body_q()[sim.body[NUT_WORLD]], baseline.body_q()[baseline.body[NUT_WORLD]], rtol=0.0, atol=1.0e-4
    )


def _assert_switched_world_matches_reference(test, sim, atol=POSE_ATOL):
    """World 1 must rest and respond to a push like world 2, the variant built in from the start."""
    q = sim.body_q()
    np.testing.assert_allclose(q[sim.body[SWITCH_WORLD]], q[sim.body[ROD_WORLD]], rtol=0.0, atol=atol)
    sim.push((SWITCH_WORLD, ROD_WORLD), PUSH_STEPS)
    sim.run(COAST_STEPS)
    q = sim.body_q()
    test.assertGreater(float(q[sim.body[SWITCH_WORLD], 0]), 0.02, "the push should have moved the switched body")
    np.testing.assert_allclose(q[sim.body[SWITCH_WORLD]], q[sim.body[ROD_WORLD]], rtol=0.0, atol=atol)


def test_mesh_switch_matches_fresh_scene(test, device):
    """Verify a mesh switched at reset rests and moves like the same mesh built in, and the other world is untouched."""
    sim, baseline = _settle_reset_and_switch(_support_scene(device), device)
    q, q_base = sim.body_q(), baseline.body_q()
    # The switched world carries the rod now: it rests at the rod's half height, not the nut's.
    test.assertAlmostEqual(float(q[sim.body[SWITCH_WORLD], 2]), ROD_HALF_HEIGHT, delta=POSE_ATOL)
    test.assertAlmostEqual(float(q_base[baseline.body[SWITCH_WORLD], 2]), NUT_HALF_HEIGHT, delta=POSE_ATOL)
    test.assertAlmostEqual(float(q[sim.body[NUT_WORLD], 2]), NUT_HALF_HEIGHT, delta=POSE_ATOL)
    _assert_untouched_world_unchanged(sim, baseline)
    _assert_switched_world_matches_reference(test, sim)
    baseline.push((SWITCH_WORLD,), PUSH_STEPS)
    baseline.run(COAST_STEPS)
    nut_travel = float(baseline.body_q()[baseline.body[SWITCH_WORLD], 0])
    rod_travel = float(sim.body_q()[sim.body[SWITCH_WORLD], 0])
    test.assertGreater(nut_travel - rod_travel, 0.01, "the heavier rod should travel less than the nut it replaced")


def test_mesh_switch_with_sdf_contacts_and_warm_start(test, device):
    """Verify the switch under mesh-on-mesh SDF contacts with contact matching and a warm start."""
    sim, baseline = _settle_reset_and_switch(
        "sdf", device, solver_kwargs={"contact_warmstart": True}, pipeline_kwargs={"contact_matching": "latest"}
    )
    test.assertAlmostEqual(float(sim.body_q()[sim.body[SWITCH_WORLD], 2]), ROD_HALF_HEIGHT, delta=POSE_ATOL)
    # The masked notification and reset leave world 0's warm start alone.
    _assert_untouched_world_unchanged(sim, baseline)
    _assert_switched_world_matches_reference(test, sim)


def test_mesh_switch_from_captured_graphs(test, device):
    """Verify the set path and the step replay from CUDA graphs captured before the switch."""
    model = _build_model("sdf", device)
    sim = _Sim(model, solver_kwargs={"contact_warmstart": True}, pipeline_kwargs={"contact_matching": "latest"})
    sim.capture_step()
    sim.run(SETTLE_STEPS)
    sim.teleport_reset_worlds()
    with wp.ScopedCapture(device) as capture:
        sim.switch()
    wp.capture_launch(capture.graph)
    sim.run(SETTLE_STEPS)
    q = sim.body_q()
    test.assertAlmostEqual(float(q[sim.body[SWITCH_WORLD], 2]), ROD_HALF_HEIGHT, delta=POSE_ATOL)
    test.assertAlmostEqual(float(q[sim.body[NUT_WORLD], 2]), NUT_HALF_HEIGHT, delta=POSE_ATOL)
    _assert_switched_world_matches_reference(test, sim)


def test_inertia_switch_matches_fresh_scene(test, device):
    """Verify a mass and inertia switch at reset, with the mesh kept, moves like the heavier body built in."""
    sim, baseline = _settle_reset_and_switch(_support_scene(device), device, switch_mesh=False)
    _assert_untouched_world_unchanged(sim, baseline)
    sim.push((NUT_WORLD, SWITCH_WORLD, ROD_WORLD), PUSH_STEPS)
    sim.run(COAST_STEPS)
    q = sim.body_q()
    np.testing.assert_allclose(q[sim.body[SWITCH_WORLD]], q[sim.body[ROD_WORLD]], rtol=0.0, atol=POSE_ATOL)
    light_travel, heavy_travel = float(q[sim.body[NUT_WORLD], 0]), float(q[sim.body[SWITCH_WORLD], 0])
    test.assertGreater(light_travel - heavy_travel, 0.01, "the lighter nut should travel further under the same push")


def test_copy_shape_geometry_validates_selections(test, device):
    """Verify copy_shape_geometry rejects mismatched selections and skips out-of-range rows."""
    model = _build_model("plane", device)
    sources = wp.array([0, 1], dtype=wp.int32, device=device)
    with test.assertRaises(ValueError):
        model.copy_shape_geometry(sources, wp.array([0], dtype=wp.int32, device=device))
    with test.assertRaises(ValueError):
        model.copy_shape_geometry(sources, wp.array([0, 1], dtype=wp.int64, device=device))
    before = model.shape_source_ptr.numpy().copy()
    skipped = wp.array([-1, model.shape_count], dtype=wp.int32, device=device)
    model.copy_shape_geometry(skipped, wp.array([0, 1], dtype=wp.int32, device=device))
    model.copy_shape_geometry(sources, skipped)
    np.testing.assert_array_equal(model.shape_source_ptr.numpy(), before)


class TestSteelMeshSwitch(unittest.TestCase):
    pass


add_function_test(
    TestSteelMeshSwitch,
    "test_mesh_switch_matches_fresh_scene",
    test_mesh_switch_matches_fresh_scene,
    devices=get_test_devices(),
)
add_function_test(
    TestSteelMeshSwitch,
    "test_mesh_switch_with_sdf_contacts_and_warm_start",
    test_mesh_switch_with_sdf_contacts_and_warm_start,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestSteelMeshSwitch,
    "test_mesh_switch_from_captured_graphs",
    test_mesh_switch_from_captured_graphs,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestSteelMeshSwitch,
    "test_inertia_switch_matches_fresh_scene",
    test_inertia_switch_matches_fresh_scene,
    devices=get_test_devices(),
)
add_function_test(
    TestSteelMeshSwitch,
    "test_copy_shape_geometry_validates_selections",
    test_copy_shape_geometry_validates_selections,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
