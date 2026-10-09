# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelFlags
from newton.solvers import SolverFeatherPGS

WORLD_COUNT = 3
GEOMETRY_FIELDS = (
    "shape_body",
    "shape_transform",
    "shape_scale",
    "shape_type",
    "shape_source_ptr",
    "shape_margin",
    "shape_is_solid",
)


class TestFeatherPGSNotifyDevice(unittest.TestCase):
    def test_armature_notify_matches_host_reference_and_fresh_solver(self):
        """Refresh the effective armature and its size groups exactly as a host rebuild would."""
        device = wp.get_device()
        model = _build_model(device)
        options = {"pgs_mode": "matrix_free", "enable_joint_friction": True} if device.is_cuda else {}
        solver = SolverFeatherPGS(model, **options)
        rng = np.random.default_rng(0)
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        _assert_armature_matches(self, solver, SolverFeatherPGS(model, **options))

    def test_kinematic_notify_matches_host_reference_and_fresh_solver(self):
        """Give newly kinematic joints the large armature and drop it when they become dynamic again."""
        model = _build_model(wp.get_device())
        solver = SolverFeatherPGS(model)
        model.joint_armature.assign(np.full(model.joint_dof_count, 0.25, dtype=np.float32))
        chain_link = int(model.joint_child.numpy()[1])
        for flag in (BodyFlags.KINEMATIC, BodyFlags.DYNAMIC):
            body_flags = model.body_flags.numpy()
            body_flags[chain_link] = int(flag)
            model.body_flags.assign(body_flags)
            solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            _assert_armature_matches(self, solver, SolverFeatherPGS(model))

    def test_shape_notify_matches_host_reference(self):
        """Refresh body radii and retire exactly the anchors whose geometry changed."""
        model = _build_model(wp.get_device())
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        rng = np.random.default_rng(1)
        seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
        for edit in ("shape_transform", "shape_scale", "shape_margin", "shape_body", "shape_collision_radius", None):
            _randomize_history(patches, model, rng)
            valid_before = patches.previous.valid.numpy()
            shape = int(rng.integers(model.shape_count))
            if edit == "shape_body":
                values = model.shape_body.numpy()
                values[shape] = (values[shape] + 1) % model.body_count
                model.shape_body.assign(values)
            elif edit is not None:
                values = getattr(model, edit).numpy()
                values[shape] = values[shape] * 1.5 + 0.01
                getattr(model, edit).assign(values)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            expected_radius, expected_valid, seen = _host_update_geometry(model, patches, seen, valid_before)
            np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
            np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_armature_and_shape_notify_capture_into_a_graph(self):
        """Replay a captured notify and pick up armature and geometry written after the capture."""
        device = wp.get_device("cuda:0")
        model = _build_model(device)
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", enable_joint_friction=True, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        flags = (
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(flags)
        seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
        rng = np.random.default_rng(2)
        _randomize_history(patches, model, rng)
        valid_before = patches.previous.valid.numpy()
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2.0)
        scale = model.shape_scale.numpy()
        scale[2] *= 1.25
        model.shape_scale.assign(scale)
        wp.capture_launch(capture.graph)
        _assert_armature_matches(
            self, solver, SolverFeatherPGS(model, pgs_mode="matrix_free", enable_joint_friction=True)
        )
        expected_radius, expected_valid, _ = _host_update_geometry(model, patches, seen, valid_before)
        np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
        np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
        self.assertLess(np.count_nonzero(expected_valid), np.count_nonzero(valid_before))


def _build_model(device):
    """Worlds with a free sphere, a 3-link and a 2-link revolute chain: three response sizes."""
    builder = newton.ModelBuilder()
    for world in range(WORLD_COUNT):
        builder.begin_world()
        sphere = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(sphere, radius=0.1)
        for link_count, x in ((3, 0.0), (2, -1.0)):
            parent, joints = -1, []
            for link_index in range(link_count):
                link = builder.add_link()
                builder.add_shape_capsule(
                    link, radius=0.05, half_height=0.1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
                )
                joint_z = 1.0 if parent < 0 else 0.2
                joints.append(
                    builder.add_joint_revolute(
                        parent,
                        link,
                        parent_xform=wp.transform(wp.vec3(x if parent < 0 else 0.0, 0.0, joint_z), wp.quat_identity()),
                        axis=newton.Axis.Y,
                        armature=0.01 * (world + link_index + 1),
                    )
                )
                parent = link
            builder.add_articulation(joints)
        builder.end_world()
    return builder.finalize(device=device)


def _assert_armature_matches(test, solver, fresh):
    """Compare the effective armature and grouped copies with the host rebuild and a fresh solver."""
    model = solver.model
    armature = model.joint_armature.numpy().copy()
    kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
    joint_qd_start = model.joint_qd_start.numpy()
    for joint, child in enumerate(model.joint_child.numpy()):
        if kinematic[child]:
            armature[joint_qd_start[joint] : joint_qd_start[joint + 1]] = 1.0e10
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), armature)
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), fresh._joint_armature_device.numpy())
    plan = solver._model_plan
    test.assertEqual(len(solver.size_groups), 3)
    for size in solver.size_groups:
        expected = np.zeros((solver.n_arts_by_size[size], size), dtype=np.float32)
        for group, art in enumerate(np.flatnonzero(plan.response_dof_count == size)):
            start, count = plan.articulation_dof_start[art], plan.articulation_dof_count[art]
            expected[group, :count] = armature[start : start + count]
        np.testing.assert_array_equal(solver.R_by_size[size].numpy(), expected)
        np.testing.assert_array_equal(solver.R_by_size[size].numpy(), fresh.R_by_size[size].numpy())


def _randomize_history(patches, model, rng):
    """Fill the previous patch frame with valid anchors on random shapes and bodies, including -1 sentinels."""
    n = patches.previous.valid.shape[0]
    patches.previous.valid.fill_(1)
    for field, count in (("shape", model.shape_count), ("body", model.body_count)):
        for side in ("a", "b"):
            getattr(patches.previous, f"{field}_{side}").assign(rng.integers(-1, count, n).astype(np.int32))


def _host_update_geometry(model, patches, seen, valid):
    """Host reference for the friction-anchor geometry refresh: body radii and retired history."""
    geometry = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
    radii = np.zeros(model.body_count, dtype=np.float32)
    for body, radius, transform in zip(
        geometry["shape_body"], model.shape_collision_radius.numpy(), geometry["shape_transform"], strict=True
    ):
        if body >= 0:
            radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
    changed = np.zeros(model.shape_count, dtype=bool)
    for name, values in geometry.items():
        difference = values != seen[name]
        if difference.ndim > 1:
            difference = np.any(difference, axis=tuple(range(1, difference.ndim)))
        changed |= difference
    bodies = np.zeros(model.body_count, dtype=bool)
    for shape_body in (seen["shape_body"], geometry["shape_body"]):
        affected = shape_body[changed]
        bodies[affected[affected >= 0]] = True
    valid = valid.copy()
    previous = patches.previous
    for ids, mask in (
        (previous.shape_a.numpy(), changed),
        (previous.shape_b.numpy(), changed),
        (previous.body_a.numpy(), bodies),
        (previous.body_b.numpy(), bodies),
    ):
        hit = ids >= 0
        valid[hit & mask[np.maximum(ids, 0)]] = 0
    return radii, valid, geometry


if __name__ == "__main__":
    unittest.main(verbosity=2)
