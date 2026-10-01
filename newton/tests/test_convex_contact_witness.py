# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover MPR portals whose normal projection lies outside the portal triangle."""

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _surface_error(vertices, indices, point):
    """Measure signed distance to the most violated convex face plane."""
    triangles = vertices[indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    inward = np.sum(normals * (vertices.mean(axis=0) - triangles[:, 0]), axis=1) > 0.0
    normals[inward] *= -1.0
    return float(np.max(np.sum(normals * (point - triangles[:, 0]), axis=1)))


def test_convex_contact_witness(test, device, *, swap=False):
    """Keep zero-margin witnesses on both hulls for the saved pose and nearby poses."""
    # Public two-hull fixture from https://github.com/newton-physics/newton/issues/4414.
    with np.load(Path(__file__).parent / "assets" / "convex_contact_4414.npz", allow_pickle=False) as fixture:
        vertices = [fixture[key + "_vertices"].copy() for key in ("torso", "elbow")]
        indices = [fixture[key + "_indices"].astype(np.int32) for key in ("torso", "elbow")]
        position = fixture["position"].copy()
        rotation = wp.quat(*fixture["rotation"])
    if swap:
        vertices.reverse()
        indices.reverse()

    poses = [(position, rotation)]
    for axis in range(3):
        direction = np.eye(3, dtype=np.float32)[axis]
        for sign in (-1.0, 1.0):
            poses.append((position + sign * 0.0001 * direction, rotation))
            poses.append((position, wp.quat_from_axis_angle(wp.vec3(direction), sign * 0.001) * rotation))

    # These wider perturbations require multiple retries; shrinking the seed
    # into each new tetrahedron makes the final portal numerically degenerate.
    wider_perturbations = (
        ((1.5673162869, -0.1565069156, 0.7797705622), (0.0035926571, 0.0044446284, 0.0084114846), 0.0910780397),
        ((0.1183510903, -0.5681262162, 0.9866573833), (0.0007874704, 0.0086510388, -0.0045985128), -0.0468000330),
    )
    for axis, delta_position, angle in wider_perturbations:
        direction = np.asarray(axis)
        direction /= np.linalg.norm(direction)
        delta_rotation = wp.quat_from_axis_angle(wp.vec3(direction), angle)
        poses.append((position + delta_position, delta_rotation * rotation))

    offset = wp.vec3(0.1, -0.03, 0.02)
    builder = newton.ModelBuilder()
    for side in range(2):
        body = builder.add_body()
        mesh = newton.Mesh(vertices[side], indices[side])
        builder.add_shape_convex_hull(
            body,
            mesh=mesh,
            xform=wp.transform(offset, wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(gap=0.01, margin=0.0),
        )
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    for pose_index, (shape_position, shape_rotation) in enumerate(poses):
        with test.subTest(pose=pose_index):
            body_transforms = [
                wp.transform(-offset, wp.quat_identity()),
                wp.transform(wp.vec3(shape_position) - wp.quat_rotate(shape_rotation, offset), shape_rotation),
            ]
            if swap:
                body_transforms.reverse()
            state.body_q.assign(np.asarray(body_transforms, dtype=np.float32))
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            shapes = [contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()]
            points = [contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()]
            normals = contacts.rigid_contact_normal.numpy()
            for contact in range(count):
                world_points = []
                for side in range(2):
                    shape = int(shapes[side][contact])
                    local_point = points[side][contact].astype(np.float64) - np.asarray(offset, dtype=np.float64)
                    error = _surface_error(vertices[shape].astype(np.float64), indices[shape], local_point)
                    test.assertLessEqual(abs(error), 1.0e-5, f"shape {shape}: surface error {error} m")
                    world_points.append(
                        np.asarray(wp.transform_point(body_transforms[shape], wp.vec3(points[side][contact])))
                    )
                test.assertAlmostEqual(float(np.linalg.norm(normals[contact])), 1.0, delta=1.0e-6)
                depth = -float(np.dot(world_points[1] - world_points[0], normals[contact]))
                test.assertGreater(depth, 0.0)
                if pose_index == 0:
                    # Independent FP64 convex Minkowski-difference boundary calculation.
                    test.assertAlmostEqual(depth, 0.05254986867157328, delta=1.0e-5)


class TestConvexContactWitness(unittest.TestCase):
    pass


add_function_test(
    TestConvexContactWitness, "test_convex_contact_witness", test_convex_contact_witness, devices=get_test_devices()
)


def test_convex_contact_witness_swapped(test, device):
    """Preserve surface membership when the two convex hulls are inserted in reverse order."""
    test_convex_contact_witness(test, device, swap=True)


add_function_test(
    TestConvexContactWitness,
    "test_convex_contact_witness_swapped",
    test_convex_contact_witness_swapped,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
