# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for convex-mesh manifolds clipped against a box or plane reference face."""

import subprocess
import sys
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.utils import remesh_convex_hull
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

SHAPE_GAP = 0.0015


def _rounded_hull(radii, longitudes=24, latitudes=12):
    """Return outward-wound vertices and triangles of a convex hull sampled from an ellipsoid."""
    points = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for i in range(1, latitudes):
        theta = np.pi * i / latitudes
        for j in range(longitudes):
            phi = 2.0 * np.pi * (j + 0.5 * (i % 2)) / longitudes
            points.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return _hull(np.asarray(points) * np.asarray(radii))


def _cube_hull(half_extent):
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return _hull(corners * half_extent)


def _hull(points):
    vertices, faces = remesh_convex_hull(np.asarray(points, dtype=np.float64))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1)


def _prism(sides, first_side=0):
    """Return a regular prism whose cap vertices are numbered starting at ``first_side``."""
    angles = 2.0 * np.pi * np.roll(np.arange(sides), first_side) / sides
    ring = np.stack([0.02 * np.cos(angles), 0.02 * np.sin(angles)], axis=1)
    bottom = np.hstack([ring, np.full((sides, 1), -0.01)])
    top = np.hstack([ring, np.full((sides, 1), 0.01)])
    vertices = np.vstack([bottom, top]).astype(np.float32)
    order = np.argsort(np.roll(np.arange(sides), first_side))
    faces = []
    for i in range(1, sides - 1):
        faces += [[order[0], order[i + 1], order[i]], [sides + order[0], sides + order[i], sides + order[i + 1]]]
    for i in range(sides):
        a, b = order[i], order[(i + 1) % sides]
        faces += [[a, b, sides + b], [a, sides + b, sides + a]]
    return vertices, np.asarray(faces, dtype=np.int32).reshape(-1)


def _ring_hull(sides, ring_height, jitters, phase=0.0):
    """Return a two-ring convex hull whose bottom ring heights deviate by ``jitters`` [m] from ``ring_height``."""
    angles = 2.0 * np.pi * np.arange(sides) / sides + phase
    bottom = np.stack([0.02 * np.cos(angles), 0.02 * np.sin(angles), ring_height + np.asarray(jitters)], axis=1)
    top = np.stack([0.02 * np.cos(angles), 0.02 * np.sin(angles), np.full(sides, 0.02)], axis=1)
    return _hull(np.vstack([bottom, top]))


def _ring_scene_hulls():
    """Return the alternating-height hull from review plus seeded rings with random sub-tolerance bottom heights."""
    hulls = [_ring_hull(16, 0.003, 4.0e-6 * np.where(np.arange(16) % 2, 1.0, -1.0))]
    rng = np.random.default_rng(7)
    for sides in (8, 12, 16, 20, 24, 32, 40, 48):
        for _ in range(2):
            hulls.append(_ring_hull(sides, 0.003, rng.uniform(-4.5e-6, 4.5e-6, sides), rng.uniform(0.0, 2.0 * np.pi)))
    return hulls


def _check_ring_scene(test, device):
    """Check that rings whose bottom heights straddle the contact gap within tolerance keep bounded manifolds."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=SHAPE_GAP)
    builder.add_ground_plane(cfg=cfg)
    hull_shapes = []
    for k, (vertices, indices) in enumerate(_ring_scene_hulls()):
        body = builder.add_body(xform=wp.transform((0.1 * k, 0.0, 0.0), wp.quat_identity()))
        hull_shapes.append(builder.add_shape_convex_hull(body, mesh=newton.Mesh(vertices, indices), cfg=cfg))
    model = builder.finalize(device=device)
    shape0, shape1, _, point1, normal, _, _ = _raw_contacts(model, True)
    for shape in hull_shapes:
        rows = (shape0 == shape) | (shape1 == shape)
        assert 3 <= rows.sum() <= 4, (shape, int(rows.sum()))
        np.testing.assert_allclose(normal[rows], np.tile([0.0, 0.0, 1.0], (int(rows.sum()), 1)), atol=1.0e-5)
        span = np.ptp(point1[rows, :2], axis=0)
        assert span.min() > 0.02, (shape, span)


def _quat_from_euler_deg(rx, ry, rz):
    q = wp.quat_identity()
    for axis, angle in ((wp.vec3(1.0, 0.0, 0.0), rx), (wp.vec3(0.0, 1.0, 0.0), ry), (wp.vec3(0.0, 0.0, 1.0), rz)):
        q = wp.quat_from_axis_angle(axis, float(np.radians(angle))) * q
    return q


def _rotate(q, points):
    return np.array([np.array(wp.quat_rotate(q, wp.vec3(*p))) for p in points])


def _build(device, support, vertices, indices, rotation, penetration, offset_xy=(0.0, 0.0), scale=(1.0, 1.0, 1.0)):
    builder = newton.ModelBuilder()
    support_cfg = newton.ModelBuilder.ShapeConfig(gap=SHAPE_GAP)
    if support == "plane":
        builder.add_ground_plane(cfg=support_cfg)
    elif support == "finite_plane":
        builder.add_shape_plane(body=-1, width=0.5, length=0.5, cfg=support_cfg)
    else:
        builder.add_shape_box(
            body=-1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05, cfg=support_cfg
        )
    z = -_rotate(rotation, vertices * np.asarray(scale))[:, 2].min() - penetration
    body = builder.add_body(xform=wp.transform((offset_xy[0], offset_xy[1], z), rotation))
    builder.add_shape_convex_hull(
        body, mesh=newton.Mesh(vertices, indices), scale=scale, cfg=newton.ModelBuilder.ShapeConfig(gap=SHAPE_GAP)
    )
    return builder.finalize(device=device)


def _world_contacts(model, convex_face_clip):
    """Return world contact points on the hull, normals and signed distances, sorted by position."""
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True, convex_face_clip=convex_face_clip)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    point0 = contacts.rigid_contact_point0.numpy()[:count]
    point1 = contacts.rigid_contact_point1.numpy()[:count]
    normal = contacts.rigid_contact_normal.numpy()[:count]
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()

    def to_world(shape, p):
        body = shape_body[shape]
        if body < 0:
            return np.asarray(p, dtype=np.float64)
        q = wp.quat(*body_q[body][3:])
        return body_q[body][:3] + np.array(wp.quat_rotate(q, wp.vec3(*p)))

    world0 = np.array([to_world(shape0[i], point0[i]) for i in range(count)]).reshape(-1, 3)
    world1 = np.array([to_world(shape1[i], point1[i]) for i in range(count)]).reshape(-1, 3)
    distance = np.einsum("ij,ij->i", world1 - world0, normal) if count else np.zeros(0)
    order = np.lexsort((world1[:, 1], world1[:, 0])) if count else np.zeros(0, dtype=int)
    return world1[order], normal[order], distance[order]


def _raw_contacts(model, convex_face_clip):
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True, convex_face_clip=convex_face_clip)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return _contact_arrays(contacts)


def _contact_arrays(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    return tuple(
        getattr(contacts, name).numpy()[:count].copy()
        for name in (
            "rigid_contact_shape0",
            "rigid_contact_shape1",
            "rigid_contact_point0",
            "rigid_contact_point1",
            "rigid_contact_normal",
            "rigid_contact_margin0",
            "rigid_contact_margin1",
        )
    )


def test_rounded_hull_rests_on_face(test, device):
    vertices, indices = _rounded_hull((0.012, 0.009, 0.008))
    for support in ("plane", "box", "finite_plane"):
        for euler in ((0.0, 0.0, 0.0), (7.0, 3.0, 0.0), (20.0, 0.0, 10.0), (90.0, 0.0, 30.0)):
            rotation = _quat_from_euler_deg(*euler)
            model = _build(device, support, vertices, indices, rotation, penetration=2.0e-4)
            legacy_points, _, _ = _world_contacts(model, False)
            points, normals, distances = _world_contacts(model, True)
            label = f"{support} {euler}"
            test.assertLess(len(legacy_points), 3, f"{label}: legacy manifold already spans a face")
            test.assertGreaterEqual(len(points), 3, f"{label}: face clip kept {len(points)} contacts")
            np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (len(points), 1)), atol=1.0e-6)
            test.assertLessEqual(distances.max(), 2.0 * SHAPE_GAP + 1.0e-6, label)
            test.assertAlmostEqual(distances.min(), -2.0e-4, delta=2.0e-5, msg=label)
            hull_vertices = _rotate(rotation, vertices)
            hull_vertices[:, 2] += -hull_vertices[:, 2].min() - 2.0e-4
            for p in points:
                gap = np.min(np.linalg.norm(hull_vertices - p, axis=1))
                test.assertLess(gap, 1.0e-5, f"{label}: contact {p} is not a hull vertex")
            edge_1, edge_2 = points[1, :2] - points[0, :2], points[2, :2] - points[0, :2]
            span = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
            test.assertGreater(abs(float(span)), 1.0e-7, f"{label}: contacts are collinear")


def test_hull_clipped_to_box_face(test, device):
    """Check that a hull overhanging the box edge keeps only contacts on the box's top face."""
    vertices, indices = _cube_hull(0.02)
    rotation = wp.quat_identity()
    model = _build(device, "box", vertices, indices, rotation, penetration=1.0e-4, offset_xy=(0.1, 0.0))
    points, normals, distances = _world_contacts(model, True)
    test.assertEqual(len(points), 4)
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (4, 1)), atol=1.0e-6)
    np.testing.assert_allclose(sorted(set(np.round(points[:, 0], 6))), [0.08, 0.1], atol=1.0e-6)
    np.testing.assert_allclose(sorted(set(np.round(points[:, 1], 6))), [-0.02, 0.02], atol=1.0e-6)
    np.testing.assert_allclose(distances, -1.0e-4, atol=1.0e-6)


def test_cube_hull_matches_legacy_corners(test, device):
    """Check that face clipping keeps the corners perturbed support mapping already finds."""
    vertices, indices = _cube_hull(0.02)
    for support in ("plane", "box"):
        for yaw in (0.0, 30.0):
            model = _build(device, support, vertices, indices, _quat_from_euler_deg(0.0, 0.0, yaw), penetration=1.0e-4)
            legacy_points, legacy_normals, legacy_distances = _world_contacts(model, False)
            points, normals, distances = _world_contacts(model, True)
            test.assertEqual(len(points), 4)
            test.assertEqual(len(legacy_points), 4)
            np.testing.assert_allclose(points, legacy_points, atol=1.0e-6)
            np.testing.assert_allclose(normals, legacy_normals, atol=1.0e-4)
            np.testing.assert_allclose(distances, legacy_distances, atol=1.0e-6)


def test_cube_hull_on_edge(test, device):
    """Check that a cube hull balanced on an edge touches along that edge only."""
    vertices, indices = _cube_hull(0.02)
    model = _build(device, "plane", vertices, indices, _quat_from_euler_deg(45.0, 0.0, 0.0), penetration=1.0e-4)
    points, _, distances = _world_contacts(model, True)
    touching = points[distances < 0.0]
    test.assertEqual(len(touching), 2)
    np.testing.assert_allclose(np.sort(touching[:, 0]), [-0.02, 0.02], atol=1.0e-6)
    np.testing.assert_allclose(touching[:, 1], 0.0, atol=1.0e-6)


def test_many_sided_face_spans_footprint(test, device):
    """Check that a face with more vertices than the patch capacity keeps contacts around its footprint."""
    for sides in (16, 32, 64):
        contact_sets = []
        for first_side in (0, sides // 4, sides // 2):
            vertices, indices = _prism(sides, first_side)
            model = _build(device, "plane", vertices, indices, wp.quat_identity(), penetration=1.0e-4)
            points, _, _ = _world_contacts(model, True)
            label = f"{sides}-gon from side {first_side}"
            test.assertGreaterEqual(len(points), 3, label)
            centroid = points[:, :2].mean(axis=0)
            test.assertLess(np.linalg.norm(centroid), 0.006, f"{label}: contacts bunch at {centroid}")
            test.assertGreater(np.ptp(points[:, 0]), 0.03, label)
            test.assertGreater(np.ptp(points[:, 1]), 0.03, label)
            contact_sets.append(points)
        for points in contact_sets[1:]:
            np.testing.assert_allclose(points, contact_sets[0], atol=1.0e-6)


def test_non_planar_face_stays_within_capacity(test, device):
    """Check that faces with heights inside the coplanarity tolerance keep 3 to 4 spanning contacts."""
    _check_ring_scene(test, device)


def test_non_planar_face_clip_is_bounds_checked(test, device):
    """Check that the non-planar ring scene runs on CPU with Warp's array bounds checks enabled."""
    script = (
        "import warp as wp\n"
        "wp.config.mode = 'debug'\n"
        "from newton.tests import test_convex_face_clip as t\n"
        "t._check_ring_scene(None, 'cpu')\n"
        "print('RING_SCENE_OK', flush=True)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=1200, check=False)
    output = result.stdout + result.stderr
    test.assertEqual(result.returncode, 0, output[-4000:])
    test.assertIn("RING_SCENE_OK", result.stdout)
    test.assertNotIn("Assertion failed", output)


def test_face_clip_leaves_other_pairs_unchanged(test, device):
    """Check that pairs other than box/plane versus convex mesh keep their contacts under the flag."""
    hull_vertices, hull_indices = _rounded_hull((0.012, 0.009, 0.008), 12, 7)
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = SHAPE_GAP
    builder.add_ground_plane()
    tilt = _quat_from_euler_deg(3.0, 2.0, 0.0)
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0199), tilt))
    builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
    top = builder.add_body(xform=wp.transform((0.003, 0.0, 0.0598), _quat_from_euler_deg(0.0, 0.0, 20.0)))
    builder.add_shape_box(top, hx=0.02, hy=0.02, hz=0.02)
    hull_a = builder.add_body(xform=wp.transform((0.5, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_convex_hull(hull_a, mesh=newton.Mesh(hull_vertices, hull_indices))
    hull_b = builder.add_body(xform=wp.transform((0.5, 0.0, 0.1159), _quat_from_euler_deg(10.0, 0.0, 0.0)))
    builder.add_shape_convex_hull(hull_b, mesh=newton.Mesh(hull_vertices, hull_indices))
    model = builder.finalize(device=device)
    legacy = _raw_contacts(model, False)
    clipped = _raw_contacts(model, True)
    pairs = set(zip(legacy[0].tolist(), legacy[1].tolist(), strict=True))
    test.assertTrue(
        {(0, 1), (1, 2), (3, 4)} <= pairs, f"expected box-plane, box-box and hull-hull contacts, got {pairs}"
    )
    # The flag compiles a separate kernel, so the GPU code generator may round the shared
    # manifold arithmetic differently; geometry is compared to well below any clipping change.
    for index, (before, after) in enumerate(zip(legacy, clipped, strict=True)):
        if index in (2, 3, 4):  # point0, point1, normal
            np.testing.assert_allclose(before, after, rtol=0.0, atol=1.0e-6)
        else:
            np.testing.assert_array_equal(before, after)


def test_face_clip_deterministic(test, device):
    """Check that repeated collides into one persistent buffer match a fresh pipeline bitwise."""
    vertices, indices = _rounded_hull((0.012, 0.009, 0.008))
    model = _build(device, "box", vertices, indices, _quat_from_euler_deg(20.0, 0.0, 10.0), penetration=2.0e-4)
    fresh = _raw_contacts(model, True)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True, convex_face_clip=True)
    contacts = pipeline.contacts()
    for _ in range(3):
        pipeline.collide(state, contacts)
        for before, after in zip(fresh, _contact_arrays(contacts), strict=True):
            np.testing.assert_array_equal(before, after)


def test_runtime_scale_matches_baked_vertices(test, device):
    """Check that an anisotropic shape scale gives the contacts of the equivalently scaled vertices."""
    vertices, indices = _rounded_hull((1.0, 1.0, 1.0))
    radii = np.array([0.012, 0.009, 0.008], dtype=np.float32)
    rotation = _quat_from_euler_deg(20.0, 0.0, 10.0)
    baked = _world_contacts(_build(device, "box", vertices * radii, indices, rotation, penetration=2.0e-4), True)
    scaled = _world_contacts(
        _build(device, "box", vertices, indices, rotation, penetration=2.0e-4, scale=tuple(radii)), True
    )
    test.assertGreaterEqual(len(scaled[0]), 3)
    for a, b in zip(baked, scaled, strict=True):
        np.testing.assert_allclose(a, b, atol=1.0e-6)


def test_rounded_hull_settles(test, device):
    """Check that a small rounded hull settles with face clipping and keeps rolling without it."""
    vertices, indices = _rounded_hull((0.012, 0.009, 0.008))
    tail_speed = {}
    for convex_face_clip in (False, True):
        builder = newton.ModelBuilder()
        builder.add_shape_box(
            body=-1,
            xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()),
            hx=0.3,
            hy=0.3,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.001),
        )
        rotation = _quat_from_euler_deg(10.0, 5.0, 0.0)
        z = -_rotate(rotation, vertices)[:, 2].min() + 0.005
        body = builder.add_body(xform=wp.transform((0.0, 0.0, z), rotation))
        builder.add_shape_convex_hull(
            body,
            mesh=newton.Mesh(vertices, indices),
            cfg=newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=0.002),
        )
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            convex_face_clip=convex_face_clip,
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(model, iterations=32, contact_regularization=0.01)
        state_0, state_1, control = model.state(), model.state(), model.control()
        speeds = []
        for _ in range(480):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            speeds.append(float(np.linalg.norm(state_0.body_qd.numpy()[0, 3:])))
        tail_speed[convex_face_clip] = float(np.median(speeds[-120:]))
    test.assertLess(tail_speed[True], 0.15, f"face-clipped hull still rotates at {tail_speed[True]:.3f} rad/s")
    test.assertGreater(tail_speed[False], 1.0, f"legacy control settled ({tail_speed[False]:.3f} rad/s)")


class TestConvexFaceClip(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()
add_function_test(
    TestConvexFaceClip, "test_rounded_hull_rests_on_face", test_rounded_hull_rests_on_face, devices=devices
)
add_function_test(TestConvexFaceClip, "test_hull_clipped_to_box_face", test_hull_clipped_to_box_face, devices=devices)
add_function_test(
    TestConvexFaceClip, "test_cube_hull_matches_legacy_corners", test_cube_hull_matches_legacy_corners, devices=devices
)
add_function_test(
    TestConvexFaceClip, "test_many_sided_face_spans_footprint", test_many_sided_face_spans_footprint, devices=devices
)
add_function_test(TestConvexFaceClip, "test_cube_hull_on_edge", test_cube_hull_on_edge, devices=devices)
add_function_test(
    TestConvexFaceClip,
    "test_face_clip_leaves_other_pairs_unchanged",
    test_face_clip_leaves_other_pairs_unchanged,
    devices=devices,
)
add_function_test(
    TestConvexFaceClip,
    "test_runtime_scale_matches_baked_vertices",
    test_runtime_scale_matches_baked_vertices,
    devices=devices,
)
add_function_test(TestConvexFaceClip, "test_face_clip_deterministic", test_face_clip_deterministic, devices=devices)
add_function_test(
    TestConvexFaceClip,
    "test_non_planar_face_stays_within_capacity",
    test_non_planar_face_stays_within_capacity,
    devices=devices,
)
add_function_test(
    TestConvexFaceClip,
    "test_non_planar_face_clip_is_bounds_checked",
    test_non_planar_face_clip_is_bounds_checked,
    devices=["cpu"],
)
add_function_test(TestConvexFaceClip, "test_rounded_hull_settles", test_rounded_hull_settles, devices=cuda_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
