# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact-witness vertex patches between convex meshes whose contact features are not flat."""

import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.multicontact import create_build_manifold
from newton._src.geometry.support_function import GenericShapeData, pack_mesh_ptr
from newton._src.geometry.utils import remesh_convex_hull
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

SHAPE_GAP = 1.0e-3
# Witness distance [m] to its own hull surface; the unverified-hull defect this guards against is 5e-5.
SURFACE_TOLERANCE = 5.0e-6


def _hull(points):
    vertices, faces = remesh_convex_hull(np.asarray(points, dtype=np.float64))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32).reshape(-1)


def _disc(radius=0.036, height=0.002, segments=32, cap=0.0, jitter=0.0, seed=0):
    """Return a disc hull whose bottom is a spherical cap ``cap`` [m] deep with random vertical ``jitter`` [m]."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    bottom = [np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros(segments)], axis=1)]
    for k in range(1, 5):
        r = radius * (1.0 - k / 5.0)
        z = -cap * (1.0 - (r / radius) ** 2)
        bottom.append(np.stack([r * np.cos(angles + 0.1 * k), r * np.sin(angles + 0.1 * k), np.full(segments, z)], 1))
    bottom.append([[0.0, 0.0, -cap]])
    bottom = np.vstack(bottom)
    bottom[:, 2] += rng.uniform(-1.0, 1.0, len(bottom)) * jitter
    top = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(segments, height)], axis=1)
    return _hull(np.vstack([bottom, top]))


def _box(hx, hy, hz, bump=0.0):
    corners = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    corners[0, 2] += bump
    return _hull(corners)


def _ring(radius, z, sections):
    angles = 2.0 * np.pi * np.arange(sections) / sections
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(sections, z)], axis=1)


def _sphere(radius, longitudes=24, latitudes=12):
    points = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
    for i in range(1, latitudes):
        theta = np.pi * i / latitudes
        for j in range(longitudes):
            phi = 2.0 * np.pi * (j + 0.5 * (i % 2)) / longitudes
            points.append([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return _hull(radius * np.asarray(points))


def _quat_from_euler_deg(rx, ry, rz):
    q = wp.quat_identity()
    for axis, angle in ((wp.vec3(1.0, 0.0, 0.0), rx), (wp.vec3(0.0, 1.0, 0.0), ry), (wp.vec3(0.0, 0.0, 1.0), rz)):
        q = wp.quat_from_axis_angle(axis, float(np.radians(angle))) * q
    return q


def _rotate(q, points):
    return np.asarray(points, dtype=np.float64) @ np.array(wp.quat_to_matrix(q), dtype=np.float64).reshape(3, 3).T


def _witness_cases():
    """Return (label, body hull, body rotation, penetration [m], support hull) for the witness matrix."""
    slab = _box(0.3, 0.3, 0.005)
    rng = np.random.default_rng(1)
    relief = np.vstack(
        [
            np.array([[sx * 0.25, sy * 0.2, sz * 0.05] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]),
            rng.uniform([-0.25, -0.2, -0.0505], [0.25, 0.2, -0.05], (40, 3)),
        ]
    )
    identity = wp.quat_identity()
    cases = [
        (f"cap_disc_tilt{tilt}", _disc(cap=1.0e-4, jitter=3.0e-5), _quat_from_euler_deg(tilt, 0, 0), 2.0e-4, slab)
        for tilt in (0.0, 0.3, 2.0, 10.0)
    ]
    cases += [
        ("flat_disc_jitter30", _disc(jitter=3.0e-5), identity, 2.0e-4, slab),
        ("thin_disc", _disc(height=3.0e-4, jitter=2.0e-5), identity, 1.0e-4, slab),
        (
            "thin_plate_on_thin_plate",
            _box(0.05, 0.04, 0.0004, bump=3.0e-5),
            _quat_from_euler_deg(0, 0, 17),
            1.0e-4,
            _box(0.06, 0.06, 0.0004),
        ),
    ]
    cases += [
        (f"sole_bump{bump}", _box(0.0205, 0.027, 0.0065, bump=bump), identity, 2.0e-4, slab)
        for bump in (2.0e-5, 1.0e-4, 2.5e-4)
    ]
    cases += [
        ("small_cube_bump", _box(0.005, 0.005, 0.005, bump=2.0e-5), _quat_from_euler_deg(0, 0, 30), 5.0e-5, slab),
        ("small_cube_tilt", _box(0.005, 0.005, 0.005), _quat_from_euler_deg(1.0, 0.5, 0), 5.0e-5, slab),
        ("large_box_relief", _hull(relief), identity, 3.0e-4, _box(0.6, 0.6, 0.01)),
        ("sphere", _sphere(0.03), identity, 3.0e-4, slab),
        (
            "cylinder_on_side",
            _hull(np.vstack([_ring(0.02, -0.05, 24), _ring(0.02, 0.05, 24)])),
            _quat_from_euler_deg(90, 0, 0),
            2.0e-4,
            slab,
        ),
        ("box_edge_on", _box(0.03, 0.03, 0.03), _quat_from_euler_deg(45, 0, 0), 2.0e-4, slab),
        ("box_corner_on", _box(0.03, 0.03, 0.03), _quat_from_euler_deg(45, 35.264, 0), 2.0e-4, slab),
        (
            "cone_tip",
            _hull(np.vstack([_ring(0.03, 0.0, 24), [[0.0, 0.0, 0.06]]])),
            _quat_from_euler_deg(180, 0, 0),
            2.0e-4,
            slab,
        ),
        (
            "disc_on_disc_offset",
            _disc(jitter=3.0e-5, seed=3),
            _quat_from_euler_deg(0, 0, 11),
            1.0e-4,
            _disc(radius=0.03, jitter=3.0e-5, seed=4),
        ),
    ]
    return cases


# Pairs whose hulls touch at a vertex, an edge or an exact face in every pose, so no patch is expected.
_POINT_OR_LINE_CASES = {"cap_disc_tilt10.0", "small_cube_tilt", "sphere", "cylinder_on_side", "box_edge_on"}
_POINT_OR_LINE_CASES |= {"box_corner_on", "cone_tip"}

_GLOBAL_ROTATIONS = (
    wp.quat_identity(),
    *(wp.normalize(wp.quat(*np.random.default_rng(seed).normal(size=4))) for seed in (11, 12)),
)


def _build_pair(device, body_hull, body_rotation, penetration, support_hull, global_rotation, swap, scale):
    """Place a body hull ``penetration`` [m] into a static support hull whose top is at z = 0 before rotation."""
    builder = newton.ModelBuilder()
    _add_pair(builder, body_hull, body_rotation, penetration, support_hull, global_rotation, swap, scale)
    return builder.finalize(device=device)


def _add_pair(builder, body_hull, body_rotation, penetration, support_hull, global_rotation, swap, scale):
    """Add one witness-matrix pair and return its two shape indices."""
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=SHAPE_GAP)
    support_vertices, support_indices = support_hull
    body_vertices, body_indices = body_hull
    scale = np.asarray(scale, dtype=np.float32)
    support_xform = wp.transform(
        wp.quat_rotate(global_rotation, wp.vec3(0.0, 0.0, -float(support_vertices[:, 2].max()))),
        global_rotation,
    )
    lowest = _rotate(body_rotation, body_vertices)[:, 2].min()
    body_xform = wp.transform(
        wp.quat_rotate(global_rotation, wp.vec3(0.002, -0.001, float(-lowest - penetration))),
        global_rotation * body_rotation,
    )
    body = builder.add_body(xform=body_xform)
    body_mesh = newton.Mesh(body_vertices / scale, body_indices, compute_inertia=False)
    support_mesh = newton.Mesh(support_vertices, support_indices, compute_inertia=False)
    if swap:
        body_shape = builder.add_shape_convex_hull(body, mesh=body_mesh, scale=tuple(scale.tolist()), cfg=cfg)
        support_shape = builder.add_shape_convex_hull(-1, xform=support_xform, mesh=support_mesh, cfg=cfg)
    else:
        support_shape = builder.add_shape_convex_hull(-1, xform=support_xform, mesh=support_mesh, cfg=cfg)
        body_shape = builder.add_shape_convex_hull(body, mesh=body_mesh, scale=tuple(scale.tolist()), cfg=cfg)
    return support_shape, body_shape


def _collide(model, convex_mesh_patch, convex_face_clip=False, split=None):
    state = model.state()
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        deterministic=True,
        convex_face_clip=convex_face_clip,
        convex_mesh_patch=convex_mesh_patch,
    )
    if split is not None:
        assert pipeline.narrow_phase.split_gjk_mpr == split
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return state, contacts


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
        )
    )


def _shape_world(model, state, shape):
    """Return the world rotation and translation of a shape."""
    xform = model.shape_transform.numpy()[shape]
    q = wp.quat(*xform[3:])
    t = np.asarray(xform[:3], dtype=np.float64)
    body = model.shape_body.numpy()[shape]
    if body >= 0:
        body_q = state.body_q.numpy()[body]
        qb = wp.quat(*body_q[3:])
        t = body_q[:3] + np.array(wp.quat_rotate(qb, wp.vec3(*t)))
        q = qb * q
    return q, t


def _surface_error(vertices, indices, point):
    """Return the distance of a point to a convex hull's surface measured by its most violated face plane."""
    triangles = vertices[indices.reshape(-1, 3)].astype(np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    inward = np.sum(normals * (vertices.mean(axis=0) - triangles[:, 0]), axis=1) > 0.0
    normals[inward] *= -1.0
    return abs(float(np.max(np.sum(normals * (point - triangles[:, 0]), axis=1))))


def _witness_report(model, state, contacts, shapes=None):
    """Return per-contact witness surface errors [m], signed distances [m] and world witnesses on shape 1.

    ``shapes`` restricts the report to the contacts between that pair of shape indices.
    """
    shape0, shape1, point0, point1, normal = _contact_arrays(contacts)
    if shapes is not None:
        keep = (np.minimum(shape0, shape1) == min(shapes)) & (np.maximum(shape0, shape1) == max(shapes))
        shape0, shape1, point0, point1, normal = shape0[keep], shape1[keep], point0[keep], point1[keep], normal[keep]
    hulls = {}
    for shape in set(shape0.tolist()) | set(shape1.tolist()):
        mesh = model.shape_source[shape]
        q, t = _shape_world(model, state, shape)
        scaled = np.asarray(mesh.vertices, dtype=np.float64) * model.shape_scale.numpy()[shape]
        hulls[shape] = (_rotate(q, scaled) + t, np.asarray(mesh.indices, dtype=np.int32))
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()

    def to_world(shape, p):
        body = shape_body[shape]
        if body < 0:
            return np.asarray(p, dtype=np.float64)
        return body_q[body][:3] + np.array(wp.quat_rotate(wp.quat(*body_q[body][3:]), wp.vec3(*p)))

    errors, distances, witnesses = [], [], []
    for i in range(len(shape0)):
        w0, w1 = to_world(shape0[i], point0[i]), to_world(shape1[i], point1[i])
        errors.append(max(_surface_error(*hulls[shape0[i]], w0), _surface_error(*hulls[shape1[i]], w1)))
        distances.append(float(np.dot(w1 - w0, normal[i])))
        witnesses.append(w1)
    return np.asarray(errors), np.asarray(distances), np.asarray(witnesses).reshape(-1, 3)


def _check_witness_matrix(test, device, swap, scale, split=None):
    # Each pair is its own world of one model, so one pipeline collides them all.
    builder = newton.ModelBuilder()
    pairs = []
    for label, body_hull, body_rotation, penetration, support_hull in _witness_cases():
        for g, global_rotation in enumerate(_GLOBAL_ROTATIONS):
            builder.begin_world()
            shapes = _add_pair(
                builder, body_hull, body_rotation, penetration, support_hull, global_rotation, swap, scale
            )
            builder.end_world()
            pairs.append((f"{label} g{g} swap={swap} scale={scale}", label, penetration, shapes))
    model = builder.finalize(device=device)
    state, contacts = _collide(model, True, split=split)
    for name, label, penetration, shapes in pairs:
        errors, distances, _ = _witness_report(model, state, contacts, shapes)
        test.assertGreater(len(errors), 0, name)
        test.assertLessEqual(len(errors), 8, name)
        test.assertLess(errors.max(), SURFACE_TOLERANCE, f"{name}: witness {errors.max():.3e} m off its hull")
        # No contact is deeper than the prescribed placement.
        test.assertGreater(distances.min(), -penetration - 1.0e-6, name)
        if label not in _POINT_OR_LINE_CASES:
            test.assertGreaterEqual(len(errors), 3, f"{name}: patch kept {len(errors)} contacts")


def test_witnesses_on_hull_surfaces(test, device):
    """Check witness surface accuracy and depth on 19 hull pairs in 3 global poses."""
    _check_witness_matrix(test, device, swap=False, scale=(1.0, 1.0, 1.0))


def test_witnesses_on_hull_surfaces_swapped_scaled(test, device):
    """Check the witness matrix with exchanged shape roles and a non-uniform body scale."""
    _check_witness_matrix(test, device, swap=True, scale=(1.0, 1.0, 1.0))
    _check_witness_matrix(test, device, swap=False, scale=(1.3, 0.8, 0.6))
    _check_witness_matrix(test, device, swap=True, scale=(1.3, 0.8, 0.6))


def test_witnesses_on_hull_surfaces_split(test, device):
    """Check the witness matrix through the split GJK/MPR kernels that large CUDA scenes use."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with (
        patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0),
        patch("newton._src.sim.collide._SPLIT_GJK_MPR_FULL_PAIR_COUNT_THRESHOLD", 0),
    ):
        _check_witness_matrix(test, device, swap=False, scale=(1.0, 1.0, 1.0), split=True)
        _check_witness_matrix(test, device, swap=True, scale=(1.3, 0.8, 0.6), split=True)


def test_patch_contact_capacity(test, device):
    """Check that the automatic capacity counts 8 contacts per convex-mesh pair with the flag."""

    def model():
        builder = newton.ModelBuilder()
        vertices, indices = _disc(jitter=3.0e-5)
        for _ in range(250):
            builder.begin_world()
            builder.add_shape_convex_hull(-1, mesh=newton.Mesh(vertices, indices))
            body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0025), wp.quat_identity()))
            builder.add_shape_convex_hull(body, mesh=newton.Mesh(vertices, indices))
            builder.end_world()
        return builder.finalize(device=device)

    off_model, on_model = model(), model()
    test.assertEqual(int(off_model.shape_contact_pair_count), 250)
    off = newton.CollisionPipeline(off_model, broad_phase="nxn")
    on = newton.CollisionPipeline(on_model, broad_phase="nxn", convex_mesh_patch=True)
    test.assertEqual(off.rigid_contact_max, 250 * 5)
    test.assertEqual(on.rigid_contact_max, 250 * 8)


def test_patch_spans_disc_on_disc(test, device):
    """Check that two rough discs, neither with a flat face, get a spanning patch instead of one contact."""
    lower = _disc(radius=0.03, jitter=3.0e-5, seed=4)
    upper = _disc(jitter=3.0e-5, seed=3)
    rotation = _quat_from_euler_deg(0.0, 0.0, 11.0)
    model = _build_pair(device, upper, rotation, 1.0e-4, lower, wp.quat_identity(), False, (1.0, 1.0, 1.0))
    state, contacts = _collide(model, False)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    state, contacts = _collide(model, True)
    _, _, witnesses = _witness_report(model, state, contacts)
    test.assertGreaterEqual(len(witnesses), 3)
    span = np.ptp(witnesses[:, :2], axis=0)
    test.assertGreater(span.min(), 0.02, f"patch spans only {span}")


def test_unreferenced_vertices_are_not_witnesses(test, device):
    """Check that hull vertices no triangle references, some just inside the hull, never become witnesses."""
    points = _puck_points().astype(np.float32)
    # The hull's triangles over every input point: about half of the bottom points are left inside it.
    hull_vertices, hull_indices = _hull(points)
    rows = {tuple(p): i for i, p in enumerate(points.tolist())}
    indices = np.array([rows[tuple(p)] for p in hull_vertices[hull_indices].tolist()], np.int32)
    test.assertLess(len(np.unique(indices)), len(points))
    slab = _box(0.2, 0.15, 0.0017)
    model = _build_pair(
        device, (points, indices), wp.quat_identity(), 2.0e-4, slab, wp.quat_identity(), False, (1.0, 1.0, 1.0)
    )
    state, contacts = _collide(model, True)
    errors, _, _ = _witness_report(model, state, contacts)
    test.assertGreaterEqual(len(errors), 3)
    test.assertLess(errors.max(), SURFACE_TOLERANCE, f"witness {errors.max():.3e} m off its hull")


def _unverified_slab_model(device):
    """Return a disc resting on a slab whose top triangles meet 50 um below the slab's vertex hull."""
    p = [[-0.1, -0.1, -0.01], [0.1, -0.1, -0.01], [0.1, 0.1, -0.01], [-0.1, 0.1, -0.01]]
    p += [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0], [0.0, 0.0, -5.0e-5]]
    faces = [0, 2, 1, 0, 3, 2, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    faces += [4, 5, 8, 5, 6, 8, 6, 7, 8, 7, 4, 8]
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=SHAPE_GAP)
    builder.add_shape_convex_hull(
        -1, mesh=newton.Mesh(np.asarray(p, np.float32), np.asarray(faces, np.int32), compute_inertia=False), cfg=cfg
    )
    vertices, indices = _disc(cap=1.0e-4, jitter=3.0e-5)
    body = builder.add_body(
        xform=wp.transform((0.002, -0.001, float(-vertices[:, 2].min() - 2.0e-4)), wp.quat_identity())
    )
    builder.add_shape_convex_hull(body, mesh=newton.Mesh(vertices, indices, compute_inertia=False), cfg=cfg)
    return builder.finalize(device=device)


def test_unverified_hull_keeps_vertex_face_contacts(test, device):
    """Check that a hull with non-supporting triangles skips the patch and keeps its vertex-face contacts."""
    model = _unverified_slab_model(device)
    off = _contact_arrays(_collide(model, False)[1])
    on = _contact_arrays(_collide(model, True)[1])
    test.assertGreaterEqual(len(on[0]), 3)
    for index, (before, after) in enumerate(zip(off, on, strict=True)):
        if index >= 2:
            np.testing.assert_allclose(before, after, rtol=0.0, atol=1.0e-6)
        else:
            np.testing.assert_array_equal(before, after)
    shape0, _, point0, point1, _ = on
    ground = np.where(shape0 == 0, point0[:, 2], point1[:, 2])
    test.assertLess(np.abs(ground).max(), 1.0e-6, f"slab witnesses at z = {ground}")


def _ridge_and_wedge():
    """Return a 100-vertex collinear ridge and a wedge with one vertex in the widened slab only."""
    ridge_x = np.linspace(-0.04, 0.04, 100)
    ridge = np.vstack(
        [
            np.stack([ridge_x, np.zeros(100), -1.0e-6 * (ridge_x / 0.04) ** 2], axis=1),
            [[-0.05, -0.1, -0.01], [0.05, -0.1, -0.01], [-0.05, 0.1, -0.01], [0.05, 0.1, -0.01]],
        ]
    )
    wedge = [[-0.08, -0.05, 6.0e-5], [0.08, -0.05, 6.0e-5], [-0.08, 0.2, 0.0], [0.08, 0.2, 0.0], [0.0, 0.001, 4.0e-5]]
    wedge += [[-0.08, -0.05, 0.01], [0.08, -0.05, 0.01], [-0.08, 0.2, 0.01], [0.08, 0.2, 0.01]]
    return _hull(ridge), _hull(np.asarray(wedge))


def _check_saturation(device):
    """Check a bounded, surface-accurate patch for two discs that each fill their half of the candidate buffer."""
    lower_vertices, lower_indices = _disc(segments=96, jitter=1.2e-5, seed=5)
    lower_vertices = lower_vertices - np.array([0.0, 0.0, lower_vertices[:, 2].max()], np.float32)
    upper_vertices, upper_indices = _disc(segments=96, jitter=1.2e-5, seed=6)
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=SHAPE_GAP)
    builder.add_shape_convex_hull(-1, mesh=newton.Mesh(lower_vertices, lower_indices, compute_inertia=False), cfg=cfg)
    z = float(-upper_vertices[:, 2].min() - 1.0e-4)
    body = builder.add_body(xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
    builder.add_shape_convex_hull(body, mesh=newton.Mesh(upper_vertices, upper_indices, compute_inertia=False), cfg=cfg)
    model = builder.finalize(device=device)
    state, contacts = _collide(model, True)
    errors, _, _ = _witness_report(model, state, contacts)
    assert len(errors) == 8, len(errors)
    assert errors.max() < SURFACE_TOLERANCE, errors.max()


def _check_widened_slab(device):
    """Return the patch contact counts for the ridge/wedge pair in both roles, with a prescribed normal and witness."""

    @wp.func
    def count_contact(contact: ContactData, out: wp.array[int], index: int):
        wp.atomic_add(out, 0, 1)

    @wp.func
    def keep_contact(
        contact: ContactData,
        geom_a: GenericShapeData,
        position_a: wp.vec3,
        orientation_a: wp.quat,
        geom_b: GenericShapeData,
        position_b: wp.vec3,
        orientation_b: wp.quat,
    ):
        return contact

    build = create_build_manifold(None, count_contact, keep_contact, convex_mesh_patch=True)
    closure = dict(zip(build.func.__code__.co_freevars, (c.cell_contents for c in build.func.__closure__), strict=True))
    patch = closure["build_convex_mesh_patch_manifold"]

    @wp.kernel(module="unique", enable_backward=False)
    def run(
        mesh_a: wp.uint64,
        mesh_b: wp.uint64,
        position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        out: wp.array[int],
    ):
        geom_a = GenericShapeData()
        geom_a.scale = wp.vec3(1.0)
        geom_a.auxiliary = pack_mesh_ptr(mesh_a)
        geom_b = GenericShapeData()
        geom_b.scale = wp.vec3(1.0)
        geom_b.auxiliary = pack_mesh_ptr(mesh_b)
        template = ContactData()
        template.gap_sum = 0.001
        identity = wp.quat_identity()
        out[1] = patch(geom_a, geom_b, identity, wp.vec3(0.0), identity, position_b, p_a, p_b, normal, out, template)

    ridge, wedge = _ridge_and_wedge()
    meshes = {}
    for name, (vertices, indices) in (("ridge", ridge), ("wedge", wedge)):
        meshes[name] = wp.Mesh(
            wp.array(vertices, dtype=wp.vec3, device=device), wp.array(indices, dtype=int, device=device)
        )
    depth = wp.vec3(0.0, 0.0, 2.0e-4)
    # The wedge overlaps the ridge by 0.2 mm; each role order supplies its own normal and witnesses.
    roles = (("ridge", "wedge", -depth, wp.vec3(0.0), -depth, wp.vec3(0.0, 0.0, 1.0)),)
    roles += (("wedge", "ridge", depth, wp.vec3(0.0), depth, wp.vec3(0.0, 0.0, -1.0)),)
    counts = []
    for a, b, position_b, p_a, p_b, normal in roles:
        out = wp.zeros(2, dtype=int, device=device)
        wp.launch(run, 1, inputs=[meshes[a].id, meshes[b].id, position_b, p_a, p_b, normal, out], device=device)
        counts.append(out.numpy().tolist())
    return counts


def test_saturated_candidates_stay_bounded(test, device):
    """Check bounded patches when both hulls fill their candidate share, and after a collinear first pass."""
    _check_saturation(device)
    # The ridge fills its share with collinear candidates, so the slab widens and the wedge adds its vertex.
    # No point lies across the ridge from the wedge's, so the quadrilateral step fails and the fill tops up to 7.
    test.assertEqual(_check_widened_slab(device), [[7, 7], [7, 7]])


def test_saturated_candidates_are_bounds_checked(test, device):
    """Check that the saturating pairs run on CPU with Warp's array bounds checks enabled."""
    script = (
        "import warp as wp\n"
        "wp.config.mode = 'debug'\n"
        "from newton.tests import test_convex_mesh_patch as t\n"
        "t._check_saturation('cpu')\n"
        "print('SATURATION_OK', t._check_widened_slab('cpu'), flush=True)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=1200, check=False)
    output = result.stdout + result.stderr
    test.assertEqual(result.returncode, 0, output[-4000:])
    test.assertIn("SATURATION_OK [[7, 7], [7, 7]]", result.stdout)
    test.assertNotIn("Assertion failed", output)


def test_patch_leaves_other_pairs_unchanged(test, device):
    """Check that primitive and box versus convex-mesh pairs keep their contacts under the flag."""
    hull_vertices, hull_indices = _sphere(0.01, 12, 7)
    for face_clip in (False, True):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = SHAPE_GAP
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0199), _quat_from_euler_deg(3.0, 2.0, 0.0)))
        builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
        top = builder.add_body(xform=wp.transform((0.003, 0.0, 0.0598), _quat_from_euler_deg(0.0, 0.0, 20.0)))
        builder.add_shape_box(top, hx=0.02, hy=0.02, hz=0.02)
        hull = builder.add_body(xform=wp.transform((0.3, 0.0, 0.0099), wp.quat_identity()))
        builder.add_shape_convex_hull(hull, mesh=newton.Mesh(hull_vertices, hull_indices))
        ball = builder.add_body(xform=wp.transform((0.6, 0.0, 0.0099), wp.quat_identity()))
        builder.add_shape_sphere(ball, radius=0.01)
        model = builder.finalize(device=device)
        off = _contact_arrays(_collide(model, False, face_clip)[1])
        on = _contact_arrays(_collide(model, True, face_clip)[1])
        pairs = set(zip(off[0].tolist(), off[1].tolist(), strict=True))
        test.assertTrue({(0, 1), (1, 2), (0, 3), (0, 4)} <= pairs, f"face_clip={face_clip}: got {pairs}")
        # The flag compiles a separate kernel, so the GPU code generator may round shared arithmetic differently.
        for index, (before, after) in enumerate(zip(off, on, strict=True)):
            if index >= 2:
                np.testing.assert_allclose(before, after, rtol=0.0, atol=1.0e-6)
            else:
                np.testing.assert_array_equal(before, after)


def test_patch_deterministic(test, device):
    """Check that repeated collides into one persistent buffer match a fresh pipeline bitwise."""
    model = _build_pair(
        device,
        _disc(cap=1.0e-4, jitter=3.0e-5),
        _quat_from_euler_deg(0.3, 0.0, 0.0),
        2.0e-4,
        _box(0.3, 0.3, 0.005),
        _GLOBAL_ROTATIONS[1],
        False,
        (1.0, 1.0, 1.0),
    )
    state, fresh_contacts = _collide(model, True)
    fresh = _contact_arrays(fresh_contacts)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True, convex_mesh_patch=True)
    contacts = pipeline.contacts()
    for _ in range(3):
        pipeline.collide(state, contacts)
        for before, after in zip(fresh, _contact_arrays(contacts), strict=True):
            np.testing.assert_array_equal(before, after)


def test_patch_rejects_prebuilt_narrow_phase(test, device):
    """Check that the flag cannot be combined with a prebuilt narrow phase."""
    model = _unverified_slab_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    with test.assertRaises(ValueError):
        newton.CollisionPipeline(
            model, broad_phase=pipeline.broad_phase, narrow_phase=pipeline.narrow_phase, convex_mesh_patch=True
        )


def _puck_points():
    """Return the points of a 36 mm puck whose bottom is a 0.1 mm dome with 30 um vertical noise."""
    rng = np.random.default_rng(0)
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    bottom = [_ring(0.036, -0.03, 32)]
    for k in range(1, 5):
        r = 0.036 * (1.0 - k / 5.0)
        phase = np.pi / 32 * (k % 2)
        z = -0.03 - 1.0e-4 * (1.0 - (r / 0.036) ** 2)
        bottom.append(np.stack([r * np.cos(angles + phase), r * np.sin(angles + phase), np.full(32, z)], axis=1))
    bottom.append([[0.0, 0.0, -0.0301]])
    bottom = np.vstack(bottom)
    bottom[:, 2] += rng.uniform(-3.0e-5, 3.0e-5, len(bottom))
    return np.vstack([bottom, _ring(0.036, 0.03, 32)])


def _puck_motion(device, convex_mesh_patch, yaw, steps=4800):
    """Return the late mean |yaw rate| and rms tilt rate [rad/s] of the puck resting on a convex-hull slab."""
    puck_vertices, puck_indices = _hull(_puck_points())
    slab_vertices, slab_indices = _box(0.2, 0.15, 0.0017)
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.002, margin=0.0)
    slab_xform = wp.transform((0.0, 0.0, 0.0017), wp.quat_identity())
    builder.add_shape_convex_hull(-1, xform=slab_xform, mesh=newton.Mesh(slab_vertices, slab_indices), cfg=cfg)
    z = 0.0034 - float(puck_vertices[:, 2].min()) + 3.0e-4
    body = builder.add_body(xform=wp.transform((0.01, 0.0, z), _quat_from_euler_deg(0.0, 0.0, yaw)))
    puck_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.002, margin=0.0, density=0.13 / (np.pi * 0.036**2 * 0.06))
    builder.add_shape_convex_hull(body, mesh=newton.Mesh(puck_vertices, puck_indices), cfg=puck_cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", contact_matching="sticky", convex_mesh_patch=convex_mesh_patch
    )
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverSteel(
        model,
        iterations=32,
        contact_regularization=0.01,
        solve_mode="serial",
        contact_shared_anchor=True,
        enable_sleeping=True,
    )
    state_0, state_1, control = model.state(), model.state(), model.control()

    def tick():
        # Contacts are refreshed once per 4 substeps; an even substep count leaves state_0 current.
        nonlocal state_0, state_1
        pipeline.collide(state_0, contacts)
        for _ in range(4):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 0.00125)
            state_0, state_1 = state_1, state_0

    tick()
    graph = None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device) as capture:
            tick()
        graph = capture.graph
    rates = []
    for k in range(steps // 4 - 2):
        if graph is None:
            tick()
        else:
            wp.capture_launch(graph)
        if k >= steps // 8:
            rates.append(state_0.body_qd.numpy()[0, 3:].copy())
    late = np.asarray(rates)
    return float(np.abs(late[:, 2]).mean()), float(np.sqrt(np.mean(np.sum(late[:, :2] ** 2, axis=1))))


def test_puck_rocking_drops(test, device):
    """Check that the patch cuts the rocking of a puck whose rough bottom has no flat face.

    The patched puck need not sleep: some poses keep a limit cycle near 0.1 rad/s on some GPUs.
    """
    for yaw in (0.0, 37.0):
        _, rock_off = _puck_motion(device, False, yaw)
        spin, rock = _puck_motion(device, True, yaw)
        test.assertGreater(rock_off, 0.3, f"yaw {yaw}: the flag-off puck no longer rocks ({rock_off:.3f} rad/s)")
        test.assertLess(rock, 0.15, f"yaw {yaw}: puck rocks at {rock:.3f} rad/s")
        test.assertLess(rock, 0.25 * rock_off, f"yaw {yaw}: puck rocks at {rock:.3f} rad/s vs {rock_off:.3f} off")
        test.assertLess(spin, 0.05, f"yaw {yaw}: puck spins at {spin:.3f} rad/s")


class TestConvexMeshPatch(unittest.TestCase):
    pass


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()
for _function in (
    test_witnesses_on_hull_surfaces,
    test_witnesses_on_hull_surfaces_split,
    test_patch_contact_capacity,
    test_witnesses_on_hull_surfaces_swapped_scaled,
    test_patch_spans_disc_on_disc,
    test_unverified_hull_keeps_vertex_face_contacts,
    test_unreferenced_vertices_are_not_witnesses,
    test_saturated_candidates_stay_bounded,
    test_patch_leaves_other_pairs_unchanged,
    test_patch_deterministic,
    test_patch_rejects_prebuilt_narrow_phase,
):
    add_function_test(TestConvexMeshPatch, _function.__name__, _function, devices=devices)
add_function_test(
    TestConvexMeshPatch,
    "test_saturated_candidates_are_bounds_checked",
    test_saturated_candidates_are_bounds_checked,
    devices=["cpu"],
)
add_function_test(TestConvexMeshPatch, "test_puck_rocking_drops", test_puck_rocking_drops, devices=cuda_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
