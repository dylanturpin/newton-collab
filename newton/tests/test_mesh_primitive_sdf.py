# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``CollisionPipeline(mesh_primitive_sdf=True)``.

Routes triangle-mesh vs analytic-primitive pairs through the SDF contact kernels:
the primitive's closed-form signed distance is sampled along the mesh edges, and a
face pass emits the primitive's support point against each mesh face it pokes
through, instead of running one GJK/MPR test per overlapping triangle.

Texture SDFs need CUDA, so on CPU the mesh runs without one (BVH fallback), which
the route supports; dynamics and graph-capture tests are CUDA only.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.geometry.types import GeoType
from newton._src.sim.collide import write_contact
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

MESH_HALF = 0.25
PENETRATION = 0.01
SLAB_HALF_Z = 0.25
# Apex of the curved primitives: strictly inside a bottom-face triangle of the
# coarse (two triangles per face) mesh, 0.14 m from the diagonal and 0.15 m from
# the outer edges, so that none of the small primitives below reach an edge.
FACE_INTERIOR_OFFSET = (0.1, -0.1)
# Ellipsoids stay on the mesh-convex path: Newton's ellipsoid distance is a first-order
# approximation that the edge sampling would inherit (see test_ellipsoid_stays_on_legacy_route).
ROUTED_PRIMITIVES = (GeoType.BOX, GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.CONE)
CONE_RADIUS = 0.5
CONE_HALF_HEIGHT = 0.4


def _expected_depth(geo):
    """Separation reported at the deepest point for a primitive whose apex is PENETRATION inside the cube.

    The face pass measures the support point against the face plane, so every primitive,
    the cone apex included, reports the vertical overlap.
    """
    return -PENETRATION


# -----------------------------------------------------------------------------
# Scene helpers
# -----------------------------------------------------------------------------


def _grid_box_mesh(half: float, n: int) -> newton.Mesh:
    """Closed box of half-extent ``half`` with ``n x n`` quads per face (12 n^2 triangles)."""
    verts = []
    tris = []
    for u_axis, v_axis, w_axis in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for sign in (-1.0, 1.0):
            base = len(verts)
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[u_axis] = -half + 2.0 * half * i / n
                    p[v_axis] = -half + 2.0 * half * j / n
                    p[w_axis] = sign * half
                    verts.append(p)
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b = a + 1
                    c = a + (n + 1)
                    d = c + 1
                    if sign > 0.0:
                        tris += [a, c, d, a, d, b]
                    else:
                        tris += [a, d, c, a, b, d]
    return newton.Mesh(np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int32), compute_inertia=False)


def _mesh_cube(device, with_sdf: bool, subdiv: int = 1) -> newton.Mesh:
    mesh = _grid_box_mesh(MESH_HALF, subdiv)
    if with_sdf and wp.get_device(device).is_cuda:
        mesh.build_sdf(max_resolution=32, device=device)
    return mesh


def _add_primitive(builder, body, geo, top_z, offset=(0.0, 0.0), rot=None, small=False, cfg=None):
    """Add an analytic primitive whose topmost point sits at ``(offset, top_z)``.

    ``small=True`` picks a footprint that fits inside a bottom-face triangle of the
    coarse mesh, so its contact can only come from the face interior.
    """
    q = wp.quat_identity() if rot is None else rot
    ox, oy = offset
    if geo == GeoType.BOX:
        h = 0.05 if small else 2.0
        xform = wp.transform(wp.vec3(ox, oy, top_z - SLAB_HALF_Z), q)
        return builder.add_shape_box(body, xform=xform, hx=h, hy=h, hz=SLAB_HALF_Z, cfg=cfg)
    if geo == GeoType.SPHERE:
        xform = wp.transform(wp.vec3(ox, oy, top_z - 0.5), q)
        return builder.add_shape_sphere(body, xform=xform, radius=0.5, cfg=cfg)
    if geo == GeoType.CAPSULE:
        # Z-up capsule: top = center + half_height + radius
        xform = wp.transform(wp.vec3(ox, oy, top_z - 0.5 - 0.25), q)
        return builder.add_shape_capsule(body, xform=xform, radius=0.25, half_height=0.5, cfg=cfg)
    if geo == GeoType.CYLINDER:
        r = 0.05 if small else 0.6
        xform = wp.transform(wp.vec3(ox, oy, top_z - 0.4), q)
        return builder.add_shape_cylinder(body, xform=xform, radius=r, half_height=0.4, cfg=cfg)
    if geo == GeoType.CONE:
        # Z-up cone: apex = center + half_height
        xform = wp.transform(wp.vec3(ox, oy, top_z - CONE_HALF_HEIGHT), q)
        return builder.add_shape_cone(body, xform=xform, radius=CONE_RADIUS, half_height=CONE_HALF_HEIGHT, cfg=cfg)
    if geo == GeoType.ELLIPSOID:
        xform = wp.transform(wp.vec3(ox, oy, top_z - 0.2), q)
        return builder.add_shape_ellipsoid(body, xform=xform, rx=0.5, ry=0.3, rz=0.2, cfg=cfg)
    raise NotImplementedError(geo)


def _build(
    device,
    geo,
    *,
    mesh_sdf=True,
    penetration=PENETRATION,
    mesh_primitive_sdf=True,
    reduce_contacts=True,
    subdiv=8,
    offset=(0.0, 0.0),
    rot=None,
    small=False,
    mesh_scale=None,
    reverse_order=False,
    prim_cfg=None,
    gap=None,
    speculative_config=None,
    prim_velocity=None,
    max_triangle_pairs=None,
):
    """Static mesh cube with its bottom face at ``z = -penetration``; primitive on a body, top at ``z = 0``."""
    builder = newton.ModelBuilder()
    mesh = _mesh_cube(device, mesh_sdf, subdiv)
    half_z = MESH_HALF * (mesh_scale[2] if mesh_scale is not None else 1.0)
    mesh_xform = wp.transform(wp.vec3(0.0, 0.0, half_z - penetration), wp.quat_identity())
    mesh_cfg = None
    if gap is not None:
        mesh_cfg = newton.ModelBuilder.ShapeConfig(gap=gap)
        prim_cfg = newton.ModelBuilder.ShapeConfig(gap=gap, margin=prim_cfg.margin if prim_cfg is not None else 0.0)

    def add_mesh():
        return builder.add_shape_mesh(body=-1, xform=mesh_xform, mesh=mesh, scale=mesh_scale, cfg=mesh_cfg)

    def add_prim():
        body = builder.add_body(xform=wp.transform_identity())
        if prim_velocity is not None:
            builder.body_qd[body] = (*prim_velocity, 0.0, 0.0, 0.0)
        return _add_primitive(builder, body, geo, top_z=0.0, offset=offset, rot=rot, small=small, cfg=prim_cfg)

    if reverse_order:
        prim_shape = add_prim()
        mesh_shape = add_mesh()
    else:
        mesh_shape = add_mesh()
        prim_shape = add_prim()
    model = builder.finalize(device=device)
    state = model.state()
    kwargs = {} if max_triangle_pairs is None else {"max_triangle_pairs": max_triangle_pairs}
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=4096,
        reduce_contacts=reduce_contacts,
        mesh_primitive_sdf=mesh_primitive_sdf,
        speculative_config=speculative_config,
        **kwargs,
    )
    contacts = pipeline.contacts()
    return model, state, pipeline, contacts, mesh_shape, prim_shape


def _routing_counts(pipeline):
    np_ = pipeline.narrow_phase
    sdf_pairs = int(np_.shape_pairs_mesh_sdf_count.numpy()[0]) if np_.shape_pairs_mesh_sdf_count is not None else 0
    convex_pairs = int(np_.shape_pairs_mesh_count.numpy()[0]) if np_.shape_pairs_mesh_count is not None else 0
    return sdf_pairs, convex_pairs


def _to_world(points, shapes, model, state):
    """Contact points are stored in their body's frame; map them to world."""
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy() if state.body_q is not None else np.zeros((0, 7))
    out = np.array(points, dtype=np.float64)
    for i, shape in enumerate(shapes):
        body = int(shape_body[int(shape)])
        if body >= 0:
            q = body_q[body]
            x, y, z, w = q[3], q[4], q[5], q[6]
            rot = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            out[i] = rot @ out[i] + q[:3]
    return out


def _contact_arrays(contacts, pipeline, state):
    n = int(contacts.rigid_contact_count.numpy()[0])
    normal = contacts.rigid_contact_normal.numpy()[:n]
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    p0 = _to_world(contacts.rigid_contact_point0.numpy()[:n], shape0, pipeline.model, state)
    p1 = _to_world(contacts.rigid_contact_point1.numpy()[:n], shape1, pipeline.model, state)
    margin = contacts.rigid_contact_margin0.numpy()[:n] + contacts.rigid_contact_margin1.numpy()[:n]
    return n, p0, p1, normal, shape0, shape1, margin


def _separations(p0, p1, normal, margin):
    """Signed surface separation along the contact normal (negative = penetration).

    Sphere and capsule contact points sit on the shape's centre line with the radius
    folded into the contact margin, so the margin is subtracted here.
    """
    return np.einsum("ij,ij->i", p1 - p0, normal) - margin


def _collide_min_sep(pipeline, state, contacts, **collide_kwargs):
    pipeline.collide(state, contacts, **collide_kwargs)
    n, p0, p1, normal, _, _, margin = _contact_arrays(contacts, pipeline, state)
    if n == 0:
        return 0, None, normal
    return n, float(_separations(p0, p1, normal, margin).min()), normal


# -----------------------------------------------------------------------------
# Routing
# -----------------------------------------------------------------------------


def test_box_routes_to_sdf_kernel(test, device):
    """Flag on: mesh-box pair enters the mesh-SDF buffer, not the mesh-convex one."""
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, mesh_primitive_sdf=True)
    pipeline.collide(state, contacts)
    test.assertEqual(_routing_counts(pipeline), (1, 0))
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_box_legacy_route_when_flag_off(test, device):
    """Flag off: the same pair takes the legacy mesh-convex path."""
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, mesh_primitive_sdf=False)
    pipeline.collide(state, contacts)
    test.assertEqual(_routing_counts(pipeline), (0, 1))
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_box_box_pair_keeps_primitive_path(test, device):
    """Two analytic boxes are untouched by the flag: no mesh in the pair, no SDF route."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 - PENETRATION), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
    # a far-away mesh keeps the mesh kernels compiled in, as in a mixed scene
    far = wp.transform(wp.vec3(10.0, 0.0, 0.0), wp.quat_identity())
    builder.add_shape_mesh(body=-1, xform=far, mesh=_mesh_cube(device, True))
    model = builder.finalize(device=device)
    state = model.state()
    results = {}
    for flag in (True, False):
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=1024, mesh_primitive_sdf=flag)
        contacts = pipeline.contacts()
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertEqual(_routing_counts(pipeline), (0, 0))
        test.assertGreater(n, 0)
        results[flag] = (n, sep)
    test.assertEqual(results[True], results[False])


# -----------------------------------------------------------------------------
# Contact geometry: edges and face interiors, every primitive
# -----------------------------------------------------------------------------


def test_edge_contacts_all_primitives(test, device):
    """Dense mesh: every primitive's apex sits one penetration depth inside the cube."""
    for geo in ROUTED_PRIMITIVES:
        _, state, pipeline, contacts, _, _ = _build(device, geo, subdiv=8, offset=FACE_INTERIOR_OFFSET)
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertEqual(_routing_counts(pipeline), (1, 0), msg=f"{geo!r} not routed")
        test.assertGreater(n, 0, msg=f"{geo!r} produced no contacts")
        test.assertAlmostEqual(sep, _expected_depth(geo), delta=3.0e-3, msg=f"{geo!r} min separation {sep}")


def test_face_interior_contacts_all_primitives(test, device):
    """Coarse mesh, primitive footprint inside one face triangle: no mesh edge touches it.

    Only the face pass can see these; it must recover the same depth as the dense case.
    """
    for geo in ROUTED_PRIMITIVES:
        _, state, pipeline, contacts, _, _ = _build(device, geo, subdiv=1, offset=FACE_INTERIOR_OFFSET, small=True)
        n, sep, normal = _collide_min_sep(pipeline, state, contacts)
        test.assertGreater(n, 0, msg=f"{geo!r}: face-interior contact missed")
        test.assertAlmostEqual(sep, _expected_depth(geo), delta=3.0e-3, msg=f"{geo!r} min separation {sep}")
        # the face pass reports the face normal
        test.assertGreater(float(np.abs(normal[:, 2]).max()), 0.99, msg=f"{geo!r} normals {normal}")


def test_box_slab_normals_and_plane(test, device):
    """Flat slab: vertical normals, contact midpoints on the cube's bottom face plane."""
    _, state, pipeline, contacts, mesh_shape, prim_shape = _build(device, GeoType.BOX)
    pipeline.collide(state, contacts)
    n, p0, p1, normal, shape0, shape1, margin = _contact_arrays(contacts, pipeline, state)
    test.assertGreater(n, 0)
    pair = {mesh_shape, prim_shape}
    for a, b in zip(shape0, shape1, strict=True):
        test.assertEqual({int(a), int(b)}, pair)
    test.assertTrue(np.all(np.abs(normal[:, 2]) > 0.99), msg=f"normals not vertical: {normal[:3]}")
    sep = _separations(p0, p1, normal, margin)
    test.assertAlmostEqual(float(sep.min()), -PENETRATION, delta=2.0e-3, msg=f"separations {sep}")
    mid_z = 0.5 * (p0[:, 2] + p1[:, 2])
    test.assertLess(float(np.abs(mid_z + PENETRATION).max()), 2.0e-3, msg=f"midpoint z {mid_z}")


def test_rotated_primitives(test, device):
    """A rotated slab and a rotated sphere give the same depth as the axis-aligned ones."""
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6)
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, rot=yaw)
    _, sep, normal = _collide_min_sep(pipeline, state, contacts)
    test.assertAlmostEqual(sep, -PENETRATION, delta=2.0e-3)
    test.assertTrue(np.all(np.abs(normal[:, 2]) > 0.99))
    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 1.1)
    _, state, pipeline, contacts, _, _ = _build(
        device, GeoType.SPHERE, subdiv=1, offset=FACE_INTERIOR_OFFSET, rot=tilt, small=True
    )
    _, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertAlmostEqual(sep, -PENETRATION, delta=3.0e-3)


def test_nonuniformly_scaled_mesh(test, device):
    """Mesh shape scale (1, 2, 0.5): the scaled bottom face still lands one depth inside the slab."""
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, mesh_scale=(1.0, 2.0, 0.5))
    n, sep, normal = _collide_min_sep(pipeline, state, contacts)
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=2.0e-3)
    test.assertTrue(np.all(np.abs(normal[:, 2]) > 0.99))


def test_reversed_pair_order(test, device):
    """Primitive added before the mesh (lower shape index): routing and depth are unchanged."""
    for geo, kwargs in ((GeoType.BOX, {}), (GeoType.SPHERE, {"subdiv": 1, "small": True})):
        _, state, pipeline, contacts, mesh_shape, prim_shape = _build(
            device, geo, reverse_order=True, offset=FACE_INTERIOR_OFFSET, **kwargs
        )
        test.assertLess(prim_shape, mesh_shape)
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertEqual(_routing_counts(pipeline), (1, 0))
        test.assertGreater(n, 0)
        test.assertAlmostEqual(sep, -PENETRATION, delta=3.0e-3, msg=f"{geo!r}")


def test_mesh_without_sdf_still_contacts(test, device):
    """A mesh with no SDF of its own still gets contacts: the analytic direction needs none."""
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, mesh_sdf=False)
    n, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertEqual(_routing_counts(pipeline), (1, 0))
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=2.0e-3)


def test_reduction_disabled_variant(test, device):
    """The non-reducing pipeline carries both the edge and the face pass."""
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, reduce_contacts=False)
    n, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertEqual(_routing_counts(pipeline), (1, 0))
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=2.0e-3)
    _, state, pipeline, contacts, _, _ = _build(
        device, GeoType.SPHERE, reduce_contacts=False, subdiv=1, offset=FACE_INTERIOR_OFFSET, small=True
    )
    n, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=3.0e-3)


# -----------------------------------------------------------------------------
# Margins and speculative contacts
# -----------------------------------------------------------------------------


def test_margin_matches_legacy_route(test, device):
    """A primitive with a collision margin reports the same separation on both routes."""
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.02)
    results = {}
    for flag in (True, False):
        _, state, pipeline, contacts, _, _ = _build(
            device, GeoType.BOX, penetration=-0.005, mesh_primitive_sdf=flag, prim_cfg=cfg, gap=0.001
        )
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertGreater(n, 0, msg=f"flag={flag}: gap inside the margin produced no contact")
        results[flag] = sep
    test.assertAlmostEqual(results[True], results[False], delta=2.0e-3, msg=f"{results}")
    # without the margin the same 5 mm gap is outside the 1 mm contact gap
    _, state, pipeline, contacts, _, _ = _build(device, GeoType.BOX, penetration=-0.005, gap=0.001)
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_speculative_contacts(test, device):
    """Separated pair approaching at speed: a contact appears only when the horizon reaches it."""
    config = newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.25)
    for geo, kwargs in ((GeoType.BOX, {}), (GeoType.SPHERE, {"subdiv": 1, "small": True})):
        _, state, pipeline, contacts, _, _ = _build(
            device,
            geo,
            penetration=-0.05,
            offset=FACE_INTERIOR_OFFSET,
            speculative_config=config,
            prim_velocity=(0.0, 0.0, 1.0),
            gap=0.001,
            **kwargs,
        )
        pipeline.collide(state, contacts, dt=0.005)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0, msg=f"{geo!r}: early contact")
        n, sep, _ = _collide_min_sep(pipeline, state, contacts, dt=0.2)
        test.assertGreater(n, 0, msg=f"{geo!r}: speculative contact missing")
        test.assertAlmostEqual(sep, 0.05, delta=3.0e-3, msg=f"{geo!r} separation {sep}")


# -----------------------------------------------------------------------------
# Parity with the legacy route
# -----------------------------------------------------------------------------


def test_depth_and_normal_parity_with_legacy(test, device):
    """Dense slab and coarse-face sphere: both routes agree on depth and normal."""
    for geo, kwargs in ((GeoType.BOX, {}), (GeoType.SPHERE, {"subdiv": 1, "small": True})):
        results = {}
        for flag in (True, False):
            _, state, pipeline, contacts, _, _ = _build(
                device, geo, mesh_primitive_sdf=flag, offset=FACE_INTERIOR_OFFSET, **kwargs
            )
            n, sep, normal = _collide_min_sep(pipeline, state, contacts)
            test.assertGreater(n, 0, msg=f"{geo!r} flag={flag}")
            results[flag] = (sep, float(np.abs(normal[:, 2]).max()))
        test.assertAlmostEqual(results[True][0], results[False][0], delta=3.0e-3, msg=f"{geo!r} {results}")
        test.assertGreater(results[True][1], 0.95)
        test.assertGreater(results[False][1], 0.95)


def _plate_mesh(size: float) -> newton.Mesh:
    """Two-triangle square plate in the XY plane, wound so its normal points down (-z)."""
    h = size / 2.0
    verts = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], dtype=np.float32)
    tris = np.array([0, 2, 1, 0, 3, 2], dtype=np.int32)
    return newton.Mesh(verts, tris, compute_inertia=False)


def _tilted_scene(device, geo, rot, top_offset, mesh_primitive_sdf, penetration=PENETRATION, gap=None):
    """A 3 m flat plate at ``z = -penetration`` over a rotated primitive whose topmost point sits at ``z = 0``.

    ``top_offset`` is the height of the primitive's topmost point above its frame origin after
    ``rot`` is applied, so the body is placed at ``z = -top_offset``. A single large face keeps the
    topmost point inside the face interior for every tilt, which is the case under test.
    """
    builder = newton.ModelBuilder()
    cfg = None if gap is None else newton.ModelBuilder.ShapeConfig(gap=gap)
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -penetration), wp.quat_identity()),
        mesh=_plate_mesh(3.0),
        cfg=cfg,
    )
    ox, oy = FACE_INTERIOR_OFFSET
    body = builder.add_body(xform=wp.transform(wp.vec3(ox, oy, -top_offset), rot))
    if geo == GeoType.BOX:
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    elif geo == GeoType.CYLINDER:
        builder.add_shape_cylinder(body, radius=0.05, half_height=0.08, cfg=cfg)
    elif geo == GeoType.CONE:
        builder.add_shape_cone(body, radius=CONE_RADIUS, half_height=CONE_HALF_HEIGHT, cfg=cfg)
    else:
        raise NotImplementedError(geo)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", rigid_contact_max=1024, mesh_primitive_sdf=mesh_primitive_sdf
    )
    return model, model.state(), pipeline, pipeline.contacts()


def _frame_z_in_local(rot):
    """World +z expressed in the rotated primitive's own frame."""
    x, y, z, w = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
    rot_mat = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot_mat.T @ np.array([0.0, 0.0, 1.0])


def _support_height(geo, rot):
    """Height of the rotated primitive's topmost point above its origin (its support along +z)."""
    d = _frame_z_in_local(rot)
    if geo == GeoType.BOX:
        return float(np.abs(d) @ np.array([0.05, 0.05, 0.05]))
    if geo == GeoType.CYLINDER:
        return float(0.05 * np.hypot(d[0], d[1]) + 0.08 * abs(d[2]))
    if geo == GeoType.CONE:
        apex = CONE_HALF_HEIGHT * d[2]
        base = CONE_RADIUS * np.hypot(d[0], d[1]) - CONE_HALF_HEIGHT * d[2]
        return float(max(apex, base))
    raise NotImplementedError(geo)


TILTED_CASES = (
    (GeoType.BOX, (1.0, 0.3, 0.0), 0.5),
    (GeoType.CYLINDER, (1.0, 0.0, 0.0), 0.45),
    (GeoType.CONE, (1.0, 0.2, 0.0), 0.35),
)


def test_tilted_features_match_legacy(test, device):
    """Tilted box corner, cylinder rim and cone apex against a large flat face.

    The face pass reports the face normal and the plane depth of the primitive's support point,
    so it must agree with the legacy mesh-convex path and with the geometric overlap.
    """
    for geo, axis, angle in TILTED_CASES:
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle)
        top = _support_height(geo, rot)
        results = {}
        for flag in (True, False):
            _, state, pipeline, contacts = _tilted_scene(device, geo, rot, top, flag)
            n, sep, normal = _collide_min_sep(pipeline, state, contacts)
            test.assertGreater(n, 0, msg=f"{geo!r} flag={flag}: no contact")
            results[flag] = (sep, float(np.abs(normal[:, 2]).max()))
        test.assertAlmostEqual(results[True][0], -PENETRATION, delta=1.0e-3, msg=f"{geo!r} {results}")
        test.assertAlmostEqual(results[True][0], results[False][0], delta=1.0e-3, msg=f"{geo!r} {results}")
        test.assertGreater(results[True][1], 0.99, msg=f"{geo!r} normals {results}")


def test_shallow_contacts(test, device):
    """0.1 mm overlaps with a 1 mm contact gap are found, for an upright cone apex and a tilted box corner."""
    shallow = 1.0e-4
    cases = (
        (GeoType.CONE, wp.quat_identity()),
        (GeoType.BOX, wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.5)),
    )
    for geo, rot in cases:
        top = _support_height(geo, rot)
        _, state, pipeline, contacts = _tilted_scene(device, geo, rot, top, True, penetration=shallow, gap=1.0e-3)
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertGreater(n, 0, msg=f"{geo!r}: shallow contact missed")
        test.assertAlmostEqual(sep, -shallow, delta=5.0e-5, msg=f"{geo!r} separation {sep}")


def test_face_candidate_overflow_is_reported(test, device):
    """A face-candidate buffer too small for the scene is counted past capacity and the verifier warns.

    Registered with ``check_output=False``: the verifier's ``wp.printf`` warning is the expected output.
    """
    _, state, pipeline, contacts, _, _ = _build(
        device, GeoType.SPHERE, subdiv=4, offset=FACE_INTERIOR_OFFSET, small=True, max_triangle_pairs=1
    )
    pipeline.collide(state, contacts)
    wp.synchronize()
    np_ = pipeline.narrow_phase
    test.assertEqual(np_.mesh_face_candidates.shape[0], 1)
    test.assertGreater(int(np_.mesh_face_candidate_count.numpy()[0]), 1)
    # with capacity the contact is found
    _, state, pipeline, contacts, _, _ = _build(
        device, GeoType.SPHERE, subdiv=4, offset=FACE_INTERIOR_OFFSET, small=True
    )
    n, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=3.0e-3)


def _box_on_plate(device, rot, mesh_primitive_sdf, half=0.05, offset=(0.3, -0.2), gap=1.0e-3):
    """Cube resting ``PENETRATION`` into a large plate, contact reduction off.

    The offset keeps every corner inside one plate triangle, so the face pass alone decides
    the manifold and the full contact set can be compared.
    """
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=gap)
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -PENETRATION), wp.quat_identity()),
        mesh=_plate_mesh(3.0),
        cfg=cfg,
    )
    top = float(np.abs(_frame_z_in_local(rot)) @ np.array([half, half, half]))
    ox, oy = offset
    body = builder.add_body(xform=wp.transform(wp.vec3(ox, oy, -top), rot))
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=1024,
        reduce_contacts=False,
        mesh_primitive_sdf=mesh_primitive_sdf,
    )
    return model, model.state(), pipeline, pipeline.contacts()


def test_box_manifold_is_supporting_corners_only(test, device):
    """Unreduced box manifold: only corners extremal along the face normal are emitted.

    A cube resting flat gives its four bottom corners at one penetration depth; tilting it
    onto a single corner gives exactly that one. Corners on the far side of the box are not
    supporting and must never appear, even though they lie inside the contact gap.
    """
    _, state, pipeline, contacts = _box_on_plate(device, wp.quat_identity(), True)
    pipeline.collide(state, contacts)
    n, p0, p1, normal, _, _, margin = _contact_arrays(contacts, pipeline, state)
    sep = np.sort(_separations(p0, p1, normal, margin))
    test.assertEqual(n, 4, msg=f"flat cube should rest on four corners, got {n}: {sep}")
    np.testing.assert_allclose(sep, np.full(4, -PENETRATION), atol=1.0e-4)
    test.assertTrue(np.all(np.abs(normal[:, 2]) > 0.99), msg=f"normals {normal}")

    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.5)
    _, state, pipeline, contacts = _box_on_plate(device, tilt, True)
    n, sep_min, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertEqual(n, 1, msg=f"tilted cube should rest on one corner, got {n}")
    test.assertAlmostEqual(sep_min, -PENETRATION, delta=1.0e-4)


def test_ellipsoid_stays_on_legacy_route(test, device):
    """Ellipsoid pairs are not routed: the edge pass would inherit the approximate distance.

    Newton's ellipsoid SDF is a first-order approximation whose error away from the principal
    axes reaches tens of percent of the penetration depth, so these pairs keep the mesh-convex
    path and must produce the same contacts whether the flag is set or not.
    """
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -PENETRATION), wp.quat_identity()),
        mesh=_plate_mesh(3.0),
    )
    rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.6, 0.0)), 0.7)
    top = float(np.linalg.norm(np.array([0.5, 0.1, 0.05]) * _frame_z_in_local(rot)))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.3, -top), rot))
    builder.add_shape_ellipsoid(body, rx=0.5, ry=0.1, rz=0.05)
    model = builder.finalize(device=device)
    state = model.state()
    results = {}
    for flag in (True, False):
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", rigid_contact_max=1024, mesh_primitive_sdf=flag)
        contacts = pipeline.contacts()
        n, sep, _ = _collide_min_sep(pipeline, state, contacts)
        test.assertEqual(_routing_counts(pipeline), (0, 1), msg=f"flag={flag}: ellipsoid pair was routed")
        test.assertGreater(n, 0)
        results[flag] = sep
    test.assertAlmostEqual(results[True], results[False], delta=1.0e-6)
    test.assertAlmostEqual(results[True], -PENETRATION, delta=1.0e-3)


def _drop_rest_height(device, mesh_primitive_sdf: bool, subdiv: int) -> float:
    """Drop a mesh cube onto a static box slab under FeatherPGS and return its resting height."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, -SLAB_HALF_Z), wp.quat_identity()), hx=2.0, hy=2.0, hz=SLAB_HALF_Z
    )
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, MESH_HALF + 0.02), wp.quat_identity()))
    mesh = _grid_box_mesh(MESH_HALF, subdiv)
    builder.add_shape_mesh(body, mesh=mesh)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", rigid_contact_max=2048, mesh_primitive_sdf=mesh_primitive_sdf
    )
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, pgs_iterations=16)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    dt = 1.0 / 240.0
    for _ in range(240):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
    return float(state_in.body_q.numpy()[0][2])


def test_dynamics_parity_with_legacy(test, device):
    """A mesh cube settles at the same height on the slab under either route."""
    for subdiv in (1, 4):
        z_sdf = _drop_rest_height(device, True, subdiv)
        z_legacy = _drop_rest_height(device, False, subdiv)
        test.assertAlmostEqual(z_sdf, MESH_HALF, delta=1.0e-2, msg=f"subdiv={subdiv} SDF route rest height {z_sdf}")
        test.assertAlmostEqual(z_sdf, z_legacy, delta=2.0e-3, msg=f"subdiv={subdiv} rest heights {z_sdf} vs {z_legacy}")


# -----------------------------------------------------------------------------
# Expert construction and graph capture
# -----------------------------------------------------------------------------


def test_expert_narrow_phase_is_authoritative(test, device):
    """A user-built NarrowPhase carries the flag; passing it to the pipeline as well is an error."""
    builder = newton.ModelBuilder()
    mesh = _mesh_cube(device, True)
    builder.add_shape_mesh(
        body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, MESH_HALF - PENETRATION), wp.quat_identity()), mesh=mesh
    )
    body = builder.add_body(xform=wp.transform_identity())
    _add_primitive(builder, body, GeoType.BOX, top_z=0.0)
    model = builder.finalize(device=device)
    state = model.state()

    broad_phase = BroadPhaseAllPairs(model.shape_world, model.shape_flags, device=device)

    def make_narrow_phase(**kwargs):
        return NarrowPhase(
            max_candidate_pairs=64,
            max_triangle_pairs=100000,
            device=device,
            shape_aabb_lower=wp.zeros(model.shape_count, dtype=wp.vec3, device=device),
            shape_aabb_upper=wp.zeros(model.shape_count, dtype=wp.vec3, device=device),
            contact_writer_warp_func=write_contact,
            shape_voxel_resolution=model._shape_voxel_resolution,
            **kwargs,
        )

    narrow_phase = make_narrow_phase(mesh_primitive_sdf=True)
    with test.assertRaisesRegex(ValueError, "mesh_primitive_sdf"):
        newton.CollisionPipeline(
            model, broad_phase=broad_phase, narrow_phase=narrow_phase, rigid_contact_max=1024, mesh_primitive_sdf=True
        )
    pipeline = newton.CollisionPipeline(
        model, broad_phase=broad_phase, narrow_phase=narrow_phase, rigid_contact_max=1024
    )
    test.assertTrue(pipeline.mesh_primitive_sdf)
    contacts = pipeline.contacts()
    n, sep, _ = _collide_min_sep(pipeline, state, contacts)
    test.assertEqual(_routing_counts(pipeline), (1, 0))
    test.assertGreater(n, 0)
    test.assertAlmostEqual(sep, -PENETRATION, delta=2.0e-3)

    plain = make_narrow_phase()
    pipeline = newton.CollisionPipeline(model, broad_phase=broad_phase, narrow_phase=plain, rigid_contact_max=1024)
    test.assertFalse(pipeline.mesh_primitive_sdf)
    pipeline.collide(state, pipeline.contacts())
    test.assertEqual(_routing_counts(pipeline), (0, 1))


def test_graph_capture(test, device):
    """Edge and face passes replay under CUDA-graph capture with a stable contact count."""
    _, state, pipeline, contacts, _, _ = _build(
        device, GeoType.SPHERE, subdiv=1, offset=FACE_INTERIOR_OFFSET, small=True
    )
    pipeline.collide(state, contacts)
    eager_count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(eager_count, 0)
    with wp.ScopedCapture(device) as capture:
        pipeline.collide(state, contacts)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), eager_count)


class TestMeshPrimitiveSDF(unittest.TestCase):
    pass


all_devices = get_test_devices()
cuda_devices = [d for d in all_devices if d.is_cuda]
CUDA_ONLY = {"test_dynamics_parity_with_legacy", "test_graph_capture"}
PRINTS_WARNING = {"test_face_candidate_overflow_is_reported"}
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        add_function_test(
            TestMeshPrimitiveSDF,
            _name,
            _fn,
            devices=cuda_devices if _name in CUDA_ONLY else all_devices,
            check_output=_name not in PRINTS_WARNING,
        )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
