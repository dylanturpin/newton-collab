# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mesh vs analytic primitive closed-form contacts, compared against the GJK/MPR path.

``_analytic_mesh_features``: 2 is the new path, 1 the band cull alone, 0 plain GJK/MPR.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

LEGACY = 0
BAND_ONLY = 1
EXACT = 2

PEN = 0.01
MESH_HALF = 0.25

# routed primitives: small ones press into a large face, large ones carry the mesh
ROUTED = (
    (GeoType.BOX, (0.05, 0.05, 0.05), 0.05),
    (GeoType.SPHERE, (0.30, 0.00, 0.00), 0.30),
    (GeoType.CAPSULE, (0.10, 0.20, 0.00), 0.30),
    (GeoType.CYLINDER, (0.20, 0.15, 0.00), 0.15),
    (GeoType.CONE, (0.25, 0.20, 0.00), 0.20),
)
LARGE = (
    (GeoType.BOX, (1.00, 1.00, 0.50)),
    (GeoType.SPHERE, (1.00, 0.00, 0.00)),
    (GeoType.CAPSULE, (0.60, 0.50, 0.00)),
    (GeoType.CYLINDER, (1.00, 0.40, 0.00)),
    (GeoType.CONE, (1.00, 0.60, 0.00)),
)


def _plate_mesh(size=3.0):
    """Two-triangle square plate in the XY plane, wound so its normal points down."""
    h = size / 2.0
    verts = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]], dtype=np.float32)
    tris = np.array([0, 2, 1, 0, 3, 2], dtype=np.int32)
    return newton.Mesh(verts, tris, compute_inertia=False)


def _grid_box_mesh(half, n):
    """Closed box with ``n x n`` quads per face, so contacts land on real interior triangles."""
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
                    tris += [a, c, d, a, d, b] if sign > 0.0 else [a, d, c, a, b, d]
    return newton.Mesh(np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int32), compute_inertia=False)


def _add_primitive(builder, body, geo, scale, xform, cfg=None):
    if geo == GeoType.BOX:
        return builder.add_shape_box(body, xform=xform, hx=scale[0], hy=scale[1], hz=scale[2], cfg=cfg)
    if geo == GeoType.SPHERE:
        return builder.add_shape_sphere(body, xform=xform, radius=scale[0], cfg=cfg)
    if geo == GeoType.CAPSULE:
        return builder.add_shape_capsule(body, xform=xform, radius=scale[0], half_height=scale[1], cfg=cfg)
    if geo == GeoType.CYLINDER:
        return builder.add_shape_cylinder(body, xform=xform, radius=scale[0], half_height=scale[1], cfg=cfg)
    if geo == GeoType.CONE:
        return builder.add_shape_cone(body, xform=xform, radius=scale[0], half_height=scale[1], cfg=cfg)
    if geo == GeoType.ELLIPSOID:
        return builder.add_shape_ellipsoid(body, xform=xform, rx=scale[0], ry=scale[1], rz=scale[2], cfg=cfg)
    raise NotImplementedError(geo)


def _support_height(geo, scale, rot):
    """Height of the rotated primitive's topmost point above its own origin."""
    x, y, z, w = (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
    rot_mat = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    d = rot_mat.T @ np.array([0.0, 0.0, 1.0])
    if geo == GeoType.BOX:
        return float(np.abs(d) @ np.array(scale))
    if geo == GeoType.SPHERE:
        return float(scale[0])
    if geo == GeoType.CAPSULE:
        return float(scale[0] + scale[1] * abs(d[2]))
    if geo == GeoType.CYLINDER:
        return float(scale[0] * np.hypot(d[0], d[1]) + scale[1] * abs(d[2]))
    if geo == GeoType.CONE:
        return float(max(scale[1] * d[2], scale[0] * np.hypot(d[0], d[1]) - scale[1] * d[2]))
    if geo == GeoType.ELLIPSOID:
        return float(np.linalg.norm(np.array(scale) * d))
    raise NotImplementedError(geo)


def _to_world(points, shapes, model, state):
    """Contact points are stored in their body's frame; map them to world."""
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy() if state.body_q is not None else np.zeros((0, 7))
    out = np.array(points, dtype=np.float64)
    for i, shape in enumerate(shapes):
        body = int(shape_body[int(shape)])
        if body < 0:
            continue
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


def _contacts(pipeline, state, contacts):
    """Return (count, separations, normals) with sphere and capsule radii accounted for."""
    pipeline.collide(state, contacts)
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return 0, np.zeros(0), np.zeros((0, 3))
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    p0 = _to_world(contacts.rigid_contact_point0.numpy()[:n], shape0, pipeline.model, state)
    p1 = _to_world(contacts.rigid_contact_point1.numpy()[:n], shape1, pipeline.model, state)
    normal = contacts.rigid_contact_normal.numpy()[:n]
    thickness = contacts.rigid_contact_margin0.numpy()[:n] + contacts.rigid_contact_margin1.numpy()[:n]
    return n, np.einsum("ij,ij->i", p1 - p0, normal) - thickness, normal


def _plate_scene(device, geo, scale, rot, features, penetration=PEN, gap=1.0e-3, reduce_contacts=True):
    """A 3 m plate with the primitive's topmost point ``penetration`` past it."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=gap)
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -penetration), wp.quat_identity()),
        mesh=_plate_mesh(),
        cfg=cfg,
    )
    top = _support_height(geo, scale, rot)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.3, -0.2, -top), rot))
    _add_primitive(builder, body, geo, scale, wp.transform_identity(), cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=2048,
        reduce_contacts=reduce_contacts,
        _analytic_mesh_features=features,
    )
    return model, model.state(), pipeline, pipeline.contacts()


def _deepest(device, geo, scale, rot, features, **kwargs):
    _, state, pipeline, contacts = _plate_scene(device, geo, scale, rot, features, **kwargs)
    n, sep, normal = _contacts(pipeline, state, contacts)
    if n == 0:
        return 0, None, None
    k = int(np.argmin(sep))
    return n, float(sep[k]), normal[k]


TILTS = (
    ("upright", (0.0, 0.0, 1.0), 0.0),
    ("tilted", (1.0, 0.3, 0.0), 0.5),
    ("steep", (1.0, 0.6, 0.2), 1.1),
)


def test_tilted_features_match_legacy(test, device):
    """Verify every routed primitive, upright and tilted, reports the legacy depth and normal."""
    for geo, scale, _top in ROUTED:
        for name, axis, angle in TILTS:
            rot = wp.quat_identity() if angle == 0.0 else wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle)
            n_new, sep_new, nrm_new = _deepest(device, geo, scale, rot, EXACT)
            n_old, sep_old, _ = _deepest(device, geo, scale, rot, LEGACY)
            msg = f"{geo!r} {name}: new {n_new} contacts {sep_new}, legacy {n_old} contacts {sep_old}"
            test.assertGreater(n_new, 0, msg=msg)
            test.assertGreater(n_old, 0, msg=msg)
            test.assertAlmostEqual(sep_new, -PEN, delta=1.0e-4, msg=msg)
            test.assertAlmostEqual(sep_new, sep_old, delta=2.0e-4, msg=msg)
            test.assertGreater(float(abs(nrm_new[2])), 0.999, msg=f"{geo!r} {name} normal {nrm_new}")


def test_shallow_contacts(test, device):
    """Verify a 0.1 mm overlap inside a 1 mm contact gap is found at the right depth."""
    shallow = 1.0e-4
    for geo, scale, _top in ROUTED:
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.5)
        n, sep, _ = _deepest(device, geo, scale, rot, EXACT, penetration=shallow, gap=1.0e-3)
        test.assertGreater(n, 0, msg=f"{geo!r}: shallow contact missed")
        test.assertAlmostEqual(sep, -shallow, delta=2.0e-5, msg=f"{geo!r} separation {sep}")


def test_box_manifold_is_supporting_corners_only(test, device):
    """Verify an unreduced flat box rests on four corners and a tilted one on its supporting corner."""
    scale = (0.05, 0.05, 0.05)
    _, state, pipeline, contacts = _plate_scene(
        device, GeoType.BOX, scale, wp.quat_identity(), EXACT, reduce_contacts=False
    )
    n, sep, normal = _contacts(pipeline, state, contacts)
    test.assertEqual(n, 4, msg=f"flat box should rest on four corners, got {n}: {np.sort(sep)}")
    np.testing.assert_allclose(np.sort(sep), np.full(4, -PEN), atol=1.0e-4)
    test.assertTrue(np.all(np.abs(normal[:, 2]) > 0.999), msg=f"normals {normal}")

    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.5)
    _, state, pipeline, contacts = _plate_scene(device, GeoType.BOX, scale, tilt, EXACT, reduce_contacts=False)
    n, sep, _ = _contacts(pipeline, state, contacts)
    test.assertEqual(n, 1, msg=f"tilted box should rest on one corner, got {n}: {np.sort(sep)}")
    test.assertAlmostEqual(float(sep[0]), -PEN, delta=1.0e-4)


def _cube_on_primitive(device, geo, scale, features, subdiv, reduce_contacts=True):
    """A closed mesh cube resting ``PEN`` on the primitive's topmost point."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=1.0e-3)
    top = _support_height(geo, scale, wp.quat_identity())
    builder.add_shape_mesh(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, top + MESH_HALF - PEN), wp.quat_identity()),
        mesh=_grid_box_mesh(MESH_HALF, subdiv),
        cfg=cfg,
    )
    body = builder.add_body(xform=wp.transform_identity())
    _add_primitive(builder, body, geo, scale, wp.transform_identity(), cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=4096,
        reduce_contacts=reduce_contacts,
        _analytic_mesh_features=features,
    )
    return model, model.state(), pipeline, pipeline.contacts()


def test_mesh_features_against_primitive_match_legacy(test, device):
    """Verify mesh vertices and edges pressing into each primitive match the legacy path."""
    for geo, scale in LARGE:
        for subdiv in (1, 6):
            results = {}
            for features in (EXACT, LEGACY):
                _, state, pipeline, contacts = _cube_on_primitive(device, geo, scale, features, subdiv)
                n, sep, normal = _contacts(pipeline, state, contacts)
                test.assertGreater(n, 0, msg=f"{geo!r} subdiv {subdiv} features {features}: no contact")
                k = int(np.argmin(sep))
                results[features] = (float(sep[k]), float(abs(normal[k][2])))
            msg = f"{geo!r} subdiv {subdiv}: {results}"
            test.assertAlmostEqual(results[EXACT][0], -PEN, delta=3.0e-4, msg=msg)
            test.assertGreater(results[EXACT][1], 0.99, msg=msg)
            # GJK/MPR's separating translation tilts off the face normal on a pointed feature
            test.assertAlmostEqual(results[EXACT][0], results[LEGACY][0], delta=1.5e-3, msg=msg)


def test_band_cull_never_changes_contacts_and_tightens_with_density(test, device):
    """Verify the band cull leaves contacts unchanged and drops more candidates on a denser mesh."""
    counts = {}
    depths = {}
    for subdiv in (12, 24):
        for features in (LEGACY, BAND_ONLY, EXACT):
            _, state, pipeline, contacts = _cube_on_primitive(
                device, GeoType.SPHERE, (1.0, 0.0, 0.0), features, subdiv=subdiv
            )
            n, sep, _ = _contacts(pipeline, state, contacts)
            test.assertGreater(n, 0)
            counts[(subdiv, features)] = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
            depths[(subdiv, features)] = float(sep.min())
    for subdiv in (12, 24):
        test.assertLessEqual(counts[(subdiv, BAND_ONLY)], counts[(subdiv, LEGACY)], msg=f"{counts}")
        test.assertEqual(counts[(subdiv, BAND_ONLY)], counts[(subdiv, EXACT)], msg=f"{counts}")
        test.assertAlmostEqual(depths[(subdiv, BAND_ONLY)], depths[(subdiv, LEGACY)], delta=1.0e-6, msg=f"{depths}")
        test.assertAlmostEqual(depths[(subdiv, EXACT)], depths[(subdiv, LEGACY)], delta=3.0e-4, msg=f"{depths}")
    # the denser mesh must be culled proportionally harder than the coarse one
    coarse = counts[(12, BAND_ONLY)] / max(counts[(12, LEGACY)], 1)
    dense = counts[(24, BAND_ONLY)] / max(counts[(24, LEGACY)], 1)
    test.assertLess(dense, coarse, msg=f"cull did not tighten with density: {counts}")


def test_flat_face_keeps_every_candidate(test, device):
    """Verify a dense face resting flat on a large box is not culled at all."""
    counts = {}
    for features in (LEGACY, BAND_ONLY):
        _, state, pipeline, contacts = _cube_on_primitive(device, GeoType.BOX, (2.0, 2.0, 0.25), features, subdiv=12)
        n, _sep, _ = _contacts(pipeline, state, contacts)
        test.assertGreater(n, 0)
        counts[features] = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
    test.assertEqual(counts[BAND_ONLY], counts[LEGACY], msg=f"{counts}")


def test_ellipsoid_is_not_routed(test, device):
    """Verify ellipsoid pairs keep GJK/MPR, so the switch makes no difference."""
    scale = (0.30, 0.12, 0.06)
    rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.6, 0.0)), 0.7)
    results = {}
    for features in (EXACT, LEGACY):
        n, sep, _ = _deepest(device, GeoType.ELLIPSOID, scale, rot, features)
        test.assertGreater(n, 0)
        results[features] = (n, sep)
    test.assertEqual(results[EXACT][0], results[LEGACY][0], msg=f"{results}")
    test.assertAlmostEqual(results[EXACT][1], results[LEGACY][1], delta=1.0e-9, msg=f"{results}")


def test_convex_mesh_pair_is_not_routed(test, device):
    """Verify a mesh against a convex mesh keeps GJK/MPR."""
    results = {}
    for features in (EXACT, LEGACY):
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(gap=1.0e-3)
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, MESH_HALF - PEN), wp.quat_identity()),
            mesh=_grid_box_mesh(MESH_HALF, 2),
            cfg=cfg,
        )
        body = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_mesh(body, mesh=_grid_box_mesh(0.2, 1), cfg=cfg)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", rigid_contact_max=2048, _analytic_mesh_features=features
        )
        n, sep, _ = _contacts(pipeline, model.state(), pipeline.contacts())
        results[features] = (n, float(sep.min()) if n else None)
    test.assertEqual(results[EXACT], results[LEGACY], msg=f"{results}")


def test_randomized_differential_against_legacy(test, device):
    """Verify random poses of every routed primitive against a plate match GJK/MPR."""
    rng = np.random.default_rng(17)
    for geo, scale, _top in ROUTED:
        for _trial in range(12):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rot = wp.quat_from_axis_angle(wp.vec3(*[float(a) for a in axis]), float(rng.uniform(0.0, np.pi)))
            pen = float(rng.uniform(2.0e-4, 8.0e-3))
            n_new, sep_new, nrm_new = _deepest(device, geo, scale, rot, EXACT, penetration=pen)
            n_old, sep_old, nrm_legacy = _deepest(device, geo, scale, rot, LEGACY, penetration=pen)
            msg = f"{geo!r} pen {pen:.5f}: new {sep_new} legacy {sep_old}"
            test.assertGreater(n_new, 0, msg=msg)
            test.assertGreater(n_old, 0, msg=msg)
            test.assertAlmostEqual(sep_new, -pen, delta=2.0e-4, msg=msg)
            test.assertAlmostEqual(sep_new, sep_old, delta=3.0e-4, msg=msg)
            test.assertGreater(float(np.dot(nrm_new, nrm_legacy)), 0.999, msg=f"{msg} normals")


def test_non_reducing_pipeline(test, device):
    """Verify every routed primitive reports the right deepest contact with reduction off."""
    for geo, scale, _top in ROUTED:
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 0.5)
        n, sep, _ = _deepest(device, geo, scale, rot, EXACT, reduce_contacts=False)
        test.assertGreater(n, 0, msg=f"{geo!r}: no contact")
        test.assertAlmostEqual(sep, -PEN, delta=1.0e-4, msg=f"{geo!r} separation {sep}")


def test_speculative_contacts(test, device):
    """Verify a separated pair closing fast reports a contact only once the horizon reaches it."""
    config = newton.CollisionPipeline.SpeculativeContactConfig(max_speculative_extension=0.25)
    for geo, scale, _top in ROUTED:
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(gap=1.0e-3)
        builder.add_shape_mesh(
            body=-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), mesh=_plate_mesh(), cfg=cfg
        )
        top = _support_height(geo, scale, wp.quat_identity())
        body = builder.add_body(xform=wp.transform(wp.vec3(0.3, -0.2, -top), wp.quat_identity()))
        builder.body_qd[body] = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        _add_primitive(builder, body, geo, scale, wp.transform_identity(), cfg)
        model = builder.finalize(device=device)
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            rigid_contact_max=2048,
            speculative_config=config,
            _analytic_mesh_features=EXACT,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts, dt=0.002)
        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0, msg=f"{geo!r}: early contact")
        pipeline.collide(state, contacts, dt=0.2)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, msg=f"{geo!r}: no speculative contact")


def _rest_height(device, geo, scale, features):
    """Drop a mesh cube onto the primitive under FeatherPGS and return its resting height."""
    builder = newton.ModelBuilder()
    top = _support_height(geo, scale, wp.quat_identity())
    _add_primitive(builder, -1, geo, scale, wp.transform_identity())
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, top + MESH_HALF + 0.02), wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=_grid_box_mesh(MESH_HALF, 4))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", rigid_contact_max=4096, _analytic_mesh_features=features
    )
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(model, pgs_iterations=16)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    for _ in range(240):
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        state_in, state_out = state_out, state_in
    return float(state_in.body_q.numpy()[0][2]), top


def test_dynamics_parity_with_legacy(test, device):
    """Verify a cube settles on each primitive at the same height as with GJK/MPR."""
    for geo, scale, _top in ROUTED:
        z_new, top = _rest_height(device, geo, scale, EXACT)
        z_old, _ = _rest_height(device, geo, scale, LEGACY)
        msg = f"{geo!r}: new {z_new:.5f} legacy {z_old:.5f} expected about {top + MESH_HALF:.5f}"
        test.assertAlmostEqual(z_new, top + MESH_HALF, delta=1.5e-2, msg=msg)
        test.assertAlmostEqual(z_new, z_old, delta=4.0e-3, msg=msg)


def test_graph_capture(test, device):
    """Verify the routed path replays under CUDA-graph capture with a stable contact count."""
    _, state, pipeline, contacts = _cube_on_primitive(device, GeoType.CYLINDER, (0.2, 0.15, 0.0), EXACT, subdiv=4)
    pipeline.collide(state, contacts)
    eager = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(eager, 0)
    with wp.ScopedCapture(device) as capture:
        pipeline.collide(state, contacts)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), eager)


class TestMeshAnalyticContacts(unittest.TestCase):
    pass


all_devices = get_test_devices()
cuda_devices = [d for d in all_devices if d.is_cuda]
CUDA_ONLY = {"test_dynamics_parity_with_legacy", "test_graph_capture"}
for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        add_function_test(
            TestMeshAnalyticContacts,
            _name,
            _fn,
            devices=cuda_devices if _name in CUDA_ONLY else all_devices,
        )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
