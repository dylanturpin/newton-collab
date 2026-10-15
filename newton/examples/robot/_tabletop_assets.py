# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Instance the procedural tabletop USD assets into a scene builder."""

from functools import cache

import numpy as np
import warp as wp

import newton
import newton.examples

COLORS = [(0.035, 0.43, 0.78), (0.02, 0.65, 0.47), (0.96, 0.61, 0.065)]


@cache
def _asset(name, floating, mu, restitution):
    """Import a USD asset once; scenes copy it per instance and share its meshes."""
    asset = newton.ModelBuilder()
    asset.default_shape_cfg.mu = mu
    asset.default_shape_cfg.restitution = restitution
    asset.default_shape_cfg.density = 0.0
    asset.default_shape_cfg.gap = 0.00025
    asset.default_joint_cfg.armature = 1.0e-5
    # Keep every authored convex piece; the largest has 128 vertices.
    asset.add_usd(newton.examples.get_asset(f"fpgs_tabletop/{name}.usd"), floating=floating, mesh_maxhullvert=128)
    return asset


def _scene_asset(b, name, floating):
    """Import with the scene's default contact material."""
    return _asset(name, floating, b.default_shape_cfg.mu, b.default_shape_cfg.restitution)


def import_object(b, name, pose, *, free_root=False, finish_color=None):
    """Add one instance of an asset at ``pose``.

    Returns a mapping from each body's USD prim name to its index in ``b``. The
    root body is welded to the world unless ``free_root`` is set. A
    ``finish_color`` overrides the colors of the visual meshes.
    """
    asset = _scene_asset(b, name, free_root)
    body_start, shape_start = b.body_count, b.shape_count
    b.add_builder(asset, xform=pose)
    if finish_color is not None:
        for shape in range(shape_start, b.shape_count):
            if not b.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES:
                b.shape_color[shape] = finish_color
    return {label.rsplit("/", 1)[-1]: body_start + i for i, label in enumerate(asset.body_label)}


def collision_bounds(b, name, orientation):
    """Bound a single-body asset's collision geometry after rotating it by ``orientation``."""
    asset = _scene_asset(b, name, True)
    rotation = wp.transform(wp.vec3(), orientation)
    points = []
    for shape, source in enumerate(asset.shape_source):
        if not asset.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES:
            continue
        local = rotation * asset.shape_transform[shape]
        for vertex in np.asarray(source.vertices) * np.asarray(asset.shape_scale[shape]):
            points.append(np.asarray(wp.transform_point(local, wp.vec3(*vertex))))
    points = np.asarray(points)
    return points.min(axis=0), points.max(axis=0)
