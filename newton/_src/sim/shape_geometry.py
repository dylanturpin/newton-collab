# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device-side copies of per-shape collision geometry between shape slots of a finalized model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from .model import Model


@wp.kernel(enable_backward=False)
def copy_shape_geometry_kernel(
    source_shapes: wp.array[wp.int32],
    target_shapes: wp.array[wp.int32],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_collision_radius: wp.array[float],
    shape_margin: wp.array[float],
    shape_gap: wp.array[float],
    shape_is_solid: wp.array[wp.bool],
    shape_source_ptr: wp.array[wp.uint64],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    shape_mesh_properties: wp.array[wp.int32],
    shape_support_data: wp.array[wp.vec4i],
    shape_sdf_index: wp.array[wp.int32],
    shape_mesh_boundary_start: wp.array[wp.int32],
    shape_edge_range: wp.array[wp.vec2i],
    shape_interior_point: wp.array[wp.vec3],
    shape_voxel_resolution: wp.array[wp.vec3i],
    shape_heightfield_index: wp.array[wp.int32],
    bvh_shape_bounds: wp.array2d[wp.vec3],
):
    """Copy one shape slot's geometry and its finalize-time companions into another slot.

    Entries whose source or target index is out of range are skipped, so a
    fixed-size selection array with ``-1`` for unselected rows can be replayed
    from a CUDA graph.
    """
    tid = wp.tid()
    source = source_shapes[tid]
    target = target_shapes[tid]
    shape_count = shape_scale.shape[0]
    if source < 0 or target < 0 or source >= shape_count or target >= shape_count or source == target:
        return
    shape_scale[target] = shape_scale[source]
    shape_transform[target] = shape_transform[source]
    shape_collision_radius[target] = shape_collision_radius[source]
    shape_margin[target] = shape_margin[source]
    shape_gap[target] = shape_gap[source]
    shape_is_solid[target] = shape_is_solid[source]
    shape_source_ptr[target] = shape_source_ptr[source]
    if shape_collision_aabb_lower:
        shape_collision_aabb_lower[target] = shape_collision_aabb_lower[source]
        shape_collision_aabb_upper[target] = shape_collision_aabb_upper[source]
    if shape_mesh_properties:
        shape_mesh_properties[target] = shape_mesh_properties[source]
    if shape_support_data:
        shape_support_data[target] = shape_support_data[source]
    if shape_sdf_index:
        shape_sdf_index[target] = shape_sdf_index[source]
    if shape_mesh_boundary_start:
        shape_mesh_boundary_start[target] = shape_mesh_boundary_start[source]
    if shape_edge_range:
        shape_edge_range[target] = shape_edge_range[source]
    if shape_interior_point:
        shape_interior_point[target] = shape_interior_point[source]
    if shape_voxel_resolution:
        shape_voxel_resolution[target] = shape_voxel_resolution[source]
    if shape_heightfield_index:
        shape_heightfield_index[target] = shape_heightfield_index[source]
    if bvh_shape_bounds:
        bvh_shape_bounds[target, 0] = bvh_shape_bounds[source, 0]
        bvh_shape_bounds[target, 1] = bvh_shape_bounds[source, 1]


def _optional_shape_array(model: Model, name: str):
    """Return a per-shape array of *model*, or ``None`` when it is absent or an empty sentinel."""
    array = getattr(model, name, None)
    if array is None or array.shape[0] < model.shape_count:
        return None
    return array


def copy_shape_geometry(model: Model, source_shapes: wp.array, target_shapes: wp.array) -> None:
    """Launch :func:`copy_shape_geometry_kernel` for *model*; see :meth:`newton.Model.copy_shape_geometry`."""
    if source_shapes.shape != target_shapes.shape or source_shapes.ndim != 1:
        raise ValueError(
            f"source_shapes and target_shapes must be one-dimensional arrays of equal length, got "
            f"{source_shapes.shape} and {target_shapes.shape}"
        )
    if source_shapes.dtype is not wp.int32 or target_shapes.dtype is not wp.int32:
        raise ValueError("source_shapes and target_shapes must have dtype wp.int32")
    if model.shape_count == 0 or source_shapes.shape[0] == 0:
        return
    wp.launch(
        copy_shape_geometry_kernel,
        dim=source_shapes.shape[0],
        inputs=[
            source_shapes,
            target_shapes,
            model.shape_scale,
            model.shape_transform,
            model.shape_collision_radius,
            model.shape_margin,
            model.shape_gap,
            model.shape_is_solid,
            model.shape_source_ptr,
            _optional_shape_array(model, "shape_collision_aabb_lower"),
            _optional_shape_array(model, "shape_collision_aabb_upper"),
            _optional_shape_array(model, "_shape_mesh_properties"),
            _optional_shape_array(model, "_shape_support_data"),
            _optional_shape_array(model, "_shape_sdf_index"),
            _optional_shape_array(model, "_shape_mesh_boundary_start"),
            _optional_shape_array(model, "shape_edge_range"),
            _optional_shape_array(model, "_shape_interior_point"),
            _optional_shape_array(model, "_shape_voxel_resolution"),
            _optional_shape_array(model, "shape_heightfield_index"),
            _optional_shape_array(model, "bvh_shape_bounds"),
        ],
        device=model.device,
    )
