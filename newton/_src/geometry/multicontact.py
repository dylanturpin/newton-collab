# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# This code is based on the multi-contact manifold generation from Jitter Physics 2
# Original: https://github.com/notgiven688/jitterphysics2
# Copyright (c) Thorben Linneweber (MIT License)
# The code has been translated from C# to Python and modified for use in Newton.

"""
Multi-contact manifold generation for collision detection.

This module implements contact manifold generation algorithms for computing
multiple contact points between colliding shapes. It includes polygon clipping
and contact point selection algorithms.
"""

import math
from typing import Any

import warp as wp

from newton._src.math import orthonormal_basis

from .contact_data import ContactData
from .mpr import create_support_map_function
from .support_function import unpack_mesh_ptr
from .types import GeoType

# Constants
EPS = 0.00001
# The tilt angle defines how much the search direction gets tilted while searching for
# points on the contact manifold.
TILT_ANGLE_RAD = wp.static(2.0 * math.pi / 180.0)
SIN_TILT_ANGLE = wp.static(math.sin(TILT_ANGLE_RAD))
COS_TILT_ANGLE = wp.static(math.cos(TILT_ANGLE_RAD))

COS_DEEPEST_CONTACT_THRESHOLD_ANGLE = wp.static(math.cos(0.1 * math.pi / 180.0))

# Face clipping applies while the collision normal is within this angle of a reference face normal.
COS_FACE_CLIP_MAX_ANGLE = wp.constant(math.cos(15.0 * math.pi / 180.0))
# Vertices kept on the boundary of the incident face.
FACE_CLIP_PATCH_CAPACITY = wp.constant(16)
# Angle between the in-plane directions that sample an incident face with more vertices than the capacity.
FACE_CLIP_DIRECTION_STEP = wp.constant(2.0 * math.pi / 16.0)
# Face capacity plus one vertex per clip plane (four side planes and the contact gap).
FACE_CLIP_POLYGON_CAPACITY = 21
# Smallest incident-plane normal component along the reference normal for which face depths are flattened.
FACE_CLIP_MIN_PLANE_DEPTH_SLOPE = wp.constant(1.0e-3)
# Hull vertices within this distance [m] of the incident face plane belong to the face.
FACE_CLIP_COPLANAR_DISTANCE = wp.constant(1.0e-5)
# Patch points closer than this [m] are merged.
FACE_CLIP_MERGE_DISTANCE_SQ = wp.constant(1.0e-8)
# Twice the triangle area [m^2] below which a patch point adds no support.
FACE_CLIP_MIN_DOUBLE_AREA = wp.constant(1.0e-10)
# Allowed excess [m] of the patch depth over the MPR/GJK depth before falling back.
FACE_CLIP_DEPTH_TOLERANCE = wp.constant(1.0e-4)


@wp.func
def should_include_deepest_contact(normal_dot: float) -> bool:
    return normal_dot < COS_DEEPEST_CONTACT_THRESHOLD_ANGLE


@wp.func
def excess_normal_deviation(dir_a: wp.vec3, dir_b: wp.vec3) -> bool:
    """
    Check if the angle between two direction vectors exceeds the tilt angle threshold.

    This is used to detect when contact polygon normals deviate too much from the
    collision normal, indicating that the contact manifold may be unreliable.

    Args:
        dir_a: First direction vector.
        dir_b: Second direction vector.

    Returns:
        True if the angle between the vectors exceeds TILT_ANGLE_RAD (2 degrees).
    """
    dot = wp.abs(wp.dot(dir_a, dir_b))
    return dot < COS_TILT_ANGLE


@wp.func
def signed_area(a: wp.vec2, b: wp.vec2, query_point: wp.vec2) -> float:
    """
    Calculates twice the signed area for the triangle (a, b, query_point).

    The result's sign indicates the triangle's orientation and is a robust way
    to check which side of a line a point is on.

    Args:
        a: The first vertex of the triangle and the start of the line segment.
        b: The second vertex of the triangle and the end of the line segment.
        query_point: The third vertex of the triangle, the point to test against the line a-b.

    Returns:
        The result's sign determines the orientation of the points:
        - Positive (> 0): The points are in a counter-clockwise (CCW) order.
          This means query_point is to the "left" of the directed line from a to b.
        - Negative (< 0): The points are in a clockwise (CW) order.
          This means query_point is to the "right" of the directed line from a to b.
        - Zero (== 0): The points are collinear; query_point lies on the infinite line defined by a and b.
    """
    # It returns twice the signed area of the triangle
    return (b[0] - a[0]) * (query_point[1] - a[1]) - (b[1] - a[1]) * (query_point[0] - a[0])


@wp.func
def ray_plane_intersection(
    ray_origin: wp.vec3, ray_direction: wp.vec3, plane_d: float, plane_normal: wp.vec3
) -> wp.vec3:
    """
    Compute intersection of a ray with a plane.

    The plane is defined by the equation: dot(point, plane_normal) + plane_d = 0
    where plane_d = -dot(point_on_plane, plane_normal).

    Args:
        ray_origin: Starting point of the ray.
        ray_direction: Direction vector of the ray.
        plane_d: Plane distance parameter (negative dot product of any point on plane with normal).
        plane_normal: Normal vector of the plane.

    Returns:
        Intersection point of the ray with the plane.
    """
    denom = wp.dot(ray_direction, plane_normal)
    # Avoid division by zero; if denom is near zero, return origin unchanged
    if wp.abs(denom) < 1.0e-12:
        return ray_origin
    # Plane equation: dot(point, normal) + d = 0
    # Solve for t: dot(ray_origin + t*ray_direction, normal) + d = 0
    # t = -(dot(ray_origin, normal) + d) / dot(ray_direction, normal)
    t = -(wp.dot(ray_origin, plane_normal) + plane_d) / denom
    return ray_origin + ray_direction * t


@wp.struct
class BodyProjector:
    """
    Plane projector for back-projecting contact points onto shape surfaces.

    The plane is defined by the equation: dot(point, normal) + plane_d = 0
    where plane_d = -dot(point_on_plane, normal) for any point on the plane.

    This representation uses a single float instead of storing a full point_on_plane vector,
    saving 8 bytes per projector (2 floats on typical architectures with alignment).
    """

    plane_d: float
    normal: wp.vec3


@wp.struct
class IncrementalPlaneTracker:
    reference_point: wp.vec3
    previous_point: wp.vec3
    normal: wp.vec3
    largest_area_sq: float


@wp.func
def update_incremental_plane_tracker(
    tracker: IncrementalPlaneTracker,
    current_point: wp.vec3,
    current_point_id: int,
) -> IncrementalPlaneTracker:
    """
    Update the incremental plane tracker with a new point.
    """
    if current_point_id == 0:
        tracker.reference_point = current_point
        tracker.largest_area_sq = 0.0
    elif current_point_id == 1:
        tracker.previous_point = current_point
    else:
        edge1 = tracker.previous_point - tracker.reference_point
        edge2 = current_point - tracker.reference_point
        cross = wp.cross(edge1, edge2)
        area_sq = wp.dot(cross, cross)
        if area_sq > tracker.largest_area_sq:
            tracker.largest_area_sq = area_sq
            tracker.normal = cross
        tracker.previous_point = current_point
    return tracker


@wp.func
def compute_line_segment_projector_normal(
    segment_dir: wp.vec3,
    reference_normal: wp.vec3,
) -> wp.vec3:
    """
    Compute a normal for a line segment projector that is perpendicular to the segment
    and lies in the plane defined by the segment and the reference normal.

    Args:
        segment_dir: Direction vector of the line segment.
        reference_normal: Normal from the other body to use as reference.

    Returns:
        Normalized normal vector for the line segment projector.
    """
    right = wp.cross(segment_dir, reference_normal)
    normal = wp.cross(right, segment_dir)
    length = wp.length(normal)
    return normal * (1.0 / length) if length > 1.0e-12 else reference_normal


@wp.func
def create_body_projectors(
    plane_tracker_a: IncrementalPlaneTracker,
    anchor_point_a: wp.vec3,
    plane_tracker_b: IncrementalPlaneTracker,
    anchor_point_b: wp.vec3,
    contact_normal: wp.vec3,
) -> tuple[BodyProjector, BodyProjector]:
    projector_a = BodyProjector()
    projector_b = BodyProjector()

    if plane_tracker_a.largest_area_sq == 0.0 and plane_tracker_b.largest_area_sq == 0.0:
        # Both are line segments - compute normals using contact_normal as reference
        dir_a = plane_tracker_a.previous_point - plane_tracker_a.reference_point
        dir_b = plane_tracker_b.previous_point - plane_tracker_b.reference_point

        point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)
        projector_a.normal = compute_line_segment_projector_normal(dir_a, contact_normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

        point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)
        projector_b.normal = compute_line_segment_projector_normal(dir_b, contact_normal)
        projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)

        return projector_a, projector_b

    if plane_tracker_a.largest_area_sq > 0.0:
        inv_len_n = 1.0 / wp.sqrt(wp.max(1.0e-12, plane_tracker_a.largest_area_sq))
        projector_a.normal = plane_tracker_a.normal * inv_len_n
        projector_a.plane_d = -wp.dot(anchor_point_a, projector_a.normal)
    if plane_tracker_b.largest_area_sq > 0.0:
        inv_len_n = 1.0 / wp.sqrt(wp.max(1.0e-12, plane_tracker_b.largest_area_sq))
        projector_b.normal = plane_tracker_b.normal * inv_len_n
        projector_b.plane_d = -wp.dot(anchor_point_b, projector_b.normal)

    if plane_tracker_a.largest_area_sq == 0.0:
        dir = plane_tracker_a.previous_point - plane_tracker_a.reference_point
        point_on_plane_a = 0.5 * (plane_tracker_a.reference_point + plane_tracker_a.previous_point)
        projector_a.normal = compute_line_segment_projector_normal(dir, projector_b.normal)
        projector_a.plane_d = -wp.dot(point_on_plane_a, projector_a.normal)

    if plane_tracker_b.largest_area_sq == 0.0:
        dir = plane_tracker_b.previous_point - plane_tracker_b.reference_point
        point_on_plane_b = 0.5 * (plane_tracker_b.reference_point + plane_tracker_b.previous_point)
        projector_b.normal = compute_line_segment_projector_normal(dir, projector_a.normal)
        projector_b.plane_d = -wp.dot(point_on_plane_b, projector_b.normal)

    return projector_a, projector_b


@wp.func
def body_projector_project(
    proj: BodyProjector,
    input: wp.vec3,
    contact_normal: wp.vec3,
) -> wp.vec3:
    """
    Project a point back onto the original shape surface using a plane projector.

    This function casts a ray from the input point along the contact normal and
    finds where it intersects the projector's plane.

    Args:
        proj: Body projector defining the projection plane.
        input: Point to project (typically in contact plane space).
        contact_normal: Direction to cast the ray (typically the collision normal).

    Returns:
        Projected point on the shape's surface in world space.
    """
    # Only plane projection is supported
    return ray_plane_intersection(input, contact_normal, proj.plane_d, proj.normal)


@wp.func
def intersection_point(trim_seg_start: wp.vec2, trim_seg_end: wp.vec2, a: wp.vec2, b: wp.vec2) -> wp.vec2:
    """
    Calculate the intersection point between a line segment and a polygon edge.

    It is known that a and b lie on different sides of the trim segment.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        a: First point of the polygon edge.
        b: Second point of the polygon edge.

    Returns:
        The intersection point as a vec2.
    """
    # Since a and b are on opposite sides, their signed areas have opposite signs
    # We can optimize: abs(signed_a) + abs(signed_b) = abs(signed_a - signed_b)
    signed_a = signed_area(trim_seg_start, trim_seg_end, a)
    signed_b = signed_area(trim_seg_start, trim_seg_end, b)
    interp_ab = wp.abs(signed_a) / wp.abs(signed_a - signed_b)

    # Interpolate between a and b
    return (1.0 - interp_ab) * a + interp_ab * b


@wp.func
def insert_vec2(arr: wp.array[wp.vec2], arr_count: int, index: int, element: wp.vec2):
    """
    Insert an element into an array at the specified index, shifting elements to the right.

    Args:
        arr: Array to insert into.
        arr_count: Current number of elements in the array.
        index: Index at which to insert the element.
        element: Element to insert.
    """
    i = arr_count
    while i > index:
        arr[i] = arr[i - 1]
        i -= 1
    arr[index] = element


@wp.func
def trim_in_place(
    trim_seg_start: wp.vec2,
    trim_seg_end: wp.vec2,
    loop: wp.array[wp.vec2],
    loop_count: int,
) -> int:
    """
    Trim a polygon in place using a line segment (Sutherland-Hodgman clip).

    All points are in 2D contact plane space.

    Args:
        trim_seg_start: Start point of the trimming segment.
        trim_seg_end: End point of the trimming segment.
        loop: Array of loop vertices (2D).
        loop_count: Number of vertices in the loop.

    Returns:
        New number of vertices in the trimmed loop.
    """
    if loop_count < 3:
        return loop_count

    intersection_a = wp.vec2(0.0, 0.0)
    change_a = int(-1)
    intersection_b = wp.vec2(0.0, 0.0)
    change_b = int(-1)

    keep = bool(False)

    # Check first vertex
    prev_outside = bool(signed_area(trim_seg_start, trim_seg_end, loop[0]) <= 0.0)

    for i in range(loop_count):
        next_idx = (i + 1) % loop_count
        outside = signed_area(trim_seg_start, trim_seg_end, loop[next_idx]) <= 0.0

        if outside != prev_outside:
            intersection = intersection_point(trim_seg_start, trim_seg_end, loop[i], loop[next_idx])
            if change_a < 0:
                change_a = i
                keep = not prev_outside
                intersection_a = intersection
            else:
                change_b = i
                intersection_b = intersection

        prev_outside = outside

    if change_a >= 0 and change_b >= 0:
        loop_indexer = int(-1)
        new_loop_count = int(loop_count)

        i = int(0)
        while i < loop_count:
            # If the current vertex is on the side to be kept, copy it.
            if keep:
                loop_indexer += 1
                loop[loop_indexer] = loop[i]

            # If the current edge intersects the trim line, add the intersection point.
            if i == change_a or i == change_b:
                pt = intersection_a if i == change_a else intersection_b

                # Handle special case: insertion needed when loop_indexer == i and not keep.
                if loop_indexer == i and not keep:
                    loop_indexer += 1
                    insert_vec2(loop, new_loop_count, loop_indexer, pt)

                    new_loop_count += 1
                    i += 1
                    change_b += 1
                    loop_count += 1
                else:
                    loop_indexer += 1
                    loop[loop_indexer] = pt

                keep = not keep

            i += 1

        new_loop_count = loop_indexer + 1
    elif prev_outside:
        new_loop_count = 0
    else:
        new_loop_count = loop_count

    return new_loop_count


@wp.func
def trim_all_in_place(
    trim_poly: wp.array[wp.vec2],
    trim_poly_count: int,
    loop: wp.array[wp.vec2],
    loop_count: int,
) -> int:
    """
    Trim a polygon using all edges of another polygon (Sutherland-Hodgman clipping).

    Both polygons (trim_poly and loop) are in 2D contact plane space and they are both convex.

    Args:
        trim_poly: Array of vertices defining the trimming polygon (2D).
        trim_poly_count: Number of vertices in the trimming polygon.
        loop: Array of vertices in the loop to be trimmed (2D).
        loop_count: Number of vertices in the loop.

    Returns:
        New number of vertices in the trimmed loop.
    """

    if trim_poly_count <= 1:
        return wp.min(1, loop_count)  # There is no trim polygon

    move_distance = float(1e-5)

    if trim_poly_count == 2:
        # Convert line segment to thin rectangle
        p0 = trim_poly[0]
        p1 = trim_poly[1]

        dir_x = p1[0] - p0[0]
        dir_y = p1[1] - p0[1]
        dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)

        if dir_len > 1e-10:
            inv_dir_len = 1.0 / dir_len
            perp_x = -dir_y * inv_dir_len
            perp_y = dir_x * inv_dir_len

            offset_x = perp_x * move_distance
            offset_y = perp_y * move_distance

            trim_poly[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)
            trim_poly[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)
            trim_poly[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)
            trim_poly[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)
            trim_poly_count = 4
        else:
            return wp.min(1, loop_count)

    if loop_count == 2:
        # Convert line segment to thin rectangle
        p0 = loop[0]
        p1 = loop[1]

        dir_x = p1[0] - p0[0]
        dir_y = p1[1] - p0[1]
        dir_len = wp.sqrt(dir_x * dir_x + dir_y * dir_y)

        if dir_len > 1e-10:
            inv_dir_len = 1.0 / dir_len
            perp_x = -dir_y * inv_dir_len
            perp_y = dir_x * inv_dir_len

            offset_x = perp_x * move_distance
            offset_y = perp_y * move_distance

            loop[0] = wp.vec2(p0[0] - offset_x, p0[1] - offset_y)
            loop[1] = wp.vec2(p1[0] - offset_x, p1[1] - offset_y)
            loop[2] = wp.vec2(p1[0] + offset_x, p1[1] + offset_y)
            loop[3] = wp.vec2(p0[0] + offset_x, p0[1] + offset_y)

            loop_count = 4
        else:
            return wp.min(1, loop_count)

    current_loop_count = loop_count

    trim_poly_0 = trim_poly[0]  # This allows to do more memory aliasing
    for i in range(trim_poly_count):
        trim_seg_start = trim_poly[i]
        trim_seg_end = trim_poly_0 if i == trim_poly_count - 1 else trim_poly[i + 1]
        current_loop_count = trim_in_place(trim_seg_start, trim_seg_end, loop, current_loop_count)

    return current_loop_count


@wp.func
def approx_max_quadrilateral_area_with_calipers(hull: wp.array[wp.vec2], hull_count: int) -> wp.vec4i:
    """
    Finds an approximate maximum area quadrilateral inside a convex hull in O(n) time
    using the Rotating Calipers algorithm to find the hull's diameter.

    Args:
        hull: Array of hull vertices (2D).
        hull_count: Number of vertices in the hull.

    Returns:
        vec4i containing (p1, p2, p3, p4) where p1, p2, p3, p4 are the indices
        of the quadrilateral vertices that form the maximum area quadrilateral.
    """
    n = hull_count

    # --- Step 1: Find the hull's diameter using Rotating Calipers in O(n) ---
    p1 = int(0)
    p3 = int(1)
    hp1 = hull[p1]
    hp3 = hull[p3]
    diff = wp.vec2(hp1[0] - hp3[0], hp1[1] - hp3[1])
    max_dist_sq = diff[0] * diff[0] + diff[1] * diff[1]

    # Relative epsilon for tie-breaking: only update if new value is at least (1 + epsilon) times better
    # This is scale-invariant and avoids catastrophic cancellation in floating-point comparisons
    # Important for objects with circular geometry to ensure consistent point selection
    tie_epsilon_rel = 1.0e-3

    # Start with point j opposite point i=0
    j = int(1)
    for i in range(n):
        # For the current point i, find its antipodal point j by advancing j
        # while the area of the triangle formed by the edge (i, i+1) and point j increases.
        # This is equivalent to finding the point j furthest from the edge (i, i+1).
        hull_i = hull[i]
        hull_i_plus_1 = hull[(i + 1) % n]

        while True:
            hull_j = hull[j]
            hull_j_plus_1 = hull[(j + 1) % n]

            area_j_plus_1 = signed_area(hull_i, hull_i_plus_1, hull_j_plus_1)
            area_j = signed_area(hull_i, hull_i_plus_1, hull_j)

            if area_j_plus_1 > area_j:
                j = (j + 1) % n
            else:
                break

        # Now, (i, j) is an antipodal pair. Check its distance (2D)
        hi = hull[i]
        hj = hull[j]
        d1 = wp.vec2(hi[0] - hj[0], hi[1] - hj[1])
        dist_sq_1 = d1[0] * d1[0] + d1[1] * d1[1]
        # Use relative tie-breaking: only update if new distance is meaningfully larger
        if dist_sq_1 > max_dist_sq * (1.0 + tie_epsilon_rel):
            max_dist_sq = dist_sq_1
            p1 = i
            p3 = j

        # The next point, (i+1, j), is also an antipodal pair. Check its distance too (2D)
        hip1 = hull[(i + 1) % n]
        d2 = wp.vec2(hip1[0] - hj[0], hip1[1] - hj[1])
        dist_sq_2 = d2[0] * d2[0] + d2[1] * d2[1]
        # Use relative tie-breaking: only update if new distance is meaningfully larger
        if dist_sq_2 > max_dist_sq * (1.0 + tie_epsilon_rel):
            max_dist_sq = dist_sq_2
            p1 = (i + 1) % n
            p3 = j

    # --- Step 2: Find points p2 and p4 furthest from the diameter (p1, p3) ---
    p2 = int(0)
    p4 = int(0)
    max_area_1 = float(0.0)
    max_area_2 = float(0.0)

    hull_p1 = hull[p1]
    hull_p3 = hull[p3]

    for i in range(n):
        # Use the signed area to determine which side of the line the point is on.
        hull_i = hull[i]
        area = signed_area(hull_p1, hull_p3, hull_i)

        # Use relative tie-breaking: only update if new area is meaningfully larger
        if area > max_area_1 * (1.0 + tie_epsilon_rel):
            max_area_1 = area
            p2 = i
        elif -area > max_area_2 * (1.0 + tie_epsilon_rel):  # Check the other side
            max_area_2 = -area
            p4 = i

    return wp.vec4i(p1, p2, p3, p4)


@wp.func
def remove_zero_length_edges(loop: wp.array[wp.vec2], loop_count: int, eps: float) -> int:
    """
    Remove zero-length edges from a polygon loop.

    Args:
        loop: Array of loop vertices (2D).
        loop_count: Number of vertices in the loop.
        eps: Epsilon threshold for considering edges as zero-length.

    Returns:
        New number of vertices in the cleaned loop.
    """
    if loop_count < 2:
        return 0

    write_idx = int(0)

    for read_idx in range(1, loop_count):
        diff = loop[read_idx] - loop[write_idx]
        if wp.length_sq(diff) > eps:
            write_idx += 1
            loop[write_idx] = loop[read_idx]

    # Handle loop closure
    if write_idx > 0:
        diff = loop[write_idx] - loop[0]
        if wp.length_sq(diff) < eps:
            new_loop_count = write_idx
        else:
            new_loop_count = write_idx + 1
    else:
        new_loop_count = write_idx + 1

    if new_loop_count < 2:
        new_loop_count = 0

    return new_loop_count


@wp.func
def add_avoid_duplicates_vec2(arr: wp.array[wp.vec2], arr_count: int, vec: wp.vec2, eps: float) -> tuple[int, bool]:
    """
    Add a vector to an array, avoiding duplicates.

    Args:
        arr: Array to add to.
        arr_count: Current number of elements in the array.
        vec: Vector to add.
        eps: Epsilon threshold for duplicate detection.

    Returns:
        Tuple of (new_count, was_added) where was_added is True if point was added
    """
    # Check for duplicates. If the new vertex 'vec' is too close to the first or last existing vertex, ignore it.
    # This is a simple reduction step to avoid redundant points.
    if arr_count > 0:
        if wp.length_sq(arr[0] - vec) < eps:
            return arr_count, False

    if arr_count > 1:
        if wp.length_sq(arr[arr_count - 1] - vec) < eps:
            return arr_count, False

    arr[arr_count] = vec
    return arr_count + 1, True


@wp.func
def face_clip_reference_face(geom: Any, normal: wp.vec3):
    """Return the box or plane face whose outward normal is closest to ``normal``, in the shape's frame.

    Returns ``(valid, face_normal, axis_u, axis_v, face_offset, half_extent_u, half_extent_v)``;
    ``valid`` is false when ``normal`` is farther than the face-clip angle from that face normal.
    """
    axis = int(2)
    face_offset = float(0.0)
    if geom.shape_type == int(GeoType.BOX):
        abs_x = wp.abs(normal[0])
        abs_y = wp.abs(normal[1])
        abs_z = wp.abs(normal[2])
        if abs_x >= abs_y and abs_x >= abs_z:
            axis = 0
        elif abs_y >= abs_z:
            axis = 1
        face_offset = geom.scale[axis]
    axis_u_index = (axis + 1) % 3
    axis_v_index = (axis + 2) % 3
    sign = 1.0 if normal[axis] >= 0.0 else -1.0
    face_normal = wp.vec3(
        sign if axis == 0 else 0.0,
        sign if axis == 1 else 0.0,
        sign if axis == 2 else 0.0,
    )
    axis_u = wp.vec3(
        1.0 if axis_u_index == 0 else 0.0,
        1.0 if axis_u_index == 1 else 0.0,
        1.0 if axis_u_index == 2 else 0.0,
    )
    axis_v = wp.vec3(
        1.0 if axis_v_index == 0 else 0.0,
        1.0 if axis_v_index == 1 else 0.0,
        1.0 if axis_v_index == 2 else 0.0,
    )
    valid = wp.abs(normal[axis]) >= COS_FACE_CLIP_MAX_ANGLE
    return valid, face_normal, axis_u, axis_v, face_offset, geom.scale[axis_u_index], geom.scale[axis_v_index]


@wp.func
def face_clip_incident_face(
    points: wp.array[wp.vec3],
    indices: wp.array[wp.int32],
    scale: wp.vec3,
    axis_u: wp.vec3,
    axis_v: wp.vec3,
    face_normal: wp.vec3,
    offset: wp.vec3,
    face_vertices: wp.array[wp.vec3],
    polygon: wp.array[wp.vec3],
):
    """Write the hull's incident face as a convex (u, v, depth) polygon.

    Returns the vertex count and the incident plane's normal and a point on it, in (u, v, depth) coordinates.

    The incident face is the hull face through the deepest vertex most anti-parallel to the reference
    face, together with its coplanar vertices. Axes are in the hull's frame; ``offset`` holds the hull
    origin's (u, v, depth); ``face_vertices`` is scratch storage.
    """
    vertex_count = points.shape[0]
    deepest_index = int(0)
    deepest_depth = float(0.0)
    for i in range(vertex_count):
        depth = wp.dot(wp.cw_mul(points[i], scale), face_normal)
        if i == 0 or depth < deepest_depth:
            deepest_depth = depth
            deepest_index = i

    best_alignment = float(-1.0)
    plane_normal = face_normal
    plane_point = wp.cw_mul(points[deepest_index], scale)
    for t in range(indices.shape[0] // 3):
        i0 = int(indices[3 * t])
        i1 = int(indices[3 * t + 1])
        i2 = int(indices[3 * t + 2])
        if i0 != deepest_index and i1 != deepest_index and i2 != deepest_index:
            continue
        p0 = wp.cw_mul(points[i0], scale)
        n = wp.cross(wp.cw_mul(points[i1], scale) - p0, wp.cw_mul(points[i2], scale) - p0)
        length = wp.length(n)
        if length <= 0.0:
            continue
        alignment = wp.abs(wp.dot(n, face_normal)) / length
        if alignment > best_alignment:
            best_alignment = alignment
            plane_normal = n / length

    plane_uvd_normal = wp.vec3(
        wp.dot(plane_normal, axis_u), wp.dot(plane_normal, axis_v), wp.dot(plane_normal, face_normal)
    )
    plane_uvd_point = (
        wp.vec3(wp.dot(plane_point, axis_u), wp.dot(plane_point, axis_v), wp.dot(plane_point, face_normal)) + offset
    )

    face_count = int(0)
    start = wp.vec3(0.0)
    for i in range(vertex_count):
        p = wp.cw_mul(points[i], scale)
        if wp.abs(wp.dot(p - plane_point, plane_normal)) > FACE_CLIP_COPLANAR_DISTANCE:
            continue
        c = wp.vec3(wp.dot(p, axis_u), wp.dot(p, axis_v), wp.dot(p, face_normal)) + offset
        if face_count == 0 or c[0] < start[0] or (c[0] == start[0] and c[1] < start[1]):
            start = c
        if face_count < FACE_CLIP_PATCH_CAPACITY:
            face_vertices[face_count] = c
        face_count += 1

    if face_count > FACE_CLIP_PATCH_CAPACITY:
        # Too many face vertices to wrap: keep the extreme vertex along evenly spaced in-plane directions.
        polygon_count = int(0)
        for k in range(FACE_CLIP_PATCH_CAPACITY):
            angle = float(k) * FACE_CLIP_DIRECTION_STEP
            direction_u = wp.cos(angle)
            direction_v = wp.sin(angle)
            best = wp.vec3(0.0)
            best_dot = float(0.0)
            has_best = bool(False)
            for i in range(vertex_count):
                p = wp.cw_mul(points[i], scale)
                if wp.abs(wp.dot(p - plane_point, plane_normal)) > FACE_CLIP_COPLANAR_DISTANCE:
                    continue
                c = wp.vec3(wp.dot(p, axis_u), wp.dot(p, axis_v), wp.dot(p, face_normal)) + offset
                d = c[0] * direction_u + c[1] * direction_v
                if (
                    not has_best
                    or d > best_dot
                    or (d == best_dot and (c[0] < best[0] or (c[0] == best[0] and c[1] < best[1])))
                ):
                    best = c
                    best_dot = d
                    has_best = True
            if polygon_count > 0:
                du = best[0] - polygon[polygon_count - 1][0]
                dv = best[1] - polygon[polygon_count - 1][1]
                if du * du + dv * dv <= FACE_CLIP_MERGE_DISTANCE_SQ:
                    continue
                du = best[0] - polygon[0][0]
                dv = best[1] - polygon[0][1]
                if du * du + dv * dv <= FACE_CLIP_MERGE_DISTANCE_SQ:
                    continue
            polygon[polygon_count] = best
            polygon_count += 1
        return polygon_count, plane_uvd_normal, plane_uvd_point

    # Counter-clockwise Jarvis march; collinear vertices resolve to the farthest.
    current = start
    polygon_count = int(0)
    while polygon_count < FACE_CLIP_PATCH_CAPACITY:
        polygon[polygon_count] = current
        polygon_count += 1
        has_next = bool(False)
        next_point = current
        next_dist_sq = float(0.0)
        for i in range(face_count):
            c = face_vertices[i]
            du = c[0] - current[0]
            dv = c[1] - current[1]
            dist_sq = du * du + dv * dv
            if dist_sq <= FACE_CLIP_MERGE_DISTANCE_SQ:
                continue
            if not has_next:
                has_next = True
                next_point = c
                next_dist_sq = dist_sq
                continue
            cross = (next_point[0] - current[0]) * dv - (next_point[1] - current[1]) * du
            if cross < -FACE_CLIP_MIN_DOUBLE_AREA or (cross <= FACE_CLIP_MIN_DOUBLE_AREA and dist_sq > next_dist_sq):
                next_point = c
                next_dist_sq = dist_sq
        if not has_next:
            break
        du_start = next_point[0] - start[0]
        dv_start = next_point[1] - start[1]
        if du_start * du_start + dv_start * dv_start <= FACE_CLIP_MERGE_DISTANCE_SQ:
            break
        current = next_point
    return polygon_count, plane_uvd_normal, plane_uvd_point


@wp.func
def face_clip_side(
    src: wp.array[wp.vec3],
    src_count: int,
    dst: wp.array[wp.vec3],
    coordinate: int,
    sign: float,
    limit: float,
) -> int:
    """Sutherland-Hodgman clip of a (u, v, depth) polygon to ``sign * p[coordinate] <= limit``.

    Returns -1 without writing when the result would exceed ``dst``'s capacity.
    """
    if src_count <= 0:
        return src_count
    # Each kept vertex and each sign change emits one point; count them before writing.
    required = int(0)
    prev_dist = sign * src[src_count - 1][coordinate] - limit
    for i in range(src_count):
        cur_dist = sign * src[i][coordinate] - limit
        if (cur_dist <= 0.0) != (prev_dist <= 0.0):
            required += 1
        if cur_dist <= 0.0:
            required += 1
        prev_dist = cur_dist
    if required > dst.shape[0]:
        return -1
    dst_count = int(0)
    prev = src[src_count - 1]
    prev_dist = sign * prev[coordinate] - limit
    for i in range(src_count):
        cur = src[i]
        cur_dist = sign * cur[coordinate] - limit
        if (cur_dist <= 0.0) != (prev_dist <= 0.0):
            t = prev_dist / (prev_dist - cur_dist)
            dst[dst_count] = prev + t * (cur - prev)
            dst_count += 1
        if cur_dist <= 0.0:
            dst[dst_count] = cur
            dst_count += 1
        prev = cur
        prev_dist = cur_dist
    return dst_count


def create_build_manifold(
    support_func: Any,
    writer_func: Any,
    post_process_contact: Any,
    _support_funcs: Any = None,
    face_clip: bool = False,
):
    """
    Factory function to create manifold generation functions with a specific support mapping function.

    This factory creates two related functions for multi-contact manifold generation:
    - build_manifold_core: The core implementation that uses preallocated buffers
    - build_manifold: The main entry point that handles buffer allocation and result extraction

    Args:
        support_func: Support mapping function for shapes that takes
                     (geometry, direction, data_provider) and returns a support point
        writer_func: Function to write contact data (signature: (ContactData, writer_data) -> None)
        post_process_contact: Function to post-process contact data
        face_clip: Build box/plane versus convex-mesh manifolds by clipping the hull's contact patch
            against the box or plane reference face, falling back to perturbed support mapping.

    Returns:
        build_manifold function that generates up to 5 contact points between two shapes
        using perturbed support mapping and polygon clipping.
    """

    if _support_funcs is not None:
        _support_map_b = _support_funcs[0]
    else:
        _support_map_b = create_support_map_function(support_func)[0]

    @wp.func
    def extract_4_point_contact_manifolds(
        m_a: wp.array[wp.vec2],
        m_a_count: int,
        m_b: wp.array[wp.vec2],
        m_b_count: int,
        normal_local: wp.vec3,
        cross_vector_1: wp.vec3,
        cross_vector_2: wp.vec3,
        center_local: wp.vec3,
        projector_a: BodyProjector,
        projector_b: BodyProjector,
        orientation_a: wp.quat,
        position_a_world: wp.vec3,
        normal_world: wp.vec3,
        writer_data: Any,
        contact_template: Any,
        geom_a: Any,
        geom_b: Any,
        position_a: wp.vec3,
        position_b: wp.vec3,
        quaternion_a: wp.quat,
        quaternion_b: wp.quat,
    ) -> tuple[int, float]:
        """
        Extract up to 4 contact points from two convex contact polygons and write them immediately.

        All intermediate work (clipping, projectors) operates in shape A's local frame.
        Final contact points are transformed to world space before writing.

        Args:
            m_a: Contact polygon vertices for shape A (2D contact plane space, up to 5 points).
            m_a_count: Number of vertices in polygon A.
            m_b: Contact polygon vertices for shape B (2D contact plane space, up to 5 points, space for 10).
            m_b_count: Number of vertices in polygon B.
            normal_local: Collision normal in A-local frame.
            cross_vector_1: First tangent vector in A-local frame.
            cross_vector_2: Second tangent vector in A-local frame.
            center_local: Center point for back-projection in A-local frame.
            projector_a: Body projector for shape A (in A-local frame).
            projector_b: Body projector for shape B (in A-local frame).
            orientation_a: World orientation of shape A (for final transform).
            position_a_world: World position of shape A (for final transform).
            normal_world: Contact normal in world space (for output).
            writer_data: Data structure for contact writer.
            contact_template: Pre-packed ContactData with static fields.
            geom_a: Geometry data for shape A.
            geom_b: Geometry data for shape B.
            position_a: World position of shape A (for post_process_contact).
            position_b: World position of shape B (for post_process_contact).
            quaternion_a: Orientation of shape A (for post_process_contact).
            quaternion_b: Orientation of shape B (for post_process_contact).

        Returns:
            Tuple of (loop_count, normal_dot) where:
            - loop_count: Number of valid contact points written (0-4)
            - normal_dot: Absolute dot product of polygon normals
        """

        normal_dot = wp.abs(wp.dot(projector_a.normal, projector_b.normal))

        loop_count = trim_all_in_place(m_a, m_a_count, m_b, m_b_count)

        loop_count = remove_zero_length_edges(m_b, loop_count, EPS)

        if loop_count > 1:
            result = wp.vec4i()
            if loop_count > 4:
                result = approx_max_quadrilateral_area_with_calipers(m_b, loop_count)
                loop_count = 4
            else:
                result = wp.vec4i(0, 1, 2, 3)

            for i in range(loop_count):
                ia = int(result[i])

                # Back-project from 2D to 3D in A-local frame
                p_local = m_b[ia].x * cross_vector_1 + m_b[ia].y * cross_vector_2 + center_local

                a = body_projector_project(projector_a, p_local, normal_local)
                b = body_projector_project(projector_b, p_local, normal_local)
                contact_point_local = 0.5 * (a + b)
                signed_distance = wp.dot(b - a, normal_local)

                # Transform from A-local to world space
                contact_point_world = wp.quat_rotate(orientation_a, contact_point_local) + position_a_world

                contact_data = contact_template
                contact_data.contact_point_center = contact_point_world
                contact_data.contact_normal_a_to_b = normal_world
                contact_data.contact_distance = signed_distance
                contact_data.sort_sub_key = (contact_template.sort_sub_key << 3) | i

                contact_data = post_process_contact(
                    contact_data, geom_a, position_a, quaternion_a, geom_b, position_b, quaternion_b
                )
                writer_func(contact_data, writer_data, -1)
        else:
            normal_dot = 0.0
            loop_count = 0

        return loop_count, normal_dot

    @wp.func
    def build_face_clip_manifold(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        position_a_world: wp.vec3,
        relative_orientation_b: wp.quat,
        relative_position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        """Write up to 4 contacts from the convex mesh B's patch clipped to box or plane A's reference face.

        Writes nothing and returns 0 when the reference face does not explain the MPR/GJK contact.
        """
        valid, face_normal, axis_u, axis_v, face_offset, half_u, half_v = face_clip_reference_face(geom_a, normal)
        if not valid:
            return 0

        depth_max = contact_template.gap_sum + contact_template.margin_a + contact_template.margin_b
        offset = wp.vec3(
            wp.dot(relative_position_b, axis_u),
            wp.dot(relative_position_b, axis_v),
            wp.dot(relative_position_b, face_normal) - face_offset,
        )
        mesh = wp.mesh_get(unpack_mesh_ptr(geom_b.auxiliary))
        polygon = wp.zeros(shape=(FACE_CLIP_POLYGON_CAPACITY,), dtype=wp.vec3f)
        scratch = wp.zeros(shape=(FACE_CLIP_POLYGON_CAPACITY,), dtype=wp.vec3f)
        count, plane_normal, plane_point = face_clip_incident_face(
            mesh.points,
            mesh.indices,
            geom_b.scale,
            wp.quat_rotate_inv(relative_orientation_b, axis_u),
            wp.quat_rotate_inv(relative_orientation_b, axis_v),
            wp.quat_rotate_inv(relative_orientation_b, face_normal),
            offset,
            scratch,
            polygon,
        )
        count = face_clip_side(polygon, count, scratch, 0, 1.0, half_u)
        count = face_clip_side(scratch, count, polygon, 0, -1.0, half_u)
        count = face_clip_side(polygon, count, scratch, 1, 1.0, half_v)
        count = face_clip_side(scratch, count, polygon, 1, -1.0, half_v)
        if count < 0:
            return 0
        clipped_count = face_clip_side(polygon, count, scratch, 2, 1.0, depth_max)
        if clipped_count < 0:
            # Face vertices lie within the coplanarity tolerance, so their depths can cross the gap plane many
            # times; flattening them onto the incident plane makes the depth clip add at most one vertex.
            if wp.abs(plane_normal[2]) <= FACE_CLIP_MIN_PLANE_DEPTH_SLOPE:
                return 0
            for i in range(count):
                p = polygon[i]
                depth = (
                    plane_point[2]
                    - (plane_normal[0] * (p[0] - plane_point[0]) + plane_normal[1] * (p[1] - plane_point[1]))
                    / plane_normal[2]
                )
                polygon[i] = wp.vec3(p[0], p[1], depth)
            clipped_count = face_clip_side(polygon, count, scratch, 2, 1.0, depth_max)
            if clipped_count < 0:
                return 0
        count = clipped_count

        # Reduce to the deepest point, the farthest point, then the largest triangle and quadrilateral.
        chosen = wp.zeros(shape=(4,), dtype=wp.vec3f)
        chosen_count = int(0)
        for i in range(count):
            if chosen_count == 0 or scratch[i][2] < chosen[0][2]:
                chosen[0] = scratch[i]
                chosen_count = 1
        if chosen_count == 0 or chosen[0][2] > wp.dot(p_b - p_a, normal) + FACE_CLIP_DEPTH_TOLERANCE:
            return 0

        p0 = chosen[0]
        best_dist_sq = float(FACE_CLIP_MERGE_DISTANCE_SQ)
        for i in range(count):
            du = scratch[i][0] - p0[0]
            dv = scratch[i][1] - p0[1]
            dist_sq = du * du + dv * dv
            if dist_sq > best_dist_sq:
                best_dist_sq = dist_sq
                chosen[1] = scratch[i]
                chosen_count = 2
        if chosen_count == 2:
            eu = chosen[1][0] - p0[0]
            ev = chosen[1][1] - p0[1]
            best_area = float(FACE_CLIP_MIN_DOUBLE_AREA)
            side = float(0.0)
            for i in range(count):
                area = eu * (scratch[i][1] - p0[1]) - ev * (scratch[i][0] - p0[0])
                if wp.abs(area) > best_area:
                    best_area = wp.abs(area)
                    side = wp.sign(area)
                    chosen[2] = scratch[i]
                    chosen_count = 3
            if chosen_count == 3:
                best_area = float(FACE_CLIP_MIN_DOUBLE_AREA)
                for i in range(count):
                    area = -side * (eu * (scratch[i][1] - p0[1]) - ev * (scratch[i][0] - p0[0]))
                    if area > best_area:
                        best_area = area
                        chosen[3] = scratch[i]
                        chosen_count = 4

        normal_world = wp.quat_rotate(orientation_a, face_normal)
        position_b_ws = wp.quat_rotate(orientation_a, relative_position_b) + position_a_world
        quaternion_b_ws = orientation_a * relative_orientation_b
        for slot in range(chosen_count):
            point = chosen[slot]
            center_local = point[0] * axis_u + point[1] * axis_v + (face_offset + 0.5 * point[2]) * face_normal
            contact_data = contact_template
            contact_data.contact_point_center = wp.quat_rotate(orientation_a, center_local) + position_a_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = point[2]
            contact_data.sort_sub_key = (contact_template.sort_sub_key << 3) | slot
            contact_data = post_process_contact(
                contact_data, geom_a, position_a_world, orientation_a, geom_b, position_b_ws, quaternion_b_ws
            )
            writer_func(contact_data, writer_data, -1)
        return chosen_count

    @wp.func
    def build_manifold(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        position_a_world: wp.vec3,
        relative_orientation_b: wp.quat,
        relative_position_b: wp.vec3,
        p_a: wp.vec3,
        p_b: wp.vec3,
        normal: wp.vec3,
        data_provider: Any,
        writer_data: Any,
        contact_template: ContactData,
    ) -> int:
        """
        Build a contact manifold between two convex shapes and write contacts directly.

        All intermediate work operates in shape A's local frame to avoid redundant
        quaternion transforms. Final contact points are transformed to world space
        before writing.

        Args:
            geom_a: Geometry data for the first shape.
            geom_b: Geometry data for the second shape.
            orientation_a: World orientation of shape A (for final world-space transform).
            position_a_world: World position of shape A (for final world-space transform).
            relative_orientation_b: Orientation of B relative to A.
            relative_position_b: Position of B relative to A (in A-local frame).
            p_a: Anchor contact point on shape A in A-local frame (from GJK/MPR).
            p_b: Anchor contact point on shape B in A-local frame (from GJK/MPR).
            normal: Contact normal in A-local frame pointing from A to B.
            data_provider: Support mapping data provider for shape queries.
            writer_data: Data structure for contact writer.
            contact_template: Pre-packed ContactData with static fields.

        Returns:
            Number of valid contact points written (0-5).
        """
        if wp.static(face_clip):
            if (
                geom_a.shape_type == int(GeoType.BOX) or geom_a.shape_type == int(GeoType.PLANE)
            ) and geom_b.shape_type == int(GeoType.CONVEX_MESH):
                face_clip_count = build_face_clip_manifold(
                    geom_a,
                    geom_b,
                    orientation_a,
                    position_a_world,
                    relative_orientation_b,
                    relative_position_b,
                    p_a,
                    p_b,
                    normal,
                    writer_data,
                    contact_template,
                )
                if face_clip_count > 0:
                    return face_clip_count

        # Precomputed cos/sin for 5 evenly spaced pentagonal angles (0, 72, 144, 216, 288 deg).
        PENT_COS_0 = float(1.0)
        PENT_SIN_0 = float(0.0)
        PENT_COS_1 = wp.static(math.cos(2.0 * math.pi / 5.0))
        PENT_SIN_1 = wp.static(math.sin(2.0 * math.pi / 5.0))
        PENT_COS_2 = wp.static(math.cos(4.0 * math.pi / 5.0))
        PENT_SIN_2 = wp.static(math.sin(4.0 * math.pi / 5.0))
        PENT_COS_3 = wp.static(math.cos(6.0 * math.pi / 5.0))
        PENT_SIN_3 = wp.static(math.sin(6.0 * math.pi / 5.0))
        PENT_COS_4 = wp.static(math.cos(8.0 * math.pi / 5.0))
        PENT_SIN_4 = wp.static(math.sin(8.0 * math.pi / 5.0))

        a_count = int(0)
        b_count = int(0)

        # Orthonormal basis from the collision normal (in A-local frame).
        tangent_a, tangent_b = orthonormal_basis(normal)

        plane_tracker_a = IncrementalPlaneTracker()
        plane_tracker_b = IncrementalPlaneTracker()

        center = 0.5 * (p_a + p_b)

        # Allocate buffers: 5 for A, up to 10 for B (5 + clipping headroom)
        b_buffer = wp.zeros(shape=(10,), dtype=wp.vec2f)
        a_buffer = wp.array(ptr=b_buffer.ptr + wp.uint64(5 * 8), shape=(5,), dtype=wp.vec2f)

        # --- Step 1: Find Contact Polygons using Perturbed Support Mapping ---
        # Shape A: support_func returns points in A-local frame directly, no quat_rotate needed.
        # Shape B: pre-transform basis to B-local, then transform results back to A-local.
        local_normal_b = wp.quat_rotate_inv(relative_orientation_b, -normal)
        local_ta_b = wp.quat_rotate_inv(relative_orientation_b, -tangent_a)
        local_tb_b = wp.quat_rotate_inv(relative_orientation_b, -tangent_b)

        for e in range(5):
            c = PENT_COS_0
            s = PENT_SIN_0
            if e == 1:
                c = PENT_COS_1
                s = PENT_SIN_1
            elif e == 2:
                c = PENT_COS_2
                s = PENT_SIN_2
            elif e == 3:
                c = PENT_COS_3
                s = PENT_SIN_3
            elif e == 4:
                c = PENT_COS_4
                s = PENT_SIN_4

            cos_tilt = COS_TILT_ANGLE
            c_sin = c * SIN_TILT_ANGLE
            s_sin = s * SIN_TILT_ANGLE

            # Shape A: direction and result both in A-local frame, zero quaternion ops.
            dir_a = normal * cos_tilt + c_sin * tangent_a + s_sin * tangent_b
            pt_a_3d = support_func(geom_a, dir_a, data_provider)
            projected_a = pt_a_3d - center
            pt_a_2d = wp.vec2(wp.dot(tangent_a, projected_a), wp.dot(tangent_b, projected_a))
            a_count, was_added_a = add_avoid_duplicates_vec2(a_buffer, a_count, pt_a_2d, EPS)
            if was_added_a:
                plane_tracker_a = update_incremental_plane_tracker(plane_tracker_a, pt_a_3d, a_count - 1)

            # Shape B: direction in B-local, result transformed to A-local.
            local_dir_b = local_normal_b * cos_tilt + c_sin * local_ta_b + s_sin * local_tb_b
            pt_b_local = support_func(geom_b, local_dir_b, data_provider)
            pt_b_3d = wp.quat_rotate(relative_orientation_b, pt_b_local) + relative_position_b
            projected_b = pt_b_3d - center
            pt_b_2d = wp.vec2(wp.dot(tangent_a, projected_b), wp.dot(tangent_b, projected_b))
            b_count, was_added_b = add_avoid_duplicates_vec2(b_buffer, b_count, pt_b_2d, EPS)
            if was_added_b:
                plane_tracker_b = update_incremental_plane_tracker(plane_tracker_b, pt_b_3d, b_count - 1)

        # World-space normal (computed once for all output contacts)
        normal_world = wp.quat_rotate(orientation_a, normal)

        # World-space positions/orientations for post_process_contact
        position_a_ws = position_a_world
        position_b_ws = wp.quat_rotate(orientation_a, relative_position_b) + position_a_world
        quaternion_a_ws = orientation_a
        quaternion_b_ws = orientation_a * relative_orientation_b

        if a_count < 2 or b_count < 2:
            count_out = 0
            normal_dot = 0.0
        else:
            projector_a, projector_b = create_body_projectors(plane_tracker_a, p_a, plane_tracker_b, p_b, normal)

            if excess_normal_deviation(normal, projector_a.normal) or excess_normal_deviation(
                normal, projector_b.normal
            ):
                count_out = 0
                normal_dot = 0.0
            else:
                num_manifold_points, normal_dot = extract_4_point_contact_manifolds(
                    a_buffer,
                    a_count,
                    b_buffer,
                    b_count,
                    normal,
                    tangent_a,
                    tangent_b,
                    center,
                    projector_a,
                    projector_b,
                    orientation_a,
                    position_a_world,
                    normal_world,
                    writer_data,
                    contact_template,
                    geom_a,
                    geom_b,
                    position_a_ws,
                    position_b_ws,
                    quaternion_a_ws,
                    quaternion_b_ws,
                )
                count_out = wp.min(num_manifold_points, 4)

        if should_include_deepest_contact(normal_dot) or count_out == 0:
            deepest_center_local = 0.5 * (p_a + p_b)
            deepest_signed_distance = wp.dot(p_b - p_a, normal)

            deepest_center_world = wp.quat_rotate(orientation_a, deepest_center_local) + position_a_world

            contact_data = contact_template
            contact_data.contact_point_center = deepest_center_world
            contact_data.contact_normal_a_to_b = normal_world
            contact_data.contact_distance = deepest_signed_distance
            contact_data.sort_sub_key = (contact_template.sort_sub_key << 3) | count_out

            contact_data = post_process_contact(
                contact_data, geom_a, position_a_ws, quaternion_a_ws, geom_b, position_b_ws, quaternion_b_ws
            )
            writer_func(contact_data, writer_data, -1)

            count_out += 1

        return count_out

    return build_manifold
