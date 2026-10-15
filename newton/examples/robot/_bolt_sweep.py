# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sweep a physical pile of fasteners into a collection bin."""

import numpy as np
import warp as wp

from ._tabletop_assets import COLORS, collision_bounds, import_object
from ._tabletop_common import box, plan_ik, tf


def populate(b, info, ee, down, device):
    # Sweep spare hardware off the elevated work deck into a lower bin.
    box(b, (0, 0, 0.13), (0.26, 0.008, 0.060), (0.43, 0.49, 0.53), ee, "bench_scraper")
    for vi in range(6):
        b.joint_target_ke[vi], b.joint_target_kd[vi] = 6000, 180
        b.joint_effort_limit[vi] = 150
    info["integral_drive_count"] = 6
    poses = [
        (0, (0.18, -0.72, 0.38), None),
        (1.5, (0.18, -0.72, 0.263), None),
        (3.0, (0.18, -0.47, 0.263), None),
        (9.0, (0.18, 0.44, 0.263), None),
        (10.0, (0.18, 0.44, 0.46), None),
        (15, (0.18, -0.50, 0.35), None),
    ]
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.13), poses, device, desired_rotation=down, waypoint_dt=0.12)
    info["dense_waypoints"] = True
    info["pile_bodies"] = []
    info["pile_piece_centers"] = []

    spawn_bounds = []
    for j in range(96):
        name = "hex_bolt" if j % 8 < 6 else "hex_nut" if j % 8 == 6 else "washer"
        rotation = wp.quat_rpy(0.2 * (j % 3), 0.45, j * 0.71)
        low, high = collision_bounds(b, name, rotation)
        center = np.array((0.18 + ((j % 6) - 2.5) * 0.080, -0.20 + (((j // 6) % 8) - 3.5) * 0.080, 0.0))
        p = center - (low + high) / 2
        p[2] = 0.206 + (j // 48) * 0.080 - low[2]
        bounds = (p + low, p + high)
        assert all(
            np.any(bounds[0] > other[1] + 0.002) or np.any(other[0] > bounds[1] + 0.002) for other in spawn_bounds
        )
        spawn_bounds.append(bounds)
        [body] = import_object(
            b,
            name,
            tf(p, rotation),
            free_root=True,
            finish_color=COLORS[int(np.random.default_rng(20261006 + j).integers(3))],
        ).values()
        info["pile_bodies"].append(body)
        local_low, local_high = collision_bounds(b, name, wp.quat_identity())
        info["pile_piece_centers"].append(((local_low + local_high) / 2).tolist())
    box(b, (0.18, -0.22, 0.100), (0.27, 0.42, 0.100), (0.29, 0.34, 0.38), label="hardware_work_deck")
    b.shape_material_mu[-1] = 0.20
    for x in (-0.10, 0.46):
        box(b, (x, -0.22, 0.235), (0.008, 0.42, 0.035), COLORS[0])
        box(b, (x, 0.46, 0.080), (0.008, 0.26, 0.080), COLORS[1], label="collection_bin_side")
    box(b, (0.18, 0.46, 0.008), (0.27, 0.26, 0.008), (0.60, 0.67, 0.69), label="collection_bin_floor")
    box(b, (0.18, 0.72, 0.080), (0.28, 0.008, 0.080), COLORS[1], label="collection_bin_back")
    info["pile_deck_height"] = 0.20
    info["pile_required_count"] = 90
    # Bounds come from the inner faces of the physical bin walls/floor.
    info["pile_bin_bounds"] = [[-0.092, 0.20, 0.016], [0.452, 0.712, 0.16]]
    info["tracked_body"] = info["pile_bodies"][0]
    info["task"] = "Sweep a mixed pile of proc-gen nuts, bolts and washers off a raised deck into a collection bin"
