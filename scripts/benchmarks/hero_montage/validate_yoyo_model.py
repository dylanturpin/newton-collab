# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the actual string kernel's passive drop/return and energy budget."""

import json

import numpy as np
import warp as wp
from yoyo_task import string_force


def main():
    wp.init()
    device = "cpu"
    mass, inertia, radius, length = 0.065, 0.000064, 0.012, 0.62
    dt = 0.00125
    z, velocity, spin = 0.82, 0.0, 0.0
    poses = wp.array([[0, 0, 1, 0, 0, 0, 1], [0, 0, z, 0, 0, 0, 1]], dtype=wp.transform, device=device)
    velocities = wp.zeros(2, dtype=wp.spatial_vector, device=device)
    forces = wp.zeros_like(velocities)
    com = wp.zeros(2, dtype=wp.vec3, device=device)
    winding = wp.array([-(length - 0.18) / radius], dtype=float, device=device)
    metrics = wp.zeros(4, dtype=float, device=device)
    heights, energies, tensions = [], [], []
    for _ in range(round(4 / dt)):
        poses.assign([[0, 0, 1, 0, 0, 0, 1], [0, 0, z, 0, 0, 0, 1]])
        velocities.assign([[0, 0, 0, 0, 0, 0], [0, 0, velocity, 0, spin, 0]])
        forces.zero_()
        wp.launch(
            string_force,
            dim=1,
            inputs=[poses, velocities, forces, com, winding, metrics, 0, 1, wp.vec3(), radius, length, dt],
            device=device,
        )
        wrench = forces.numpy()[1]
        velocity += (float(wrench[2]) / mass - 9.81) * dt
        spin += float(wrench[4]) / inertia * dt
        z += velocity * dt
        q = float(winding.numpy()[0])
        extension = max(0, 1 - z - (length - radius * np.sqrt(q * q + 0.15**2)))
        heights.append(z)
        energies.append(0.5 * mass * velocity**2 + 0.5 * inertia * spin**2 + mass * 9.81 * z + 0.5 * 250 * extension**2)
        tensions.append(float(metrics.numpy()[0]))
    bottom = int(np.argmin(heights[: int(1.5 / dt)]))
    returned = max(heights[bottom : int(3 / dt)])
    report = {
        "bottom_height_m": heights[bottom],
        "return_height_m": returned,
        "maximum_energy_gain_J": max(energies) - energies[0],
        "minimum_tension_N": min(tensions),
        "recovered_drop_fraction": (returned - heights[bottom]) / (0.82 - heights[bottom]),
    }
    assert report["maximum_energy_gain_J"] < 0.005, report
    assert min(tensions) >= 0 and heights[bottom] < 0.39 and report["recovered_drop_fraction"] > 0.75, report
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
