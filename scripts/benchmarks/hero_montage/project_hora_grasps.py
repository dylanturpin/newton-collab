# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Project cached grasps onto nonpenetrating fingertip contacts before simulation."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    folder = Path(__file__).parent / "assets/hora"
    root = ET.parse(folder / "hand.urdf").getroot()
    order = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
    indices = {f"joint_{number}.0": i for i, number in enumerate(order)}
    joints = []
    bounds = [None] * 16
    for item in root.findall("joint"):
        origin = item.find("origin")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", list(map(float, origin.get("rpy", "0 0 0").split()))).as_matrix()
        transform[:3, 3] = list(map(float, origin.get("xyz", "0 0 0").split()))
        index = indices.get(item.get("name"))
        axis = np.zeros(3) if index is None else np.array(list(map(float, item.find("axis").get("xyz").split())))
        joints.append((item.find("parent").get("link"), item.find("child").get("link"), transform, index, axis))
        if index is not None:
            limit = item.find("limit")
            bounds[index] = (float(limit.get("lower")), float(limit.get("upper")))
    tips = []
    for link in root.findall("link"):
        if not link.get("name").endswith("_tip"):
            continue
        collision = link.find("collision")
        center = np.array([*map(float, collision.find("origin").get("xyz").split()), 1])
        tips.append((link.get("name"), center, float(collision.find("geometry/sphere").get("radius"))))
    training_frame = np.eye(4)
    training_frame[:3, :3] = (Rotation.from_euler("y", -np.pi / 2) * Rotation.from_euler("x", np.pi / 2)).as_matrix()
    training_frame[2, 3] = 0.5
    raw = np.load(folder / "grasps.npy")
    output = raw.copy()
    evidence = []
    for i in range(3):
        cached = raw[i]
        cube = Rotation.from_quat(cached[19:])

        def gaps(q, cube=cube, cached=cached):
            transforms = {"base_link": training_frame}
            for parent, child, origin, index, axis in joints:
                rotation = np.eye(4)
                if index is not None:
                    rotation[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                transforms[child] = transforms[parent] @ origin @ rotation
            points = np.array([(transforms[name] @ center)[:3] for name, center, _ in tips])
            distance = np.abs(cube.inv().apply(points - cached[16:19])) - 0.032
            signed = np.linalg.norm(np.maximum(distance, 0), axis=1) + np.minimum(distance.max(axis=1), 0)
            return signed - np.array([radius for _, _, radius in tips])

        initial = gaps(cached[:16])
        active = initial < 0.003
        result = minimize(
            lambda q, cached=cached: np.sum((q - cached[:16]) ** 2),
            np.clip(cached[:16], np.array(bounds)[:, 0], np.array(bounds)[:, 1]),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": lambda q: gaps(q) - 0.00015},
                {"type": "ineq", "fun": lambda q, active=active: 0.00065 - gaps(q)[active]},
            ],
            options={"maxiter": 250, "ftol": 1e-12},
        )
        assert result.success, result.message
        assert gaps(result.x).min() >= 0.0001
        output[i, :16] = result.x
        evidence.append(
            {
                "cache_index": i,
                "original_tip_gaps_mm": (initial * 1000).tolist(),
                "projected_tip_gaps_mm": (gaps(result.x) * 1000).tolist(),
                "joint_adjustments_radians": (result.x - cached[:16]).tolist(),
                "object_pose_changed": False,
            }
        )
    np.save(folder / "grasps-projected.npy", output)
    (folder / "grasp-projection.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
