# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fit Shadow thumb targets to a CUDA force-reference recording.

MuJoCo supplies MJCF mesh transforms and forward kinematics only. This authoring
tool never integrates physics or writes object poses. The resulting targets must
be validated in a separate CUDA FPGS run with the reference force disabled.
Run with ``uv run --with mujoco python fit_lighter_thumb.py ...``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    import mujoco
    from scipy.optimize import differential_evolution, least_squares
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Existing grasp configuration to extend")
    parser.add_argument("--lid-contact", type=float, nargs=3, default=[0.022, -0.012, 0.052])
    parser.add_argument("--contact-body", default="rh_thdistal", choices=("rh_thdistal", "rh_thmiddle"))
    args = parser.parse_args()
    assets = json.loads(args.assets.read_text())
    config = json.loads(args.output.read_text())
    model = mujoco.MjModel.from_xml_path(str(Path(assets["wonik_allegro"]).parent / "shadow_hand/right_hand.xml"))
    data = mujoco.MjData(model)
    palm = model.body("rh_palm").id
    tip = model.body(args.contact_body).id
    for j in range(model.njnt):
        name = model.joint(j).name
        data.qpos[model.jnt_qposadr[j]] = (
            0.0
            if "WRJ" in name or name.endswith("J4")
            else 0.1
            if name.endswith("LFJ5")
            else {"3": 1.45, "2": 1.35, "1": 0.65}.get(name[-1], 0.0)
        )
        if name in config.get("closed_grasp", {}):
            data.qpos[model.jnt_qposadr[j]] = config["closed_grasp"][name]
    names = [f"rh_THJ{i}" for i in (5, 4, 3, 2, 1)]
    indices = [model.joint(name).qposadr[0] for name in names]
    bounds = np.array([model.joint(name).range for name in names]).T
    world = next(
        w for w in json.loads((args.reference_run / "model-summary.json").read_text())["worlds"] if w.get("lighter")
    )
    recording = np.load(args.reference_run / "trace.npz")
    poses, fps = recording["poses"], int(recording["fps"])
    surfaces = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != tip or not model.geom_contype[geom]:
            continue
        kind = int(model.geom_type[geom])
        if args.contact_body == "rh_thdistal" and kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            surface = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        elif args.contact_body == "rh_thmiddle" and kind in (
            mujoco.mjtGeom.mjGEOM_SPHERE,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        ):
            radius = model.geom_size[geom, 0]
            half_length = model.geom_size[geom, 1] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0
            surface = []
            for polar in np.linspace(0, np.pi, 17):
                for azimuth in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                    z = radius * np.cos(polar)
                    surface.append(
                        [
                            radius * np.sin(polar) * np.cos(azimuth),
                            radius * np.sin(polar) * np.sin(azimuth),
                            z + np.copysign(half_length, z),
                        ]
                    )
            surface = np.asarray(surface)
        else:
            continue
        qw, qx, qy, qz = model.geom_quat[geom]
        surfaces.append(Rotation.from_quat([qx, qy, qz, qw]).apply(surface) + model.geom_pos[geom])
    vertices = np.vstack(surfaces)

    def thumb_vertices(x):
        data.qpos[indices] = x
        mujoco.mj_kinematics(model, data)
        return (
            data.xmat[palm].reshape(3, 3).T
            @ (data.xmat[tip].reshape(3, 3) @ vertices.T + (data.xpos[tip] - data.xpos[palm])[:, None])
        ).T

    lid_local = np.asarray(args.lid_contact)

    def contact_target(t):
        frame = round(t * fps)
        lid = poses[frame, world["body_start"] + world["lighter_lid"]]
        palm_pose = poses[frame, world["body_start"] + world["palm_body"]]
        point = lid[:3] + Rotation.from_quat(lid[3:]).apply(lid_local)
        return Rotation.from_quat(palm_pose[3:]).inv().apply(point - palm_pose[:3])

    fit = differential_evolution(
        lambda x: np.linalg.norm(thumb_vertices(x) - contact_target(2.5), axis=1).min(),
        list(zip(*bounds, strict=True)),
        seed=7,
        popsize=9,
        maxiter=150,
        polish=True,
    )
    vertex = np.linalg.norm(thumb_vertices(fit.x) - contact_target(2.5), axis=1).argmin()
    x = fit.x
    rest = config["waypoints"][0]["joints"]
    waypoints = [{"time": 0, "joints": rest}, {"time": 1.5, "joints": rest}]
    for sample_time in np.arange(1.7, 3.51, 0.10):
        t = round(float(sample_time), 2)
        target = contact_target(t)
        fit = least_squares(
            lambda value, goal=target, previous=x: np.r_[
                thumb_vertices(value)[vertex] - goal, 0.00001 * (value - previous)
            ],
            x,
            bounds=bounds,
            max_nfev=600,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        x = fit.x
        error = float(np.linalg.norm(thumb_vertices(x)[vertex] - target))
        waypoints.append(
            {
                "time": t,
                "joints": dict(zip(names, x.tolist(), strict=True)),
                "surface_target": target.tolist(),
                "position_error_m": error,
            }
        )
    waypoints.extend({"time": t, "joints": dict.fromkeys(names, 0.0)} for t in (4.2, 15))
    config.update(
        opening_finger="thumb",
        contact_vertex=vertices[vertex].tolist(),
        contact_body=args.contact_body,
        lid_contact_local=lid_local.tolist(),
        force_fitted=False,
        waypoints=waypoints,
        force_reference_trace_sha256=hashlib.sha256((args.reference_run / "trace.npz").read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
