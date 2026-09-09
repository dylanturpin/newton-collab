# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolate twist resistance in a force-controlled, two-jaw pinch.

The jaws translate along X and apply 10 N each. Gravity, angular damping, and
rolling friction are disabled. After settling, a 0.002 N m torque about X tests
whether the actual contact witnesses constrain spin around the pinch axis.
"""

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton


def _build_pinch(shape, device, *, mu=0.8, mu_torsional=0.0, half_height=0.03):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    radius = 0.02
    extent = radius + (half_height if shape == "capsule_end" else 0.0)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=0.0, gap=0.001, margin=0.0
    )
    jaws = []
    for sign in (-1.0, 1.0):
        pose = wp.transform(wp.vec3(sign * (extent + 0.005), 0, 0), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.001))
        builder.add_shape_box(body, hx=0.005, hy=0.06, hz=0.06, cfg=cfg)
        joint = builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, parent_xform=pose)
        builder.add_articulation([joint])
        jaws.append(body)
    # A fixed, isotropic inertia makes the free-spin prediction exact across
    # all geometries and prevents gyroscopic coupling from obscuring the mode.
    inertia = 0.4 * radius * radius
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3) * inertia))
    if shape == "sphere":
        builder.add_shape_sphere(body, radius=radius, cfg=cfg)
    elif shape.startswith("capsule"):
        rotation = wp.quat_identity()
        if shape == "capsule_end":
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)
        builder.add_shape_capsule(
            body, radius=radius, half_height=half_height, xform=wp.transform(wp.vec3(0), rotation), cfg=cfg
        )
    elif shape == "box":
        builder.add_shape_box(body, hx=radius, hy=radius, hz=radius, cfg=cfg)
    else:
        raise ValueError(shape)
    return builder.finalize(device=device), jaws, body, inertia


def _contact_geometry(model, state, contacts, body):
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    p0 = contacts.rigid_contact_point0.numpy()[:count]
    p1 = contacts.rigid_contact_point1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    margin0 = contacts.rigid_contact_margin0.numpy()[:count]
    margin1 = contacts.rigid_contact_margin1.numpy()[:count]
    pose = wp.transform(*state.body_q.numpy()[body])
    com = model.body_com.numpy()[body]
    points, angular_rows = [], []
    for c in range(count):
        if shape0[c] >= 0 and shape_body[shape0[c]] == body:
            p = p0[c]
            offset = margin0[c] * normals[c]
        elif shape1[c] >= 0 and shape_body[shape1[c]] == body:
            p = p1[c]
            offset = -margin1[c] * normals[c]
        else:
            continue
        # Sphere/capsule witnesses use their center or medial axis plus a
        # radius in the contact margin. Include it just as the solver does.
        r = np.asarray(wp.transform_vector(pose, wp.vec3(*(p - com)))) + offset
        surface_point = p + np.asarray(wp.transform_vector(wp.transform_inverse(pose), wp.vec3(*offset)))
        n = normals[c]
        t0 = np.cross(n, [0, 0, 1])
        if np.linalg.norm(t0) < 1.0e-6:
            t0 = np.cross(n, [0, 1, 0])
        t0 /= np.linalg.norm(t0)
        t1 = np.cross(n, t0)
        angular_rows.extend([np.cross(r, t0), np.cross(r, t1)])
        points.append(surface_point.tolist())
    matrix = np.asarray(angular_rows)
    singular_values = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.zeros(3)
    return {
        "contacts": len(points),
        "body_local_points": points,
        "angular_rank": int(np.count_nonzero(singular_values > 1.0e-6)),
        "angular_singular_values": singular_values.tolist(),
        "pinch_axis_lever_arm": float(np.linalg.norm(matrix[:, 0])) if len(matrix) else 0.0,
    }


def _run_pinch(
    shape, device, *, solver_name="fpgs", mu=0.8, mu_torsional=0.0, iterations=64, mode="split", half_height=0.03
):
    with wp.ScopedDevice(device):
        model, jaws, body, inertia = _build_pinch(
            shape, device, mu=mu, mu_torsional=mu_torsional, half_height=half_height
        )
        if solver_name == "fpgs":
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode=mode,
                pgs_iterations=iterations,
                pgs_beta=0.05,
                angular_damping=0.0,
                dense_max_constraints=64,
                mf_max_constraints=64,
            )
        else:
            solver = newton.solvers.SolverXPBD(model, iterations=iterations, angular_damping=0.0)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32, broad_phase="nxn")
        contacts = pipeline.contacts()
        source, target = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, source)
        dt, settle, steps, torque = 0.005, 60, 100, 0.002
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[jaws[0], 0], forces[jaws[1], 0] = 10.0, -10.0
        omega, normal_loads, contact_counts = [], [], []
        geometry = None
        for step in range(settle + steps):
            forces[body, 3] = torque if step >= settle else 0.0
            source.body_f.assign(forces)
            pipeline.collide(source, contacts)
            if step == settle:
                geometry = _contact_geometry(model, source, contacts, body)
            solver.step(source, target, control, contacts, dt)
            source, target = target, source
            if step >= settle:
                omega.append(float(source.body_qd.numpy()[body, 3]))
                contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
                if solver_name == "fpgs":
                    load = 0.0
                    paths = solver.contact_path.numpy()
                    slots = solver.contact_slot.numpy()
                    dense, mf = solver.impulses.numpy(), solver.mf_impulses.numpy()
                    for c in range(contact_counts[-1]):
                        if slots[c] >= 0:
                            load += (dense if paths[c] == 0 else mf)[0, slots[c]] / dt
                    normal_loads.append(float(load))
        return {
            "shape": shape,
            "solver": solver_name,
            "device": str(device),
            "warp_version": wp.__version__,
            "mode": mode,
            "mu": mu,
            "mu_torsional_m": mu_torsional,
            "iterations": iterations,
            "capsule_half_height_m": half_height,
            "geometry": geometry,
            "final_omega_x_rad_s": omega[-1],
            "predicted_free_omega_x_rad_s": torque * steps * dt / inertia,
            "angle_x_rad": float(np.sum(omega) * dt),
            "mean_normal_load_N": float(np.mean(normal_loads)) if normal_loads else None,
            "contact_count_range": [min(contact_counts), max(contact_counts)],
            "object_position_m": source.body_q.numpy()[body, :3].tolist(),
            "finite": bool(np.isfinite(source.body_q.numpy()).all()),
        }


class TestFeatherPGSTorsionalGrip(unittest.TestCase):
    def test_point_contacts_leave_a_free_pinch_axis(self):
        """Confirm the free twist mode while keeping the two jaws loaded."""
        device = wp.get_device()
        mode = "matrix_free" if device.is_cuda else "split"
        for shape in ("sphere", "capsule_end"):
            with self.subTest(shape=shape):
                result = _run_pinch(shape, device, mode=mode)
                self.assertTrue(result["finite"])
                self.assertEqual(result["contact_count_range"], [2, 2])
                self.assertAlmostEqual(result["mean_normal_load_N"], 20.0, delta=0.02)
                self.assertAlmostEqual(result["final_omega_x_rad_s"], 6.25, delta=0.1)
                self.assertLess(result["geometry"]["pinch_axis_lever_arm"], 0.0002)

    def test_distributed_contacts_resist_twist(self):
        """Verify that box faces and both long and short capsule sides resist the same torque."""
        device = wp.get_device()
        mode = "matrix_free" if device.is_cuda else "split"
        for shape, half_height, count in (("box", 0.03, 8), ("capsule_side", 0.03, 4), ("capsule_side", 0.003, 4)):
            with self.subTest(shape=shape, half_height=half_height):
                result = _run_pinch(shape, device, mode=mode, half_height=half_height)
                self.assertTrue(result["finite"])
                self.assertEqual(result["geometry"]["contacts"], count)
                self.assertLess(abs(result["final_omega_x_rad_s"]), 0.01)
                self.assertAlmostEqual(result["mean_normal_load_N"], 20.0, delta=0.02)

    def test_more_friction_and_iterations_do_not_constrain_sphere_twist(self):
        """Distinguish a missing constraint direction from insufficient iteration or friction budgets."""
        device = wp.get_device()
        mode = "matrix_free" if device.is_cuda else "split"
        result = _run_pinch("sphere", device, mode=mode, mu=5.0, iterations=256)
        self.assertEqual(result["geometry"]["angular_rank"], 2)
        self.assertAlmostEqual(result["final_omega_x_rad_s"], 6.25, delta=0.01)

    def test_torsional_material_is_a_positive_control_in_xpbd(self):
        """Hold sphere and capsule-end twist by changing only the torsional material coefficient."""
        for shape in ("sphere", "capsule_end"):
            with self.subTest(shape=shape):
                control = _run_pinch(shape, wp.get_device(), solver_name="xpbd")
                torsion = _run_pinch(shape, wp.get_device(), solver_name="xpbd", mu_torsional=0.005)
                self.assertGreater(control["final_omega_x_rad_s"], 6.0)
                self.assertLess(abs(torsion["final_omega_x_rad_s"]), 0.01)
                self.assertEqual(torsion["contact_count_range"], [2, 2])

    @unittest.expectedFailure
    def test_feather_pgs_honors_torsional_material(self):
        """Expose FeatherPGS's missing material response without changing production solver code."""
        device = wp.get_device()
        mode = "matrix_free" if device.is_cuda else "split"
        result = _run_pinch("sphere", device, mode=mode, mu_torsional=0.005)
        self.assertLess(abs(result["final_omega_x_rad_s"]), 0.01)


def _probe():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Run one diagnostic case and print its measurements.")
    parser.add_argument("--shape", choices=("sphere", "capsule_end", "capsule_side", "box"), default="sphere")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solver", choices=("fpgs", "xpbd"), default="fpgs")
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--mu-torsional", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--mode", choices=("split", "matrix_free"), default="split")
    parser.add_argument("--capsule-half-height", type=float, default=0.03)
    parser.add_argument(
        "--matrix", action="store_true", help="Run geometry, material, and iteration controls together."
    )
    parser.add_argument("--output", type=Path, help="Write the measurements as JSON.")
    args = parser.parse_args()
    if args.matrix:
        cases = [(shape, {}) for shape in ("sphere", "capsule_end", "capsule_side", "box")]
        cases += [
            ("capsule_side", {"half_height": 0.003}),
            ("sphere", {"mu": 5.0, "iterations": 256}),
            ("sphere", {"mu_torsional": 0.005}),
            ("capsule_end", {"mu_torsional": 0.005}),
        ]
        cases += [
            (shape, {"solver_name": "xpbd", "mu_torsional": coefficient})
            for shape in ("sphere", "capsule_end")
            for coefficient in (0.0, 0.005)
        ]
    else:
        cases = [
            (
                args.shape,
                {
                    "solver_name": args.solver,
                    "mu": args.mu,
                    "mu_torsional": args.mu_torsional,
                    "iterations": args.iterations,
                    "half_height": args.capsule_half_height,
                },
            )
        ]
    results = []
    for shape, options in cases:
        result = _run_pinch(shape, args.device, mode=args.mode, **options)
        results.append(result)
        print("RESULT " + json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        _probe()
    else:
        unittest.main()
