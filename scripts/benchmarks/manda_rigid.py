# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct published Manda rigid fixtures for FPGS and native MuJoCo CPU.

Run from the repository root with ``uv run --extra dev python -m
scripts.benchmarks.manda_rigid --scene all --solver both --output /tmp/manda-run``.
Robot motion tapes and unspecified placements are our reconstruction, not the
authors' original inputs. Read manda_rigid.md for provenance and interpretation.
Diagnostic readbacks are included; wall time is not a solver throughput benchmark.
"""

import argparse
import copy
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.viewer

ARTICLE = "https://mandarobotics.com/blog/comparing-physics-engines/"
SCENES = ("slide", "drop", "hinge", "collision", "panda_effort", "grasp", "stack", "push")
HOME = np.array([0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4])
KP = np.array([80, 80, 60, 60, 20, 15, 8, 400, 400], dtype=float)
KD = np.array([18, 18, 14, 14, 5, 4, 2, 4, 4], dtype=float)
LIMITS = np.array([87, 87, 87, 87, 12, 12, 12, 4, 4], dtype=float)
CONTROL_DT = 0.001


def _mujoco():
    # MuJoCo is an existing optional dependency; no mesh download is required.
    import mujoco

    return mujoco


def _numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def _rotation(quaternion):
    """Convert a normalized scalar-first quaternion to a rotation matrix."""
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _box(world, name, position, *, side=0.04, mass=0.1):
    body = ET.SubElement(world, "body", name=name, pos=_numbers(position))
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    ET.SubElement(body, "inertial", pos="0 0 0", mass=str(mass), diaginertia=_numbers([mass * side**2 / 6] * 3))
    ET.SubElement(body, "geom", name=f"{name}_geom", type="box", size=_numbers([side / 2] * 3), rgba="0.2 0.7 0.4 1")
    return body


def _static_box(world, name, position, halfsize):
    ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=_numbers(position),
        size=_numbers(halfsize),
        contype="4",
        conaffinity="2",
        rgba="0.5 0.55 0.65 1",
    )


@dataclass
class Fixture:
    name: str
    xml: str
    duration: float
    dt: float
    tracked: tuple[str, ...]
    controlled: tuple[str, ...]
    initial_joints: np.ndarray
    initial_velocity: dict[str, tuple[float, ...]]
    metadata: dict
    native_model: object

    def native_data(self):
        """Initialize canonical joints and velocities by name."""
        mj = _mujoco()
        data = mj.MjData(self.native_model)
        for name, value in zip(self.controlled, self.initial_joints, strict=True):
            index = self.native_model.joint(name).id
            data.qpos[self.native_model.jnt_qposadr[index]] = value
        for name, velocity in self.initial_velocity.items():
            index = self.native_model.joint(name).id
            start = self.native_model.jnt_dofadr[index]
            data.qvel[start : start + len(velocity)] = velocity
        mj.mj_forward(self.native_model, data)
        return data


def build_scene(name, *, friction=None, stack_offset=0.0, grasp_offset=0.0, width=0.09, rear_offset=0.0):
    """Author shared MJCF with explicit rigid-body mass and inertia in SI units."""
    if name not in SCENES:
        raise ValueError(f"Unknown scene: {name}")
    mj = _mujoco()
    robot = name in ("panda_effort", "grasp", "stack", "push")
    mu = (0.5 if robot else 0.3 if name == "collision" else 0.4) if friction is None else friction
    root = ET.Element("mujoco", model=f"manda_reconstruction_{name}")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false")
    ET.SubElement(
        root,
        "option",
        timestep="0.001",
        gravity="0 0 -9.81",
        integrator="Euler",
        solver="Newton",
        cone="elliptic",
        iterations="100",
        tolerance="1e-10",
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0", armature="0", frictionloss="0", limited="false")
    ET.SubElement(
        default,
        "geom",
        friction=f"{mu} 0 0",
        condim="3",
        solref="0.01 1",
        solimp="0.99 0.999 0.001 0.5 2",
        contype="2",
        conaffinity="7",
        density="0",
        margin="0",
        gap="0",
    )
    visual = ET.SubElement(default, "default", **{"class": "visual"})
    ET.SubElement(visual, "geom", contype="0", conaffinity="0", group="2", rgba="0.7 0.75 0.85 1")
    world = ET.SubElement(root, "worldbody")
    controlled, initial, velocity = (), np.empty(0), {}
    metadata = {
        "source": ARTICLE + "index.html",
        "reconstruction": True,
        "friction": mu,
        "gravity_m_s2": 9.81,
        "contact_laws_matched": False,
    }
    dt = 0.002 if name == "slide" else 0.0005 if name == "collision" else 0.001
    duration = {
        "slide": 0.6,
        "drop": 1.0,
        "hinge": 1.0,
        "collision": 1.0,
        "panda_effort": 0.5,
        "grasp": 3.5,
        "stack": 6.0,
        "push": 5.0,
    }[name]

    if name in ("slide", "drop", "collision"):
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="2 2 0.1",
            contype="4",
            conaffinity="2",
            rgba="0.65 0.65 0.65 1",
        )
    if name in ("slide", "drop"):
        _box(world, "cube", (0, 0, 0.02 if name == "slide" else 0.35))
        tracked = ("cube",)
        if name == "slide":
            velocity["cube_free"] = (1, 0, 0, 0, 0, 0)
            metadata["ideal_stopping_distance_m"] = 1 / (2 * mu * 9.81) if mu > 0 else None
    elif name == "hinge":
        link = ET.SubElement(world, "body", name="pendulum", pos="0 0 0.45")
        ET.SubElement(link, "joint", name="hinge", axis="0 1 0")
        ET.SubElement(
            link,
            "inertial",
            pos="0 0 -0.15",
            mass="1",
            diaginertia="0.007633333333333 0.007633333333333 0.000266666666667",
        )
        ET.SubElement(
            link, "geom", name="link", type="box", pos="0 0 -0.15", size="0.02 0.02 0.15", **{"class": "visual"}
        )
        tracked, controlled, initial = ("pendulum",), ("hinge",), np.array([0.5])
        metadata["source"] = ARTICLE + "assets/hinge/audit.md"
    elif name == "collision":
        sphere = ET.SubElement(world, "body", name="sphere", pos="-0.18 -0.012 0.03")
        ET.SubElement(sphere, "freejoint", name="sphere_free")
        ET.SubElement(sphere, "inertial", pos="0 0 0", mass="0.15", diaginertia=_numbers([0.4 * 0.15 * 0.03**2] * 3))
        ET.SubElement(sphere, "geom", name="sphere_geom", type="sphere", size="0.03", rgba="0.8 0.4 0.2 1")
        _box(world, "cube", (0, 0, 0.03), side=0.06)
        # One vertex points along +X; the public notes omit the original mesh yaw.
        radius = 0.09 / math.sqrt(3)
        vertices = [
            (radius * math.cos(a), radius * math.sin(a), z)
            for z in (-0.03, 0.03)
            for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)
        ]
        asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "mesh",
            name="prism_mesh",
            vertex=_numbers(np.asarray(vertices).ravel()),
            face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5",
        )
        prism = ET.SubElement(world, "body", name="prism", pos="0.1 0.02 0.03")
        ET.SubElement(prism, "freejoint", name="prism_free")
        ET.SubElement(
            prism,
            "inertial",
            pos="0 0 0",
            mass="0.1",
            diaginertia=_numbers([0.1 * (0.09**2 / 24 + 0.06**2 / 12)] * 2 + [0.1 * 0.09**2 / 12]),
        )
        ET.SubElement(prism, "geom", name="prism_geom", type="mesh", mesh="prism_mesh", rgba="0.4 0.4 0.9 1")
        velocity["sphere_free"] = (1.4, 0, 0, 0, 1.4 / 0.03, 0)
        tracked = ("sphere", "cube", "prism")
        metadata.update(source=ARTICLE + "assets/collision-chain/audit.md", mesh_yaw_reconstructed=True)
    else:
        panda = copy.deepcopy(
            ET.parse(Path(__file__).with_name("assets") / "manda_panda.xml").getroot().find("worldbody/body")
        )
        world.append(panda)
        controlled = tuple(f"joint{i}" for i in range(1, 8))
        initial = HOME.copy()
        for body in panda.iter("body"):
            ET.SubElement(
                body, "geom", name=f"{body.get('name')}_display", type="sphere", size="0.025", **{"class": "visual"}
            )
            for child in body.findall("body"):
                end = np.fromstring(child.get("pos", "0 0 0"), sep=" ")
                if np.linalg.norm(end) > 0.06:
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{body.get('name')}_{child.get('name')}_display",
                        type="capsule",
                        size="0.022",
                        fromto="0 0 0 " + _numbers(end),
                        **{"class": "visual"},
                    )
        if name == "panda_effort":
            for finger_name, sign in (("left_finger", 1), ("right_finger", -1)):
                finger = panda.find(f".//body[@name='{finger_name}']")
                finger.remove(finger.find("joint"))
                finger.set("pos", f"0 {sign * 0.04} 0.0584")
            tracked = ("hand",)
        else:
            controlled += ("finger_joint1", "finger_joint2")
            initial = np.concatenate([initial, [0.04, 0.04] if name != "push" else [0.012, 0.012]])
            for finger_name in ("left_finger", "right_finger"):
                finger = panda.find(f".//body[@name='{finger_name}']")
                ET.SubElement(
                    finger,
                    "geom",
                    name=f"{finger_name}_pad",
                    type="box",
                    pos="0 0.004 0.0445",
                    size="0.012 0.004 0.016",
                    contype="1",
                    conaffinity="2",
                    rgba="0.15 0.2 0.3 1",
                )
            # Place our reconstructed tabletop under the published standard home pose.
            provisional = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
            data = mj.MjData(provisional)
            data.qpos[:9] = initial
            mj.mj_forward(provisional, data)
            hand = provisional.body("hand").id
            rotation = data.xmat[hand].reshape(3, 3)
            center = data.xpos[hand] + rotation @ np.array([0, 0, 0.1029])
            tabletop = float(center[2] - 0.02)
            metadata.update(
                tabletop_m=tabletop,
                home_hand_position_m=data.xpos[hand].tolist(),
                home_hand_rotation=rotation.tolist(),
                cube_origin_m=center.tolist(),
                motion_tape="locally generated unloaded IK/inverse-dynamics reference",
                stack_offset_m=stack_offset,
                grasp_offset_m=grasp_offset,
                channel_width_m=width,
                rear_offset_m=rear_offset,
            )
            halfsize = (0.15, 0.15, 0.025) if name == "grasp" else (0.4, 0.3, 0.025)
            _static_box(
                world, "table", (center[0] + (0.05 if name != "grasp" else 0), center[1], tabletop - 0.025), halfsize
            )
            if name in ("grasp", "stack"):
                position = center + rotation[:, 0] * grasp_offset
                _box(world, "cube", position)
                tracked = ("cube", "hand")
                if name == "stack":
                    _box(world, "support", center + np.array([0.1, 0, 0]))
                    tracked = ("cube", "support", "hand")
            else:
                rear = center + np.array([0.034, rear_offset, 0])
                _box(world, "rear", rear)
                _box(world, "front_left", center + np.array([0.078, 0.022, 0]))
                _box(world, "front_right", center + np.array([0.078, -0.022, 0]))
                entrance = float(center[0] + 0.12)
                for side in (-1, 1):
                    _static_box(
                        world,
                        f"wall_{side}",
                        (entrance + 0.09, center[1] + side * (width / 2 + 0.01), tabletop + 0.04),
                        (0.09, 0.01, 0.04),
                    )
                metadata["channel_entrance_x_m"] = entrance
                tracked = ("rear", "front_left", "front_right", "hand")
        metadata.update(
            source=ARTICLE + f"assets/{'panda' if name == 'panda_effort' else name}/audit.md",
            panda_revision="822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7",
            placements_reconstructed=name != "panda_effort",
        )
        if name != "panda_effort":
            tracked += ("left_finger", "right_finger")
    xml = ET.tostring(root, encoding="unicode")
    native = mj.MjModel.from_xml_string(xml)
    return Fixture(name, xml, duration, dt, tracked, controlled, initial, velocity, metadata, native)


def hinge_reference(duration, *, dt=0.00001):
    """Integrate the published scalar pendulum equation with independent RK4."""
    state = np.array([0.5, 0.0])
    inertia = 0.007633333333333 + 0.15**2
    for step in range(round(duration / dt)):
        torque = 0.2 if 0.1 <= step * dt < 0.2 else 0.0

        def derivative(q, torque=torque):
            return np.array([q[1], (torque - 9.81 * 0.15 * np.sin(q[0])) / inertia])

        k1 = derivative(state)
        k2 = derivative(state + dt * k1 / 2)
        k3 = derivative(state + dt * k2 / 2)
        k4 = derivative(state + dt * k3)
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state


@dataclass
class Tape:
    q: np.ndarray
    qd: np.ndarray
    feedforward: np.ndarray

    @property
    def sha256(self):
        return hashlib.sha256(
            b"".join(a.astype("<f8").tobytes() for a in (self.q, self.qd, self.feedforward))
        ).hexdigest()


def _ik(model, data, target, rotation, seed):
    mj = _mujoco()
    hand = model.body("hand").id
    q = seed.copy()
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(150):
        data.qpos[:7] = q
        mj.mj_forward(model, data)
        current = data.xmat[hand].reshape(3, 3)
        error = np.concatenate(
            [target - data.xpos[hand], 0.5 * sum(np.cross(current[:, i], rotation[:, i]) for i in range(3))]
        )
        if np.linalg.norm(error) < 1.0e-8:
            return q
        mj.mj_jacBody(model, data, jac_pos, jac_rot, hand)
        jacobian = np.vstack([jac_pos[:, :7], jac_rot[:, :7]])
        change = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 1.0e-5, error)
        q = np.clip(q + np.clip(change, -0.1, 0.1), model.jnt_range[:7, 0] + 0.001, model.jnt_range[:7, 1] - 0.001)
    raise RuntimeError(f"Reconstructed IK waypoint did not converge: {np.linalg.norm(error):.3g}")


def _blend(times, start, end):
    phase = np.clip((times - start) / (end - start), 0, 1)
    return phase**3 * (10 + phase * (-15 + 6 * phase))


def make_tape(fixture):
    """Freeze one unloaded 1 kHz reference and feedforward tape for every solver."""
    mj = _mujoco()
    count = round(fixture.duration / CONTROL_DT) + 1
    times = np.arange(count) * CONTROL_DT
    q = np.tile(fixture.initial_joints, (count, 1))
    feedforward = np.zeros_like(q)
    if not fixture.controlled or fixture.name == "hinge":
        return Tape(q, np.zeros_like(q), feedforward)
    model, data = fixture.native_model, fixture.native_data()
    if fixture.name == "panda_effort":
        feedforward[:] = data.qfrc_bias[:7]
        feedforward[(times >= 0.05) & (times < 0.15), 1] += 2
        feedforward[(times >= 0.15) & (times < 0.25), 3] -= 1
        return Tape(q, np.zeros_like(q), feedforward)
    hand = model.body("hand").id
    home = data.xpos[hand].copy()
    rotation = data.xmat[hand].reshape(3, 3).copy()
    if fixture.name == "push":
        waypoints = [(0.5, 3.5, np.array([0.24, 0, 0]))]
    else:
        q[:, 7:] = 0.04 - 0.028 * _blend(times, 0.3, 0.8)[:, None]
        waypoints = [(1.0, 1.8, np.array([0, 0, 0.08]))]
        if fixture.name == "stack":
            offset = 0.1 + fixture.metadata["stack_offset_m"]
            waypoints += [
                (1.8, 2.6, np.array([offset, 0, 0.08])),
                (2.6, 3.4, np.array([offset, 0, 0.048])),
                (4.2, 4.8, np.array([offset, 0, 0.12])),
            ]
            q[:, 7:] += 0.028 * _blend(times, 3.6, 4.0)[:, None]
        else:
            q[:, 7:] += 0.028 * _blend(times, 2.3, 2.7)[:, None]
    previous = HOME.copy()
    for start, end, offset in waypoints:
        waypoint = _ik(model, data, home + offset, rotation, previous)
        q[:, :7] += _blend(times, start, end)[:, None] * (waypoint - previous)
        previous = waypoint
    qd = np.gradient(q, CONTROL_DT, axis=0, edge_order=2)
    qdd = np.gradient(qd, CONTROL_DT, axis=0, edge_order=2)
    # Inverse dynamics on a robot-only model prevents object/contact forces from
    # leaking into the frozen reference or adapting it to a tested solver.
    robot_root = ET.fromstring(fixture.xml)
    world = robot_root.find("worldbody")
    for child in list(world):
        if child.tag != "body" or child.get("name") != "link0":
            world.remove(child)
    for geom in world.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    unloaded = mj.MjModel.from_xml_string(ET.tostring(robot_root, encoding="unicode"))
    reference = mj.MjData(unloaded)
    for i in range(count):
        reference.qpos[:] = q[i]
        reference.qvel[:] = qd[i]
        reference.qacc[:] = qdd[i]
        mj.mj_inverse(unloaded, reference)
        feedforward[i] = reference.qfrc_inverse
    return Tape(q, qd, feedforward)


class FPGSRunner:
    def __init__(self, fixture, *, device="cpu", iterations=64, mode=None):
        self.fixture = fixture
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            fixture.xml, parse_visuals=True, force_show_colliders=True, collapse_fixed_joints=False, ctrl_direct=True
        )
        self.model = builder.finalize(device=device)
        self.body_index = {label.split("/")[-1]: i for i, label in enumerate(self.model.body_label)}
        joints = {label.split("/")[-1]: i for i, label in enumerate(self.model.joint_label)}
        qstart, dstart = self.model.joint_q_start.numpy(), self.model.joint_qd_start.numpy()
        self.qindices = np.array([qstart[joints[name]] for name in fixture.controlled], dtype=int)
        self.dindices = np.array([dstart[joints[name]] for name in fixture.controlled], dtype=int)
        q, qd = self.model.joint_q.numpy(), self.model.joint_qd.numpy()
        q[self.qindices] = fixture.initial_joints
        for name, velocity in fixture.initial_velocity.items():
            # Importer labels free roots as <body>/floating_base.
            body_name = name.removesuffix("_free")
            joint = next(
                i
                for i, label in enumerate(self.model.joint_label)
                if self.model.joint_child.numpy()[i] == self.body_index[body_name]
            )
            qd[dstart[joint] : dstart[joint] + len(velocity)] = velocity
        self.model.joint_q.assign(q)
        self.model.joint_qd.assign(qd)
        self.state, self.output = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.output)
        self.control = self.model.control()
        self.effort = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=256, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            pgs_mode=mode or ("matrix_free" if self.model.device.is_cuda else "split"),
            pgs_iterations=iterations,
            dense_max_constraints=768,
            mf_max_constraints=768,
            angular_damping=0.0,
            enable_joint_limits=False,
            friction_anchor_beta=0.0,
            contact_torsion_radius=0.0,
            row_watermark=True,
        )
        self.audit = self._audit()
        self.shape_body = self.model.shape_body.numpy()
        self.tracked_indices = [self.body_index[name] for name in fixture.tracked]

    def _audit(self):
        mj = _mujoco()
        native, reference = self.fixture.native_model, self.fixture.native_data()
        mass, com, inertia, pose = (
            self.model.body_mass.numpy(),
            self.model.body_com.numpy(),
            self.model.body_inertia.numpy(),
            self.state.body_q.numpy(),
        )
        errors = {
            "mass_max_error_kg": 0.0,
            "com_max_error_m": 0.0,
            "inertia_max_error_kg_m2": 0.0,
            "fk_max_error_m": 0.0,
            "rotation_max_error": 0.0,
        }
        for name, index in self.body_index.items():
            original = native.body(name).id
            rotation = _rotation(native.body_iquat[original])
            tensor = rotation @ np.diag(native.body_inertia[original]) @ rotation.T
            for key, error in (
                ("mass_max_error_kg", abs(mass[index] - native.body_mass[original])),
                ("com_max_error_m", np.max(np.abs(com[index] - native.body_ipos[original]))),
                ("inertia_max_error_kg_m2", np.max(np.abs(inertia[index] - tensor))),
                ("fk_max_error_m", np.linalg.norm(pose[index, :3] - reference.xpos[original])),
                (
                    "rotation_max_error",
                    np.linalg.norm(_rotation(pose[index, [6, 3, 4, 5]]) - reference.xmat[original].reshape(3, 3)),
                ),
            ):
                errors[key] = max(errors[key], float(error))
        if any(
            errors[key] > limit
            for key, limit in (
                ("mass_max_error_kg", 1e-6),
                ("com_max_error_m", 1e-6),
                ("inertia_max_error_kg_m2", 1e-6),
                ("fk_max_error_m", 1e-5),
                ("rotation_max_error", 1e-4),
            )
        ):
            raise AssertionError(f"Imported fixture differs from canonical MJCF: {errors}")
        # Audit realized collision pairs, including exclusions for pads vs table/walls.
        active = self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES) != 0
        pairs = self.model.shape_collision_filter_pairs
        groups = self.model.shape_collision_group.numpy()
        geom_index = {label.split("/")[-1]: i for i, label in enumerate(self.model.shape_label)}
        count = 0
        for a in range(native.ngeom):
            for b in range(a):
                allowed = bool(
                    (native.geom_contype[a] & native.geom_conaffinity[b])
                    or (native.geom_contype[b] & native.geom_conaffinity[a])
                )
                ia, ib = (
                    geom_index[mj.mj_id2name(native, mj.mjtObj.mjOBJ_GEOM, a)],
                    geom_index[mj.mj_id2name(native, mj.mjtObj.mjOBJ_GEOM, b)],
                )
                ga, gb = int(groups[ia]), int(groups[ib])
                group_allowed = ga != 0 and gb != 0 and ((ga == gb or gb < 0) if ga > 0 else ga != gb)
                realized = bool(active[ia] and active[ib] and group_allowed and tuple(sorted((ia, ib))) not in pairs)
                # Same-body and articulation-neighbor exclusions are irrelevant:
                # this fixture only enables pad-object/object-static/object-object pairs.
                if allowed != realized:
                    raise AssertionError(f"Collision mask mismatch for geom pair {a}, {b}")
                count += int(allowed)
        errors["allowed_collision_pairs"] = count
        return errors

    def joints(self):
        return self.state.joint_q.numpy()[self.qindices], self.state.joint_qd.numpy()[self.dindices]

    def step(self, efforts, dt):
        self.effort[self.dindices] = efforts
        self.control.joint_f.assign(self.effort)
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.output, self.control, self.contacts, dt)
        self.solver.update_contacts(self.contacts)
        self.solver.check_constraint_capacity()
        self.state, self.output = self.output, self.state

    def observe(self):
        pose = self.state.body_q.numpy()[self.tracked_indices]
        velocity = self.state.body_qd.numpy()[self.tracked_indices, :3]
        return pose[:, :3], pose[:, 3:], velocity

    def forces(self):
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        force = self.contacts.rigid_contact_force.numpy()[:count]
        # Force export acts on shape 0; reactions on shape 1 have opposite sign.
        total = np.zeros((self.model.body_count, 3))
        for shapes, sign in ((self.contacts.rigid_contact_shape0, 1), (self.contacts.rigid_contact_shape1, -1)):
            bodies = self.shape_body[shapes.numpy()[:count]]
            valid = bodies >= 0
            np.add.at(total, bodies[valid], sign * force[valid])
        return total[self.tracked_indices]


class MuJoCoRunner:
    def __init__(self, fixture):
        self.fixture, self.model, self.data = fixture, fixture.native_model, fixture.native_data()
        self.qindices = [self.model.jnt_qposadr[self.model.joint(name).id] for name in fixture.controlled]
        self.dindices = [self.model.jnt_dofadr[self.model.joint(name).id] for name in fixture.controlled]
        self.tracked_indices = [self.model.body(name).id for name in fixture.tracked]
        self.audit = {"canonical_native_mjcf": True}

    def joints(self):
        return self.data.qpos[self.qindices], self.data.qvel[self.dindices]

    def step(self, efforts, dt):
        self.model.opt.timestep = dt
        self.data.qfrc_applied[:] = 0
        self.data.qfrc_applied[self.dindices] = efforts
        _mujoco().mj_step(self.model, self.data)

    def observe(self):
        # Refresh post-step kinematics without replacing the solved contact data.
        mj = _mujoco()
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)
        mj.mj_comVel(self.model, self.data)
        velocity = np.zeros((len(self.tracked_indices), 3))
        for i, body in enumerate(self.tracked_indices):
            spatial = np.zeros(6)
            mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, body, spatial, 0)
            velocity[i] = spatial[3:]
        quaternion = self.data.xquat[self.tracked_indices][:, [1, 2, 3, 0]]
        return self.data.xpos[self.tracked_indices].copy(), quaternion.copy(), velocity

    def forces(self):
        mj = _mujoco()
        total = np.zeros((self.model.nbody, 3))
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            local = np.zeros(6)
            mj.mj_contactForce(self.model, self.data, i, local)
            world = contact.frame.reshape(3, 3).T @ local[:3]
            total[self.model.geom_bodyid[contact.geom[0]]] -= world
            total[self.model.geom_bodyid[contact.geom[1]]] += world
        return total[self.tracked_indices]


def _metrics(fixture, times, positions, rotations, velocities, forces, joint_q):
    result = {
        "finite": bool(all(np.isfinite(a).all() for a in (positions, rotations, velocities, forces, joint_q))),
        "final_positions_m": dict(zip(fixture.tracked, positions[-1].tolist(), strict=True)),
        "peak_net_contact_force_N": float(np.linalg.norm(forces, axis=2).max()),
        "max_linear_speed_m_s": float(np.linalg.norm(velocities, axis=2).max()),
    }
    if fixture.name in ("grasp", "stack", "push"):
        axes = np.array([[_rotation(q[[3, 0, 1, 2]])[:, 1] for q in pair[-2:]] for pair in rotations])
        closing_force = np.abs(np.sum(axes * forces[:, -2:], axis=2))
        result["peak_mean_finger_closing_force_N"] = float(closing_force.mean(axis=1).max())
        if fixture.name == "grasp":
            holding = (times >= 1.8) & (times <= 2.3)
            result["min_finger_closing_force_during_hold_N"] = (
                float(closing_force[holding].min()) if holding.any() else None
            )
    if fixture.name != "hinge" and fixture.controlled:
        lower, upper = fixture.native_model.jnt_range[:7].T
        result["max_arm_limit_violation_rad"] = float(
            max(0, (lower - joint_q[:, :7]).max(), (joint_q[:, :7] - upper).max())
        )
    # Only free objects with origin at COM participate in this linear impulse audit.
    objects = [
        i for i, name in enumerate(fixture.tracked) if name not in ("hand", "left_finger", "right_finger", "pendulum")
    ]
    if objects and len(times) > 1:
        masses = np.array([fixture.native_model.body(fixture.tracked[i]).mass[0] for i in objects])
        impulse = np.diff(velocities[:, objects], axis=0) * masses[None, :, None]
        net_force = forces[1:, objects].copy()
        net_force[:, :, 2] -= masses[None, :] * 9.81
        residual = impulse - net_force * np.diff(times)[:, None, None]
        result["max_linear_impulse_residual_Ns"] = float(np.linalg.norm(residual, axis=2).max())
    if fixture.name == "slide":
        result.update(
            travel_m=float(positions[-1, 0, 0] - positions[0, 0, 0]),
            final_speed_m_s=float(np.linalg.norm(velocities[-1, 0])),
        )
    elif fixture.name == "drop":
        result.update(final_height_m=float(positions[-1, 0, 2]), min_center_height_m=float(positions[:, 0, 2].min()))
    elif fixture.name == "hinge":
        result["final_angle_error_rad"] = float(abs(joint_q[-1, 0] - hinge_reference(times[-1])[0]))
    elif fixture.name in ("grasp", "stack"):
        result["max_cube_lift_m"] = float(positions[:, 0, 2].max() - positions[0, 0, 2])
        if fixture.name == "grasp":
            holding = (times >= 1.8) & (times <= 2.3)
            result["held_during_hold"] = (
                bool(np.all(positions[holding, 0, 2] - positions[0, 0, 2] > 0.04)) if holding.any() else None
            )
        else:
            settling = times >= max(4.0, fixture.duration - 0.5)
            relative = positions[:, 0] - positions[:, 1]
            # Transform displacement into the support's local coordinates.
            local = np.array([_rotation(q[[3, 0, 1, 2]]).T @ r for q, r in zip(rotations[:, 1], relative, strict=True)])
            upright = np.array(
                [
                    [_rotation(q[[3, 0, 1, 2]])[2, 2] > math.cos(math.radians(10)) for q in pair[:2]]
                    for pair in rotations
                ]
            )
            survives = (
                (np.abs(local[:, :2]) <= 0.02).all(axis=1) & (np.abs(local[:, 2] - 0.04) <= 0.005) & upright.all(axis=1)
            )
            result["stack_survives_final_half_second"] = bool(survives[settling].all()) if settling.any() else None
            result["max_final_cube_speed_m_s"] = (
                float(np.linalg.norm(velocities[settling, 0], axis=1).max()) if settling.any() else None
            )
    elif fixture.name == "push":
        goal = fixture.metadata["channel_entrance_x_m"] + 0.06
        result["blocks_past_goal"] = int(np.count_nonzero(positions[-1, :3, 0] >= goal))
        result["all_blocks_passed"] = result["blocks_past_goal"] == 3
    return result


def run(fixture, tape, *, solver, dt, iterations, device, directory, duration=None, viewer=None, source_hashes=None):
    """Record every native step and preserve physical failures in an explicit result."""
    duration = fixture.duration if duration is None else duration
    steps = round(duration / dt)
    runner = FPGSRunner(fixture, device=device, iterations=iterations) if solver == "fpgs" else MuJoCoRunner(fixture)
    if viewer is not None:
        viewer.set_model(runner.model)
        viewer.set_camera(pos=wp.vec3(1.2, -1.3, 1.0), pitch=-20, yaw=130)
    positions, rotations, velocities, forces, joints, efforts, times = [], [], [], [], [], [], []
    command = np.zeros(len(fixture.controlled))
    ticks = round(CONTROL_DT / dt) if dt <= CONTROL_DT else 1
    started = time.perf_counter()
    status, failure = "completed", None
    for step in range(steps + 1):
        now = step * dt
        position, rotation, velocity = runner.observe()
        q, qd = runner.joints()
        force = runner.forces()
        if not all(np.isfinite(a).all() for a in (position, rotation, velocity, q, qd, force)):
            status, failure = "failed", f"Nonfinite state or contact force at {now:.6f} s"
            break
        if fixture.name in ("panda_effort", "grasp", "stack", "push"):
            lower, upper = fixture.native_model.jnt_range[:7].T
            if np.any(q[:7] < lower - 1e-5) or np.any(q[:7] > upper + 1e-5):
                status, failure = "failed", f"Arm joint limit violation at {now:.6f} s"
                break
        times.append(now)
        positions.append(position)
        rotations.append(rotation)
        velocities.append(velocity)
        forces.append(force)
        joints.append(q.copy())
        efforts.append(command.copy())
        if viewer is not None and (step % max(1, round(1 / (60 * dt))) == 0 or step == steps):
            viewer.begin_frame(now)
            viewer.log_state(runner.state)
            viewer.end_frame()
        if step == steps:
            break
        if step % ticks == 0:
            index = min(round(now / CONTROL_DT), len(tape.q) - 1)
            if fixture.name == "hinge":
                command[:] = 0.2 if 0.1 <= now < 0.2 else 0
            elif fixture.name == "panda_effort":
                command = tape.feedforward[index].copy()
            elif len(q):
                command = np.clip(
                    tape.feedforward[index] + KP * (tape.q[index] - q) + KD * (tape.qd[index] - qd), -LIMITS, LIMITS
                )
        try:
            runner.step(command, dt)
        except (AssertionError, RuntimeError, ValueError) as error:
            status, failure = "failed", f"{type(error).__name__} at {now:.6f} s: {error}"
            break
    arrays = {
        "time_s": np.asarray(times),
        "positions_m": np.asarray(positions),
        "rotations_xyzw": np.asarray(rotations),
        "linear_velocity_m_s": np.asarray(velocities),
        "net_contact_force_N": np.asarray(forces),
        "controlled_joint_q": np.asarray(joints),
        "commanded_effort": np.asarray(efforts),
    }
    np.savez_compressed(directory / "trace.npz", **arrays)
    metrics = _metrics(
        fixture,
        arrays["time_s"],
        arrays["positions_m"],
        arrays["rotations_xyzw"],
        arrays["linear_velocity_m_s"],
        arrays["net_contact_force_N"],
        arrays["controlled_joint_q"],
    )
    result = {
        "scene": fixture.name,
        "solver": solver,
        "status": status,
        "failure": failure,
        "dt_s": dt,
        "duration_s": duration,
        "recorded_steps": len(times),
        "iterations": iterations if solver == "fpgs" else 100,
        "device": str(runner.model.device) if solver == "fpgs" else "cpu",
        "elapsed_diagnostic_run_s": time.perf_counter() - started,
        "audit": runner.audit,
        "metrics": metrics,
        "scene_sha256": hashlib.sha256(fixture.xml.encode()).hexdigest(),
        "tape_sha256": tape.sha256,
        "metadata": fixture.metadata,
        "warp": wp.__version__,
        "mujoco": _mujoco().__version__,
        "force_timing": "force from preceding integration step, post-step body poses",
        "velocity_reference": "body COM",
        "reconstruction": True,
        "source_hashes": source_hashes,
        "control": {"dt_s": CONTROL_DT, "kp": KP.tolist(), "kd": KD.tolist(), "effort_limits": LIMITS.tolist()}
        if fixture.name in ("grasp", "stack", "push")
        else None,
        "fpgs_options": {
            "mode": runner.solver.pgs_mode,
            "angular_damping": 0.0,
            "joint_limits": False,
            "pgs_beta": 0.2,
            "pgs_cfm": 1e-6,
            "warmstart": False,
            "torsion_radius_m": 0.0,
            "friction_anchor_beta": 0.0,
            "shape_gap_m": runner.model.shape_gap.numpy().tolist(),
        }
        if solver == "fpgs"
        else None,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=(*SCENES, "all"), default="all")
    parser.add_argument("--solver", choices=("fpgs", "mujoco", "both"), default="fpgs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--dt", type=float, help="Physics timestep; robot control remains at 1 kHz")
    parser.add_argument("--duration", type=float, help="Short diagnostic duration; omitted for complete protocols")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--stack-offset", type=float, default=0.0, help="Placement offset [m]; published cases 0, .016, .020, .022"
    )
    parser.add_argument("--grasp-offset", type=float, default=0.0, help="Initial cube offset along hand X [m]")
    parser.add_argument("--width", type=float, default=0.09, help="Push-channel clear width [m]")
    parser.add_argument("--rear-offset", type=float, default=0.0, help="Initial rear block Y offset [m]")
    parser.add_argument("--viewer", choices=("null", "gl", "viser"), default="null")
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory; existing directories are rejected"
    )
    args = parser.parse_args()
    if (
        args.iterations < 1
        or args.repeats < 1
        or args.width <= 0
        or not np.isfinite([args.width, args.stack_offset, args.grasp_offset, args.rear_offset]).all()
    ):
        parser.error("iterations, repeats and width must be positive; geometry must be finite")
    if args.dt is not None and (not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("dt must be finite and positive")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration < (args.dt or 0.002)):
        parser.error("duration must be finite and at least one physics step")
    if args.viewer != "null" and (args.scene == "all" or args.solver != "fpgs" or args.repeats != 1):
        parser.error("a live viewer requires one scene, FPGS and one repeat")
    names = SCENES if args.scene == "all" else (args.scene,)
    for name in names:
        if name in ("panda_effort", "grasp", "stack", "push") and args.dt is not None:
            ratio = CONTROL_DT / args.dt
            if ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
                parser.error("robot dt must divide 0.001 s to preserve the 1 kHz controller")
    args.output.mkdir(parents=True, exist_ok=False)
    wp.init()
    root = Path(newton.__file__).resolve().parents[1]
    sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("assets") / "manda_panda.xml",
        root / "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
        root / "newton/_src/solvers/feather_pgs/kernels.py",
    )
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    viewer = None
    if args.viewer != "null":
        viewer = newton.viewer.ViewerGL() if args.viewer == "gl" else newton.viewer.ViewerViser()
    results = []
    try:
        for name in names:
            fixture = build_scene(
                name,
                stack_offset=args.stack_offset,
                grasp_offset=args.grasp_offset,
                width=args.width,
                rear_offset=args.rear_offset,
            )
            tape = make_tape(fixture)
            scene_dir = args.output / name
            scene_dir.mkdir()
            (scene_dir / "scene.xml").write_text(fixture.xml + "\n")
            np.savez_compressed(
                scene_dir / "reference.npz", q=tape.q, qd=tape.qd, feedforward=tape.feedforward, control_dt_s=CONTROL_DT
            )
            for repeat in range(args.repeats):
                for solver in ("fpgs", "mujoco") if args.solver == "both" else (args.solver,):
                    directory = scene_dir / f"{solver}-{repeat}"
                    directory.mkdir()
                    print(f"Running {name} / {solver} / repeat {repeat}", flush=True)
                    result = run(
                        fixture,
                        tape,
                        solver=solver,
                        dt=args.dt or fixture.dt,
                        iterations=args.iterations,
                        device=args.device,
                        directory=directory,
                        duration=args.duration,
                        viewer=viewer,
                        source_hashes=source_hashes,
                    )
                    results.append(result)
                    print(
                        json.dumps(
                            {"scene": name, "solver": solver, "status": result["status"], "metrics": result["metrics"]},
                            allow_nan=False,
                        ),
                        flush=True,
                    )
    finally:
        if viewer is not None:
            viewer.close()
        (args.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if any(result["status"] != "completed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
