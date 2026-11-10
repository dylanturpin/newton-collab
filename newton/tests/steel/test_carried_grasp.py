# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A two-finger pinch of a free box under SolverSteel: held still, and carried up and back down."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_FPS, _SUBSTEPS = 60, 3
_HALF_WIDTH = 0.039  # box half extent across the fingers
_SQUEEZE = 0.004  # finger target inside the box face


def _build_pinch(device, worlds: int, lift: bool):
    """Fingers on a hand that is welded to the world, or rides a stiff z slider when ``lift`` is set.

    Each finger carries a 2 x 2 grid of small pads. The box's centre of mass sits 6 cm below and 3 cm beside
    the pads, so gravity twists it about the pinch axis and only the pads' friction holds it.
    """
    b = newton.ModelBuilder()
    pad = newton.ModelBuilder.ShapeConfig(mu=1.1, gap=0.001, density=0.0)
    wall = newton.ModelBuilder.ShapeConfig(mu=0.45, gap=0.00015, density=0.0)
    hand_z = 0.4
    hand = b.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, hand_z), wp.quat_identity()), mass=0.65, inertia=wp.mat33(np.eye(3) * 1e-3)
    )
    hand_xform = wp.transform(wp.vec3(0.0, 0.0, hand_z), wp.quat_identity())
    if lift:
        slider = newton.ModelBuilder.JointDofConfig(
            axis=(0.0, 0.0, 1.0),
            target_ke=1.0e5,
            target_kd=2.0e3,
            armature=0.1,
            effort_limit=1.0e6,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        joints = [
            b.add_joint_prismatic(-1, hand, parent_xform=hand_xform, child_xform=wp.transform_identity(), axis=slider)
        ]
    else:
        joints = [b.add_joint_fixed(-1, hand, parent_xform=hand_xform)]
    for side in (-1.0, 1.0):
        finger = b.add_link(mass=0.03, inertia=wp.mat33(np.eye(3) * 1e-5))
        for dx in (-0.006, 0.006):
            for dz in (-0.012, 0.012):
                b.add_shape_box(
                    finger,
                    xform=wp.transform(wp.vec3(dx, side * 0.0075, dz), wp.quat_identity()),
                    hx=0.005,
                    hy=0.0075,
                    hz=0.01,
                    cfg=pad,
                )
        dof = newton.ModelBuilder.JointDofConfig(
            axis=(0.0, side, 0.0),
            limit_lower=0.0,
            limit_upper=0.04,
            target_q=0.04,
            target_ke=2500.0,
            target_kd=30.0,
            armature=0.08,
            effort_limit=35.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        joints.append(
            b.add_joint_prismatic(
                hand,
                finger,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=dof,
            )
        )
    b.add_articulation(joints)
    mass, hx, hy, hz, t = 0.16, 0.065, _HALF_WIDTH, 0.1, 0.0015
    box = b.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, hand_z - 0.08), wp.quat_identity()),
        mass=mass,
        com=wp.vec3(0.03, 0.0, -0.06),
        inertia=wp.mat33(np.diag([mass * (hy**2 + hz**2) / 3, mass * (hx**2 + hz**2) / 3, mass * (hx**2 + hy**2) / 3])),
    )
    for xf, he in (
        ((0, 0, -hz + t), (hx, hy, t)),
        ((-hx + t, 0, 0), (t, hy, hz)),
        ((hx - t, 0, 0), (t, hy, hz)),
        ((0, -hy + t, 0), (hx, t, hz)),
        ((0, hy - t, 0), (hx, t, hz)),
    ):
        b.add_shape_box(
            box, xform=wp.transform(wp.vec3(*xf), wp.quat_identity()), hx=he[0], hy=he[1], hz=he[2], cfg=wall
        )
    b.add_ground_plane()
    top = newton.ModelBuilder()
    top.replicate(b, worlds)
    return top.finalize(device=device), b.body_count, len(b.joint_q), box


def _twist_deg(
    device,
    *,
    warmstart: bool,
    lift_speed: float,
    deterministic: bool = False,
    color_order: str = "contact",
    seconds: float = 2.5,
    worlds: int = 4,
) -> tuple[np.ndarray, float | None]:
    """Per-world rotation of the box in the hand frame over ``seconds``, and the share of finger contacts matched.

    The rotation is measured from the start for a held pinch and from the end of the 0.5 s squeeze for a lift.

    With ``lift_speed`` the hand rises and returns on a cosine profile whose peak speed is ``lift_speed``.
    """
    lift = lift_speed > 0.0
    model, bodies, coords, box = _build_pinch(device, worlds, lift)
    solver = newton.solvers.SolverSteel(
        model,
        solve_mode="colored",
        articulated_contact_response="propagation",
        iterations=8,
        enable_joint_limits=True,
        contact_warmstart=warmstart,
        friction_anchor_beta=0.0,
        propagation_constraint_max=4096,
        propagation_color_order=color_order,
    )
    matching = "latest" if warmstart else "disabled"
    pipeline = newton.CollisionPipeline(
        model, contact_matching=matching, contact_matching_body_frame=warmstart, deterministic=deterministic
    )
    contacts = pipeline.contacts()
    control = model.control()
    first = 1 if lift else 0
    q = model.joint_q.numpy()
    target = model.joint_target_q.numpy()
    for w in range(worlds):
        for k in (0, 1):
            q[w * coords + first + k] = _HALF_WIDTH + 0.0005
            target[w * coords + first + k] = _HALF_WIDTH - _SQUEEZE
    model.joint_q.assign(q)
    control.joint_target_q.assign(target)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    def box_in_hand(state):
        bq = state.body_q.numpy().reshape(worlds, bodies, 7)
        return bq[:, 0, 3:7], bq[:, box, 3:7]

    finger_bodies = np.array([w * bodies + f for w in range(worlds) for f in (1, 2)])
    shape_body = model.shape_body.numpy()
    matched, total = 0, 0
    frames = int(seconds * _FPS)
    period = (seconds - 0.5) / 2
    rel0 = None if lift else _relative_rotation(*box_in_hand(state_0))
    for f in range(frames):
        u = (f + 1) / _FPS - 0.5
        if lift and u > 0.0:
            # settle the squeeze for 0.5 s, then up and back down
            z = lift_speed * period / math.pi * (1.0 - math.cos(math.pi * min(u, 2 * period) / period))
            target[[w * coords for w in range(worlds)]] = z
            control.joint_target_q.assign(target)
        for _ in range(_SUBSTEPS):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            if warmstart and u > 0.1:
                n = int(contacts.rigid_contact_count.numpy()[0])
                s0 = contacts.rigid_contact_shape0.numpy()[:n]
                s1 = contacts.rigid_contact_shape1.numpy()[:n]
                sel = np.isin(shape_body[s0], finger_bodies) | np.isin(shape_body[s1], finger_bodies)
                matched += int((contacts.rigid_contact_match_index.numpy()[:n][sel] >= 0).sum())
                total += int(sel.sum())
            solver.step(state_0, state_1, control, contacts, 1.0 / (_FPS * _SUBSTEPS))
            state_0, state_1 = state_1, state_0
        if lift and f == 29:
            rel0 = _relative_rotation(*box_in_hand(state_0))
    rel = _relative_rotation(*box_in_hand(state_0))
    angle = np.degrees(2.0 * np.arccos(np.clip(np.abs((rel * rel0).sum(axis=1)), 0.0, 1.0)))
    return angle, (matched / total if total else None)


def _relative_rotation(q_hand, q_box):
    """Box rotation in the hand frame (xyzw)."""
    inv = q_hand * np.array([-1.0, -1.0, -1.0, 1.0])
    x1, y1, z1, w1 = inv.T
    x2, y2, z2, w2 = q_box.T
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=1,
    )


def test_warmstart_holds_carried_pinch(test, device):
    """Keep the warm start of a pinch carried at 0.2 m/s with body-frame matching, so the box does not twist."""
    twist, matched = _twist_deg(device, warmstart=True, lift_speed=0.2)
    test.assertGreater(matched, 0.9, f"only {matched:.1%} of the carried finger contacts kept their warm start")
    test.assertLess(float(np.median(twist)), 1.0, f"carried box twisted {np.round(twist, 2)} deg in the grasp")


def test_rotating_color_order_holds_pinch_from_sorted_contacts(test, device):
    """Hold the pinch from a sorted contact stream with the rotating color order as well as from an unsorted stream.

    The fixed contact order over the same sorted stream is the control: it twists the box several times more.
    """
    unsorted, _ = _twist_deg(device, warmstart=False, lift_speed=0.0, seconds=2.0)
    fixed, _ = _twist_deg(device, warmstart=False, lift_speed=0.0, seconds=2.0, deterministic=True)
    rotating, _ = _twist_deg(
        device, warmstart=False, lift_speed=0.0, seconds=2.0, deterministic=True, color_order="rotating"
    )
    bound = 2.0 * float(np.median(unsorted))
    test.assertGreater(float(np.median(fixed)), bound, f"control: fixed order twisted only {np.round(fixed, 2)} deg")
    test.assertLess(
        float(np.median(rotating)),
        bound,
        f"rotating order twisted the box {np.round(rotating, 2)} deg against {np.round(unsorted, 2)} unsorted",
    )


class TestSteelCarriedGrasp(unittest.TestCase):
    pass


for _fn in (test_warmstart_holds_carried_pinch, test_rotating_color_order_holds_pinch_from_sorted_contacts):
    add_function_test(TestSteelCarriedGrasp, _fn.__name__, _fn, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
