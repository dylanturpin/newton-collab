# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact-operated remote fader driving the supplied pelican automaton."""

import math

import numpy as np
import warp as wp

import newton

from ._tabletop_assets import import_object
from ._tabletop_common import box, plan_ik, relative_angles, rotate_vectors, tf

SCALE = 2.5
TRAVEL = 0.24
MAX_SPEED = math.tau


@wp.kernel
def fader_drive(fader: int, wheel: int, q: wp.array[float], velocity_targets: wp.array[float]):
    fraction = wp.clamp(q[fader] / 0.24, 0.0, 1.0)
    # This is an electrical controller reading the physical slider encoder.
    # It never sets the slider pose or prescribes a mechanism trajectory.
    velocity_targets[wheel] = 6.28318530718 * fraction


def populate(b, info, ee, down, fingers, device):

    # The separate remote sits in front of the complete, unchanged bicycle.
    start = np.array([0.02, -0.36, 0.12])
    orientation = wp.quat_rpy(math.pi, 0.0, 0.0)
    poses = []
    for t in np.linspace(0, 14, 281):
        u = np.clip((t - 2.0) / 5.0, 0, 1)
        displacement = TRAVEL * (u * u * (3 - 2 * u))
        aperture = float(0.035 - 0.023 * np.clip((t - 0.5) / 1.0, 0, 1))
        poses.append((float(t), np.add(start, [0, displacement, 0]), aperture))
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i] = 16000, 220
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 1200, 20, 12
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=orientation)
    blue, teal, gold, dark = (0.06, 0.38, 0.70), (0.06, 0.57, 0.43), (0.92, 0.66, 0.20), (0.14, 0.18, 0.22)
    box(b, (0.02, -0.24, 0.04), (0.11, 0.20, 0.04), blue, label="pelican/remote-case")
    box(b, (0.02, -0.24, 0.081), (0.012, 0.14, 0.001), dark, label="pelican/fader-slot")
    for k in range(11):
        box(
            b,
            (-0.052, start[1] + TRAVEL * k / 10, 0.082),
            (0.012 if k % 5 == 0 else 0.006, 0.001, 0.001),
            gold,
            label="pelican/fader-tick",
        )
    for y in (start[1] - 0.018, start[1] + TRAVEL + 0.018):
        box(b, (start[0], y, 0.095), (0.045, 0.004, 0.01), dark, label="pelican/end-stop")
    fader_q = b.joint_coord_count
    knob = b.add_link(xform=tf(start), label="pelican/fader")
    joint = b.add_joint_prismatic(
        -1,
        knob,
        parent_xform=tf(start),
        axis=(0, 1, 0),
        limit_lower=0,
        limit_upper=TRAVEL,
        damping=2.0,
        target_ke=0,
        target_kd=0,
        actuator_mode=newton.JointTargetMode.NONE,
        label="pelican/fader-slide",
    )
    box(b, (0, 0, 0), (0.035, 0.014, 0.018), teal, knob, "pelican/fader-cap")
    box(b, (0, 0, 0.0185), (0.027, 0.002, 0.0005), gold, knob, "pelican/fader-mark")
    b.add_articulation([joint], label="pelican/remote")
    first_dof = b.joint_dof_count
    bodies = import_object(b, "pelican_bicycle", tf((0.12, 0.25, 0)))
    for i in range(first_dof, b.joint_dof_count):
        b.joint_target_ke[i], b.joint_target_kd[i] = 0, 0
        b.joint_target_mode[i] = int(newton.JointTargetMode.NONE)
    wheel_joint = next(i for i, name in enumerate(b.joint_label) if name.endswith("wheel_angle"))
    wheel_v = b.joint_qd_start[wheel_joint]
    b.joint_target_mode[wheel_v] = int(newton.JointTargetMode.VELOCITY)
    b.joint_target_kd[wheel_v] = 0.12
    b.joint_effort_limit[wheel_v] = 0.20
    bodies["fader"] = knob
    info.update(
        pelican_fader_q=fader_q,
        pelican_wheel_v=wheel_v,
        pelican_bodies=bodies,
        tracked_body=bodies["input"],
        task="Move a physical remote fader to control the pelican drive speed",
    )


def check(info, poses, fps):
    """Measure the completed physical interaction from its recorded trajectory."""

    def angle(name, axis):
        return relative_angles(poses[:, info["pelican_bodies"][name], 3:], axis)

    fader = poses[:, info["pelican_bodies"]["fader"], 1]
    wheel, cam = angle("input", 1), angle("output", 1)
    result = {
        "fader_travel_m": float(fader[-1] - fader[0]),
        "wheel_turns": float(wheel[-1] / math.tau),
        "gear_ratio_error_rad": float(np.max(np.abs(cam + wheel * 0.5))),
        "wing_sweeps_deg": [float(np.rad2deg(np.ptp(angle(n, 0)))) for n in ("wing_left", "wing_right")],
        "jaw_sweep_deg": float(np.rad2deg(np.ptp(angle("jaw", 1)))),
        "initial_wheel_motion_rad": float(np.ptp(wheel[: int(fps)])),
    }

    def point(name, local):
        body = poses[:, info["pelican_bodies"][name]]
        return body[:, :3] + rotate_vectors(body[:, 3:], np.asarray(local) * SCALE)

    errors = [
        np.linalg.norm(point("shin_" + side, (0, 0, -0.076)) - point("input", pedal), axis=1)
        for side, pedal in (("left", (0.022, -0.072, 0)), ("right", (-0.022, 0.072, 0)))
    ]
    result["maximum_ankle_closure_error_m"] = float(np.max(errors))
    actual_speed = np.gradient(wheel, 1 / fps)
    commanded_speed = MAX_SPEED * np.clip((fader - fader[0]) / TRAVEL, 0, 1)
    result["encoder_speed_tracking_rms_rad_s"] = float(
        np.sqrt(np.mean((actual_speed[int(2 * fps) : -1] - commanded_speed[int(2 * fps) : -1]) ** 2))
    )
    full = commanded_speed > 0.95 * MAX_SPEED
    result["full_scale_speed_rpm"] = float(np.mean(actual_speed[full]) * 60 / math.tau) if np.any(full) else 0.0
    result["pass"] = bool(
        result["encoder_speed_tracking_rms_rad_s"] < 0.25
        and 55 < result["full_scale_speed_rpm"] < 65
        and result["maximum_ankle_closure_error_m"] < 0.002
        and 0.22 < result["fader_travel_m"] < 0.25
        and result["wheel_turns"] > 4
        and result["gear_ratio_error_rad"] < 0.15
        and min(result["wing_sweeps_deg"]) > 35
        and result["jaw_sweep_deg"] > 12
        and result["initial_wheel_motion_rad"] < 0.1
    )
    return result
