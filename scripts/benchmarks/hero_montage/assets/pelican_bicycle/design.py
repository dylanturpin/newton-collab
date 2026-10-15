"""Pelican penny-farthing automaton. CAD in mm; generalized angles in radians.
Original dimensioned mechanism. One input angle; gears and leg loops are mechanical.
"""

from __future__ import annotations

import math

import numpy as np

P = {
    "front_x": 38.0,
    "front_z": 90.0,
    "front_radius": 76.0,
    "rear_x": -111.0,
    "rear_z": 44.0,
    "rear_radius": 30.0,
    "cam_x": -22.0,
    "cam_z": 90.0 + math.sqrt(72.0**2 - 60.0**2),
    "gear_y": 34.0,
    "module": 2.0,
    "pressure_angle": 20.0,
    "input_teeth": 24,
    "output_teeth": 48,
    "backlash": 0.20,
    "gear_width": 6.0,
    "hip_x": -10.0,
    "hip_z": 188.0,
    "thigh": 68.0,
    "shin": 76.0,
    "pedal_radius": 22.0,
    "thigh_y": 66.0,
    "shin_y": 72.0,
    "wing_x": -17.0,
    "wing_z": 233.0,
    "shoulder_y": 23.0,
    "wing_arm": 18.0,
    "wing_angle_deg": 30.0,
    "wing_cam_y": 50.0,
    "wing_cam_r": 19.0,
    "wing_shoe_r": 3.0,
    "fork_x": -64.0,
    "fork_y": 39.5,
    "jaw_x": 55.0,
    "jaw_z": 264.0,
    "jaw_arm": 24.0,
    "jaw_angle_deg": 22.0,
    "jaw_shoe_r": 2.5,
    "beak_cam_y": 65.0,
    "beak_cam_r": 26.0,
    "jaw_weight_x": -34.0,
    "jaw_weight_y": -8.0,
    "nominal_wheel_rpm": 60.0,
}
P["wing_e"] = P["wing_arm"] * math.sin(math.radians(P["wing_angle_deg"]))
P["beak_lift"] = P["jaw_arm"] * math.sin(math.radians(P["jaw_angle_deg"]))
P["wing_foot_z"] = P["cam_z"] + P["wing_cam_r"] - P["wing_e"]
P["beak_foot_z"] = P["cam_z"] + P["beak_cam_r"]
P["wing_pad_offset"] = P["wing_z"] - P["wing_e"] - P["wing_shoe_r"] - P["wing_foot_z"]
P["jaw_pad_offset"] = P["jaw_z"] - P["jaw_shoe_r"] - P["beak_foot_z"]
LINE_ANGLE = math.atan2(P["cam_z"] - P["front_z"], P["cam_x"] - P["front_x"])
GEAR_PHASE = (LINE_ANGLE, LINE_ANGLE + math.pi - math.pi / P["output_teeth"])  # positive XZ polar angles


def beak_lift(theta):
    """C2 lift law, 180-degree closed dwell; q=H*(-cos theta)^3 on back half."""
    c = np.cos(theta)
    s = np.sin(theta)
    inside = c < 0
    H = P["beak_lift"]
    return (
        np.where(inside, -H * c**3, 0.0),
        np.where(inside, 3 * H * c * c * s, 0.0),
        np.where(inside, 3 * H * (3 * c**3 - 2 * c), 0.0),
    )


def beak_profile(n=960):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    q, dq, _ = beak_lift(t)
    s = P["beak_cam_r"] + q
    # The output rotates about +Y by -theta. This is the flat-follower envelope.
    return np.column_stack((s * np.sin(t) + dq * np.cos(t), s * np.cos(t) - dq * np.sin(t)))


def leg(phi, side=-1):
    """Exact two-circle intersection; forward-knee branch, not animated joint targets."""
    phi = np.asarray(phi)
    a = phi + (np.pi if side == 1 else 0.0)
    hx, hz = P["hip_x"], P["hip_z"]
    l1, l2 = P["thigh"], P["shin"]
    r = P["pedal_radius"]
    px = P["front_x"] + r * np.cos(a)
    pz = P["front_z"] - r * np.sin(a)
    vx, vz = px - hx, pz - hz
    d = np.hypot(vx, vz)
    along = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(np.maximum(0, l1 * l1 - along * along))
    kx = hx + along * vx / d - h * vz / d
    kz = hz + along * vz / d + h * vx / d
    thigh = np.arctan2(-(kx - hx), -(kz - hz))
    shin = np.arctan2(-(px - kx), -(pz - kz))
    return {"px": px, "pz": pz, "kx": kx, "kz": kz, "thigh": thigh, "shin": shin, "d": d}


def pose(phi, **passive):
    theta = phi / 2.0
    qw = P["wing_e"] * (1 - math.cos(theta))
    qb = float(beak_lift(theta)[0])
    p = {
        "phi": float(phi),
        "theta": theta,
        "qw": qw,
        "qb": qb,
        "alpha_left": math.asin((qw - P["wing_e"]) / P["wing_arm"]),
        "alpha_right": math.asin((qw - P["wing_e"]) / P["wing_arm"]),
        "beta": math.asin(qb / P["jaw_arm"]),
    }
    p.update(passive)
    return p


def transform_for(group, p):
    """World translation (mm), rotation axis and absolute rotation (rad) for a rigid group."""
    if group == "fixed":
        return np.zeros(3), "y", 0.0
    if group == "input":
        return np.array([P["front_x"], 0, P["front_z"]]), "y", p["phi"]
    if group == "output":
        return np.array([P["cam_x"], 0, P["cam_z"]]), "y", -p["phi"] / 2.0
    if group == "rear":
        return np.array([P["rear_x"], 0, P["rear_z"]]), "y", 0.0
    if group == "fork":
        return np.array([0, 0, P["wing_foot_z"] + p["qw"]]), "z", 0.0
    if group == "beak_slider":
        return np.array([0, 0, P["beak_foot_z"] + p["qb"]]), "z", 0.0
    if group == "jaw":
        return np.array([P["jaw_x"], 0, P["jaw_z"]]), "y", p["beta"]
    side = -1 if group.endswith("_left") else 1
    if group.startswith("wing_"):
        return (
            np.array([P["wing_x"], side * P["shoulder_y"], P["wing_z"]]),
            "x",
            side * p["alpha_left" if side == -1 else "alpha_right"],
        )
    l = leg(p["phi"], side)
    if group.startswith("thigh_"):
        return np.array([P["hip_x"], side * P["thigh_y"], P["hip_z"]]), "y", float(l["thigh"])
    if group.startswith("shin_"):
        return np.array([l["kx"], side * P["shin_y"], l["kz"]]), "y", float(l["shin"])
    raise ValueError(group)


def gear_tooth(n, samples=16):
    m = P["module"]
    rp = n * m / 2
    rb = rp * math.cos(math.radians(P["pressure_angle"]))
    ra = rp + m
    rr = rp - 1.25 * m

    def inv(r):
        return math.sqrt(max(0, (r / rb) ** 2 - 1)) - math.acos(min(1, rb / r))

    half = math.pi / (2 * n) - P["backlash"] / (2 * rp)
    rs = np.linspace(max(rb, rr), ra, samples)
    off = np.array([half + inv(rp) - inv(r) for r in rs])
    pts = [(rr * math.cos(off[0]), -rr * math.sin(off[0]))]
    pts += [(r * math.cos(a), -r * math.sin(a)) for r, a in zip(rs, off, strict=False)]
    pts += [(ra * math.cos(a), ra * math.sin(a)) for a in np.linspace(-off[-1], off[-1], 6)[1:]]
    pts += [(r * math.cos(a), r * math.sin(a)) for r, a in zip(rs[-2::-1], off[-2::-1], strict=False)]
    pts += [(rr * math.cos(off[0]), rr * math.sin(off[0]))]
    return np.array(pts)


if __name__ == "__main__":
    print(P)
