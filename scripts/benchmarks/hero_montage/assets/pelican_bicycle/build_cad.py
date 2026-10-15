"""Actual B-rep solids and named, colored STEP assembly; local-coordinate STL parts.
Run this first. All component transformations come from design.py's mechanism.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import cadquery as cq
import numpy as np
from design import GEAR_PHASE, P, beak_profile, gear_tooth, pose, transform_for

ROOT = Path(__file__).resolve().parent
CAD = ROOT / "cad"
CAD.mkdir(exist_ok=True)
C = {
    "frame": (0.12, 0.34, 0.34),
    "stand": (0.21, 0.24, 0.27),
    "base": (0.30, 0.23, 0.16),
    "rim": (0.63, 0.68, 0.68),
    "tire": (0.10, 0.12, 0.14),
    "gear": (0.78, 0.57, 0.25),
    "cam": (0.88, 0.43, 0.18),
    "slider": (0.15, 0.53, 0.68),
    "body": (0.93, 0.91, 0.82),
    "wing": (0.90, 0.90, 0.85),
    "tip": (0.18, 0.21, 0.24),
    "beak": (0.98, 0.68, 0.20),
    "pouch": (0.87, 0.42, 0.15),
    "leg": (0.88, 0.57, 0.24),
    "metal": (0.65, 0.68, 0.69),
    "eye": (0.055, 0.065, 0.07),
}
DENS = {
    "frame": 2700,
    "stand": 2700,
    "base": 650,
    "rim": 2700,
    "tire": 1100,
    "gear": 750,
    "cam": 1100,
    "slider": 1100,
    "body": 650,
    "wing": 650,
    "tip": 650,
    "beak": 650,
    "pouch": 650,
    "leg": 750,
    "metal": 7800,
    "eye": 1100,
}


def box(x, y, z, c=(0, 0, 0)):
    return cq.Workplane("XY").box(x, y, z).val().translate(tuple(c))


def cyl(r, l, c=(0, 0, 0), axis="z"):
    axes = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
    v = np.array(axes[axis])
    a = np.array(c) - v * l / 2
    return cq.Solid.makeCylinder(r, l, cq.Vector(*a), cq.Vector(*v))


def tube_between(a, b, r):
    a = np.array(a, float)
    d = np.array(b, float) - a
    return cq.Solid.makeCylinder(r, float(np.linalg.norm(d)), cq.Vector(*a), cq.Vector(*d))


def union(items):
    s = items[0]
    for t in items[1:]:
        s = s.fuse(t)
    return s.clean()


def compound(items):
    return cq.Compound.makeCompound(items)


def poly_xz(points, y0, w):
    p = np.array(points)
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]]
    if np.linalg.norm(p[0] - p[-1]) < 1e-7:
        p = p[:-1]
    return cq.Workplane("XZ", origin=(0, y0 + w, 0)).polyline([tuple(v) for v in p]).close().extrude(w).val()


def poly_xy(points, z0, w):
    return cq.Workplane("XY", origin=(0, 0, z0)).polyline([tuple(p) for p in points]).close().extrude(w).val()


def smooth(points, n=2):
    pts = np.array(points, float)
    for _ in range(n):
        nex = np.roll(pts, -1, axis=0)
        pts = np.stack((0.75 * pts + 0.25 * nex, 0.25 * pts + 0.75 * nex), axis=1).reshape(-1, 2)
    return pts


def gear(n, phase):
    rr = n * P["module"] / 2 - 1.25 * P["module"]
    w = P["gear_width"]
    inner = max(9, rr - 7)
    ring = cyl(rr, w, axis="y").cut(cyl(inner, w + 2, axis="y"))
    hub = cyl(7, w, axis="y")
    spokes = []
    for a in np.arange(6) * math.pi / 3:
        spokes.append(
            tube_between(
                (5 * math.cos(a), 0, 5 * math.sin(a)), ((inner + 2) * math.cos(a), 0, (inner + 2) * math.sin(a)), 2.5
            )
        )
    # Square-edged radial spokes stay within the gear face width.
    items = [ring, hub, *spokes]
    t = poly_xz(gear_tooth(n), -w / 2, w)
    items += [t.rotate((0, 0, 0), (0, 1, 0), -360 * k / n) for k in range(n)]
    return (
        union(items)
        .cut(cyl(4.12, w + 4, axis="y"))
        .rotate((0, 0, 0), (0, 1, 0), -math.degrees(phase))
        .translate((0, P["gear_y"], 0))
    )


def wheel_parts(R, rear=False):
    width = 8 if rear else 10
    ir = R - 4 if rear else R - 5
    rim_in = ir - 2 if rear else ir - 5
    tire = cyl(R, width, axis="y").cut(cyl(ir, width + 2, axis="y"))
    rim = cyl(ir, width - 2, axis="y").cut(cyl(rim_in, width + 2, axis="y"))
    spokes = [cyl(7 if rear else 9, 18 if rear else 22, axis="y").cut(cyl(4.1, 26, axis="y"))]
    for a in np.arange(10 if rear else 12) * 2 * np.pi / (10 if rear else 12):
        spokes.append(
            tube_between(
                (5 * math.cos(a), 0, 5 * math.sin(a)),
                ((rim_in + 1) * math.cos(a), 0, (rim_in + 1) * math.sin(a)),
                1.1 if rear else 1.45,
            )
        )
    return tire, rim, compound(spokes)


def bicycle_frame():
    parts = []
    fx, fz = P["front_x"], P["front_z"]
    rx, rz = P["rear_x"], P["rear_z"]
    for sy in [-1, 1]:
        y = sy * 15
        pts = [(rx, y, rz), (-70, y, 122), (-10, y, 180), (42, y, 170), (fx, y, fz)]
        parts += [tube_between(a, b, 3) for a, b in itertools.pairwise(pts)]
        parts += [tube_between((-10, y, 180), (-10, sy * 14, 188), 3), tube_between((42, y, 170), (60, y, 193), 3)]
        parts += [
            cyl(7.5, 8, (fx, y, fz), "y").cut(cyl(4.2, 10, (fx, y, fz), "y")),
            cyl(6, 8, (rx, y, rz), "y").cut(cyl(4.2, 10, (rx, y, rz), "y")),
        ]
    parts += [
        tube_between((60, -43, 193), (60, 43, 193), 3),
        tube_between((-10, -15, 180), (-10, 15, 180), 3),
        tube_between((rx, -22, rz), (rx, 22, rz), 4),
        cyl(3, 154, (P["hip_x"], 0, P["hip_z"]), "y"),
    ]
    return compound(parts).cut(cyl(4.2, 140, (fx, 0, fz), "y"))


def support_and_guides():
    s = []
    # Separate bearing stand keeps the bicycle stationary, with tires clear of the plinth.
    for sy in [-1, 1]:
        s.append(box(9, 8, 72, (P["front_x"], sy * 25, 50)))
        s.append(box(10, 17, 8, (P["front_x"], sy * 21, 90)).cut(cyl(4.2, 22, (P["front_x"], sy * 21, 90), "y")))
        s.append(box(9, 8, 28, (P["rear_x"], sy * 19, 28)))
        # Fork guides bolted to the rigid torso/saddle frame.
        center = (P["fork_x"], sy * P["fork_y"], 205)
        guide = box(10, 10, 14, center).cut(box(4.4, 4.4, 18, center))
        s += [guide, tube_between((-36, sy * 18, 205), (-58, sy * P["fork_y"], 205), 2.5)]
        # Fixed wing journal blocks.
        for x in [-27, 0]:
            b = box(7, 12, 12, (x, sy * 23, 233)).cut(cyl(3.2, 10, (x, sy * 23, 233), "x"))
            s.append(b)
    for y in [22.0, 78.0]:
        collar = box(14, 7, 14, (P["cam_x"], y, P["cam_z"])).cut(cyl(4.2, 10, (P["cam_x"], y, P["cam_z"]), "y"))
        post = box(8, 7, P["cam_z"] - 14, (P["cam_x"], y, (P["cam_z"] + 14) / 2)).cut(
            cyl(4.2, 10, (P["cam_x"], y, P["cam_z"]), "y")
        )
        s += [collar, post]
    for x, z in [(P["cam_x"], 220), (31.0, 252)]:
        center = (x, 9, z)
        height = 6 if x == 31 else 12
        s += [
            box(9, 9, height, center).cut(box(4.4, 4.4, height + 4, center)),
            tube_between((x, -16, z), (x, 3.8, z), 2.5),
        ]
    return compound(s)


def body_parts():
    outline = [
        (-69, 206),
        (-51, 218),
        (-39, 235),
        (-22, 245),
        (-1, 243),
        (14, 235),
        (21, 241),
        (24, 264),
        (32, 279),
        (45, 286),
        (58, 284),
        (64, 277),
        (64, 265),
        (58, 258),
        (43, 253),
        (32, 235),
        (25, 208),
        (14, 194),
        (-11, 187),
        (-35, 192),
        (-52, 200),
    ]
    pieces = []
    for y in [-20, 16]:
        p = poly_xz(smooth(outline, 2), y, 4)
        p = p.cut(cyl(3.2, 8, (55, y + 2, 264), "y"))
        if y > 0:
            p = p.cut(box(4.8, 8, 15, (-22, y + 2, 200.5)))  # moving crosshead clearance slot
        # Deliberate small side window exposes the rear jaw counterweight.
        p = p.cut(poly_xz(smooth([(14, 253), (28, 253), (32, 259), (30, 273), (23, 275), (17, 266)], 1), y - 1, 6))
        pieces.append(p)
    pieces += [
        box(18, 32, 6, (-10, 0, 193)),
        box(12, 32, 5, (-33, 0, 212)),
        box(10, 32, 5, (53, 0, 280)),
        cyl(3, 44, (55, 0, 264), "y"),
    ]
    return compound(pieces)


def wing(side):
    outline = [
        (12, 10),
        (26, 23),
        (22, 46),
        (14, 74),
        (0, 91),
        (-8, 89),
        (-6, 70),
        (-19, 84),
        (-25, 79),
        (-17, 58),
        (-31, 69),
        (-36, 61),
        (-25, 39),
        (-36, 43),
        (-37, 32),
        (-26, 15),
        (-14, 10),
    ]
    blade = poly_xy([(x, side * y) for x, y in smooth(outline, 1)], -1.6, 3.2)
    # Shaft is along the body. Its external arm and shoe sit behind the wing.
    items = [
        blade,
        box(6, 14, 4, (-2, side * 7, 0)),
        cyl(3, 70, (-13, 0, 0), "x"),
        box(4, 18, 4, (-39, side * 9, 0)),
        cyl(4.5, 4, (-39, 0, 0), "x"),
        cyl(3, 12, (-45, side * 18, 0), "x"),
    ]
    return union(items)


def wing_tips(side):
    shapes = []
    for poly in [
        [(13, 71), (0, 89), (-7, 87), (-5, 71), (1, 66)],
        [(-7, 69), (-19, 82), (-23, 78), (-15, 61)],
        [(-23, 58), (-32, 67), (-34, 60), (-26, 50)],
    ]:
        shapes.append(poly_xy([(x, side * y) for x, y in poly], 1.61, 0.8))
    return compound(shapes)


def wing_fork():
    H = P["wing_pad_offset"]
    f0 = P["wing_foot_z"]
    cross = 170.0 - f0
    y = P["wing_cam_y"]
    s = [
        box(36, 10, 4, (P["cam_x"], y, 2)),
        box(4, 4, cross - 4, (P["cam_x"], y, (cross + 4) / 2)),
        box(46, 6, 4, (-43, y, cross)),
        box(6, 98, 4, (-64, 7, cross)),
    ]
    for sy in [-1, 1]:
        s += [
            box(4, 4, H - cross, (-64, sy * P["fork_y"], (H + cross) / 2)),
            box(8, 12, 4, (-64, sy * P["fork_y"], H - 2)),
        ]
    return union(s)


def jaw_slider():
    H = P["jaw_pad_offset"]
    f0 = P["beak_foot_z"]
    cross = 235.0 - f0
    transfer = 196.0 - f0
    y = P["beak_cam_y"]
    return union(
        [
            box(32, 10, 4, (P["cam_x"], y, 2)),
            box(4, 4, transfer - 4, (P["cam_x"], y, (transfer + 4) / 2)),
            box(4, 60, 4, (P["cam_x"], 37, transfer)),
            box(4, 4, cross - transfer, (P["cam_x"], 9, (cross + transfer) / 2)),
            box(57, 4, 4, (4.5, 9, cross)),
            box(4, 4, H - cross, (31, 9, (H + cross) / 2)),
            box(10, 7, 4, (33, 9, H - 2)),
        ]
    )


def lower_jaw():
    front = poly_xz(smooth([(0, 1), (81, 1), (79, -3), (56, -11), (33, -22), (12, -19), (-1, -6)], 1), -6, 12)
    arm = box(34, 4, 3, (-17, 0, 0))
    hub = cyl(5, 14, axis="y")
    shoe = cyl(2.5, 14, (-24, 6, 0), "y")
    return union([front, arm, hub, shoe]).cut(cyl(3.2, 18, axis="y"))


def leg_piece(L, thigh=True):
    beam = poly_xz([(-3, 0), (3, 0), (2.4, -L), (-2.4, -L)], -2, 4)
    s = union([beam, cyl(5, 4, axis="y"), cyl(5, 4, (0, 0, -L), "y")])
    return s.cut(cyl(3.2 if thigh else 2.7, 8, axis="y")).cut(cyl(2.7 if thigh else 3.05, 8, (0, 0, -L), "y"))


def build():
    parts = {}

    def add(name, shape, color, group="fixed", density=None):
        if not shape.isValid():
            raise RuntimeError("Invalid B-rep " + name)
        rho = DENS[color] if density is None else density
        cq.exporters.export(shape, str(CAD / (name + ".step")))
        cq.exporters.export(shape, str(CAD / (name + ".stl")), tolerance=0.10, angularTolerance=0.16)
        c = np.array(shape.Center().toTuple()) * 0.001
        m = shape.Volume() * rho * 1e-9
        I = np.array(cq.Shape.matrixOfInertia(shape)) * rho * 1e-15
        parts[name] = {
            "shape": shape,
            "color": C[color],
            "group": group,
            "density_kg_m3": rho,
            "mass_kg": m,
            "com_m": c.tolist(),
            "inertia_com_kg_m2": I.tolist(),
            "volume_mm3": shape.Volume(),
            "valid": shape.isValid(),
        }
        print(name, flush=True)

    add("display_plinth", box(290, 176, 12, (-2, 0, 6)), "base")
    add("bicycle_frame", bicycle_frame(), "frame")
    add("bearings_stands_guides", support_and_guides(), "stand")
    add("saddle", box(35, 36, 6, (-10, 0, 181)), "tire")
    for sy in [-1, 1]:
        add("handle_grip_" + str(sy), cyl(4.5, 14, (60, sy * 41, 193), "y"), "tire")
    add("pelican_body", body_parts(), "body")
    upper = poly_xz([(55, 270), (132, 268), (145, 267), (146, 269), (134, 274), (65, 277), (55, 276)], -7, 14)
    add("upper_bill", upper, "beak")
    for sy in [-1, 1]:
        add("eye_" + str(sy), cyl(2.5, 1.5, (53, sy * 20.5, 275.5), "y"), "eye")
    for front in [True, False]:
        tire, rim, spokes = wheel_parts(P["front_radius"] if front else P["rear_radius"], not front)
        prefix = "front" if front else "rear"
        group = "input" if front else "rear"
        add(prefix + "_tire", tire, "tire", group)
        add(prefix + "_rim", rim, "rim", group)
        add(prefix + "_spokes", spokes, "metal", group, density=2700)
    rotor = [cyl(4, 122, axis="y")]
    for sy in [-1, 1]:
        px = P["pedal_radius"] * (-sy)
        rotor += [
            box(22, 6, 6, (px / 2, sy * 56, 0)),
            cyl(5.5, 6, (0, sy * 56, 0), "y"),
            cyl(2.8, 26, (px, sy * 66, 0), "y"),
        ]
    add("axle_cranks_pedal_pins", compound(rotor), "metal", "input")
    add("wheel_drive_gear_24T", gear(24, GEAR_PHASE[0]), "gear", "input")
    add("camshaft_gear_48T", gear(48, GEAR_PHASE[1]), "gear", "output")
    add("camshaft", cyl(4, 66, (0, 51, 0), "y"), "metal", "output")
    add(
        "wing_eccentric_cam",
        cyl(P["wing_cam_r"], 6, (0, P["wing_cam_y"], -P["wing_e"]), "y").cut(
            cyl(4.12, 8, (0, P["wing_cam_y"], 0), "y")
        ),
        "cam",
        "output",
    )
    add(
        "beak_C2_dwell_cam",
        poly_xz(beak_profile(), P["beak_cam_y"] - 3, 6).cut(cyl(4.12, 8, (0, P["beak_cam_y"], 0), "y")),
        "cam",
        "output",
    )
    add("wing_lifting_fork", wing_fork(), "slider", "fork")
    add("beak_lifting_rod", jaw_slider(), "slider", "beak_slider")
    add("lower_bill_pouch", lower_jaw(), "pouch", "jaw")
    add("jaw_counterweight", cyl(8, 12, (-34, -8, 0), "y"), "metal", "jaw")
    for sy, side in [(-1, "left"), (1, "right")]:
        add("wing_" + side, wing(sy), "wing", "wing_" + side)
        add("feather_tips_" + side, wing_tips(sy), "tip", "wing_" + side)
        add("thigh_" + side, leg_piece(P["thigh"]), "leg", "thigh_" + side)
        add("knee_pin_" + side, cyl(2.5, 14, (0, sy * 4, -P["thigh"]), "y"), "metal", "thigh_" + side)
        add("shin_" + side, leg_piece(P["shin"], False), "leg", "shin_" + side)
        # Webbed foot is rigid with shin; ankle/pedal pin closes each linkage.
        foot = poly_xy(
            [(-3, -4), (10, -7), (24, -8), (20, -2), (24, 1), (19, 3), (22, 7), (10, 6), (-3, 4)], -P["shin"] - 3, 3
        ).cut(cyl(3.05, 20, (0, 0, -P["shin"]), "y"))
        add("webbed_foot_" + side, foot, "pouch", "shin_" + side)
    assy = cq.Assembly(name="Pelican_Penny_Farthing_Zero_Pose")
    manifest = {}
    for name, d in parts.items():
        assy.add(transform(d, pose(0)), name=name, color=cq.Color(*d["color"]))
        manifest[name] = {k: v for k, v in d.items() if k != "shape"}
    assy.export(str(CAD / "pelican_bicycle.step"))
    (CAD / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (ROOT / "parameters.json").write_text(json.dumps(P, indent=2))
    print("Built", len(parts), "valid CAD components", flush=True)
    return parts


def transform(d, p):
    o, axis, a = transform_for(d["group"], p)
    s = d["shape"]
    if a:
        s = s.rotate((0, 0, 0), {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis], math.degrees(a))
    return s.translate(tuple(o))


if __name__ == "__main__":
    build()
