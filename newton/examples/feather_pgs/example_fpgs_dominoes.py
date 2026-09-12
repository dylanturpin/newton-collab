# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Dominoes
#
# Dominoes laid out as the letters of a word, one chain per pen stroke of
# each glyph. Every chain is tipped on the same frame, so the whole word
# collapses at once and each letter reads while it falls. Nothing is
# scripted after the tip: the chains are pure contact propagation.
#
# Command: python -m newton.examples fpgs_dominoes
#
###########################################################################

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite

TEXT = "JAR 3D"
# Glyph cell in metres and the gap between cells. Strokes below are authored in a
# unit cell, x to the right and y up, and scaled into the ground plane.
CELL_W, CELL_H, CELL_GAP = 5.0, 7.5, 1.6
# A word space advances less than a glyph cell, so "JAR" and "3D" read as one line.
SPACE_W = 2.2
HEIGHT, WIDTH, THICKNESS = 1.2, 0.65, 0.18
# Tiles fall onto the next one when the spacing is well under their height. The
# minimum keeps two strokes that meet at a corner from stacking tiles on top of
# each other.
SPACING = 0.42
# Clearance between the footprints of tiles from two different strokes. Strokes that
# meet at a corner or an apex would otherwise place tiles across each other, and the
# solver opens that overlap with a shove that starts the chain early.
TILE_CLEARANCE = 0.04
# Corners are rounded at about this radius before the tiles are placed.
CORNER_RADIUS = 0.9
# A chain is cut whenever consecutive tiles end up further apart than this, and each
# piece gets its own starter, so a hole can never leave part of a letter standing.
MAX_LINK = 0.75
TIP_RATE = 3.5
TIP_AT = 0.6
# Tiles 0.18 m thick need a 1/240 s step to hand the fall on without tunnelling.
SOLVERS = {
    "feather_pgs": {"pgs_iterations": 4, "mf_max_constraints": 8192, "substeps": 4},
    "mujoco": {"njmax": 8192, "nconmax": 4096},
}

# One polyline per pen stroke, in a unit cell. Each stroke becomes one chain and is
# tipped at its first point, so a glyph falls from several places at once.
GLYPHS: dict[str, list[list[tuple[float, float]]]] = {
    "J": [
        [(0.18, 1.0), (0.95, 1.0)],
        [(0.62, 0.95), (0.62, 0.3), (0.55, 0.12), (0.36, 0.04), (0.14, 0.12), (0.08, 0.3)],
    ],
    "A": [
        [(0.5, 1.0), (0.08, 0.0)],
        [(0.5, 1.0), (0.92, 0.0)],
        # The crossbar sits low so the falling diagonals clear it; at mid height its
        # tiles land against them and prop three of them upright.
        [(0.28, 0.3), (0.72, 0.3)],
    ],
    "R": [
        [(0.12, 0.0), (0.12, 1.0)],
        [(0.2, 1.0), (0.66, 1.0), (0.86, 0.85), (0.86, 0.72), (0.66, 0.57), (0.2, 0.57)],
        [(0.5, 0.46), (0.92, 0.0)],
    ],
    "3": [
        [(0.12, 0.88), (0.3, 1.0), (0.62, 1.0), (0.82, 0.84), (0.78, 0.65), (0.5, 0.55)],
        [(0.5, 0.45), (0.8, 0.36), (0.86, 0.17), (0.66, 0.02), (0.3, 0.02), (0.1, 0.14)],
    ],
    "D": [
        [(0.12, 0.0), (0.12, 1.0)],
        [(0.2, 1.0), (0.62, 1.0), (0.88, 0.72), (0.88, 0.28), (0.62, 0.0), (0.2, 0.0)],
    ],
    " ": [],
}


def _overlaps(a, b) -> bool:
    """Separating-axis test between two tile footprints in the ground plane.

    Each footprint is ``(centre, thin axis, half extents)``; the tiles are boxes on
    end, so overlap in plan is overlap in space.
    """
    for axes, half in ((a, b), (b, a)):
        for axis, extent in zip(axes[1], axes[2], strict=True):
            centre = float(np.dot(b[0] - a[0], axis))
            reach = extent + sum(abs(float(np.dot(axis, other))) * e for other, e in zip(half[1], half[2], strict=True))
            if abs(centre) > reach:
                return False
    return True


def _footprint(pos, tangent):
    normal = np.array([-tangent[1], tangent[0]])
    return (
        pos,
        (np.asarray(tangent[:2], dtype=float), normal),
        (THICKNESS / 2 + TILE_CLEARANCE, WIDTH / 2 + TILE_CLEARANCE),
    )


def _densify(pts, step):
    """Split every segment so none is longer than ``step``."""
    out = [pts[0]]
    for a, b in pairwise(pts):
        length = float(np.linalg.norm(b - a))
        for k in range(1, max(1, int(np.ceil(length / step))) + 1):
            out.append(a + (b - a) * (k / max(1, int(np.ceil(length / step)))))
    return out


def _chaikin(pts, iterations=2):
    """Round a polyline by corner cutting, keeping its two endpoints.

    A tile takes the tangent of the stroke it sits on, so a hard corner puts two
    neighbours across each other and one of them has to be dropped. That leaves a
    double gap at exactly the point where the chain also has to turn, which is where
    it stops. Rounded corners keep every tile.
    """
    for _ in range(iterations):
        rounded = [pts[0]]
        for a, b in pairwise(pts):
            rounded.append(0.75 * a + 0.25 * b)
            rounded.append(0.25 * a + 0.75 * b)
        rounded.append(pts[-1])
        pts = rounded
    return pts


def _stroke_points(stroke, origin, spacing):
    """Walk a rounded polyline at a fixed arc length, returning (position, tangent) pairs."""
    pts = [np.array([origin[0] + u * CELL_W, origin[1] + v * CELL_H]) for u, v in stroke]
    if len(pts) > 2:
        pts = _chaikin(_densify(pts, CORNER_RADIUS))
    out, carry = [], 0.0
    for a, b in pairwise(pts):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1.0e-9:
            continue
        direction = seg / length
        s = carry
        while s <= length + 1.0e-9:
            out.append((a + direction * s, direction))
            s += spacing
        carry = s - length
    if not out:
        out.append((pts[0], np.array([1.0, 0.0])))
    return out


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.test_mode = bool(getattr(args, "test", False))
        self.tipped = False
        self.tip_rate = TIP_RATE

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.002
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        tile = newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.4, restitution=0.0)

        self.tiles = []
        self.starters = []  # (body, tangent) tipped together
        self.chain_bodies: dict[tuple[str, int], list[int]] = {}
        placed: list[tuple] = []
        advance = [SPACE_W if c == " " else CELL_W + CELL_GAP for c in TEXT]
        width = sum(advance) - CELL_GAP
        cursor = -width / 2.0
        for index, char in enumerate(TEXT):
            origin = (cursor, -CELL_H / 2.0)
            cursor += advance[index]
            for stroke_index, stroke in enumerate(GLYPHS[char]):
                chain, chains, previous = [], [], None
                for pos, tangent in _stroke_points(stroke, origin, SPACING):
                    footprint = _footprint(pos, tangent)
                    near = [f for f in placed if float(np.linalg.norm(pos - f[0])) < WIDTH + THICKNESS]
                    if any(_overlaps(footprint, f) for f in near):
                        if chain:
                            chains.append(chain)
                        chain, previous = [], None
                        continue
                    if previous is not None and float(np.linalg.norm(pos - previous)) > MAX_LINK:
                        chains.append(chain)
                        chain = []
                    previous = pos
                    placed.append(footprint)
                    yaw = math.atan2(float(tangent[1]), float(tangent[0]))
                    body = builder.add_body(
                        xform=wp.transform(
                            wp.vec3(float(pos[0]), float(pos[1]), HEIGHT / 2.0),
                            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                        )
                    )
                    t = index / max(len(TEXT) - 1, 1)
                    color = wp.vec3(0.95 - 0.7 * t, 0.35 + 0.4 * t, 0.3 + 0.6 * t)
                    builder.add_shape_box(body, hx=THICKNESS / 2, hy=WIDTH / 2, hz=HEIGHT / 2, cfg=tile, color=color)
                    self.tiles.append(body)
                    chain.append((body, tangent))
                if chain:
                    chains.append(chain)
                for piece_index, piece in enumerate(chains):
                    self.starters.append(piece[0])
                    self.chain_bodies[(f"{char}{index}", stroke_index * 10 + piece_index)] = [b for b, _ in piece]

        self.model = builder.finalize()
        self.model.rigid_contact_max = 24 * (len(self.tiles) + 1)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVERS, solver=str(getattr(args, "solver", "feather_pgs")))
        self.initial_q = self.state_0.body_q.numpy().copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -16.0, 31.0), pitch=-62.0, yaw=90.0)

    def tip(self):
        """Tip the first tile of every stroke on the same frame.

        A tile topples about its bottom edge, so the angular velocity comes with the
        matching centre-of-mass velocity: the chains start from a rotation, not a
        shove, and the word falls the way a hand would start it.
        """
        qd = self.state_0.body_qd.numpy()
        for body, tangent in self.starters:
            axis = np.array([-tangent[1], tangent[0], 0.0])
            qd[body, :3] = np.array([tangent[0], tangent[1], 0.0]) * (self.tip_rate * HEIGHT / 2.0)
            qd[body, 3:] = axis * self.tip_rate
        self.state_0.body_qd.assign(qd)
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        self.stepper.notify_state_edit()
        self.tipped = True

    def on_reset(self):
        """Re-arm the tip after the panel resets the scene."""
        self.tipped = False

    def fallen(self) -> int:
        q = self.state_0.body_q.numpy()[self.tiles]
        up = np.array([wp.quat_rotate(wp.quat(*row[3:7]), wp.vec3(0.0, 0.0, 1.0))[2] for row in q])
        return int(np.count_nonzero(up < 0.5))

    def step(self):
        if not self.tipped and self.sim_time >= TIP_AT:
            self.tip()
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.stepper.collide()
        self.stepper.solve()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("tiles fallen", self.fallen())
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f'"{TEXT}" in {len(self.tiles)} tiles, {len(self.starters)} chains, {self.fallen()} fallen')
        _, self.tip_rate = ui.slider_float("Tip rate [rad/s]", self.tip_rate, 0.5, 10.0)
        if ui.button("Topple the word"):
            self.tip()
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a tile fell through the ground")
        fallen = self.fallen()
        if fallen < 0.8 * len(self.tiles):
            raise ValueError(f"only {fallen} of {len(self.tiles)} tiles fell after {self.sim_time:.0f} s")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="feather_pgs", choices=list(SOLVERS), help="Rigid-body solver.")
        parser.set_defaults(num_frames=420)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
