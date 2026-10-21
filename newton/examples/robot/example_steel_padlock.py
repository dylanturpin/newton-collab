# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Insert a profiled key, align five pin stacks and release a padlock's shackle.

Command: python -m newton.examples steel_padlock
Wrong key: python -m newton.examples steel_padlock --key wrong

The cutaway changes display meshes only. Contact drives the passive lock mechanism;
only the free key receives bounded insertion forces and turning torque.
"""

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSteel

from . import _padlock


class Example:
    def __init__(self, viewer, args):
        if not wp.get_device().is_cuda:
            raise RuntimeError("The Steel padlock example requires a CUDA device")
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.key = args.key
        self.cutaway = args.cutaway
        newton.use_coord_layout_targets = True
        scene, self.design, key_body, self.coordinates, self.views = _padlock.build(self.key)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_world(scene)
        self.model = builder.finalize()
        self.pipeline = newton.CollisionPipeline(
            self.model, rigid_contact_max=4096, deterministic=True, convex_face_clip=False
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverSteel(
            self.model,
            iterations=args.iterations,
            velocity_iterations=0,
            solve_mode="matrix_free",
            articulated_contact_response="immediate",
            row_schedule="interleaved",
            drive_mode="augmented",
            articulation_constraint_max=1024,
            free_body_constraint_max=1024,
            enable_joint_limits=True,
            contact_friction_gap_max=self.design["scale"] * 0.01,
            contact_regularization=0.001,
            baumgarte=0.2,
            enable_restitution=False,
            deterministic=True,
            row_watermark=True,
        )
        self.observables = self.solver.observables([self.solver.ObservableKind.CONTACT_F], contacts=self.contacts)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.initial = self.model.state()
        self.initial.assign(self.state_0)
        self.control = self.model.control()
        self.clock = wp.zeros(1, dtype=float)
        self.playing = wp.ones(1, dtype=int)
        self.metrics = wp.zeros(4, dtype=int)
        self.key_ids = wp.array([key_body], dtype=int)
        self.key_mass = wp.array([scene.body_mass[key_body]], dtype=float)
        self.key_inertia = wp.array([scene.body_inertia[key_body]], dtype=wp.mat33)
        self.target = wp.array(
            [0, 0, self.design["seated_target_z"] * _padlock.LENGTH, self.design["start_z"] * _padlock.LENGTH],
            dtype=float,
        )
        self.rotation = wp.array([wp.quat_identity()], dtype=wp.quat)
        self.command = wp.zeros(1, dtype=wp.spatial_vector)
        self.opacities = self.model.shape_opacity.numpy()
        self.set_cutaway(self.cutaway)
        self.viewer.set_model(self.model)
        self.viewer.show_ground = False
        self.viewer.set_camera(pos=wp.vec3(-60, -85, 70), pitch=-20, yaw=55)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.look_at((0, 0, 32))
            self.viewer.camera.fov = 70
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.shadow_extents = 150
            renderer.spotlight_enabled = False
            renderer.exposure = 2.0
        self.graph = None
        self.simulate()
        self.reset()
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self.reset()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=216)
        parser.add_argument("--substeps", type=int, default=32, help="Physics substeps per 60 Hz frame")
        parser.add_argument("--iterations", type=int, default=16, help="Steel sweeps per substep")
        parser.add_argument("--key", choices=("matched", "wrong"), default="matched")
        parser.add_argument("--cutaway", action=argparse.BooleanOptionalAction, default=True)
        return parser

    def set_cutaway(self, enabled):
        """Switch visual opacity without touching physical geometry or state."""
        self.cutaway = enabled
        self.opacities[self.views["full"]] = float(not enabled)
        self.opacities[self.views["cutaway"]] = float(enabled)
        self.model.shape_opacity.assign(self.opacities)

    def reset(self):
        """Restore the initial physical state and clear controller/solver history."""
        self.state_0.assign(self.initial)
        self.state_1.assign(self.initial)
        self.solver.reset(self.state_0)
        self.pipeline.reset_contact_matching()
        self.clock.zero_()
        self.playing.fill_(1)
        self.metrics.zero_()
        self.sim_time = 0.0
        self.has_turned = False

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _padlock.drive,
                1,
                inputs=[
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.key_ids,
                    self.model.body_com,
                    self.clock,
                    self.target,
                    self.rotation,
                    self.key_mass,
                    self.key_inertia,
                    self.command,
                    self.sim_dt,
                    self.design["turn"],
                ],
            )
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt, observables=self.observables
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                _padlock.monitor,
                max(self.model.joint_coord_count, len(self.solver.constraint_overflow)),
                inputs=[
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    self.solver.constraint_overflow,
                    self.contacts.rigid_contact_count,
                    self.metrics,
                    self.contacts.rigid_contact_max,
                ],
            )
            wp.launch(_padlock.tick, 1, inputs=[self.clock, self.playing, self.sim_dt])
        if self.sim_substeps % 2:
            self.state_1.assign(self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1, observables=self.observables)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Insert key, align pin stacks, turn plug, release shackle.")
        changed, cutaway = ui.checkbox("Cutaway", self.cutaway)
        if changed:
            self.set_cutaway(cutaway)
        playing = bool(self.playing.numpy()[0])
        changed, playing = ui.checkbox("Run key path", playing)
        if changed:
            self.playing.fill_(int(playing))
        changed, phase = ui.slider_float("Key path [s]", float(self.clock.numpy()[0]), 0.0, 3.6)
        if changed:
            self.clock.fill_(phase)
            self.playing.zero_()
        if ui.button("Reset scene"):
            self.reset()
        q = self.state_0.joint_q.numpy()
        ui.text(f"Plug: {np.rad2deg(q[self.coordinates['plug_turn']]):.1f} deg")
        ui.text(f"Shackle lift: {q[self.coordinates['shackle_lift']]:.2f} mm")
        ui.text(f"Applied key force: {np.linalg.norm(self.command.numpy()[0, :3]):.2f} N")

    def test_post_step(self):
        """Keep the shackle latched until contact has turned the plug."""
        q = self.state_0.joint_q.numpy()
        self.has_turned |= q[self.coordinates["plug_turn"]] > 0.05
        if not self.has_turned:
            assert q[self.coordinates["shackle_lift"]] < 1.0, "Shackle released before the plug turned"

    def test_final(self):
        """Verify matched-key release or wrong-key blocking without lost solver rows."""
        assert self.sim_time >= 3.6 - 1e-6, "Run at least 216 frames to complete the demonstration"
        assert not self.metrics.numpy()[1:].any(), "Nonfinite state or contact/constraint capacity overflow"
        stats = self.solver.constraint_row_watermarks()
        assert all(v == 0 for k, v in stats.items() if "overflow" in k or "dropped" in k), stats
        q = self.state_0.joint_q.numpy()
        pose = self.state_0.body_q.numpy()[int(self.key_ids.numpy()[0])]
        assert abs(pose[2]) < 0.2, f"Key did not seat: z={pose[2]} mm"
        plug, shackle = q[self.coordinates["plug_turn"]], q[self.coordinates["shackle_lift"]]
        if self.key == "matched":
            assert plug > 1.0 and shackle >= self.design["open_travel"] * _padlock.LENGTH, (plug, shackle)
        else:
            assert abs(plug) < 0.05 and shackle < 1.0, (plug, shackle)
        before = self.model.shape_flags.numpy()
        cutaway = self.cutaway
        self.set_cutaway(not self.cutaway)
        assert np.array_equal(before, self.model.shape_flags.numpy()), "Cutaway changed collision flags"
        self.set_cutaway(cutaway)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
