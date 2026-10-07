# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frame stepping and runtime material updates of the Franka coupled examples."""

import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.examples.multiphysics import (
    example_franka_cable_ik_pick_place,
    example_franka_cloth_block,
    example_franka_shirt_fold_stack,
    example_mujoco_franka_vbd_cable_admm_solver,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

FRAME_GRAPH_EXAMPLES = {
    "franka_cable_ik_pick_place": example_franka_cable_ik_pick_place,
    "franka_cloth_block": example_franka_cloth_block,
    "franka_shirt_fold_stack": example_franka_shirt_fold_stack,
    "mujoco_franka_vbd_cable_admm_solver": example_mujoco_franka_vbd_cable_admm_solver,
}


class TestCoupledFrankaExamples(unittest.TestCase):
    def test_replayed_frame_matches_eager_substep_count(self):
        """Replaying one recorded simulate() advances as many substeps per frame as eager stepping."""
        frames = 4
        with (
            mock.patch.object(newton.examples, "apply_coupled_viewer_forces"),
            mock.patch.object(newton, "eval_ik"),
        ):
            for name, module in FRAME_GRAPH_EXAMPLES.items():
                for substeps in (1, 2, 3):
                    with self.subTest(example=name, substeps=substeps):
                        eager = _counting_example(module, substeps, use_graph=False, recorder=_Recorder())
                        eager_counts = []
                        for _ in range(frames):
                            eager.simulate()
                            eager_counts.append(eager.state_0.substeps)

                        recorder = _Recorder()
                        captured = _counting_example(module, substeps, use_graph=True, recorder=recorder)
                        recorder.tape = []
                        captured.simulate()
                        tape, recorder.tape = recorder.tape, None
                        replay_counts = []
                        for _ in range(frames):
                            for op in tape:
                                op()
                            replay_counts.append(captured.state_0.substeps)

                        self.assertEqual(eager_counts, [substeps * (frame + 1) for frame in range(frames)])
                        self.assertEqual(replay_counts, eager_counts)


def test_cloth_block_graph_matches_eager(test, device):
    """CUDA graph replay of the cloth-block example tracks eager stepping for odd and even substeps."""
    frames = 3
    for substeps in (1, 2, 3):
        results = []
        for graph in (True, False):
            args = ["--viewer", "null", "--rigid-solver", "featherpgs", "--substeps", str(substeps)]
            if not graph:
                args.append("--no-graph-capture")
            with wp.ScopedDevice(device):
                example = _build_example(example_franka_cloth_block, args)
                test.assertEqual(example.graph is not None, graph)
                for _ in range(frames):
                    example.step()
                results.append((example.state_0.body_q.numpy(), example.state_0.particle_q.numpy()))
        (graph_body_q, graph_particle_q), (eager_body_q, eager_particle_q) = results
        np.testing.assert_allclose(graph_body_q, eager_body_q, atol=1.0e-4, err_msg=f"substeps={substeps}")
        np.testing.assert_allclose(graph_particle_q, eager_particle_q, atol=1.0e-4, err_msg=f"substeps={substeps}")


def test_shirt_finger_friction_reaches_coupled_entries(test, device):
    """The shirt's pinch and release friction reaches every coupled entry's shape materials in place."""
    module = example_franka_shirt_fold_stack
    for rigid_solver in ("mujoco", "featherpgs"):
        with wp.ScopedDevice(device):
            example = _build_example(module, ["--viewer", "null", "--rigid-solver", rigid_solver])
        solver = example.solver
        fingers = example.finger_shape_ids.numpy()
        mu_arrays = [example.model.shape_material_mu] + [
            solver.view(name).shape_material_mu for name in solver.entry_names()
        ]
        pointers = [array.ptr for array in mu_arrays]

        # The open-hand start, then mid-lift of the first sleeve pinch.
        for sim_time, expected in ((0.0, module.FINGER_RELEASE_MU), (5.5, module.CLOTH_MU)):
            example.sim_time = sim_time
            example.update_ik_targets()
            for name, array in zip(("parent", *solver.entry_names()), mu_arrays, strict=True):
                np.testing.assert_allclose(
                    array.numpy()[fingers], expected, err_msg=f"{rigid_solver}: {name} at t={sim_time}"
                )
        test.assertEqual([array.ptr for array in mu_arrays], pointers)
        example.step()
        test.assertTrue(np.all(np.isfinite(example.state_0.body_q.numpy())))


devices = get_cuda_test_devices()
add_function_test(
    TestCoupledFrankaExamples,
    "test_cloth_block_graph_matches_eager",
    test_cloth_block_graph_matches_eager,
    devices=devices,
)
add_function_test(
    TestCoupledFrankaExamples,
    "test_shirt_finger_friction_reaches_coupled_entries",
    test_shirt_finger_friction_reaches_coupled_entries,
    devices=devices,
)


class _Recorder:
    """Runs buffer operations at once, or records them for replay like a captured graph while ``tape`` is a list."""

    def __init__(self):
        self.tape = None

    def run(self, op):
        if self.tape is None:
            op()
        else:
            self.tape.append(op)


class _CountingState:
    """State whose only content is the number of substeps integrated into it."""

    def __init__(self, recorder: _Recorder):
        self.recorder = recorder
        self.substeps = 0
        self.joint_q = self.joint_qd = None

    def clear_forces(self):
        pass

    def assign(self, other: "_CountingState"):
        self.recorder.run(lambda: setattr(self, "substeps", other.substeps))


class _CountingSolver:
    def __init__(self, recorder: _Recorder):
        self.recorder = recorder

    def step(self, state_in, state_out, control, contacts, dt):
        self.recorder.run(lambda: setattr(state_out, "substeps", state_in.substeps + 1))


def _counting_example(module, substeps: int, *, use_graph: bool, recorder: _Recorder):
    """An example whose simulate() runs against counting states and a counting solver on the CPU."""
    example = module.Example.__new__(module.Example)
    coords = 9
    example.sim_substeps = substeps
    example.sim_dt = 1.0 / (60.0 * substeps)
    example.use_graph = use_graph
    example.device = "cpu"
    example.world_count = 1
    example.n_coords = coords
    example.finger_idx0, example.finger_idx1 = coords - 2, coords - 1
    example.ik_iters = 1
    example.ik_solver = types.SimpleNamespace(step=lambda *args, **kwargs: None)
    example.ik_joint_q = wp.zeros((1, coords), dtype=float, device="cpu")
    example.control_joint_target_q = wp.zeros((1, coords), dtype=float, device="cpu")
    example.finger_pos_buf = wp.zeros(1, dtype=float, device="cpu")
    example.collision_pipeline = types.SimpleNamespace(collide=lambda *args: None)
    example.model = example.control = example.contacts = None
    example.solver = _CountingSolver(recorder)
    example.state_0 = _CountingState(recorder)
    example.state_1 = _CountingState(recorder)
    return example


def _build_example(module, args: list[str]):
    parsed = module.Example.create_parser().parse_args(args)
    return module.Example(newton.viewer.ViewerNull(num_frames=1000), parsed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
