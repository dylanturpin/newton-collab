# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that public FK launch layout preserves its numerical and selection API."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.sim import articulation
from newton.tests.test_ik_fk_kernels import _randomize_joint_q
from newton.tests.test_kinematics import _build_dynamic_and_kinematic_single_joint_model
from newton.tests.unittest_utils import add_function_test, get_test_devices

BLOCKS = (16, 256)


def _mixed_model(device, *, requires_grad=False):
    """Mix branched and serial articulations in a batch that leaves partial blocks."""
    builder = newton.ModelBuilder()
    for art in range(35):
        bodies, joints = [], []
        parents = (-1, 0, 1, 0, 3, 4) if art % 2 else (-1, 0, 1, 2, 3, 4)
        for index, kind in enumerate(("free", "revolute", "prismatic", "ball", "d6", "fixed")):
            body = builder.add_link(mass=1.0, com=wp.vec3(0.03, -0.02, 0.04), inertia=wp.mat33(np.eye(3)))
            options = {
                "parent": -1 if parents[index] < 0 else bodies[parents[index]],
                "child": body,
                "parent_xform": wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_rpy(0.1, -0.2, 0.3)),
                "child_xform": wp.transform(wp.vec3(-0.02, 0.04, 0.01), wp.quat_rpy(-0.1, 0.2, 0.05)),
            }
            if kind in ("revolute", "prismatic"):
                options["axis"] = newton.Axis.Y
            elif kind == "d6":
                axes = [newton.ModelBuilder.JointDofConfig(axis=axis) for axis in newton.Axis]
                options.update(linear_axes=axes, angular_axes=axes)
            joints.append(getattr(builder, f"add_joint_{kind}")(**options))
            bodies.append(body)
        builder.add_articulation(joints)
    model = builder.finalize(device=device, requires_grad=requires_grad)
    _randomize_joint_q(model, seed=17)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


def _fk(model, state, block, **selection):
    """Exercise the public wrapper, overriding only the kernel launch block size."""
    launch = wp.launch

    def configured(*args, **kwargs):
        if kwargs.get("kernel") is not articulation.eval_articulation_fk:
            raise AssertionError("Unexpected kernel in public FK")
        return launch(*args, **dict(kwargs, block_dim=block))

    with patch.object(wp, "launch", side_effect=configured):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state, **selection)


def _sentinel_state(model):
    state = model.state()
    state.body_q.fill_(wp.transform(wp.vec3(7.0, -3.0, 2.0), wp.quat_identity()))
    state.body_qd.fill_(wp.spatial_vector(2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
    return state


def test_fk_layout_selection(test, device):
    """Preserve exact mixed-tree poses, velocities, masks and indexed partial updates."""
    with wp.ScopedDevice(device):
        model = _mixed_model(device)
        inputs = (model.joint_q.numpy().copy(), model.joint_qd.numpy().copy())
        selected = np.arange(model.articulation_count) % 2 == 0
        choices = (
            ({}, np.ones(model.articulation_count, dtype=bool)),
            ({"mask": wp.array(selected, dtype=wp.bool, device=device)}, selected),
            (
                {"indices": wp.array([-1, 100, *np.flatnonzero(selected)[::-1]], dtype=wp.int32, device=device)},
                selected,
            ),
        )
        child_bodies = model.joint_child.numpy()
        joint_articulations = model.joint_articulation.numpy()
        for selection, active in choices:
            reference = _sentinel_state(model)
            untouched = (reference.body_q.numpy().copy(), reference.body_qd.numpy().copy())
            _fk(model, reference, 256, **selection)
            expected = (reference.body_q.numpy(), reference.body_qd.numpy())
            excluded = child_bodies[~active[joint_articulations]]
            for before, after in zip(untouched, expected, strict=True):
                np.testing.assert_array_equal(after[excluded], before[excluded])
            for block in BLOCKS:
                with test.subTest(block=block, selection=tuple(selection)):
                    state = _sentinel_state(model)
                    _fk(model, state, block, **selection)
                    np.testing.assert_array_equal(state.body_q.numpy(), expected[0])
                    np.testing.assert_array_equal(state.body_qd.numpy(), expected[1])
        np.testing.assert_array_equal(model.joint_q.numpy(), inputs[0])
        np.testing.assert_array_equal(model.joint_qd.numpy(), inputs[1])


def test_fk_layout_body_flags(test, device):
    """Retain unmatched dynamic or kinematic body state at every block size."""
    with wp.ScopedDevice(device):
        model = _build_dynamic_and_kinematic_single_joint_model(device)
        model.joint_q.assign(np.array([0.3, -0.4], dtype=np.float32))
        model.joint_qd.assign(np.array([0.7, -0.8], dtype=np.float32))
        for flag in (newton.BodyFlags.DYNAMIC, newton.BodyFlags.KINEMATIC):
            reference = _sentinel_state(model)
            before = (reference.body_q.numpy().copy(), reference.body_qd.numpy().copy())
            _fk(model, reference, 256, body_flag_filter=int(flag))
            excluded = (model.body_flags.numpy() & int(flag)) == 0
            expected = (reference.body_q.numpy(), reference.body_qd.numpy())
            for initial, final in zip(before, expected, strict=True):
                np.testing.assert_array_equal(final[excluded], initial[excluded])
            for block in BLOCKS:
                state = _sentinel_state(model)
                _fk(model, state, block, body_flag_filter=int(flag))
                np.testing.assert_array_equal(state.body_q.numpy(), expected[0])
                np.testing.assert_array_equal(state.body_qd.numpy(), expected[1])


def test_fk_layout_gradients(test, device):
    """Preserve position and velocity adjoints across independent-articulation layouts."""
    with wp.ScopedDevice(device):
        model = _mixed_model(device, requires_grad=True)
        reference = None
        for block in (256, *BLOCKS[:-1]):
            state = model.state()
            with wp.Tape() as tape:
                _fk(model, state, block)
            tape.backward(grads={state.body_q: wp.ones_like(state.body_q), state.body_qd: wp.ones_like(state.body_qd)})
            gradients = (model.joint_q.grad.numpy().copy(), model.joint_qd.grad.numpy().copy())
            if reference is None:
                reference = gradients
                for gradient in gradients:
                    test.assertTrue(np.isfinite(gradient).all())
                    test.assertGreater(float(np.max(np.abs(gradient))), 0.01)
            else:
                for actual, expected in zip(gradients, reference, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            tape.zero()


def test_fk_layout_graph_replay(test, device):
    """Replay captured FK layouts after changing joint state without recapturing."""
    if not device.is_cuda:
        test.skipTest("CUDA graphs require CUDA")
    with wp.ScopedDevice(device):
        model = _mixed_model(device)
        states, graphs = [], []
        mask = wp.array(np.arange(model.articulation_count) % 2 == 0, dtype=wp.bool, device=device)
        for block in BLOCKS:
            state = _sentinel_state(model)
            _fk(model, state, block, mask=mask)
            with wp.ScopedCapture(device=device) as capture:
                _fk(model, state, block, mask=mask)
            states.append(state)
            graphs.append(capture.graph)
        _randomize_joint_q(model, seed=29)
        model.joint_qd.fill_(0.35)
        reference = _sentinel_state(model)
        _fk(model, reference, 256, mask=mask)
        for state, graph in zip(states, graphs, strict=True):
            wp.capture_launch(graph)
            np.testing.assert_array_equal(state.body_q.numpy(), reference.body_q.numpy())
            np.testing.assert_array_equal(state.body_qd.numpy(), reference.body_qd.numpy())


class TestFKLaunchLayout(unittest.TestCase):
    def test_launch_policy_uses_selected_articulation_count(self):
        """Use narrow CUDA blocks only when selection supplies multiple blocks."""
        model = _mixed_model("cpu")
        state = model.state()
        choices = [({}, 35), ({"mask": wp.ones(35, dtype=wp.bool, device="cpu")}, 35)]
        choices.extend(
            ({"indices": wp.array(np.arange(count), dtype=wp.int32, device="cpu")}, count) for count in (0, 1, 16, 17)
        )
        for is_cuda in (False, True):
            # Only host dispatch is inspected; all arrays remain on CPU and no
            # kernel launches against this stand-in device.
            selected_model = SimpleNamespace(**vars(model))
            selected_model.device = SimpleNamespace(is_cuda=is_cuda)
            for selection, count in choices:
                with self.subTest(cuda=is_cuda, selection=tuple(selection), count=count):
                    with patch.object(wp, "launch") as launch:
                        newton.eval_fk(selected_model, model.joint_q, model.joint_qd, state, **selection)
                    launch.assert_called_once()
                    self.assertEqual(launch.call_args.kwargs["dim"], count)
                    expected = 16 if is_cuda and count > 16 else 256
                    self.assertEqual(launch.call_args.kwargs.get("block_dim", 256), expected)


for function in (
    test_fk_layout_selection,
    test_fk_layout_body_flags,
    test_fk_layout_gradients,
    test_fk_layout_graph_replay,
):
    add_function_test(TestFKLaunchLayout, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
