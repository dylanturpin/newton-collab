# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Latency mode's merged setup launches write exactly what the per-array and per-route launches write."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel
from newton.tests.steel.test_propagation import DT, _build_latency_scene

# merged launch -> the launches it replaces; both are methods with the same signature
_PAIRS = {
    "_clear_step_row_state": "_clear_step_row_state_per_array",
    "_snapshot_stage3": "_snapshot_stage3_per_array",
    "_prescale_joint_velocities_warp": "_prescale_joint_velocities",
    "_allocate_velocity_limit_slots_warp": "_allocate_velocity_limit_slots",
    "_populate_velocity_limit_rows_warp": "_populate_velocity_limit_rows",
    "_link_patch_rows": "_link_patch_rows_per_route",
    "_finish_patch_impulses": "_finish_patch_impulses_per_route",
    "_accumulate_row_watermarks_fused": "_accumulate_row_watermarks",
    "_update_qdd_and_integrate_joints": "_update_qdd_then_integrate_joints",
}


def _arrays(obj, prefix="", depth=0, seen=None, out=None) -> dict[str, wp.array]:
    """Every non-empty CUDA value array reachable through attributes, containers and Warp struct fields."""
    seen = set() if seen is None else seen
    out = {} if out is None else out
    if id(obj) in seen or depth > 4:
        return out
    seen.add(id(obj))
    if isinstance(obj, wp.array):
        if obj.size > 0 and obj.device.is_cuda and not wp.types.type_is_struct(obj.dtype):
            out.setdefault(prefix, obj)
        return out
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    elif hasattr(getattr(obj, "_cls", None), "vars"):  # a Warp struct instance
        items = ((name, getattr(obj, name)) for name in obj._cls.vars)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type) and type(obj).__module__.startswith("newton"):
        items = vars(obj).items()
    else:
        return out
    for key, value in items:
        if isinstance(value, (wp.array, dict, list, tuple)) or hasattr(value, "__dict__"):
            _arrays(value, f"{prefix}.{key}" if prefix else str(key), depth + 1, seen, out)
    return out


def _snapshot(objs) -> dict[str, np.ndarray]:
    wp.synchronize_device()
    return {k: v.numpy().copy() for k, v in _arrays(objs).items()}


def _restore(objs, values: dict[str, np.ndarray]) -> None:
    for k, v in _arrays(objs).items():
        if k in values:
            v.assign(values[k])


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)
    )


def _long_chain_scene(device) -> newton.Model:
    """A 40-joint revolute chain swinging under gravity: one articulation wider than a warp of joints."""
    builder = newton.ModelBuilder()
    prev = -1
    joints = []
    for i in range(40):
        link = builder.add_link()
        builder.add_shape_capsule(link, radius=0.01, half_height=0.02)
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0 if i % 2 else 0.0, 0.0 if i % 2 else 1.0),
                parent_xform=wp.transform(wp.vec3(0.05 if i else 0.0, 0.0, 2.0 if i == 0 else 0.0), wp.quat_identity()),
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
        prev = link
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return _with_tight_velocity_limits(builder.finalize(device=device))


def _with_tight_velocity_limits(model: newton.Model) -> newton.Model:
    # tight velocity limits on the revolute arms so the PhysX prescale ratio drops below one
    limits = model.joint_velocity_limit.numpy()
    joint_type = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    for j in np.flatnonzero(joint_type == int(newton.JointType.REVOLUTE)):
        limits[qd_start[j]] = 0.05
    model.joint_velocity_limit.assign(limits)
    return model


def _scene(device) -> newton.Model:
    return _with_tight_velocity_limits(_build_latency_scene(device))


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestFusedSetup(unittest.TestCase):
    """Run the per-pass launches and the merged launch from identical state on real step calls."""

    def _run(self, model=None, pairs=_PAIRS, **solver_options):
        model = _scene("cuda:0") if model is None else model
        solver = SolverSteel(
            model,
            solve_mode="colored",
            articulated_contact_response="propagation",
            iterations=8,
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            row_watermark=True,
            **solver_options,
        )
        self.assertTrue(solver._latency_fused)
        self.assertTrue(solver._fused_setup_active())
        compared = dict.fromkeys(pairs, 0)
        changed = dict.fromkeys(pairs, 0)
        test = self

        def make(fused_name, reference_name):
            fused = getattr(SolverSteel, fused_name)
            reference = getattr(SolverSteel, reference_name)

            def compare(self_, *args):
                objs = {"solver": self_, "args": args}
                before = _snapshot(objs)
                reference(self_, *args)
                expected = _snapshot(objs)
                _restore(objs, before)
                fused(self_, *args)
                actual = _snapshot(objs)
                for k, v in expected.items():
                    test.assertTrue(_bitwise_equal(v, actual[k]), f"{fused_name}: {k} differs")
                compared[fused_name] += 1
                changed[fused_name] += sum(not _bitwise_equal(v, before[k]) for k, v in expected.items())
                if fused_name == "_populate_velocity_limit_rows_warp":
                    # rows that persist from the previous step are rewritten with the same values: count the rows
                    changed[fused_name] += int((self_._velocity_limit_slot.numpy() >= 0).sum())

            return compare

        patches = [patch.object(SolverSteel, f, make(f, pairs[f])) for f in pairs]
        for p in patches:
            p.start()
        try:
            pipeline = newton.CollisionPipeline(model, contact_matching="latest")
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            for _ in range(30):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
        finally:
            for p in patches:
                p.stop()
        for name in pairs:
            self.assertGreater(compared[name], 0, f"{name} never ran")
            # the comparison covered real work: the launches changed the state they started from
            self.assertGreater(changed[name], 0, f"{name} never changed anything")

    def test_merged_launches_match_per_pass_launches(self):
        self._run()

    def test_merged_launches_match_with_contact_warmstart(self):
        # warm start seeds the patch impulses from history, so finishing reads seeded rows
        self._run(contact_warmstart=True)

    def test_merged_launches_match_with_gated_deterministic_rows(self):
        # the limit-gating configuration, with rows reserved in a fixed order (count and write passes)
        self._run(deterministic=True, joint_limit_activation_gap=0.1, velocity_limit_activation_fraction=0.5)

    def test_velocity_limit_launches_match_past_a_warp_of_joints(self):
        # 40 joints: the warp kernels take two rounds of 32 and carry the slot offset between them
        pairs = {
            name: _PAIRS[name]
            for name in (
                "_prescale_joint_velocities_warp",
                "_allocate_velocity_limit_slots_warp",
                "_populate_velocity_limit_rows_warp",
            )
        }
        for options in ({}, {"deterministic": True, "velocity_limit_activation_fraction": 0.5}):
            with self.subTest(**options):
                self._run(_long_chain_scene("cuda:0"), pairs, articulation_constraint_max=512, **options)

    def test_velocity_limit_rows_skip_free_body_groups(self):
        # free bodies have no prismatic, revolute or D6 joint, so their size group never holds velocity-limit rows
        model = _scene("cuda:0")
        solver = SolverSteel(
            model, solve_mode="colored", articulated_contact_response="propagation", enable_joint_velocity_limits=True
        )
        joint_type = model.joint_type.numpy()
        art_start = model.articulation_start.numpy()
        limited_types = (int(newton.JointType.PRISMATIC), int(newton.JointType.REVOLUTE), int(newton.JointType.D6))
        for size in solver._size_groups:
            arts = solver._group_to_art[size].numpy()
            types = np.concatenate([joint_type[art_start[a] : art_start[a + 1]] for a in arts])
            self.assertEqual(size in solver._velocity_limit_row_sizes, bool(np.isin(types, limited_types).any()))
        self.assertLess(len(solver._velocity_limit_row_sizes), len(solver._size_groups))

    def test_per_pass_launches_off_latency_mode(self):
        # more worlds than SMs take the per-pass launches; so does an explicit opt-out
        model = _scene("cuda:0")
        solver = SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")
        self.assertTrue(solver._fused_setup_active())
        solver._latency_fused = False
        self.assertFalse(solver._fused_setup_active())


if __name__ == "__main__":
    unittest.main(verbosity=2)
