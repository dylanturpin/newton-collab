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
    "_link_patch_rows": "_link_patch_rows_per_route",
    "_finish_patch_impulses": "_finish_patch_impulses_per_route",
    "_accumulate_row_watermarks_fused": "_accumulate_row_watermarks",
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


def _scene(device) -> newton.Model:
    model = _build_latency_scene(device)
    # tight velocity limits on the revolute arms so the PhysX prescale ratio drops below one
    limits = model.joint_velocity_limit.numpy()
    joint_type = model.joint_type.numpy()
    qd_start = model.joint_qd_start.numpy()
    for j in np.flatnonzero(joint_type == int(newton.JointType.REVOLUTE)):
        limits[qd_start[j]] = 0.05
    model.joint_velocity_limit.assign(limits)
    return model


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestFusedSetup(unittest.TestCase):
    """Run the per-pass launches and the merged launch from identical state on real step calls."""

    def _run(self, **solver_options):
        model = _scene("cuda:0")
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
        compared = dict.fromkeys(_PAIRS, 0)
        changed = dict.fromkeys(_PAIRS, 0)
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

            return compare

        patches = [patch.object(SolverSteel, f, make(f, r)) for f, r in _PAIRS.items()]
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
        for name in _PAIRS:
            self.assertGreater(compared[name], 0, f"{name} never ran")
            # the comparison covered real work: the launches changed the state they started from
            self.assertGreater(changed[name], 0, f"{name} never changed anything")

    def test_merged_launches_match_per_pass_launches(self):
        self._run()

    def test_merged_launches_match_with_contact_warmstart(self):
        # warm start seeds the patch impulses from history, so finishing reads seeded rows
        self._run(contact_warmstart=True)

    def test_per_pass_launches_off_latency_mode(self):
        # more worlds than SMs take the per-pass launches; so does an explicit opt-out
        model = _scene("cuda:0")
        solver = SolverSteel(model, solve_mode="colored", articulated_contact_response="propagation")
        self.assertTrue(solver._fused_setup_active())
        solver._latency_fused = False
        self.assertFalse(solver._fused_setup_active())


if __name__ == "__main__":
    unittest.main(verbosity=2)
