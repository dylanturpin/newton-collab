# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact adhesion of SolverSteel (``steel:adhesion``)."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0
HALF = 0.1
DENSITY = 100.0
GAP = 0.002
# Weight of the hanging box [N]; a box face touches a box face at four contact points.
WEIGHT = (2.0 * HALF) ** 3 * DENSITY * 9.81
CONTACT_POINTS = 4


def _hanging_box(device, kind="free", drop=0.0, adhesion=0.0, mu=0.5, register=True, wall=False):
    """Return a box under a static ceiling (or beside a static wall) and the ceiling's shape index.

    ``kind="free"`` gives the box a free joint (free-body rows); ``"slider"`` a prismatic joint along
    the contact normal (articulated rows). ``drop`` [m] opens a gap below the ceiling.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    if register:
        SolverSteel.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, gap=GAP)
    if wall:
        support = builder.add_shape_box(
            -1, xform=wp.transform((-0.05, 0.0, 1.0), wp.quat_identity()), hx=0.05, hy=0.5, hz=0.5, cfg=cfg
        )
        position = (HALF + drop, 0.0, 1.0)
        axis = newton.Axis.X
    else:
        support = builder.add_shape_box(
            -1, xform=wp.transform((0.0, 0.0, 1.0 + 0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05, cfg=cfg
        )
        position = (0.0, 0.0, 1.0 - HALF - drop)
        axis = newton.Axis.Z
    xform = wp.transform(position, wp.quat_identity())
    if kind == "free":
        body = builder.add_body(xform=xform)
    else:
        body = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, body, axis=axis, parent_xform=xform)
        builder.add_articulation([joint])
    box_cfg = newton.ModelBuilder.ShapeConfig(mu=mu, gap=GAP, density=DENSITY)
    builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=box_cfg)
    model = builder.finalize(device=device)
    if register and adhesion:
        _set_adhesion(model, support, adhesion)
    return model, support


def _set_adhesion(model, shape, value):
    values = model.steel.adhesion.numpy()
    values[shape] = value
    model.steel.adhesion.assign(values)


class _Run:
    """A solver, pipeline and state pair stepping ``model`` eagerly."""

    def __init__(self, model, contact_matching="disabled", observe=False, **options):
        self.model = model
        self.solver = SolverSteel(model, **options)
        deterministic = options.get("deterministic") is not None
        self.pipeline = newton.CollisionPipeline(model, contact_matching=contact_matching, deterministic=deterministic)
        self.contacts = self.pipeline.contacts()
        self.state, self.next = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        self.observables = None
        if observe:
            self.observables = self.solver.observables({SolverSteel.ObservableKind.CONTACT_F}, contacts=self.contacts)

    def step(self, count=1):
        for _ in range(count):
            self.pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.next, self.control, self.contacts, DT, observables=self.observables)
            self.state, self.next = self.next, self.state

    def box_position(self):
        return self.state.body_q.numpy()[0, :3]


_MODES = {
    "immediate": {},
    "propagation": {"articulated_contact_response": "propagation"},
    "colored": {"solve_mode": "colored", "articulated_contact_response": "propagation"},
}


def _modes(device):
    """Return the solve modes to test; CPU runs the colored mode as the serial propagation sweep."""
    if wp.get_device(device).is_cuda:
        return _MODES
    return {mode: options for mode, options in _MODES.items() if mode != "colored"}


def test_holds_until_pull_off(test, device):
    """Hold a box under a ceiling when the summed pull exceeds its weight and drop it otherwise."""
    for kind in ("free", "slider"):
        for mode, options in _modes(device).items():
            for factor, holds in ((1.1, True), (0.9, False)):
                adhesion = factor * WEIGHT / CONTACT_POINTS
                with test.subTest(kind=kind, mode=mode, holds=holds):
                    run = _Run(_hanging_box(device, kind, adhesion=adhesion)[0], **options)
                    run.step(120)
                    test.assertEqual(int(run.contacts.rigid_contact_count.numpy()[0]), CONTACT_POINTS if holds else 0)
                    z = float(run.box_position()[2])
                    if holds:
                        test.assertAlmostEqual(z, 1.0 - HALF, delta=1.0e-4)
                    else:
                        test.assertLess(z, 0.7)


def test_solver_options(test, device):
    """Hold the box with warm start, velocity-only iterations, point friction and late friction."""
    for name, options in (
        ("warmstart", {"contact_warmstart": True}),
        ("velocity_iterations", {"velocity_iterations": 2}),
        ("point_friction", {"friction_anchor_beta": 0.0}),
        ("late_friction", {"contact_friction_position_iterations": 2}),
    ):
        matching = "latest" if options.get("contact_warmstart") else "disabled"
        for factor, holds in ((1.1, True), (0.9, False)):
            with test.subTest(options=name, holds=holds):
                model = _hanging_box(device, "slider", adhesion=factor * WEIGHT / CONTACT_POINTS)[0]
                run = _Run(model, contact_matching=matching, **options)
                run.step(120)
                z = float(run.box_position()[2])
                if holds:
                    test.assertAlmostEqual(z, 1.0 - HALF, delta=1.0e-4)
                else:
                    test.assertLess(z, 0.7)


def test_release_and_attract(test, device):
    """Drop a held box when its adhesion is zeroed in place, and pull a box up across the gap."""
    adhesion = 1.5 * WEIGHT / CONTACT_POINTS
    for kind in ("free", "slider"):
        with test.subTest(kind=kind, case="release"):
            model, ceiling = _hanging_box(device, kind, adhesion=adhesion)
            run = _Run(model)
            run.step(60)
            test.assertAlmostEqual(float(run.box_position()[2]), 1.0 - HALF, delta=1.0e-4)
            _set_adhesion(model, ceiling, 0.0)
            run.step(60)
            test.assertLess(float(run.box_position()[2]), 0.8)
        for value, attracted in ((adhesion, True), (0.0, False)):
            with test.subTest(kind=kind, case="attract", attracted=attracted):
                model, ceiling = _hanging_box(device, kind, drop=0.5 * GAP, adhesion=adhesion)
                _set_adhesion(model, ceiling, value)
                run = _Run(model)
                run.step(60)
                z = float(run.box_position()[2])
                if attracted:
                    test.assertAlmostEqual(z, 1.0 - HALF, delta=1.0e-4)
                else:
                    test.assertLess(z, 0.8)


def test_friction_includes_adhesion(test, device):
    """Hold a box on a wall when ``mu`` times the summed pull exceeds its weight and slide otherwise."""
    mu = 0.5
    for factor, holds in ((1.25, True), (0.75, False)):
        adhesion = factor * WEIGHT / (mu * CONTACT_POINTS)
        with test.subTest(holds=holds):
            run = _Run(_hanging_box(device, "free", adhesion=adhesion, mu=mu, wall=True)[0])
            run.step(120)
            x, _, z = (float(v) for v in run.box_position())
            test.assertAlmostEqual(x, HALF, delta=1.0e-3)
            if holds:
                test.assertAlmostEqual(z, 1.0, delta=2.0e-3)
            else:
                test.assertLess(z, 0.9)


def test_contact_force_is_net(test, device):
    """Report a hanging box's tensile contacts as the net force, and a separating contact as the full pull."""
    for kind in ("free", "slider"):
        with test.subTest(kind=kind, case="hanging"):
            adhesion = 1.5 * WEIGHT / CONTACT_POINTS
            run = _Run(_hanging_box(device, kind, adhesion=adhesion)[0], observe=True)
            run.step(120)
            count = int(run.contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, CONTACT_POINTS)
            force = run.observables.contact_f.numpy()[:count, :3]
            # Shape 0 is the ceiling, so its force is the box's pull on it, downward.
            test.assertAlmostEqual(float(force[:, 2].sum()), -WEIGHT, delta=1.0e-3 * WEIGHT)
            test.assertTrue(np.all(force[:, 2] >= -adhesion * (1.0 + 1.0e-5)))
        with test.subTest(kind=kind, case="separating"):
            # Too weak to hold: in the gap every contact pulls with its full adhesion and no friction.
            adhesion = 0.25 * WEIGHT / CONTACT_POINTS
            run = _Run(_hanging_box(device, kind, drop=0.5 * GAP, adhesion=adhesion)[0], observe=True)
            run.step(1)
            count = int(run.contacts.rigid_contact_count.numpy()[0])
            test.assertEqual(count, CONTACT_POINTS)
            pull = run.solver._contact_adhesion_impulse.numpy()[:count]
            np.testing.assert_allclose(pull, adhesion * DT, rtol=1.0e-6)
            force = run.observables.contact_f.numpy()[:count, :3]
            np.testing.assert_allclose(force[:, 2], -adhesion, rtol=1.0e-5)
            np.testing.assert_allclose(force[:, :2], 0.0, atol=1.0e-6)


def test_determinism(test, device):
    """Repeat a chain of adhesive boxes hanging from a ceiling bitwise."""

    def simulate(options):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverSteel.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(gap=GAP, density=DENSITY)
        builder.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 1.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05)
        for level in range(3):
            z = 1.0 - HALF - level * (2.0 * HALF + 0.0005)
            body = builder.add_body(xform=wp.transform((0.01 * level, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)
        model = builder.finalize(device=device)
        model.steel.adhesion.fill_(2.0 * WEIGHT)
        run = _Run(model, deterministic=wp.DeterministicMode.RUN_TO_RUN, **options)
        run.step(60)
        return run.state.body_q.numpy()

    for mode, options in _modes(device).items():
        with test.subTest(mode=mode):
            first, second = simulate(options), simulate(options)
            test.assertEqual(first.tobytes(), second.tobytes())
            # The chain hangs together under the ceiling.
            np.testing.assert_allclose(first[:, 2], [0.9, 0.7, 0.5], atol=2.0e-3)


def test_graph_capture(test, device):
    """Replay a captured step, then release the box by zeroing its adhesion between replays."""
    model, ceiling = _hanging_box(device, "slider", adhesion=1.5 * WEIGHT / CONTACT_POINTS)
    run = _Run(model)
    run.step(2)
    with wp.ScopedCapture(device) as capture:
        run.step(2)
    for _ in range(30):
        wp.capture_launch(capture.graph)
    test.assertAlmostEqual(float(run.box_position()[2]), 1.0 - HALF, delta=1.0e-4)
    _set_adhesion(model, ceiling, 0.0)
    for _ in range(30):
        wp.capture_launch(capture.graph)
    test.assertLess(float(run.box_position()[2]), 0.8)


def test_unused_is_bitwise_identical(test, device):
    """Step bitwise like a model without the attribute when every shape's adhesion is zero."""
    for mode, options in _modes(device).items():
        with test.subTest(mode=mode):
            results = []
            for register in (False, True):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                if register:
                    SolverSteel.register_custom_attributes(builder)
                builder.add_ground_plane()
                for i in range(3):
                    body = builder.add_body(xform=wp.transform((0.05 * i, 0.0, 0.11 + 0.21 * i), wp.quat_identity()))
                    builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF)
                run = _Run(builder.finalize(device=device), **options)
                test.assertFalse(run.solver._adhesion_enabled)
                run.step(30)
                results.append(run.state.body_q.numpy())
            test.assertEqual(results[0].tobytes(), results[1].tobytes())


def test_mjcf_import(test, device):
    """Import MuJoCo's geom ``adhesion``, also through a default class, into ``steel:adhesion``."""
    mjcf = """
    <mujoco>
      <default>
        <default class="magnet"><geom adhesion="1.5" gap=".003"/></default>
      </default>
      <worldbody>
        <geom name="frame" type="box" size=".1 .1 .1" class="magnet"/>
        <body name="door" pos="0 0 .5">
          <freejoint/>
          <geom name="plain" type="box" size=".1 .1 .1"/>
          <geom name="strip" type="box" size=".1 .1 .01" pos="0 0 -.1" adhesion="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    builder.add_mjcf(mjcf)
    model = builder.finalize(device=device)
    by_label = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(model.shape_label)}
    adhesion = model.steel.adhesion.numpy()
    test.assertAlmostEqual(float(adhesion[by_label["frame"]]), 1.5, places=6)
    test.assertEqual(float(adhesion[by_label["plain"]]), 0.0)
    test.assertAlmostEqual(float(adhesion[by_label["strip"]]), 2.0, places=6)
    test.assertAlmostEqual(float(model.shape_gap.numpy()[by_label["frame"]]), 0.003, places=6)


def test_validation(test, device):
    """Reject invalid adhesion values and options the adhesion law does not cover."""
    for value in (-1.0, float("nan")):
        with test.subTest(value=value), test.assertRaisesRegex(ValueError, "steel:adhesion"):
            SolverSteel(_hanging_box(device, adhesion=value)[0])
    model, ceiling = _hanging_box(device, adhesion=1.0)
    for options in ({"contact_regularization": 0.1}, {"enable_sleeping": True}):
        with test.subTest(options=options), test.assertRaisesRegex(NotImplementedError, "steel:adhesion"):
            SolverSteel(model, **options)
    solver = SolverSteel(model)
    _set_adhesion(model, ceiling, -1.0)
    with test.assertRaisesRegex(ValueError, "steel:adhesion"):
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    model, ceiling = _hanging_box(device)
    solver = SolverSteel(model)
    _set_adhesion(model, ceiling, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertTrue(any("steel:adhesion is ignored" in str(w.message) for w in caught))


class TestSteelAdhesion(unittest.TestCase):
    pass


devices = get_test_devices()
for _fn in (
    test_holds_until_pull_off,
    test_solver_options,
    test_release_and_attract,
    test_friction_includes_adhesion,
    test_contact_force_is_net,
    test_determinism,
    test_unused_is_bitwise_identical,
    test_mjcf_import,
    test_validation,
):
    add_function_test(TestSteelAdhesion, _fn.__name__, _fn, devices=devices)
add_function_test(TestSteelAdhesion, "test_graph_capture", test_graph_capture, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
