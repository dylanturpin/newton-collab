# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo-style fixed and spatial tendons in SolverSteel, checked against closed-form mechanics."""

import math
import unittest
import warnings

import numpy as np

import newton
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

G = 9.81
_HEAD = (
    '<option gravity="0 0 -9.81"><flag contact="disable"/></option>'
    '<default><geom contype="0" conaffinity="0"/></default>'
)


def _slider(name, pos, axis="0 0 1", mass=1.0):
    return (
        f'<body name="{name}" pos="{pos}"><joint name="{name}" type="slide" axis="{axis}"/>'
        f'<geom type="box" size="0.05 0.05 0.05" mass="{mass}"/><site name="{name}_s" pos="0 0 0.05"/></body>'
    )


def _frame(children):
    return f'<body name="frame"><geom type="sphere" size="0.02" mass="0.1"/>{children}</body>'


def _atwood_mjcf(m1, m2, r=0.1, zp=2.5, z0=1.0):
    length = 2.0 * (zp - (z0 + 0.05)) + math.pi * r
    return f"""<mujoco>{_HEAD}<worldbody>
      <geom name="pulley" type="cylinder" size="{r} 0.02" pos="0 0 {zp}" euler="90 0 0"/>
      <site name="side" pos="0 0 {zp + 0.3}"/>
      {_frame(_slider("m1", f"{-r} 0 {z0}", mass=m1) + _slider("m2", f"{r} 0 {z0}", mass=m2))}
    </worldbody><tendon><spatial limited="true" range="0 {length:.9f}">
      <site site="m1_s"/><geom geom="pulley" sidesite="side"/><site site="m2_s"/></spatial></tendon></mujoco>"""


class _Run:
    """A one-world model from MJCF stepped by SolverSteel, with joint access by name."""

    def __init__(
        self,
        mjcf,
        device,
        *,
        solve_mode="serial",
        qd=None,
        q=None,
        ctrl=None,
        joint_f=None,
        parent_body=False,
        **solver_kwargs,
    ):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -G))
        parent = -1
        if parent_body:
            parent = builder.add_link(mass=0.0)
            builder.add_articulation([builder.add_joint_fixed(parent=-1, child=parent)])
        builder.add_mjcf(mjcf, parent_body=parent)
        builder.joint_armature = [0.0] * len(builder.joint_armature)
        self.model = builder.finalize(device=device)
        self.labels = list(self.model.joint_label)
        joint_q, joint_qd = self.model.joint_q.numpy(), self.model.joint_qd.numpy()
        for values, target, start in ((q, joint_q, "joint_q_start"), (qd, joint_qd, "joint_qd_start")):
            for name, value in (values or {}).items():
                target[getattr(self.model, start).numpy()[self.joint(name)]] = value
        self.model.joint_q.assign(joint_q)
        self.model.joint_qd.assign(joint_qd)
        if solve_mode == "colored":
            solver_kwargs.setdefault("articulated_contact_response", "propagation")
        self.solver = SolverSteel(self.model, solve_mode=solve_mode, **solver_kwargs)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        if ctrl is not None:
            self.control.mujoco.ctrl.assign(ctrl)
        if joint_f:
            forces = self.control.joint_f.numpy()
            for name, value in joint_f.items():
                forces[self.model.joint_qd_start.numpy()[self.joint(name)]] = value
            self.control.joint_f.assign(forces)

    def joint(self, name):
        return next(i for i, label in enumerate(self.labels) if label.endswith(name))

    def step(self, dt, steps):
        for _ in range(steps):
            self.solver.step(self.state_0, self.state_1, self.control, None, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def q(self, name):
        return float(self.state_0.joint_q.numpy()[self.model.joint_q_start.numpy()[self.joint(name)]])

    def qd(self, name):
        return float(self.state_0.joint_qd.numpy()[self.model.joint_qd_start.numpy()[self.joint(name)]])


_MODES = ("serial", "colored")


def test_tendon_atwood_machine(test, device):
    """A rope over a cylinder wrap: a = (m1 - m2) g / (m1 + m2) and tension 2 m1 m2 g / (m1 + m2)."""
    m1, m2, dt, steps = 3.0, 1.0, 1.0e-3, 300
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(_atwood_mjcf(m1, m2), device, solve_mode=mode, iterations=32)
            run.step(dt, steps - 100)
            # one step's impulse carries the float32 position round-off over dt^2; average the force instead
            tension = 0.0
            for _ in range(100):
                run.step(dt, 1)
                tension -= float(run.solver.tendon_forces(dt).numpy()[0, 0]) / 100.0
            accel = (m1 - m2) * G / (m1 + m2)
            # semi-implicit Euler integrates a constant acceleration exactly in velocity
            test.assertAlmostEqual(run.qd("m1"), -accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("m2"), accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(tension, 2.0 * m1 * m2 * G / (m1 + m2), delta=0.02)
            length = float(run.solver.tendon_lengths().numpy()[0])
            test.assertAlmostEqual(length, 2.0 * 1.45 + math.pi * 0.1, delta=1.0e-3)


def test_tendon_slack_rope_jerks_plastically(test, device):
    """A thrown mass slackens the rope (free fall), which then snaps taut with an inelastic impulse."""
    m1, m2, v0, dt = 1.5, 1.0, 2.0, 1.0e-3
    t_star = v0 / G
    run = _Run(_atwood_mjcf(m1, m2), device, qd={"m2": v0}, iterations=32)
    run.step(dt, int(0.15 / dt))
    test.assertAlmostEqual(run.qd("m1"), -G * 0.15, delta=2.0e-3)  # no rope force while slack
    test.assertAlmostEqual(run.qd("m2"), v0 - G * 0.15, delta=2.0e-3)
    run.step(dt, int(0.15 / dt))
    impulse = v0 / (1.0 / m1 + 1.0 / m2)
    accel = (m1 - m2) * G / (m1 + m2)
    after = 0.3 - t_star
    test.assertAlmostEqual(run.qd("m1"), -v0 + impulse / m1 - accel * after, delta=0.02)
    test.assertAlmostEqual(run.qd("m2"), impulse / m2 + accel * after, delta=0.02)


def test_tendon_pulley_divisor_tackle(test, device):
    """A 2:1 tackle, L = |A - s2| + |B - s1| / 2, moves the effort mass twice as far as the load."""
    m1, m2, dt, steps = 1.0, 3.0, 1.0e-3, 300
    length = (2.5 - 1.05) + (2.5 - 1.25) / 2.0
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      <site name="A" pos="0.3 0 2.5"/><site name="B" pos="-0.3 0 2.5"/>
      {_frame(_slider("m1", "-0.3 0 1.2", mass=m1) + _slider("m2", "0.3 0 1.0", mass=m2))}
    </worldbody><tendon><spatial limited="true" range="0 {length:.9f}">
      <site site="A"/><site site="m2_s"/><pulley divisor="2"/><site site="B"/><site site="m1_s"/>
    </spatial></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, iterations=32)
            run.step(dt, steps)
            accel = (m2 - 2.0 * m1) * G / (2.0 * m1 + 0.5 * m2)
            test.assertAlmostEqual(run.qd("m1"), accel * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("m2"), -0.5 * accel * steps * dt, delta=2.0e-3)


def test_tendon_fixed_spring_damper(test, device):
    """A fixed tendon L = q1 + 2 q2 with stiffness and damping rings as the damped oscillator of L."""
    m1, m2, k, b, q10, dt = 1.0, 2.0, 50.0, 1.0, 0.1, 5.0e-4
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0", mass=m1) + _slider("m2", "0 0.2 1", axis="1 0 0", mass=m2))}
    </worldbody><tendon><fixed stiffness="{k}" damping="{b}" springlength="0">
      <joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed></tendon></mujoco>"""
    inv_mass = 1.0 / m1 + 4.0 / m2
    wn = math.sqrt(k * inv_mass)
    zeta = b * inv_mass / (2.0 * wn)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    for mode, passive in ((m, p) for m in _MODES for p in ("implicit_damping", "implicit")):
        with test.subTest(solve_mode=mode, tendon_passive_mode=passive):
            run = _Run(mjcf, device, solve_mode=mode, q={"m1": q10}, iterations=32, tendon_passive_mode=passive)
            for t in (0.2, 0.4, 0.6):
                run.step(dt, int(round(0.2 / dt)))
                exact = q10 * math.exp(-zeta * wn * t) * (math.cos(wd * t) + zeta * wn / wd * math.sin(wd * t))
                length = run.q("m1") + 2.0 * run.q("m2")
                # implicit Euler adds O(wn dt) numerical damping
                test.assertAlmostEqual(length, exact, delta=3.0e-3)
            # the tendon force acts along J = (1, 2), so the momentum combination 2 p1 - p2 stays zero
            test.assertAlmostEqual(m1 * run.qd("m1") * 2.0 - m2 * run.qd("m2"), 0.0, delta=1.0e-4)


def test_tendon_friction_loss_sticks_and_slips(test, device):
    """Tendon friction loss holds an incline below its bound and slows a slip above it by f / m."""
    theta, dt, steps = math.radians(30.0), 1.0e-3, 400
    axis = f"{math.cos(theta)} 0 {-math.sin(theta)}"
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("stick", "0 -0.4 1", axis=axis) + _slider("slip", "0 0 1", axis=axis) + _slider("stop", "0 0.4 1", axis="1 0 0"))}
    </worldbody><tendon>
      <fixed frictionloss="6"><joint joint="stick" coef="1"/></fixed>
      <fixed frictionloss="3"><joint joint="slip" coef="1"/></fixed>
      <fixed frictionloss="2"><joint joint="stop" coef="1"/></fixed>
    </tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, qd={"stop": 1.0}, iterations=32)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("stick"), 0.0, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("slip"), (G * math.sin(theta) - 3.0) * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(run.qd("stop"), 1.0 - 2.0 * steps * dt, delta=2.0e-3)
            run.step(dt, 200)  # past t = m v0 / f = 0.5 s the slide has stopped and stays stopped
            test.assertAlmostEqual(run.qd("stop"), 0.0, delta=1.0e-4)


def test_tendon_across_articulations_is_ignored(test, device):
    """A tendon coupling two articulations warns and is not enforced."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_slider("a", "-0.2 0 1")}{_slider("b", "0.2 0 1")}
    </worldbody><tendon><fixed stiffness="100"><joint joint="a" coef="1"/><joint joint="b" coef="1"/></fixed></tendon></mujoco>"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _Run(mjcf, device, iterations=8)
    test.assertTrue(any("cross-articulation" in str(w.message) for w in caught))
    test.assertIsNone(run.solver.tendon_lengths())
    run.step(1.0e-3, 100)
    test.assertAlmostEqual(run.qd("a"), -G * 0.1, delta=1.0e-3)


def _winch_mjcf(actuator):
    return f"""<mujoco>{_HEAD}<worldbody><site name="top" pos="0 0 2.5"/>
      {_frame(_slider("m", "0 0 1"))}
    </worldbody><tendon><spatial name="rope"><site site="top"/><site site="m_s"/></spatial></tendon>
    <actuator>{actuator}</actuator></mujoco>"""


def test_tendon_motor_lifts_a_mass(test, device):
    """A tendon motor's force acts along dL/dq: a negative (shortening) force lifts a hanging mass."""
    pull, dt, steps = 14.715, 1.0e-3, 300
    mjcf = _winch_mjcf('<motor tendon="rope" ctrlrange="-100 100"/>')
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[-pull], iterations=32)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("m"), (pull - G) * steps * dt, delta=2.0e-3)
            test.assertAlmostEqual(float(run.solver.tendon_actuator_forces().numpy()[0]), -pull, delta=1.0e-3)


def test_tendon_servo_saturates_at_its_force_range(test, device):
    """A saturated tendon servo reports its force-range bound and accelerates just below the bound's rate.

    Like a joint drive's effort limit, the range clamps the explicit force while the servo's stiffness and
    damping stay implicit in the mass matrix, which slows the saturated lift slightly (2 % here).
    """
    kp, fmax, dt = 200.0, 30.0, 1.0e-3
    mjcf = _winch_mjcf(
        f'<general tendon="rope" biastype="affine" gainprm="{kp}" biasprm="0 {-kp} -20" ctrlrange="0 3" '
        f'forcerange="{-fmax} {fmax}"/>'
    )
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, ctrl=[1.15], iterations=32)  # 0.3 m short of L = 1.45
            run.step(dt, 50)
            test.assertAlmostEqual(float(run.solver.tendon_actuator_forces().numpy()[0]), -fmax, delta=1.0e-3)
            saturated = (fmax - G) * 50 * dt
            test.assertLess(run.qd("m"), saturated)
            test.assertGreater(run.qd("m"), 0.95 * saturated)


def test_tendon_armature_couples_joint_inertia(test, device):
    """Fixed-tendon armature adds a J^T J to the mass matrix, so a force on q1 alone also accelerates q2."""
    a, dt, steps = 0.5, 1.0e-3, 200
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0") + _slider("m2", "0 0.2 1", axis="1 0 0", mass=2.0))}
    </worldbody><tendon><fixed armature="{a}"><joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed>
    </tendon></mujoco>"""
    jac = np.array([1.0, 2.0])
    accel = np.linalg.solve(np.diag([1.0, 2.0]) + a * np.outer(jac, jac), [1.0, 0.0])
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={"m1": 1.0}, iterations=8)
            run.step(dt, steps)
            test.assertAlmostEqual(run.qd("m1"), accel[0] * steps * dt, delta=1.0e-4)
            test.assertAlmostEqual(run.qd("m2"), accel[1] * steps * dt, delta=1.0e-4)


def test_tendon_implicit_spring_survives_large_steps(test, device):
    """With tendon_passive_mode="implicit" a stiff tendon spring stays bounded at omega * dt = 4.6."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0", mass=0.02) + _slider("m2", "0 0.2 1", axis="1 0 0", mass=0.04))}
    </worldbody><tendon><fixed stiffness="2000" springlength="0">
      <joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed></tendon></mujoco>"""
    run = _Run(mjcf, device, q={"m1": 0.01}, iterations=8, tendon_passive_mode="implicit")
    run.step(1.0 / 120.0, 60)  # omega = 548 rad/s, dt = 8.3 ms
    test.assertLessEqual(abs(run.q("m1") + 2.0 * run.q("m2")), 0.01)


def test_tendon_passive_mode_is_validated(test, device):
    """An unknown tendon_passive_mode raises."""
    mjcf = f"""<mujoco>{_HEAD}<worldbody>{_frame(_slider("m", "0 0 1"))}</worldbody>
    <tendon><fixed stiffness="1"><joint joint="m" coef="1"/></fixed></tendon></mujoco>"""
    with test.assertRaises(ValueError):
        _Run(mjcf, device, tendon_passive_mode="explicit")


def test_tendon_lengths_include_tendons_without_rows(test, device):
    """Tendons without a limit, spring, friction loss, or actuator still report their length, as MuJoCo's ten_length."""
    dt = 1.0e-3
    mjcf = f"""<mujoco>{_HEAD}<worldbody>
      {_frame(_slider("m1", "0 -0.2 1", axis="1 0 0") + _slider("m2", "0 0.2 1", axis="1 0 0"))}
    </worldbody><tendon><fixed><joint joint="m1" coef="1"/><joint joint="m2" coef="2"/></fixed>
      <fixed armature="0.5"><joint joint="m1" coef="-1"/></fixed>
      <spatial><site site="m1_s"/><site site="m2_s"/></spatial></tendon></mujoco>"""
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={"m1": 1.0}, iterations=8)
            run.step(dt, 99)
            q1, q2 = run.q("m1"), run.q("m2")  # lengths are evaluated at the input state of the step
            run.step(dt, 1)
            test.assertGreater(q1, 1.0e-3)
            lengths = run.solver.tendon_lengths().numpy()
            np.testing.assert_allclose(lengths, [q1 + 2.0 * q2, -q1, math.hypot(q2 - q1, 0.4)], atol=1.0e-5)


def test_tendon_rows_beyond_the_contact_budget_are_kept(test, device):
    """More tendon rows than articulation_constraint_max: every tendon still holds its limit."""
    count, dt, steps = 40, 1.0e-3, 200
    sliders = "".join(_slider(f"m{i}", f"0 {0.2 * i} 1", axis="1 0 0") for i in range(count))
    tendons = "".join(
        f'<fixed limited="true" range="-0.05 0.05"><joint joint="m{i}" coef="1"/></fixed>' for i in range(count)
    )
    mjcf = f"<mujoco>{_HEAD}<worldbody>{_frame(sliders)}</worldbody><tendon>{tendons}</tendon></mujoco>"
    for mode in _MODES:
        with test.subTest(solve_mode=mode):
            run = _Run(mjcf, device, solve_mode=mode, joint_f={f"m{i}": 10.0 for i in range(count)}, iterations=16)
            run.step(dt, steps)
            q = np.array([run.q(f"m{i}") for i in range(count)])
            # a free 1 kg slider pushed by 10 N would travel 0.2 m; the limit holds it at 0.05 m
            np.testing.assert_allclose(q, 0.05, atol=2.0e-3)


class TestSteelTendon(unittest.TestCase):
    pass


for _name in (
    "test_tendon_atwood_machine",
    "test_tendon_slack_rope_jerks_plastically",
    "test_tendon_pulley_divisor_tackle",
    "test_tendon_fixed_spring_damper",
    "test_tendon_friction_loss_sticks_and_slips",
    "test_tendon_across_articulations_is_ignored",
    "test_tendon_motor_lifts_a_mass",
    "test_tendon_servo_saturates_at_its_force_range",
    "test_tendon_armature_couples_joint_inertia",
    "test_tendon_implicit_spring_survives_large_steps",
    "test_tendon_passive_mode_is_validated",
    "test_tendon_lengths_include_tendons_without_rows",
    "test_tendon_rows_beyond_the_contact_budget_are_kept",
):
    add_function_test(TestSteelTendon, _name, globals()[_name], devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
