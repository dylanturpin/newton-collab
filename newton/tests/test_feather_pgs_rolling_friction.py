# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate opt-in per-contact torsional and rolling friction in FeatherPGS."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.contact_angular_friction import angular_friction_gs_sources
from newton.solvers import SolverFeatherPGS

GRAVITY = 9.81
DT = 0.00125
MU = 0.8
RADIUS = 0.02
MU_ROLLING = 0.01
SOLVER_OPTIONS = {
    "pgs_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 16,
    "friction_anchor_beta": 0.0,
    "angular_damping": 0.0,
    "dense_max_constraints": 32,
    "mf_max_constraints": 16,
    "double_buffer": False,
}


def ball(
    *,
    radius=RADIUS,
    mu=MU,
    mu_torsional=0.0,
    mu_rolling=MU_ROLLING,
    speed=1.0,
    rolling=True,
    spin=0.0,
    tilt=0.0,
    enable=True,
    cone="pyramidal",
    **solver_overrides,
):
    """Build a unit-mass solid sphere on the ground, articulated by slide x/z and hinge y/z joints.

    The joint chain keeps the slides world-aligned, so the generalized velocities are the
    center speed (0), vertical speed (1), rolling rate (2) and spin rate (3).
    """
    b = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius)
    body = b.add_link(xform=pose, mass=1.0, inertia=inertia)
    identity = wp.transform_identity()
    joints = [
        b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0)),
        b.add_joint_prismatic(
            carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
        ),
        b.add_joint_revolute(
            carriers[1], carriers[2], parent_xform=identity, child_xform=identity, axis=(0.0, 1.0, 0.0)
        ),
        b.add_joint_revolute(carriers[2], body, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)),
    ]
    b.add_articulation(joints)
    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_shape_sphere(body, radius=radius, cfg=sphere)
    model = b.finalize(device="cuda:0")
    model.set_gravity((GRAVITY * math.sin(tilt), 0.0, -GRAVITY * math.cos(tilt)))
    state = model.state()
    state.joint_qd.assign(np.array([speed, 0.0, speed / radius if rolling else 0.0, spin], np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    options = dict(SOLVER_OPTIONS)
    options.update(solver_overrides)
    solver = SolverFeatherPGS(
        model, enable_torsional_rolling_friction=enable, torsional_rolling_friction_cone=cone, **options
    )
    return Scene(model, state, solver)


def responsive_ball(*, radius, mu, mu_torsional, mu_rolling, velocity, cone, dt, **solver_overrides):
    """Build a unit-mass solid sphere on three world-aligned slides and a ball joint, so all five friction rows respond.

    ``velocity`` is the generalized velocity ``(vx, vy, vz, wx, wy, wz)``.
    """
    b = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    body = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius))
    identity = wp.transform_identity()
    joints = [b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0))]
    for parent, child, axis in ((0, 1, (0.0, 1.0, 0.0)), (1, 2, (0.0, 0.0, 1.0))):
        joints.append(
            b.add_joint_prismatic(
                carriers[parent], carriers[child], parent_xform=identity, child_xform=identity, axis=axis
            )
        )
    joints.append(b.add_joint_ball(carriers[2], body, parent_xform=identity, child_xform=identity))
    b.add_articulation(joints)
    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_shape_sphere(body, radius=radius, cfg=sphere)
    model = b.finalize(device="cuda:0")
    model.set_gravity((0.0, 0.0, -GRAVITY))
    state = model.state()
    state.joint_qd.assign(np.asarray(velocity, np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    options = dict(SOLVER_OPTIONS)
    options.update(solver_overrides)
    solver = SolverFeatherPGS(
        model, enable_torsional_rolling_friction=True, torsional_rolling_friction_cone=cone, **options
    )
    return Scene(model, state, solver, dt)


def solved_block(scene):
    """Return the normal impulse, friction impulses, coefficients, and the friction block's Delassus and free velocity."""
    solver = scene.solver
    rows = int(solver.constraint_count.numpy()[0])
    impulse = solver.impulses.numpy()[0, :rows].astype(float)
    jacobian = solver.J_world.numpy()[0, :rows].astype(float)
    response = solver.Y_world.numpy()[0, :rows].astype(float)
    mu = solver.row_mu.numpy()[0, :rows].astype(float)
    free_velocity = solver.v_out.numpy().astype(float) - response.T @ impulse
    delassus = jacobian @ response.T
    # Velocity of the friction rows with the normal impulse applied and friction removed.
    velocity = delassus[1:, :1] @ impulse[:1] + (jacobian @ free_velocity)[1:]
    return impulse[0], impulse[1:], mu[1:], delassus[1:, 1:], velocity


BLOCK_GROUPS = ([0, 1], [2], [3, 4])


def cone_norm(normalized, cone):
    """The norm whose ball of radius ``lambda_n`` is the cone, over coefficient-normalized friction impulses."""
    if cone == "elliptic":
        return float(np.linalg.norm(normalized))
    return float(np.linalg.norm(normalized[:2]) + abs(normalized[2]) + np.linalg.norm(normalized[3:]))


def elliptic_kkt_minimum(delassus, velocity, mu, load):
    """Minimize ``0.5 l'Al + v'l`` over the elliptic cone by eigendecomposition and a bisected KKT multiplier."""
    active = (mu > 0.0) & (np.diag(delassus) > 1.0e-12 * np.diag(delassus).max())
    hessian = (mu[:, None] * delassus * mu[None, :])[np.ix_(active, active)]
    gradient = (mu * velocity)[active]
    eigenvalues, basis = np.linalg.eigh(hessian)
    rotated = basis.T @ gradient

    def minimizer(multiplier):
        return -basis @ (rotated / np.maximum(eigenvalues + multiplier, 1.0e-300))

    if eigenvalues.min() > 1.0e-12 * eigenvalues.max() and np.linalg.norm(minimizer(0.0)) <= load:
        best = minimizer(0.0)
    else:
        low, high = 0.0, np.linalg.norm(gradient) / load
        for _ in range(200):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if np.linalg.norm(minimizer(middle)) > load else (low, middle)
        best = minimizer(high)
    out = np.zeros(len(mu))
    out[active] = mu[active] * best
    return out


def pyramidal_gap(delassus, velocity, mu, load, impulse):
    """Duality gap of ``impulse`` on the block-L1 cone, which bounds its objective excess, relative to the objective."""
    gradient = delassus @ impulse + velocity
    scaled = mu * gradient
    gap = gradient @ impulse + load * max(np.linalg.norm(scaled[g]) for g in BLOCK_GROUPS)
    return gap / abs(0.5 * impulse @ delassus @ impulse + velocity @ impulse)


def certified_pyramidal_minimum(delassus, velocity, mu, load):
    """Minimize over the block-L1 cone in float64 by group-scaled accelerated projection until the gap certifies it."""
    active = (mu > 0.0) & (np.diag(delassus) > 1.0e-12 * np.diag(delassus).max())
    hessian = mu[:, None] * delassus * mu[None, :] * np.outer(active, active)
    gradient = mu * velocity * active
    scale = np.ones(len(mu))
    for g in BLOCK_GROUPS:
        curvature = hessian[g, g].max()
        scale[g] = math.sqrt(curvature) if curvature > 0.0 else 1.0
    hessian = hessian / np.outer(scale, scale)
    gradient = gradient / scale
    weights = np.array([1.0 / scale[g[0]] for g in BLOCK_GROUPS])
    step = 1.0 / np.linalg.eigvalsh(hessian).max()

    def project(u):
        norms = np.array([np.linalg.norm(u[g]) for g in BLOCK_GROUPS])
        if weights @ norms <= load:
            return u
        low, high = 0.0, (norms / weights).max()
        for _ in range(200):
            middle = 0.5 * (low + high)
            if weights @ np.maximum(norms - middle * weights, 0.0) > load:
                low = middle
            else:
                high = middle
        out = u.copy()
        for g, norm, weight in zip(BLOCK_GROUPS, norms, weights, strict=True):
            out[g] *= max(norm - high * weight, 0.0) / norm if norm > 0.0 else 0.0
        return out

    u = np.zeros(len(mu))
    z = u.copy()
    momentum = 1.0
    for iteration in range(1, 200001):
        following = project(z - step * (gradient + hessian @ z)) * active
        momentum = 1.0 if (z - following) @ (following - u) > 0.0 else momentum
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        z = following + (momentum - 1.0) / next_momentum * (following - u)
        u, momentum = following, next_momentum
        impulse = mu * u / scale
        if iteration % 200 == 0 and pyramidal_gap(delassus, velocity, mu, load, impulse) < 1.0e-10:
            return impulse
    raise AssertionError("The block-L1 reference did not certify")


_BLOCK_KERNELS = {}


def block_impulses(cone, jacobian, velocity, mu, load):
    """Run the emitted friction block once in a single warp on rows ``jacobian`` (5 x 3) with unit inverse mass."""
    if cone not in _BLOCK_KERNELS:
        snippet = (
            "#if defined(__CUDA_ARCH__)\n"
            + angular_friction_gs_sources(cone, 0.0, 3)["helpers"]
            + """
    int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ float v[3], lam[5], mu[5], rhs[5];
    if (lane < 3) v[lane] = velocity.data[lane];
    if (lane < 5) {
        lam[lane] = 0.f;
        rhs[lane] = 0.f;
        mu[lane] = coefficients.data[lane];
    }
    __syncwarp();
    int changed = 0;
    float first = AngularFrictionBlock::solve(
        v, lam, mu, rhs, jacobian.data, jacobian.data, lane, 0xffffffffu, load, 1.f, &changed);
    __syncwarp();
    if (lane == 0) lam[0] = first;
    __syncwarp();
    if (lane < 5) out.data[lane] = lam[lane];
#endif
"""
        )

        @wp.func_native(snippet)
        def block(
            jacobian: wp.array(dtype=float),
            velocity: wp.array(dtype=float),
            coefficients: wp.array(dtype=float),
            load: float,
            out: wp.array(dtype=float),
        ): ...

        @wp.kernel(enable_backward=False, module="unique")
        def probe(
            jacobian: wp.array(dtype=float),
            velocity: wp.array(dtype=float),
            coefficients: wp.array(dtype=float),
            load: float,
            out: wp.array(dtype=float),
        ):
            block(jacobian, velocity, coefficients, load, out)

        _BLOCK_KERNELS[cone] = probe
    out = wp.zeros(5, dtype=float, device="cuda:0")
    arrays = [wp.array(np.asarray(a, np.float32).ravel(), device="cuda:0") for a in (jacobian, velocity, mu)]
    wp.launch(_BLOCK_KERNELS[cone], dim=32, block_dim=32, inputs=[*arrays, float(load), out], device="cuda:0")
    return out.numpy().astype(float)


class Scene:
    """Step one model with its own collision pipeline and record generalized velocities."""

    def __init__(self, model, state, solver, dt=DT):
        self.model = model
        self.dt = dt
        self.solver = solver
        self.state = state
        self.next = model.state()
        self.control = model.control()
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, broad_phase="explicit")
        self.contacts = self.pipeline.contacts()

    def step(self):
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next, self.control, self.contacts, self.dt)
        self.state, self.next = self.next, self.state

    def run(self, steps):
        """Return the generalized velocity after each step, shape (steps, dofs)."""
        out = []
        for _ in range(steps):
            self.step()
            out.append(self.state.joint_qd.numpy().copy())
        return np.array(out)


def project_cone(y, load, cone, groups):
    """Euclidean projection onto the elliptic ball or the L1-of-group-norms ball of radius ``load``."""
    if cone == "elliptic":
        norm = np.linalg.norm(y)
        return y * (load / norm) if norm > load else y
    norms = np.array([np.linalg.norm(y[g]) for g in groups])
    if norms.sum() <= load:
        return y
    ordered = np.sort(norms)[::-1]
    theta = 0.0
    for k in range(len(ordered)):
        candidate = (ordered[: k + 1].sum() - load) / (k + 1)
        if ordered[k] - candidate > 0.0:
            theta = candidate
    out = y.copy()
    for g, norm in zip(groups, norms, strict=True):
        out[g] = y[g] * (max(norm - theta, 0.0) / norm if norm > 0.0 else 0.0)
    return out


def cone_minimum(delassus, velocity, coefficients, load, cone, groups, iterations=3000):
    """Approximate the impulse minimizing ``0.5 l'Al + v'l`` over the cone by a fixed number of float64 iterations."""
    scale = np.asarray(coefficients, float)
    hessian = scale[:, None] * delassus * scale[None, :]
    gradient = scale * velocity
    lipschitz = np.abs(hessian).sum(1).max()
    y = np.zeros(len(scale))
    z = y.copy()
    momentum = 1.0
    for _ in range(iterations):
        step = project_cone(z - (gradient + hessian @ z) / lipschitz, load, cone, groups)
        step[scale == 0.0] = 0.0
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        z = step + (momentum - 1.0) / next_momentum * (step - y)
        y, momentum = step, next_momentum
    return scale * y


def reference_velocities(
    cone, steps, *, radius=RADIUS, mu=MU, mu_torsional=0.0, mu_rolling=MU_ROLLING, speed=1.0, rolling=True, spin=0.0
):
    """Velocities (center speed, rolling rate, spin rate) under a numerically solved joint cone each step, float64."""
    mass = 1.0 + 3.0e-4
    inertia = 0.4 * radius * radius
    inverse_mass = np.diag([1.0 / mass, 1.0 / (inertia + 1.0e-8), 1.0 / inertia])
    rows = np.array([[1.0, -radius, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # slip, spin, roll
    delassus = rows @ inverse_mass @ rows.T
    velocity = np.array([speed, speed / radius if rolling else 0.0, spin])
    out = []
    for _ in range(steps):
        impulse = cone_minimum(
            delassus, rows @ velocity, (mu, mu_torsional, mu_rolling), GRAVITY * DT, cone, [[0], [1], [2]]
        )
        velocity = velocity + inverse_mass @ rows.T @ impulse
        out.append(velocity.copy())
    return np.array(out)


def holding_slope(cone, radius=RADIUS, mu=MU, mu_rolling=MU_ROLLING):
    """Largest incline slope on which the cone holds the sphere at rest."""
    if cone == "pyramidal":
        return 1.0 / (1.0 / mu + radius / mu_rolling)
    return 1.0 / math.hypot(1.0 / mu, radius / mu_rolling)


@unittest.skipUnless(wp.is_cuda_available(), "Torsional/rolling friction requires CUDA")
class TestFeatherPGSRollingFriction(unittest.TestCase):
    def test_trajectories_match_the_reference_joint_cone(self):
        """Rolling, sliding and spinning starts follow a float64 reference that solves the cone per step."""
        cases = (
            {"radius": 0.02, "mu_rolling": 0.01},
            {"radius": 0.1, "mu_rolling": 0.002},
            {"radius": 0.02, "mu_rolling": 0.01, "rolling": False},
            {"radius": 0.1, "mu_torsional": 0.02, "mu_rolling": 0.0, "speed": 0.05, "rolling": False, "spin": 5.0},
        )
        for cone in ("pyramidal", "elliptic"):
            for case in cases:
                with self.subTest(cone=cone, **case):
                    qd = ball(cone=cone, **case).run(80)[:, [0, 2, 3]]
                    expected = reference_velocities(cone, 80, **case)
                    scale = np.abs(expected[0]) + np.array([1.0, 1.0, 1.0])
                    np.testing.assert_array_less(np.abs(qd - expected).max(0), 0.005 * scale)

    def test_rolling_stops_and_rests(self):
        qd = ball(speed=0.1).run(200)
        self.assertLess(abs(qd[-1, 0]), 1e-5)
        self.assertLess(abs(qd[-1, 2]), 1e-3)

    def test_incline_holds_below_and_rolls_above_threshold(self):
        for cone in ("pyramidal", "elliptic"):
            slope = holding_slope(cone)
            for factor, rolls in ((0.9, False), (1.1, True)):
                with self.subTest(cone=cone, factor=factor):
                    scene = ball(speed=0.0, tilt=math.atan(factor * slope), cone=cone)
                    qd = scene.run(400)
                    if rolls:
                        self.assertGreater(qd[-1, 0], 0.05)
                    else:
                        self.assertLess(abs(qd[-1, 0]), 1e-4)

    def test_spin_decays_at_the_torsional_bound(self):
        radius, mu_torsional = 0.1, 0.005
        for cone in ("pyramidal", "elliptic"):
            with self.subTest(cone=cone):
                scene = ball(radius=radius, mu_torsional=mu_torsional, mu_rolling=0.0, speed=0.0, spin=5.0, cone=cone)
                qd = scene.run(160)
                t = np.arange(1, 161) * DT
                rate = -np.polyfit(t[8:], qd[8:, 3], 1)[0]
                expected = mu_torsional * GRAVITY / (0.4 * radius * radius)
                self.assertAlmostEqual(rate, expected, delta=0.005 * expected)

    def test_creep_speed_softens_stiction_by_load_fraction(self):
        """Below the bound the coefficient times the angular rate settles at creep speed times the load fraction."""
        creep_speed, mu_torsional = 2.0e-3, 0.02
        for dof, coefficient in ((3, mu_torsional), (2, MU_ROLLING)):
            for fraction in (0.5, 0.9):
                with self.subTest(dof=dof, fraction=fraction):
                    scene = ball(
                        speed=0.0, mu_torsional=mu_torsional, torsional_rolling_friction_creep_speed=creep_speed
                    )
                    scene.run(40)
                    force = np.zeros(scene.model.joint_dof_count, np.float32)
                    force[dof] = fraction * coefficient * GRAVITY
                    scene.control.joint_f.assign(force)
                    qd = scene.run(200)
                    self.assertAlmostEqual(
                        qd[-1, dof], creep_speed * fraction / coefficient, delta=0.02 * creep_speed / coefficient
                    )
        rigid = ball(speed=0.0, mu_torsional=mu_torsional)
        rigid.run(40)
        force = np.zeros(rigid.model.joint_dof_count, np.float32)
        force[3] = 0.9 * mu_torsional * GRAVITY
        rigid.control.joint_f.assign(force)
        self.assertLess(abs(rigid.run(200)[-1, 3]), 1e-5)

    def test_competing_sliding_and_spin_share_the_cone(self):
        """A sliding, spinning sphere gives spin budget within tolerance of the reference objective, and more sweeps
        do not worsen it."""
        objectives = {}
        for cone in ("pyramidal", "elliptic"):
            for iterations in (16, 128):
                scene = ball(
                    radius=0.1, speed=0.0, mu_torsional=0.02, mu_rolling=0.0, cone=cone, pgs_iterations=iterations
                )
                scene.run(40)
                scene.state.joint_qd.assign(np.array([0.05, 0.0, 0.0, 5.0], np.float32))
                newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
                scene.step()
                solver = scene.solver
                rows = int(solver.constraint_count.numpy()[0])
                impulse = solver.impulses.numpy()[0, :rows].astype(float)
                jacobian = solver.J_world.numpy()[0, :rows].astype(float)
                response = solver.Y_world.numpy()[0, :rows].astype(float)
                free_velocity = solver.v_out.numpy().astype(float) - response.T @ impulse
                delassus = jacobian @ response.T

                def objective(candidate, delassus=delassus, velocity=jacobian @ free_velocity):
                    return 0.5 * candidate @ delassus @ candidate + velocity @ candidate

                mu = solver.row_mu.numpy()[0, :rows].astype(float)
                velocity = delassus[1:, :1] @ impulse[:1] + (jacobian @ free_velocity)[1:]
                friction = cone_minimum(delassus[1:, 1:], velocity, mu[1:], impulse[0], cone, [[0, 1], [2], [3, 4]])
                best = np.concatenate([impulse[:1], friction])
                with self.subTest(cone=cone, iterations=iterations):
                    self.assertGreater(abs(impulse[3]), 0.1 * mu[3] * impulse[0])
                    self.assertLessEqual(objective(impulse), objective(best) + 1e-3 * abs(objective(best)))
                objectives[cone, iterations] = objective(impulse)
        for cone in ("pyramidal", "elliptic"):
            self.assertLessEqual(objectives[cone, 128], objectives[cone, 16] + 1e-9)

    def test_zero_and_tiny_normal_loads_keep_friction_inside_the_cone(self):
        """A separating contact applies no friction, and a barely loaded one stays inside its cone."""
        for cone in ("pyramidal", "elliptic"):
            for vertical in (1.0, -1.0e-9):
                with self.subTest(cone=cone, vertical=vertical):
                    scene = ball(radius=0.1, rolling=False, mu=1.0, mu_torsional=0.005, mu_rolling=0.0001, cone=cone)
                    scene.model.set_gravity((0.0, 0.0, 0.0))
                    scene.state.joint_qd.assign(np.array([1.0, vertical, 0.0, 0.0], np.float32))
                    newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
                    scene.step()
                    self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 6)
                    load, impulse, mu, _, _ = solved_block(scene)
                    if vertical > 0.0:
                        self.assertEqual(load, 0.0)
                        np.testing.assert_array_equal(impulse, 0.0)
                        self.assertEqual(scene.state.joint_qd.numpy()[0], 1.0)
                    else:
                        self.assertGreater(load, 0.0)
                        self.assertLess(load, 1.0e-6)
                    self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))

    def test_small_coefficients_reach_the_block_optimum(self):
        """Small spin and rolling coefficients get the optimal block impulse at modest sweep counts."""
        builders = {
            "planar": lambda cone, iterations: ball(
                radius=0.1,
                speed=0.001,
                mu=1.0,
                mu_torsional=0.005,
                mu_rolling=0.0001,
                cone=cone,
                pgs_iterations=iterations,
            ),
            "responsive": lambda cone, iterations: responsive_ball(
                radius=0.1,
                mu=1.0,
                mu_torsional=0.005,
                mu_rolling=0.0001,
                velocity=(0.001, 0.0, 0.0, 0.0, 0.01, 0.0),
                cone=cone,
                dt=1.0 / 240.0,
                pgs_iterations=iterations,
            ),
        }
        for name, build in builders.items():
            for cone in ("elliptic", "pyramidal"):
                for iterations in (16, 128):
                    with self.subTest(scene=name, cone=cone, iterations=iterations):
                        scene = build(cone, iterations)
                        scene.step()
                        load, impulse, mu, delassus, velocity = solved_block(scene)
                        self.assertGreater(load, 0.0)
                        self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))
                        if cone == "elliptic":
                            expected = elliptic_kkt_minimum(delassus, velocity, mu, load)
                        else:
                            expected = certified_pyramidal_minimum(delassus, velocity, mu, load)
                        np.testing.assert_array_less(np.abs(impulse - expected), 1.0e-3 * mu * load + 1.0e-12)
                        rolling = 3 + int(np.abs(expected[3:]).argmax())
                        self.assertAlmostEqual(impulse[rolling] / expected[rolling], 1.0, delta=1.0e-3)

    def test_dependent_sliding_rows_keep_spin_friction(self):
        """Two tangent rows driven by one diagonal slide are dependent; the independent spin row still gets its bound."""
        for cone in ("elliptic", "pyramidal"):
            for iterations in (16, 128):
                with self.subTest(cone=cone, iterations=iterations):
                    b = newton.ModelBuilder()
                    material = newton.ModelBuilder.ShapeConfig(mu=1.0, mu_torsional=1.0e-4, mu_rolling=0.0)
                    b.add_ground_plane(cfg=material)
                    pose = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
                    identity = wp.transform_identity()
                    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
                    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(2)]
                    body = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                    diagonal = (2.0**-0.5, 2.0**-0.5, 0.0)
                    joints = [
                        b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=diagonal),
                        b.add_joint_prismatic(
                            carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
                        ),
                        b.add_joint_revolute(
                            carriers[1], body, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
                        ),
                    ]
                    b.add_articulation(joints)
                    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0, mu_torsional=1.0e-4, mu_rolling=0.0)
                    b.add_shape_sphere(body, radius=0.1, cfg=sphere)
                    model = b.finalize(device="cuda:0")
                    state = model.state()
                    state.joint_qd.assign(np.array([0.0, 0.0, 1.0e-5], np.float32))
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    options = dict(SOLVER_OPTIONS, pgs_iterations=iterations)
                    solver = SolverFeatherPGS(
                        model, enable_torsional_rolling_friction=True, torsional_rolling_friction_cone=cone, **options
                    )
                    scene = Scene(model, state, solver, dt=1.0 / 240.0)
                    scene.step()
                    load, impulse, mu, delassus, velocity = solved_block(scene)
                    np.testing.assert_allclose(delassus[0, :2], delassus[1, :2], rtol=1.0e-5)
                    # The spin row is decoupled: it stops the spin or saturates at its bound.
                    expected = -min(abs(velocity[2]) / delassus[2, 2], mu[2] * load)
                    self.assertAlmostEqual(impulse[2] / expected, 1.0, delta=1.0e-3)
                    normalized = np.divide(impulse, mu, out=np.zeros_like(impulse), where=mu > 0.0)
                    self.assertLessEqual(cone_norm(normalized, cone), load * (1.0 + 1.0e-5))

    def test_large_dependency_coefficients_stay_finite(self):
        """Weak tangents made dependent by a strong row keep the independent rolling row exact, never NaN."""
        mu = [0.5, 0.5, 0.01, 1.0e-4, 1.0e-4]
        for cone in ("elliptic", "pyramidal"):
            for scale in (1.0e-4, 1.0e-5, 2.0e-6, 1.0e-6, 1.0e-7):
                with self.subTest(cone=cone, scale=scale):
                    jacobian = [[scale, 0, 0], [0, scale, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]]
                    impulse = block_impulses(cone, jacobian, [0.0, 0.0, 1.0e-5], mu, 1.0)
                    self.assertTrue(np.isfinite(impulse).all())
                    # The rolling row is independent and well inside its bound, so it stops the rotation.
                    self.assertAlmostEqual(impulse[3] / -1.0e-5, 1.0, delta=1.0e-3)

    def test_dependent_blocks_stay_finite_feasible_and_dissipative(self):
        """Rank-deficient blocks with extreme row scales give finite impulses inside the cone that remove energy."""
        rng = np.random.default_rng(3)
        for cone in ("elliptic", "pyramidal"):
            for trial in range(60):
                with self.subTest(cone=cone, trial=trial):
                    # Three generalized velocities make any five rows dependent.
                    jacobian = rng.normal(size=(5, 3)) * 10.0 ** rng.uniform(-7.0, 0.0, size=(5, 1))
                    if trial % 3 == 1:
                        jacobian[1] = jacobian[0] * 10.0 ** rng.uniform(-3.0, 3.0)
                    elif trial % 3 == 2:
                        # Two weak tangents, just above the activity cutoff, that a strong spin row combines with large
                        # coefficients, and an independent rolling row.
                        weak = 10.0 ** rng.uniform(-5.8, -5.5)
                        u, w = rng.normal(size=(2, 2))
                        jacobian[:] = 0.0
                        jacobian[0, :2], jacobian[1, :2] = weak * u, weak * w
                        jacobian[2, :2] = u + w
                        jacobian[3, 2] = rng.uniform(0.5, 2.0)
                    velocity = rng.normal(size=3) * 10.0 ** rng.uniform(-6.0, 0.0)
                    if trial % 3 == 2:
                        velocity[:2] = 0.0
                    mu = np.concatenate([[rng.uniform(0.2, 1.5)] * 2, 10.0 ** rng.uniform(-5.0, -1.0, size=3)])
                    mu[4] = mu[3]
                    if trial % 3 == 2:
                        # Normalized dependency coefficients mu_spin / (mu_slide * weak) of 1e3 to 1e5.
                        mu[:2] = rng.uniform(0.2, 0.6)
                        mu[2], mu[3:] = 10.0 ** rng.uniform(-2.0, -1.5), 10.0 ** rng.uniform(-4.5, -3.5)
                    load = 10.0 ** rng.uniform(-4.0, 0.0)
                    impulse = block_impulses(cone, jacobian, velocity, mu, load)
                    self.assertTrue(np.isfinite(impulse).all())
                    self.assertLessEqual(cone_norm(impulse / mu, cone), load * (1.0 + 1.0e-5))
                    delassus = jacobian @ jacobian.T
                    rate = jacobian @ velocity
                    objective = 0.5 * impulse @ delassus @ impulse + rate @ impulse
                    self.assertLessEqual(objective, 1.0e-6 * abs(rate @ rate))

    def test_disabled_and_zero_coefficients_match_the_baseline_bitwise(self):
        baseline = ball(enable=False).run(60)
        self.assertAlmostEqual(baseline[-1, 0], 1.0, delta=1e-4)
        zero = ball(mu_torsional=0.0, mu_rolling=0.0).run(60)
        np.testing.assert_array_equal(zero, ball(mu_torsional=0.0, mu_rolling=0.0, enable=False).run(60))
        np.testing.assert_array_equal(baseline, ball(enable=False, mu_rolling=0.5).run(60))

    def test_contact_force_reports_the_sliding_pair(self):
        """Angular contacts still report their normal and sliding force; the angular impulses are not reported."""
        scene = ball(speed=1.0, rolling=False)
        scene.model.request_contact_attributes("force")
        scene.contacts = scene.pipeline.contacts()
        scene.step()
        scene.solver.update_contacts(scene.contacts)
        force = scene.contacts.force.numpy()[0, :3]
        self.assertAlmostEqual(abs(float(force[2])), GRAVITY * 1.0003, delta=0.01)
        self.assertAlmostEqual(float(np.linalg.norm(force[:2])), MU * GRAVITY * 1.0003, delta=0.02)

    def test_zero_coefficients_allocate_no_rows(self):
        scene = ball(mu_torsional=0.0, mu_rolling=0.0)
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 3)
        scene = ball()
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 6)

    def test_runtime_coefficient_edits_and_array_replacement(self):
        scene = ball(mu_rolling=0.0)
        qd = scene.run(20)
        self.assertAlmostEqual(qd[-1, 0], 1.0, delta=1e-4)
        shapes = scene.model.shape_count
        scene.model.shape_material_mu_rolling.assign(np.full(shapes, MU_ROLLING, np.float32))
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        start = scene.state.joint_qd.numpy()
        expected = reference_velocities("pyramidal", 60, speed=float(start[0]))
        np.testing.assert_allclose(scene.run(60)[:, 0], expected[:, 0], atol=2e-3)
        scene.model.shape_material_mu_rolling = wp.zeros(shapes, dtype=wp.float32, device=scene.model.device)
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        # Sliding friction removes the rolling-friction slip, then the sphere rolls freely.
        qd = scene.run(30)
        self.assertAlmostEqual(qd[-1, 0], qd[-11, 0], delta=1e-5)
        self.assertAlmostEqual(qd[-1, 0], qd[-1, 2] * RADIUS, delta=1e-4)

    def test_graph_replay_matches_eager(self):
        eager = ball().run(40)
        scene = ball()
        scene.step()
        scene = ball()
        with wp.ScopedCapture(device=scene.model.device) as capture:
            scene.step()
            scene.step()
        replay = []
        for _ in range(20):
            wp.capture_launch(capture.graph)
            replay.append(scene.state.joint_qd.numpy().copy())
        np.testing.assert_array_equal(np.array(replay), eager[1::2])

    def test_reset_keeps_no_angular_history(self):
        fresh = ball().run(30)
        scene = ball()
        start_q = scene.state.joint_q.numpy().copy()
        start_qd = scene.state.joint_qd.numpy().copy()
        scene.run(30)
        scene.state.joint_q.assign(start_q)
        scene.state.joint_qd.assign(start_qd)
        newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
        scene.solver.reset(scene.state)
        np.testing.assert_array_equal(scene.run(30), fresh)

    def test_row_capacity_overflow_is_reported(self):
        scene = ball(dense_max_constraints=5, warn_constraint_overflow=False)
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 0)
        self.assertTrue(bool(scene.solver.constraint_overflow.numpy()[0]))

    def test_unsupported_routes_are_rejected(self):
        rejected = (
            {"pgs_mode": "split"},
            {"articulated_contact_response": "propagation"},
            {"friction_anchor_beta": 0.2},
            {"pgs_warmstart": True},
            {"pgs_velocity_iterations": 2},
            {"friction_mode": "bisection"},
            {"pgs_schedule": "contact_then_internal"},
            {"contact_torsion_radius": 0.01},
            {"enable_sleeping": True},
        )
        for overrides in rejected:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                ball(**overrides)
        with self.assertRaises(ValueError):
            ball(cone="cubic")
        with self.assertRaises(ValueError):
            ball(torsional_rolling_friction_creep_speed=-1.0)

    def test_free_rigid_route_coefficients_are_rejected(self):
        """Free spheres on a world-static floor or a floor link fixed to the world both take the matrix-free route."""
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                b = newton.ModelBuilder()
                floor_material = newton.ModelBuilder.ShapeConfig(mu=MU, mu_torsional=0.02, mu_rolling=0.01, density=0.0)
                pose = wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity())
                if fixed:
                    floor = b.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                    b.add_articulation([b.add_joint_fixed(-1, floor, parent_xform=pose)])
                else:
                    floor = -1
                floor_shape = b.add_shape_box(
                    floor, xform=None if fixed else pose, hx=2.0, hy=2.0, hz=0.05, cfg=floor_material
                )
                body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
                sphere = newton.ModelBuilder.ShapeConfig(mu=MU, mu_torsional=0.0, mu_rolling=0.0)
                b.add_shape_sphere(body, radius=0.1, cfg=sphere)
                model = b.finalize(device="cuda:0")
                with self.assertRaises(ValueError):
                    SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
                model.shape_material_mu_torsional.zero_()
                model.shape_material_mu_rolling.zero_()
                solver = SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
                rolling = model.shape_material_mu_rolling.numpy()
                rolling[floor_shape] = 0.01
                model.shape_material_mu_rolling.assign(rolling)
                with self.assertRaises(ValueError):
                    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
