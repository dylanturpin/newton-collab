# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the SolverFeatherPGS test modules."""

from __future__ import annotations

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.friction_patches import (
    _FrictionPatchState,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_propagation_colored_block_kernel
from newton.solvers import SolverBase, SolverFeatherPGS


def _build_four_bar():
    """Build a planar four-bar: two grounded revolute chains closed by a BALL loop joint.

    Ground pivots at x=0 and x=0.4; crank and rocker hang down 0.2 m; the coupler
    connects the crank tip to the rocker tip through the loop closure. The crank is
    position-driven; the rocker is undriven, so any coherent motion it does comes from
    the closed loop.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 0.6

    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

    # Loop closure: coupler tip pinned to rocker tip (a trailing BALL loop joint,
    # matching how the MJCF importer closes `connect` equalities).
    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )

    # Drive the crank only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b


def _loop_anchor_gap(model, state) -> float:
    """World-space distance between the loop joint's parent and child anchors [m]."""
    bq = state.body_q.numpy().reshape(-1, 7).astype(np.float64)
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    j = int(np.nonzero(jt == int(newton.JointType.BALL))[0][-1])

    def anchor(body, X):
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        a = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
        w = wp.transform_multiply(t, a)
        return np.array([w.p[0], w.p[1], w.p[2]])

    return float(np.linalg.norm(anchor(int(jp[j]), Xp[j]) - anchor(int(jc[j]), Xc[j])))


def make_fixture(
    *,
    articulated,
    enabled,
    iterations=8,
    stock=False,
    height=0.05,
    kinematic=False,
    world_count=1,
    solver_options=None,
):
    """Build actual Newton sphere/plane contacts, one sphere per world."""
    device = os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.005
        if world_count == 1:
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        xform = wp.transform(wp.vec3(0, 0, height), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            if solver_options and solver_options.pop("_d6", False):
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    parent_xform=xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            else:
                joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        builder.add_shape_sphere(
            body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.3 / (4 / 3 * np.pi * 0.05**3), mu=0.5)
        )
        if world_count == 1:
            model = builder.finalize()
        else:
            scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
            scene.rigid_gap = 0.005
            scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
            scene.replicate(builder, world_count)
            model = scene.finalize()
        bodies = [body + world * builder.body_count for world in range(world_count)]
        if kinematic:
            flags = model.body_flags.numpy()
            flags[bodies] |= int(newton.BodyFlags.KINEMATIC)
            model.body_flags.assign(flags)
        capacity = 32 * world_count
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=capacity)
        model.rigid_contact_max = capacity
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(capacity, dtype=float))
        extra = {} if stock else {"contact_compliance": enabled}
        options = dict(
            # Isolate the normal material law from positional patch friction.
            friction_anchor_beta=0.0,
            solve_mode="matrix_free",
            row_schedule="interleaved",
            articulated_contact_response="immediate",
            enable_restitution=False,
            iterations=iterations,
            velocity_iterations=0,
            articulation_constraint_max=32,
            free_body_constraint_max=32,
            baumgarte=0.05,
            **extra,
        )
        options.update(solver_options or {})
        solver = SolverFeatherPGS(model, **options)
        return SimpleNamespace(
            model=model, pipeline=pipeline, contacts=contacts, solver=solver, body=body, bodies=bodies, device=device
        )


def run_fixture(
    *,
    articulated,
    enabled,
    steps=200,
    dt=0.005,
    iterations=8,
    stock=False,
    lateral_force=0.0,
    friction_scale=1.0,
    stiffness=3000.0,
    height=0.05,
    kinematic=False,
    solver_options=None,
):
    """Run actual Newton sphere/plane contacts through one physical step per tick."""
    options = dict(solver_options or {})
    if lateral_force:
        options["_d6"] = True
    fixture = make_fixture(
        articulated=articulated,
        enabled=enabled,
        iterations=iterations,
        stock=stock,
        height=height,
        kinematic=kinematic,
        solver_options=options,
    )
    model, pipeline, contacts, solver, body = (
        fixture.model,
        fixture.pipeline,
        fixture.contacts,
        fixture.solver,
        fixture.body,
    )
    with wp.ScopedDevice(fixture.device):
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        trace = []
        seen_paths = set()
        control = model.control()
        for step in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(friction_scale)
            if lateral_force and step >= steps // 2:
                control.joint_f.assign(np.array([lateral_force, 0.0], dtype=np.float32))
            solver.step(state, next_state, control, contacts, dt)
            state, next_state = next_state, state
            seen_paths.update(solver._contact_path.numpy()[:count].tolist())
            trace.append(state.body_q.numpy()[body].copy())
        return np.asarray(trace), seen_paths, solver, float(model.body_mass.numpy()[body])


def _patch_fixture(
    points, *, shape0=None, normals=None, materials=(0.5, 0.5, 0.5), shape_bodies=(0, 1, 0), device="cpu"
):
    """Construct patch geometry without a collision matcher or a solver step."""
    n = len(points)
    body_count = max(shape_bodies) + 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_count=body_count,
        body_world=wp.zeros(body_count, dtype=int, device=device),
        shape_body=wp.array(shape_bodies, dtype=int, device=device),
        shape_collision_radius=wp.array([0.2] * len(shape_bodies), dtype=float, device=device),
        shape_transform=wp.array([wp.transform_identity()] * len(shape_bodies), dtype=wp.transform, device=device),
        shape_scale=wp.ones(len(shape_bodies), dtype=wp.vec3, device=device),
        shape_type=wp.zeros(len(shape_bodies), dtype=int, device=device),
        shape_source_ptr=wp.zeros(len(shape_bodies), dtype=wp.uint64, device=device),
        shape_margin=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_gap=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_is_solid=wp.ones(len(shape_bodies), dtype=bool, device=device),
        shape_material_mu=wp.array(materials, dtype=float, device=device),
    )
    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device))
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([n], dtype=int, device=device),
        rigid_contact_shape0=wp.array(shape0 if shape0 is not None else [0] * n, dtype=int, device=device),
        rigid_contact_shape1=wp.ones(n, dtype=int, device=device),
        rigid_contact_point0=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_normal=wp.array(
            normals if normals is not None else [[0.0, 0.0, -1.0]] * n, dtype=wp.vec3, device=device
        ),
        rigid_contact_margin0=wp.zeros(n, dtype=float, device=device),
        rigid_contact_margin1=wp.zeros(n, dtype=float, device=device),
    )
    patches = _FrictionPatchState(model, max(n, 8), True, wp.zeros(max(n, 8), dtype=wp.vec2, device=device))
    patches.build(model, state, contacts)
    return model, state, contacts, patches


def _build_sparse_contact_friction_model(num_branches=16, num_worlds=2, *, device=None):
    """Build the sparse pair model with tilted sliding branches resting on a static ground plane.

    Each branch slides along ``(1, 0, 1) / sqrt(2)`` and carries a small box that touches the ground, so every
    contact normal and tangent row acts on that branch's single coordinate and friction is active.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    scene.default_shape_cfg.ke = 1.0e5
    scene.default_shape_cfg.kd = 1.0e3
    scene.default_shape_cfg.mu = 0.8
    scene.default_shape_cfg.margin = 0.0
    scene.default_shape_cfg.gap = 0.0
    scene.add_ground_plane()
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
        )
    ]
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        # Four boxes keep the four-point patches within the dense row capacity of the test solvers.
        if branch % 4 == 0:
            scene.add_shape_box(child, hx=0.01, hy=0.01, hz=0.01)
        star_joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=child,
                axis=wp.normalize(wp.vec3(1.0, 0.0, 1.0)),
                parent_xform=wp.transform(wp.vec3(0.05 * branch, 0.0, -0.4905), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 1.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "feather_pgs:rigid_body_max_linear_velocity": free_body_velocity_limit,
                "feather_pgs:rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


N_WORLDS = 4


def _build_scissor_model(device: str) -> newton.Model:
    """One articulation: base link + two sibling links whose boxes overlap.

    The siblings are not directly jointed, so collision produces a
    link-link contact within a single articulation (common ancestor = base).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    j_base = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )

    joints = [j_base]
    for side in (1.0, -1.0):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _contact_rows(solver: SolverFeatherPGS, path_id: int, contact_count: int | None = None) -> dict[int, int]:
    """contact index -> row slot for contacts routed to the given path."""
    contact_path = solver._contact_path.numpy().astype(np.int32)
    contact_slot = solver._contact_slot.numpy().astype(np.int32)
    contact_world = solver._contact_world.numpy().astype(np.int32)
    out = {}
    if contact_count is None:
        contact_count = solver._test_live_contact_count
    for c in range(contact_count):
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0:
            out[c] = int(contact_slot[c])
    return out


N_LINKS = 6


def _make_floating_chain_world(*, base_z: float, contact_sphere: str | None) -> newton.ModelBuilder:
    """A floating box base (free joint) trailing N_LINKS revolute links.

    ``contact_sphere`` places a single collision sphere on the base or on the
    deepest link; all other shapes carry no collision geometry so each world
    produces at most one contact point (keeps the contact pipeline
    deterministic for the end-to-end comparison). ``None`` builds density-only
    box shapes with collision enabled (used for the contact-free tests).
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.7

    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()))
    cfg = builder.default_shape_cfg if contact_sphere is None else no_collision
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    if contact_sphere == "base":
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(N_LINKS):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=cfg)
        if contact_sphere == "deep" and i == N_LINKS - 1:
            builder.add_shape_sphere(link, radius=0.05)
        offset = 0.22 if i == 0 else 0.12
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder


def _build_free_root_chain_model(
    device: str, *, base_z: float, contact_sphere: str | None, ground: bool
) -> newton.Model:
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        scene.add_world(_make_floating_chain_world(base_z=base_z, contact_sphere=contact_sphere))
    if ground:
        scene.add_ground_plane()
    return scene.finalize(device=device)


def _sphere_on_ground(device, armature, dense, com=(0.0, 0.0, 0.0), mu=0.0):
    """A 1 kg sphere on a free joint, optionally with a massless fixed child that routes it to dense rows."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(com),
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    joints = [builder.add_joint_free(body)]
    if dense:
        child = builder.add_link(mass=0.0, inertia=wp.mat33(0.0))
        joints.append(builder.add_joint_fixed(body, child))
    builder.add_articulation(joints)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = armature
    model.joint_armature.assign(joint_armature)
    return model


def set_torsion_radius(model, radius, shapes=None):
    """Set ``feather_pgs:min_torsional_patch_radius`` [m] of ``shapes`` (every shape for ``None``), zero elsewhere.

    ``radius`` may also be one value per selected shape.
    """
    radii = np.zeros(model.shape_count, dtype=np.float32)
    radii[slice(None) if shapes is None else list(shapes)] = radius
    namespace = getattr(model, "feather_pgs", None)
    if namespace is not None and getattr(namespace, "min_torsional_patch_radius", None) is not None:
        namespace.min_torsional_patch_radius.assign(radii)
    else:
        model.add_attribute(
            "min_torsional_patch_radius",
            wp.array(radii, dtype=wp.float32, device=model.device),
            frequency=newton.Model.AttributeFrequency.SHAPE,
            namespace="feather_pgs",
        )


def fixture(
    radius=None,
    *,
    mu=0.5,
    closing=0.1,
    spin=1.0,
    sliding=0.0,
    sat=False,
    separation=0.0,
    center_only=False,
    center_count=1,
    restitution=0.0,
    dt=0.0025,
    row_limit=None,
    model_arrays=None,
    step=True,
    torsion_shapes=None,
    **solver_overrides,
):
    """Solve two equal articulated rectangular pads touching face to face.

    ``model_arrays`` maps model attribute names to values assigned before the solver is built.
    ``radius`` is the torsion radius of ``torsion_shapes`` (both pads for ``None``); ``()`` builds the
    torsion solver and then clears every radius, so no contact is admitted. With ``step=False`` the
    solver is returned before its first step, without a result.
    """
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * (0.025 + separation / 2)), wp.quat_identity())
        body = b.add_link(
            xform=pose, mass=0.3, inertia=wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        )
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        b.add_articulation([joint])
        b.add_shape_box(
            body,
            hx=0.02,
            hy=0.015,
            hz=0.025,
            cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=restitution),
        )
    model = b.finalize(device="cuda:0")
    for name, values in (model_arrays or {}).items():
        getattr(model, name).assign(values)
    a, z = model.state(), model.state()
    a.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest",
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        box_box_sat=sat,
    )
    contacts = pipeline.contacts()
    pipeline.collide(a, contacts)
    if center_only and int(contacts.rigid_contact_count.numpy()[0]) > 0:
        # Deliberately represent the known planar disk by one central witness:
        # this isolates the new material spin law from native point lever arms.
        shapes = [contacts.rigid_contact_shape0.numpy()[0], contacts.rigid_contact_shape1.numpy()[0]]
        poses = a.body_q.numpy()
        for side, name in enumerate(("rigid_contact_point0", "rigid_contact_point1")):
            values = getattr(contacts, name).numpy()
            body = model.shape_body.numpy()[shapes[side]]
            values[:center_count] = -poses[body, :3]
            values[:center_count, 2] += np.sign(poses[body, 2]) * separation / 2
            getattr(contacts, name).assign(values)
        contacts.rigid_contact_count.assign(np.array([center_count], np.int32))
    kwargs = {
        # Keep this material-law control on the same point-friction baseline
        # whether the optional spin radius is zero or positive.
        "friction_anchor_beta": 0.0,
        "solve_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "iterations": 128,
        "velocity_iterations": 0,
        "baumgarte": 0.05,
        "cfm": 0,
        "contact_regularization": 0,
        "contact_friction_position_iterations": -1,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "enable_joint_velocity_limits": True,
        "contact_warmstart": False,
        "articulation_constraint_max": 64,
        "free_body_constraint_max": 16,
        "row_watermark": True,
    }
    admit_none = torsion_shapes == ()
    if radius is not None:
        set_torsion_radius(model, radius, None if admit_none else torsion_shapes)
    kwargs.update(solver_overrides)
    tuning = {key: kwargs.pop(key) for key in ("pgs_kernel",) if key in kwargs}
    with kernel_overrides(tuning):
        solver = SolverFeatherPGS(model, **kwargs)
    solver._rigid_body_angular_damping.zero_()
    if row_limit is not None:
        solver._articulation_constraint_max = row_limit
    if admit_none:
        # Torsion stays enabled, but no shape pair has a radius left to admit.
        set_torsion_radius(model, 0.0)
    if not step:
        return None, solver, model, a, contacts
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    solver.step(a, z, model.control(), contacts, dt, observables=observables)
    result = {name: getattr(z, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    result["contact_f"] = observables.contact_f.numpy()
    result.update(
        {
            name: getattr(solver, f"_{name}").numpy()
            for name in ("impulses", "row_type", "row_parent", "row_mu", "J_world", "Y_world", "v_hat", "v_out")
        }
    )
    result["count"] = solver._constraint_count.numpy()
    return result, solver, model, a, contacts


_COLORED_SWITCHES = (
    "FEATHER_PGS_COLORED_PREFETCH",
    "FEATHER_PGS_COLORED_STAGED",
    "FEATHER_PGS_COLORED_PACKED",
    "FEATHER_PGS_COLORED_STRAIGHT_LINE",
    "FEATHER_PGS_COLORED_LANES",
)
# Each colored solve body, selected by its kernel switches, with a tag of the kernel it must build.
COLORED_VARIANTS = {
    "packed": ({"FEATHER_PGS_COLORED_PACKED": "1"}, "_pf1_st0_pk1"),
    "prefetch": ({"FEATHER_PGS_COLORED_PACKED": "0"}, "_pf1_st0_pk0"),
    "staged": ({"FEATHER_PGS_COLORED_STAGED": "1"}, "_pf1_st1"),
    "reference": ({"FEATHER_PGS_COLORED_PREFETCH": "0"}, "_pf0_st0_pk0_sl1"),
    "row loop": ({"FEATHER_PGS_COLORED_PREFETCH": "0", "FEATHER_PGS_COLORED_STRAIGHT_LINE": "0"}, "_sl0"),
    "two lanes": ({"FEATHER_PGS_COLORED_LANES": "2"}, "_r2_"),
    "four lanes": ({"FEATHER_PGS_COLORED_LANES": "4"}, "_r4_"),
    "block": ({"FEATHER_PGS_COLORED_LANES": "3"}, "_sb"),
    "block global": ({"FEATHER_PGS_COLORED_LANES": "3"}, "_gb"),
}


@contextlib.contextmanager
def colored_variant_switches(name):
    """Set only colored variant ``name``'s kernel switches while solvers are built."""
    env, _ = COLORED_VARIANTS[name]
    with mock.patch.dict(os.environ, env):
        for key in _COLORED_SWITCHES:
            if key not in env:
                os.environ.pop(key, None)
        yield


def select_colored_variant(solver, name):
    """Give ``solver`` colored variant ``name``'s solve kernel and check that it built that kernel.

    Build the solver inside :func:`colored_variant_switches`. Raises ``SkipTest`` where staging is unsupported.
    """
    if name == "block global":
        # Bodies past the staging cap keep the twists in global memory; the body list is unused then.
        solver._pgs_solve_propagation_colored_block_kernel = _get_pgs_solve_propagation_colored_block_kernel(
            solver._propagation_max_constraints,
            solver._color_world_offsets.shape[0] // solver._world_count,
            solver._propagation_colored_block_dim,
            385,
            str(solver.model.device.arch),
            free_rigid_angular=solver._free_rigid_angular_rows,
            contact_torsion=solver._propagation_torsion_rows,
        )
    warp_kernel = solver._pgs_solve_propagation_colored_warp_kernel
    kernel = solver._pgs_solve_propagation_colored_block_kernel if warp_kernel is None else warp_kernel
    if name == "staged" and warp_kernel is not None and "_pf1_st1" not in kernel.key:
        raise unittest.SkipTest("shared-memory staging needs cp.async")
    tag = COLORED_VARIANTS[name][1]
    assert tag in kernel.key, f"{name} built {kernel.key}"
    assert ("_fra" in kernel.key) == bool(solver._free_rigid_angular_rows), kernel.key
    assert ("_tor" in kernel.key) == bool(solver._propagation_torsion_rows), kernel.key
