# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Propagation contact responses, colored sweeps and tree-parallel kernels of SolverFeatherPGS."""

from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.kernels import PROPAGATION_UNIT_META, PROPAGATION_UNIT_RING
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    PROPAGATION_COLOR_TAIL,
    _colored_staging_supported,
    _FeatherPGSTreePlan,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import (
    N_WORLDS,
    _build_free_root_chain_model,
    _build_scissor_model,
    _contact_rows,
    set_torsion_radius,
)
from newton.tests.unittest_utils import StdOutCapture

DT = 1.0 / 240.0


def _add_floating_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A floating box base (free joint) trailing ``n_links`` revolute links.

    ``spheres`` selects which bodies carry a collision sphere ("base", "mid",
    "deep"); all other shapes are density-only. Multiple spheres per world
    give several simultaneously contact-active bodies on one articulation,
    which is what exercises the cached cross response blocks B_ab.
    """
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=no_collision)
    if "base" in spheres:
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
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


def _add_fixed_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A world-anchored all-revolute chain: the fixed-base extraction variant.

    No free root — the first joint is a world-rooted revolute, so every joint
    is 1-DOF and the articulation takes the cached-response extraction's
    fixed-base (single-DOF) variant, the default shape of a fixed-base arm.
    ``spheres`` selects collision spheres on the first ("base"), middle
    ("mid"), and last ("deep") link; the chain droops under gravity onto the
    ground plane, so several bodies are simultaneously contact-active and the
    cross response blocks B_ab carry real impulses.
    """
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "base" in spheres and i == 0:
            builder.add_shape_sphere(link, radius=0.1)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
        parent_xform = (
            wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity())
            if prev == -1
            else wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity())
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)


def _build_model(
    device: str, *, chains: tuple[int, ...], spheres: tuple[str, ...], fixed_base: bool = False
) -> newton.Model:
    """N_WORLDS identical worlds, each holding one chain per entry of ``chains``.

    Multiple chain lengths inside one world create several articulation
    DOF-size groups while keeping per-world DOF totals homogeneous (the
    propagation modes reject heterogeneous per-world DOF counts).
    ``fixed_base`` swaps the floating chains for world-anchored ones.
    """
    scene = newton.ModelBuilder()
    add_chain = _add_fixed_chain if fixed_base else _add_floating_chain
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        for k, n_links in enumerate(chains):
            add_chain(world, n_links=n_links, base_z=0.105, base_y=0.8 * k, spheres=spheres)
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _build_clutter_model(device: str, *, n_boxes: int) -> newton.Model:
    """A fixed-base arm plus free-rigid clutter boxes resting on the ground.

    The clutter bodies are contact-active every step but never take the
    cached GEMV path (free bodies keep the sweep-estimate + flush path), so
    they must not count against the cache capacity: only the arm's bodies
    are cache-eligible. Clutter is frictionless spheres: one normal-only
    ground row per clutter body keeps every clutter row's state fully
    disjoint, so the lockstep gate's tree-walk determinism precondition
    survives the atomic (run-varying) row ordering — box corner contacts or
    friction pairs share a body and reorder its float accumulation between
    runs.
    """
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        # Anchor low enough that every arm sphere starts pressed into the
        # ground: a marginal (flickering) tip contact makes the active-body
        # set differ between the lockstepped solvers.
        _add_fixed_chain(world, n_links=6, base_z=0.095, base_y=0.0, spheres=("base", "mid", "deep"))
        clutter_cfg = world.default_shape_cfg.copy()
        clutter_cfg.mu = 0.0
        for b in range(n_boxes):
            ball = world.add_link(
                xform=wp.transform(wp.vec3(-0.4 - 0.25 * (b % 3), 0.6 + 0.25 * (b // 3), 0.0995), wp.quat_identity())
            )
            world.add_shape_sphere(ball, radius=0.1, cfg=clutter_cfg)
            world.add_articulation([world.add_joint_free(parent=-1, child=ball)])
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _make_solver(
    model: newton.Model, response: str, *, cached: bool, cache_max_bodies: int = 8, **solver_kwargs
) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        iterations=8,
        contact_warmstart=False,
        propagation_cached_response=cached,
        propagation_cached_response_body_max=cache_max_bodies,
        **solver_kwargs,
    )


def _seed_joint_qd(model: newton.Model) -> np.ndarray:
    n = int(model.joint_dof_count)
    return (0.4 * np.sin(0.7 * np.arange(n, dtype=np.float64) + 0.3)).astype(np.float32)


def _active_body_qd(solver) -> dict[int, np.ndarray]:
    """Live COM twists keyed by body, restricted to contact-active bodies.

    The cached path deliberately leaves NON-active bodies' propagation_body_qd
    stale between forced refreshes (the GS sweep only ever reads active
    bodies, and every pass setup force-refreshes all bodies from v_out), so
    only active entries are comparable across paths.
    """
    qd = solver._propagation_body_qd.numpy()
    counts = solver._propagation_body_count.numpy()
    body_list = solver._propagation_body_list.numpy()
    out: dict[int, np.ndarray] = {}
    for world in range(body_list.shape[0]):
        n = min(int(counts[world]), body_list.shape[1])
        for slot in range(n):
            body = int(body_list[world, slot])
            if body >= 0:
                out[body] = qd[body].copy()
    return out


class _LockstepStats:
    def __init__(self):
        self.max_diffs = {"joint_q": 0.0, "joint_qd": 0.0, "v_out": 0.0, "active_body_qd": 0.0}
        self.max_active_bodies = 0
        self.total_rows = 0


def _run_lockstep(test_case, model, response, n_steps, *, cache_max_bodies: int = 8, **solver_kwargs):
    """Advance a tree-walk reference trajectory; at every step, run the cached
    solver one step from the identical input state and record output diffs.

    A second tree-walk solver steps alongside as a bitwise determinism
    precondition: if the fallback itself were not reproducible from shared
    inputs, the cached-vs-fallback diff would be meaningless.
    """
    ref = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    ref2 = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    test = _make_solver(model, response, cached=True, cache_max_bodies=cache_max_bodies, **solver_kwargs)

    state_in = model.state()
    ref_out = model.state()
    ref2_out = model.state()
    test_out = model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    stats = _LockstepStats()
    for _ in range(n_steps):
        state_in.clear_forces()
        # A fresh collide before each solver isolates them from any contact
        # buffer bookkeeping a step might do; collide is deterministic in
        # state, so all three consume identical contact sets.
        collision_pipeline.collide(state_in, contacts)
        test.step(state_in, test_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref2.step(state_in, ref2_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref.step(state_in, ref_out, control, contacts, DT)
        wp.synchronize()

        # determinism precondition
        for key in ("joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(ref_out, key).numpy(),
                getattr(ref2_out, key).numpy(),
                err_msg=f"tree-walk solve not reproducible for {key}; scene invalidates the gate",
            )

        for key in ("joint_q", "joint_qd"):
            diff = float(np.max(np.abs(getattr(test_out, key).numpy() - getattr(ref_out, key).numpy())))
            stats.max_diffs[key] = max(stats.max_diffs[key], diff)
        stats.max_diffs["v_out"] = max(
            stats.max_diffs["v_out"], float(np.max(np.abs(test._v_out.numpy() - ref._v_out.numpy())))
        )
        ref_body_qd = _active_body_qd(ref)
        test_body_qd = _active_body_qd(test)
        test_case.assertEqual(sorted(ref_body_qd), sorted(test_body_qd), "active body sets differ")
        for body, ref_qd in ref_body_qd.items():
            diff = float(np.max(np.abs(test_body_qd[body] - ref_qd)))
            stats.max_diffs["active_body_qd"] = max(stats.max_diffs["active_body_qd"], diff)

        counts = ref._propagation_body_count.numpy()
        stats.max_active_bodies = max(stats.max_active_bodies, int(counts.max()))
        stats.total_rows += int(ref._propagation_constraint_count.numpy().sum())

        # advance the shared trajectory along the reference solve
        state_in, ref_out = ref_out, state_in
    return stats, ref, test


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationCachedResponse(unittest.TestCase):
    TOL = 1.0e-4

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        cls.model_multi = _build_model(cls.device, chains=(6,), spheres=("base", "mid", "deep"))
        cls.model_two_groups = _build_model(cls.device, chains=(6, 3), spheres=("base", "deep"))
        cls.model_single = _build_model(cls.device, chains=(6,), spheres=("base",))
        cls.model_fixed = _build_model(cls.device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=True)

    def test_cached_path_detected(self):
        solver = _make_solver(self.model_multi, "propagation_colored", cached=True)
        self.assertTrue(solver._propagation_cached_response_active)
        self.assertIsNotNone(solver._propagation_cached_gemv_kernel)
        self.assertIsNotNone(solver._propagation_cache_R)
        self.assertIsNotNone(solver._propagation_cache_B)
        self.assertIsNotNone(solver._propagation_cache_qd_base)
        self.assertIsNotNone(solver._propagation_cache_world_flag)
        self.assertEqual(
            solver._propagation_cache_max_bodies,
            min(solver._max_propagation_bodies, solver._propagation_cached_response_body_max),
        )
        sizes = [int(s) for s in solver._size_groups if solver._n_arts_by_size[s] > 0]
        for size in sizes:
            self.assertIsNotNone(
                solver._cached_response_tree_warp_kernels_by_size.get(size),
                f"no cached-response kernel for size group {size}",
            )
        # eligibility must exclude nothing here: all arts are free-root trees
        self.assertTrue(np.all(solver._propagation_cache_art_eligible.numpy() == 1))

        off = _make_solver(self.model_multi, "propagation_colored", cached=False)
        self.assertFalse(off._propagation_cached_response_active)
        self.assertIsNone(off._propagation_cache_R)

    def _assert_stats(self, stats, *, expect_min_bodies):
        self.assertGreater(stats.total_rows, 0, "no propagation contact rows produced")
        self.assertGreaterEqual(
            stats.max_active_bodies,
            expect_min_bodies,
            "scene did not produce the expected number of simultaneously active bodies",
        )
        for key, diff in stats.max_diffs.items():
            self.assertLessEqual(diff, self.TOL, f"{key} per-step max abs diff {diff:.3e} exceeds {self.TOL:g}")

    def test_lockstep_multi_contact_bodies(self):
        """Three contact spheres per articulation: cross blocks B_ab are live."""
        stats, _, test = _run_lockstep(self, self.model_multi, "propagation_colored", 40)
        self.assertTrue(np.all(test._propagation_cache_world_flag.numpy() == 1))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_lockstep_second_size_group(self):
        """Two articulations per world -> two DOF-size groups, both cached."""
        stats, _, test = _run_lockstep(self, self.model_two_groups, "propagation_colored", 40)
        sizes = [int(s) for s in test._size_groups if test._n_arts_by_size[s] > 0]
        self.assertEqual(len(sizes), 2, f"expected two size groups, got {sizes}")
        for size in sizes:
            self.assertIsNotNone(test._cached_response_tree_warp_kernels_by_size.get(size))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_lockstep_serial_propagation_mode(self):
        """Non-colored 'propagation' mode; one contact per world keeps the
        serial row sweep deterministic (same constraint as the free-root
        warp tests)."""
        stats, _, _ = _run_lockstep(self, self.model_single, "propagation", 40)
        self._assert_stats(stats, expect_min_bodies=1)

    def test_lockstep_fixed_base_single_dof(self):
        """Fixed-base all-revolute chains: the single-DOF extraction variant.

        Every prior lockstep scene is floating-base, which takes the
        free-root extraction variant; a fixed-base arm — the default shape of
        every table-mounted robot — takes the other one, so gate it
        explicitly against the tree walk on the same contact-rich scene.
        """
        stats, _, test = _run_lockstep(self, self.model_fixed, "propagation_colored", 40)
        sizes = [int(s) for s in test._size_groups if test._n_arts_by_size[s] > 0]
        for size in sizes:
            # A fully single-DOF group takes the plain fixed-base extraction
            # variant (the kernel factories receive has_free_root=not
            # single_dof), even though a 1-DOF world-rooted chain also
            # satisfies the free-root SHAPE check.
            self.assertTrue(
                test._propagation_tree_single_dof_by_size.get(size, False),
                f"size group {size} did not take the fixed-base single-DOF variant",
            )
            self.assertIsNotNone(test._cached_response_tree_warp_kernels_by_size.get(size))
        self.assertTrue(np.all(test._propagation_cache_world_flag.numpy() == 1))
        self._assert_stats(stats, expect_min_bodies=2)

    def test_overflow_falls_back_to_tree_walk(self):
        """Cache capacity below the active-body count: every world must take
        the device-side tree-walk fallback and match the cached=False run."""
        stats, ref, test = _run_lockstep(self, self.model_multi, "propagation_colored", 40, cache_max_bodies=2)
        self.assertTrue(test._propagation_cached_response_active)
        self.assertEqual(test._propagation_cache_max_bodies, 2)
        flags = test._propagation_cache_world_flag.numpy()
        counts = ref._propagation_body_count.numpy()
        self.assertTrue(np.all(counts > 2), f"scene must overflow the cap, got counts {counts}")
        self.assertTrue(np.all(flags == 0), f"overflowing worlds must clear the cache flag, got {flags}")
        self._assert_stats(stats, expect_min_bodies=3)

    def test_clutter_does_not_evict_cache(self):
        """Free-rigid clutter must not count against the cache capacity.

        A fixed-base arm (three contact-active tree bodies) plus two free
        spheres on the ground against a cache capacity of 4: the TOTAL active
        count (5) exceeds the cap, but only the arm's bodies are
        cache-eligible, so every world must stay on the cached GEMV path —
        and still match the tree walk in lockstep. Before the partitioned
        capacity gate this scene silently self-disabled the cache. Clutter is
        kept small because larger free-body counts make the tree-walk
        REFERENCE itself non-reproducible run-to-run on this branch
        (reassociation-level, and near the default 32-row cap much larger),
        which invalidates the lockstep gate's determinism precondition.
        """
        model = _build_clutter_model(self.device, n_boxes=2)
        stats, ref, test = _run_lockstep(
            self, model, "propagation_colored", 40, cache_max_bodies=4, articulation_constraint_max=128
        )
        self.assertTrue(test._propagation_cached_response_active)
        counts = ref._propagation_body_count.numpy()
        self.assertTrue(np.all(counts > 4), f"scene must overflow the TOTAL active-body count, got {counts}")
        eligible_counts = test._propagation_cache_body_count.numpy()
        self.assertTrue(
            np.all((eligible_counts >= 2) & (eligible_counts <= 4)),
            f"expected 2..4 eligible arm bodies per world, got {eligible_counts}",
        )
        self.assertTrue(
            np.all(test._propagation_cache_world_flag.numpy() == 1),
            "clutter must not evict the arm from the cached path",
        )
        # Partition invariant: cache-eligible bodies occupy the slot prefix.
        art_eligible = test._propagation_cache_art_eligible.numpy()
        body_to_art = test._body_to_articulation.numpy()
        body_list = test._propagation_body_list.numpy()
        test_counts = test._propagation_body_count.numpy()
        for world in range(body_list.shape[0]):
            n_elig = int(eligible_counts[world])
            for slot in range(min(int(test_counts[world]), body_list.shape[1])):
                body = int(body_list[world, slot])
                self.assertGreaterEqual(body, 0)
                is_eligible = int(art_eligible[int(body_to_art[body])]) == 1
                self.assertEqual(is_eligible, slot < n_elig, f"world {world} slot {slot}: eligible prefix violated")
        self._assert_stats(stats, expect_min_bodies=5)

    def test_cached_matrices_match_numpy_reference(self):
        """Floating-base R/B against the float64 tree-walk mirror."""
        self._check_cached_matrices_vs_numpy(self.model_multi)

    def test_fixed_base_cached_matrices_match_numpy_reference(self):
        """Fixed-base (single-DOF variant) R/B against the same mirror."""
        self._check_cached_matrices_vs_numpy(self.model_fixed)

    def _check_cached_matrices_vs_numpy(self, model):
        """R and B against a float64 mirror of the serial tree-walk algorithm.

        R_b column j is the qdd over all articulation DOFs from a unit basis
        wrench j at body b; B_ab column j is the resulting COM twist delta at
        body a. Both fall out of _ref_propagate run at zero initial velocity.
        """
        solver = _make_solver(model, "propagation_colored", cached=True)
        state_in, state_out = model.state(), model.state()
        control = model.control()
        state_in.joint_qd.assign(_seed_joint_qd(model))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(20):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize()

        counts = solver._propagation_body_count.numpy()
        body_list = solver._propagation_body_list.numpy()
        body_to_art = solver._body_to_articulation.numpy()
        flags = solver._propagation_cache_world_flag.numpy()
        art_dof_start = solver.model.joint_qd_start.numpy()[solver.model.articulation_start.numpy()[:-1]]
        R = solver._propagation_cache_R.numpy()
        B = solver._propagation_cache_B.numpy()

        checked_pairs = 0
        for world in range(min(2, body_list.shape[0])):
            self.assertEqual(int(flags[world]), 1)
            n = int(counts[world])
            self.assertGreaterEqual(n, 2, "need at least two active bodies for cross blocks")
            for slot_b in range(n):
                body_b = int(body_list[world, slot_b])
                art = int(body_to_art[body_b])
                ti = _tree_inputs(model, solver, art)
                dof0 = int(art_dof_start[art])
                n_dofs = int(ti["qd_start"][ti["j1"]] - ti["qd_start"][ti["j0"]])
                for basis in range(6):
                    imp = np.zeros(6)
                    imp[basis] = 1.0
                    v_ref, body_qd_ref = _ref_propagate(
                        ti, {body_b: imp}, np.zeros(solver._v_out.shape[0], dtype=np.float64)
                    )
                    got_R = R[world, slot_b, :n_dofs, basis]
                    ref_R = v_ref[dof0 : dof0 + n_dofs]
                    scale = max(1.0, float(np.max(np.abs(ref_R))))
                    self.assertLess(
                        float(np.max(np.abs(got_R - ref_R))),
                        1e-4 * scale,
                        f"R mismatch world {world} slot {slot_b} basis {basis}",
                    )
                    for slot_a in range(n):
                        body_a = int(body_list[world, slot_a])
                        if int(body_to_art[body_a]) == art:
                            ref_col = body_qd_ref[body_a]
                        else:
                            ref_col = np.zeros(6)
                        got_col = B[world, slot_a, slot_b].reshape(6, 6)[:, basis]
                        scale = max(1.0, float(np.max(np.abs(ref_col))))
                        self.assertLess(
                            float(np.max(np.abs(got_col - ref_col))),
                            1e-4 * scale,
                            f"B mismatch world {world} a {slot_a} b {slot_b} basis {basis}",
                        )
                        checked_pairs += 1
        self.assertGreater(checked_pairs, 0)

    def test_impulses_consumed(self):
        """Deferred body impulses must be cleared by the GEMV each iteration."""
        model = self.model_multi
        solver = _make_solver(model, "propagation_colored", cached=True)
        state_in, state_out = model.state(), model.state()
        control = model.control()
        state_in.joint_qd.assign(_seed_joint_qd(model))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        for _ in range(10):
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            state_in, state_out = state_out, state_in
        wp.synchronize()
        self.assertEqual(float(np.max(np.abs(solver._propagation_body_impulses.numpy()))), 0.0)


def _build_arm_world_model(device: str, *, mu: float, base_z: tuple[float, ...], gravity: float = -9.81):
    """Identical fixed-base arms with contact spheres on links 0, 3 and 5, one per world, at the given base heights."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    for z in base_z:
        world = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = mu
        _add_fixed_chain(world, n_links=6, base_z=z, base_y=0.0, spheres=("base", "mid", "deep"))
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _cached_rollout(model, response, n_steps, **solver_kwargs):
    """Step a cached-response solver on deterministic contacts; return per-step joint state and the solver."""
    solver = _make_solver(model, response, cached=True, articulation_constraint_max=64, **solver_kwargs)
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    history = []
    max_active = 0
    for _ in range(n_steps):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        history.append(np.concatenate([state_out.joint_q.numpy(), state_out.joint_qd.numpy()]))
        max_active = max(max_active, int(solver._propagation_body_count.numpy().max()))
        state_in, state_out = state_out, state_in
    return np.stack(history), solver, max_active


@wp.kernel
def _rank_propagation_body_map_descending(
    propagation_body_count: wp.array[int],
    propagation_cache_body_count: wp.array[int],
    max_propagation_bodies: int,
    claimed_body_list: wp.array2d[int],
    propagation_body_list: wp.array2d[int],
    propagation_body_local_slot: wp.array[int],
):
    """Same partitions as the solver's ranking, but in descending body order: a fixed, different slot order."""
    world, i = wp.tid()
    n = wp.min(propagation_body_count[world], max_propagation_bodies)
    if i >= n:
        return
    split = wp.min(propagation_cache_body_count[world], n)
    lo = int(0)
    hi = split
    if i >= split:
        lo = split
        hi = n
    body = claimed_body_list[world, i]
    rank = lo
    for j in range(lo, hi):
        if claimed_body_list[world, j] > body:
            rank += 1
    propagation_body_list[world, rank] = body
    propagation_body_local_slot[body] = rank


@wp.kernel
def _claim_propagation_body_map_reversed(
    propagation_constraint_count: wp.array[int],
    propagation_body_a: wp.array2d[int],
    propagation_body_b: wp.array2d[int],
    propagation_max_constraints: int,
    max_propagation_bodies: int,
    body_to_articulation: wp.array[int],
    propagation_cache_art_eligible: wp.array[int],
    want_eligible: int,
    propagation_body_seen: wp.array[int],
    propagation_body_list: wp.array2d[int],
    propagation_body_count: wp.array[int],
    propagation_body_local_slot: wp.array[int],
):
    """Claim like the partitioned map, but one thread per world walking rows backwards: an adverse claim order."""
    world = wp.tid()
    m = wp.min(propagation_constraint_count[world], propagation_max_constraints)
    for r in range(m):
        i = m - 1 - r
        for side in range(2):
            body = propagation_body_a[world, i]
            if side == 0:
                body = propagation_body_b[world, i]
            if body < 0:
                continue
            art = body_to_articulation[body]
            elig = int(0)
            if art >= 0:
                elig = propagation_cache_art_eligible[art]
            if elig != want_eligible or propagation_body_seen[body] != 0:
                continue
            propagation_body_seen[body] = 1
            slot = propagation_body_count[world]
            propagation_body_count[world] = slot + 1
            if slot < max_propagation_bodies:
                propagation_body_list[world, slot] = body
                propagation_body_local_slot[body] = slot


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationBodyMapOrder(unittest.TestCase):
    """The cached response must not depend on the order in which GPU threads claim body slots.

    The solver runs with ``deterministic=False`` (its default); contacts come from
    ``CollisionPipeline(deterministic=True)`` so their order is fixed. Frictionless spheres
    give one row per contact body, so the plain serial sweep is order-free too.
    """

    STEPS = 60

    def _assert_repeatable(self, response, mu):
        model = _build_arm_world_model("cuda:0", mu=mu, base_z=(0.095,) * N_WORLDS)
        first, solver, max_active = _cached_rollout(model, response, self.STEPS)
        second, _, _ = _cached_rollout(model, response, self.STEPS)
        self.assertTrue(solver._propagation_cached_response_active)
        self.assertGreaterEqual(max_active, 2, "scene must keep several bodies of one arm contact-active")
        np.testing.assert_array_equal(first, second, err_msg="run-to-run")
        coords = model.joint_coord_count // N_WORLDS
        dofs = model.joint_dof_count // N_WORLDS
        q, qd = first[:, : model.joint_coord_count], first[:, model.joint_coord_count :]
        for world in range(1, N_WORLDS):
            np.testing.assert_array_equal(q[:, world * coords : (world + 1) * coords], q[:, :coords])
            np.testing.assert_array_equal(qd[:, world * dofs : (world + 1) * dofs], qd[:, :dofs])

    def test_colored_is_bitwise_repeatable(self):
        """Repeat colored steps bitwise across runs and identical worlds."""
        self._assert_repeatable("propagation_colored", mu=0.7)

    def test_colored_frictionless_is_bitwise_repeatable(self):
        """Repeat frictionless colored steps bitwise across runs and identical worlds."""
        self._assert_repeatable("propagation_colored", mu=0.0)

    def test_plain_frictionless_is_bitwise_repeatable(self):
        """Repeat frictionless plain propagation steps bitwise across runs and identical worlds."""
        self._assert_repeatable("propagation", mu=0.0)

    def test_body_list_is_sorted_within_partitions(self):
        """Sort each partition of the body list by body index and keep the slot map its inverse."""
        model = _build_clutter_model("cuda:0", n_boxes=2)
        for response in ("propagation", "propagation_colored"):
            _, solver, _ = _cached_rollout(model, response, 20, cache_max_bodies=4)
            counts = solver._propagation_body_count.numpy()
            split = solver._propagation_cache_body_count.numpy()
            body_list = solver._propagation_body_list.numpy()
            local_slot = solver._propagation_body_local_slot.numpy()
            for world in range(body_list.shape[0]):
                n, k = int(counts[world]), int(split[world])
                self.assertGreater(k, 0)
                self.assertGreater(n, k, "scene must fill both partitions")
                for part in (body_list[world, :k], body_list[world, k:n]):
                    self.assertTrue(np.all(np.diff(part) > 0), f"{response} world {world}: {part}")
                np.testing.assert_array_equal(local_slot[body_list[world, :n]], np.arange(n))

    def test_slot_order_changes_the_cached_sum(self):
        """Fail-before reproducer: a fixed descending slot order changes the result, so claim order mattered."""
        model = _build_arm_world_model("cuda:0", mu=0.7, base_z=(0.095,) * N_WORLDS)
        ascending, _, _ = _cached_rollout(model, "propagation_colored", self.STEPS)
        module = inspect.getmodule(SolverFeatherPGS)
        with patch.object(module, "rank_propagation_body_map", _rank_propagation_body_map_descending):
            descending, _, max_active = _cached_rollout(model, "propagation_colored", self.STEPS)
        self.assertGreaterEqual(max_active, 2)
        differs = np.any(ascending != descending, axis=1)
        self.assertTrue(differs.any(), "slot order did not reach the cached sum")
        # The first difference is reassociation-sized; later steps compound it.
        first = int(np.argmax(differs))
        self.assertLess(float(np.max(np.abs(ascending[first] - descending[first]))), 1.0e-5)

    def test_adverse_claim_order_gives_the_same_step(self):
        """Claiming bodies in reverse row order must not change any result."""
        model = _build_arm_world_model("cuda:0", mu=0.7, base_z=(0.095,) * N_WORLDS)
        reference, _, _ = _cached_rollout(model, "propagation_colored", self.STEPS)
        module = inspect.getmodule(SolverFeatherPGS)
        launch = wp.launch

        def adverse_launch(kernel, dim, *args, **kwargs):
            if kernel is module.build_propagation_body_map_partitioned:
                kernel, dim = _claim_propagation_body_map_reversed, model.world_count
            return launch(kernel, dim, *args, **kwargs)

        with patch.object(module.wp, "launch", adverse_launch):
            adverse, solver, max_active = _cached_rollout(model, "propagation_colored", self.STEPS)
        self.assertGreaterEqual(max_active, 2)
        np.testing.assert_array_equal(adverse, reference)
        claimed = solver._propagation_body_claim_list.numpy()
        ranked = solver._propagation_body_list.numpy()
        counts = solver._propagation_body_count.numpy()
        self.assertTrue(
            any(not np.array_equal(claimed[w, : counts[w]], ranked[w, : counts[w]]) for w in range(len(counts))),
            "the adverse claim order never differed from the ranked order",
        )

    def test_zero_and_one_active_body_worlds(self):
        """A world with no contacts and a world with one contact body rank without touching other slots."""
        model = _build_arm_world_model("cuda:0", mu=0.7, base_z=(1.0, 0.095), gravity=0.0)
        # Only the second world's base sphere starts in contact; its other spheres are lifted clear.
        q = model.joint_q.numpy()
        q[model.joint_coord_count // 2 :] = np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        model.joint_q.assign(q)
        first, solver, _ = _cached_rollout(model, "propagation_colored", 20)
        second, _, _ = _cached_rollout(model, "propagation_colored", 20)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(solver._propagation_body_count.numpy(), [0, 1])
        self.assertEqual(int(solver._propagation_cache_body_count.numpy()[1]), 1)
        self.assertGreaterEqual(int(solver._propagation_body_list.numpy()[1, 0]), 0)

    def test_cache_capacity_boundary(self):
        """At exactly the cap the world stays cached and matches the tree walk; one over the cap falls back."""
        model = _build_arm_world_model("cuda:0", mu=0.7, base_z=(0.095,) * N_WORLDS)
        _, solver, _ = _cached_rollout(model, "propagation_colored", self.STEPS)
        active = int(solver._propagation_cache_body_count.numpy().min())
        self.assertGreaterEqual(active, 2)
        for cap in (active, active - 1):
            stats, _, test = _run_lockstep(self, model, "propagation_colored", 30, cache_max_bodies=cap)
            flags = test._propagation_cache_world_flag.numpy()
            counts = test._propagation_cache_body_count.numpy()
            np.testing.assert_array_equal(flags, (counts <= cap).astype(flags.dtype), err_msg=f"cap {cap}")
            if cap == active:
                self.assertTrue(np.any(counts == cap), f"no world exactly at cap {cap}: counts {counts}")
            else:
                self.assertTrue(np.any(counts > cap), f"no world over cap {cap}: counts {counts}")
            for key, diff in stats.max_diffs.items():
                self.assertLessEqual(diff, TestPropagationCachedResponse.TOL, f"cap {cap} {key}: {diff:.3e}")
            first, _, _ = _cached_rollout(model, "propagation_colored", 30, cache_max_bodies=cap)
            second, _, _ = _cached_rollout(model, "propagation_colored", 30, cache_max_bodies=cap)
            np.testing.assert_array_equal(first, second, err_msg=f"cap {cap}")


PROPAGATION_RESPONSES = ("propagation", "propagation_fused")


REGULARIZED = {"baumgarte": 0.0, "cfm": 0.0, "contact_regularization": 3.0}


# (articulated_contact_response, propagation_cached_response)
PROPAGATION_ROUTES = (
    ("propagation", False),
    ("propagation_fused", False),
    ("propagation", True),
    ("propagation_colored", False),
)


def _quadruped_model(device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [builder.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = builder.add_link(xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, 0.08), wp.quat_identity()))
            builder.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                builder.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _three_foot_slider_model(device):
    """One prismatic-Z DOF carrying three spheres on fixed-linked bodies above a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.0
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    joints = [builder.add_joint_prismatic(-1, slider, axis=wp.vec3(0.0, 0.0, 1.0))]
    for x in (-0.2, 0.0, 0.2):
        foot = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(foot, radius=0.05)
        joints.append(
            builder.add_joint_fixed(
                slider,
                foot,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _solver(model, response, iterations, cached=False, **kwargs):
    return SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        propagation_cached_response=cached,
        friction_anchor_beta=0.0,
        articulation_constraint_max=96,
        free_body_constraint_max=96,
        iterations=iterations,
        **kwargs,
    )


def _rollout(model, solver, joint_qd, steps):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        yield state_0.joint_qd.numpy()


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationCoupledContacts(unittest.TestCase):
    def test_shared_dof_feet_reach_rest(self):
        """Three feet on one prismatic DOF stop at the plane for every sweep count."""
        model = _three_foot_slider_model("cuda:0")
        for response in PROPAGATION_RESPONSES:
            for iterations in (1, 2, 12, 13):
                with self.subTest(response=response, iterations=iterations):
                    solver = _solver(model, response, iterations, baumgarte=0.0, cfm=0.0)
                    (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                    self.assertLess(abs(float(qd[0])), 1.0e-3)

    def test_regularized_feet_reach_immediate_equilibrium(self):
        """Splitting leaves the regularized contact fixed point of three coupled feet unchanged."""
        model = _three_foot_slider_model("cuda:0")
        for iterations in (32, 128):
            solver = _solver(model, "immediate", iterations, **REGULARIZED)
            (expected,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
            for response, cached in PROPAGATION_ROUTES:
                with self.subTest(response=response, cached=cached, iterations=iterations):
                    solver = _solver(model, response, iterations, cached=cached, **REGULARIZED)
                    (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                    self.assertAlmostEqual(float(qd[0]), float(expected[0]), delta=1.0e-4)

    def test_graph_replay_matches_eager_steps(self):
        """A captured step pair recounts the coupled bodies on every replay."""
        model = _three_foot_slider_model("cuda:0")
        for response, cached in PROPAGATION_ROUTES:
            with self.subTest(response=response, cached=cached):
                eager = _solver(model, response, 12, cached=cached, **REGULARIZED)
                expected = list(_rollout(model, eager, np.array([-1.0], dtype=np.float32), 7))[-1]
                solver = _solver(model, response, 12, cached=cached, **REGULARIZED)
                state, output = model.state(), model.state()
                control = model.control()
                state.joint_qd.assign(np.array([-1.0], dtype=np.float32))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                solver.step(state, output, control, contacts, DT)
                with wp.ScopedCapture(device=model.device) as capture:
                    for source, target in ((output, state), (state, output)):
                        pipeline.collide(source, contacts)
                        solver.step(source, target, control, contacts, DT)
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                np.testing.assert_allclose(output.joint_qd.numpy(), expected, atol=1.0e-6)
                self.assertEqual(int(solver._propagation_coupling_group_body_count.numpy().max()), 3)

    def test_floating_quadruped_with_joint_velocities_stays_bounded(self):
        """Four feet coupled through a light free root keep their roll rate physical."""
        model = _quadruped_model("cuda:0")
        joint_qd = np.linspace(-1.5, 1.5, model.joint_dof_count).astype(np.float32)
        for response in PROPAGATION_RESPONSES:
            with self.subTest(response=response):
                solver = _solver(model, response, 12)
                trajectory = list(_rollout(model, solver, joint_qd, 120))
                self.assertLess(float(np.abs(trajectory[0][3:6]).max()), 10.0)
                self.assertTrue(np.isfinite(trajectory[-1]).all())
                self.assertLess(float(np.abs(np.stack(trajectory)).max()), 50.0)


CONTACT_ROW_TYPE = 0


DENSE_PATH = 0


PROPAGATION_PATH = 2


def _zero_iteration_step(model: newton.Model, solver: SolverFeatherPGS):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    state_in.clear_forces()
    collision_pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
    solver._test_live_contact_count = int(contacts.rigid_contact_count.numpy()[0])
    return state_in


def _make_solver_propagation_same_articulation(model: newton.Model, mode: str, **kwargs) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=mode,
        iterations=0,
        velocity_iterations=0,
        contact_friction_scale=0.0,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        contact_warmstart=False,
        **kwargs,
    )


def _analytic_H_and_X(model: newton.Model, state) -> tuple[np.ndarray, np.ndarray]:
    """Joint-space inertia H [D,D] and per-body kinematic maps X [bodies,6,D]
    for single-world revolute trees, world frame, velocities as
    [v_com_lin, omega] (the propagation body convention)."""
    body_q = state.body_q.numpy().astype(np.float64)
    body_com = model.body_com.numpy().astype(np.float64)
    body_mass = model.body_mass.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    joint_parent = model.joint_parent.numpy().astype(np.int32)
    joint_child = model.joint_child.numpy().astype(np.int32)
    joint_axis = model.joint_axis.numpy().astype(np.float64)
    joint_X_p = model.joint_X_p.numpy().astype(np.float64)
    joint_qd_start = model.joint_qd_start.numpy().astype(np.int32)
    armature = model.joint_armature.numpy().astype(np.float64)

    def rot(q):
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    D = int(model.joint_dof_count)
    n_bodies = model.body_count
    com_w = np.zeros((n_bodies, 3))
    R_w = np.zeros((n_bodies, 3, 3))
    for b in range(n_bodies):
        p, q = body_q[b, :3], body_q[b, 3:]
        R_w[b] = rot(q)
        com_w[b] = p + R_w[b] @ body_com[b]

    # world-frame joint axis and anchor per joint
    n_joints = joint_parent.shape[0]
    axis_w = np.zeros((n_joints, 3))
    anchor_w = np.zeros((n_joints, 3))
    parent_joint_of_body = np.full(n_bodies, -1, dtype=np.int32)
    for j in range(n_joints):
        parent = int(joint_parent[j])
        if parent >= 0:
            Rp = R_w[parent]
            pp = body_q[parent, :3]
        else:
            Rp = np.eye(3)
            pp = np.zeros(3)
        Xp_p, Xp_q = joint_X_p[j, :3], joint_X_p[j, 3:]
        anchor_w[j] = pp + Rp @ Xp_p
        axis_w[j] = (
            Rp @ rot(Xp_q) @ joint_axis[joint_qd_start[j]] if joint_axis.ndim == 2 else Rp @ rot(Xp_q) @ joint_axis[j]
        )
        parent_joint_of_body[int(joint_child[j])] = j

    X = np.zeros((n_bodies, 6, D))
    for b in range(n_bodies):
        j = int(parent_joint_of_body[b])
        while j >= 0:
            dof = int(joint_qd_start[j])
            a = axis_w[j]
            X[b, 0:3, dof] = np.cross(a, com_w[b] - anchor_w[j])
            X[b, 3:6, dof] = a
            parent = int(joint_parent[j])
            j = int(parent_joint_of_body[parent]) if parent >= 0 else -1

    H = np.zeros((D, D))
    for b in range(n_bodies):
        M = np.zeros((6, 6))
        M[0:3, 0:3] = body_mass[b] * np.eye(3)
        M[3:6, 3:6] = R_w[b] @ body_inertia[b] @ R_w[b].T
        H += X[b].T @ M @ X[b]
    H += np.diag(armature[:D])
    return H, X


def _propagation_operator_diag(solver: SolverFeatherPGS) -> np.ndarray:
    count = int(solver._propagation_constraint_count.numpy()[0])
    J_a = solver._propagation_J_a.numpy()[0, :count].astype(np.float64)
    J_b = solver._propagation_J_b.numpy()[0, :count].astype(np.float64)
    MiJt_a = solver._propagation_MiJt_a.numpy()[0, :count].astype(np.float64)
    MiJt_b = solver._propagation_MiJt_b.numpy()[0, :count].astype(np.float64)
    body_a = solver._propagation_body_a.numpy()[0, :count].astype(np.int32)
    body_b = solver._propagation_body_b.numpy()[0, :count].astype(np.int32)
    diag = np.zeros(count, dtype=np.float64)
    for row in range(count):
        if body_a[row] >= 0:
            diag[row] += float(np.dot(J_a[row], MiJt_a[row]))
        if body_b[row] >= 0:
            diag[row] += float(np.dot(J_b[row], MiJt_b[row]))
    return diag


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationSameArticulation(unittest.TestCase):
    def test_scissor_scene_produces_same_articulation_contact(self):
        model = _build_scissor_model("cuda:0")
        solver = _make_solver_propagation_same_articulation(model, "immediate")
        _zero_iteration_step(model, solver)
        dense_rows = _contact_rows(solver, DENSE_PATH)
        self.assertGreater(len(dense_rows), 0, "scissor scene produced no dense link-link contact")

    def test_default_routing_keeps_same_articulation_rows_dense(self):
        model = _build_scissor_model("cuda:0")
        solver = _make_solver_propagation_same_articulation(model, "propagation")
        _zero_iteration_step(model, solver)
        self.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), 0)
        self.assertGreater(len(_contact_rows(solver, DENSE_PATH)), 0)

    def test_flag_requires_propagation_mode(self):
        model = _build_scissor_model("cuda:0")
        with self.assertRaises(ValueError):
            _make_solver_propagation_same_articulation(model, "immediate", propagation_same_articulation_rows=True)

    def test_cross_response_effective_mass_matches_reference_operator(self):
        """The defining check: with same-articulation rows routed to propagation,
        each row's J M^-1 J^T must match the analytic reference including the
        cross term between the two links."""
        model = _build_scissor_model("cuda:0")

        reference = _make_solver_propagation_same_articulation(model, "immediate")
        _zero_iteration_step(model, reference)
        dense_rows = _contact_rows(reference, DENSE_PATH)
        self.assertGreater(len(dense_rows), 0)

        solver = _make_solver_propagation_same_articulation(
            model, "propagation", propagation_same_articulation_rows=True
        )
        state = _zero_iteration_step(model, solver)
        prop_rows = _contact_rows(solver, PROPAGATION_PATH)
        self.assertEqual(
            set(prop_rows.keys()),
            set(dense_rows.keys()),
            "same-articulation contacts were not routed to propagation rows",
        )
        prop_diag = _propagation_operator_diag(solver)

        # Analytic reference: generalized row Jacobian g = X_a^T j_a + X_b^T j_b
        # from the solver's own row geometry, W_kk = g^T H^-1 g.
        H, X = _analytic_H_and_X(model, state)
        count = int(solver._propagation_constraint_count.numpy()[0])
        J_a = solver._propagation_J_a.numpy()[0, :count].astype(np.float64)
        J_b = solver._propagation_J_b.numpy()[0, :count].astype(np.float64)
        body_a = solver._propagation_body_a.numpy()[0, :count].astype(np.int32)
        body_b = solver._propagation_body_b.numpy()[0, :count].astype(np.int32)

        # Convention check: the reference solver's Y rows must equal H^-1 g
        # (up to overall row sign) — this pins frames/ordering before we
        # trust the reference for the real assertion.
        width = int(reference._world_dof_count.numpy()[0])
        Y = reference._Y_world.numpy()[0].astype(np.float64)
        for contact, dense_slot in sorted(dense_rows.items()):
            row = prop_rows[contact]
            g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
            y_ref = np.linalg.solve(H, g)
            y_got = Y[dense_slot, :width]
            err = min(np.linalg.norm(y_got - y_ref), np.linalg.norm(y_got + y_ref))
            self.assertLess(
                err / max(np.linalg.norm(y_ref), 1e-12),
                1.0e-3,
                f"analytic reference disagrees with solver Y row {dense_slot}: {y_got} vs {y_ref}",
            )

        checked = 0
        for contact in sorted(dense_rows):
            row = prop_rows[contact]
            g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
            ref = float(g @ np.linalg.solve(H, g))
            got = prop_diag[row]
            if ref < 1.0e-6:
                # Kinematically locked direction: true W_kk is zero, both
                # sides must be at float noise level.
                self.assertLess(abs(got), 1.0e-6)
                continue
            checked += 1
            rel = abs(got - ref) / ref
            self.assertLess(
                rel,
                1.0e-3,
                f"contact {contact}: propagation W_kk {got:.6e} vs reference {ref:.6e} "
                f"(rel {rel:.3e}) — cross response term missing or wrong",
            )
        self.assertGreater(checked, 0, "no non-degenerate same-articulation rows were checked")


_WARP_KERNEL_DICTS = (
    "_propagate_tree_warp_kernels_by_size",
    "_factor_tree_warp_kernels_by_size",
    "_response_tree_warp_kernels_by_size",
    "_refresh_tree_warp_kernels_by_size",
)


def _make_solver_propagation_free_root_warp(model: newton.Model, response: str) -> SolverFeatherPGS:
    # propagation_cached_response=False pins the per-iteration tree-walk
    # propagation path these tests compare (warp vs serial variants). The
    # cached-response GEMV replacement has its own equivalence gate in this module.
    return SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        iterations=8,
        contact_warmstart=False,
        propagation_cached_response=False,
    )


def _force_serial_fallback(solver: SolverFeatherPGS) -> None:
    for name in _WARP_KERNEL_DICTS:
        kernels = getattr(solver, name, None)
        if kernels:
            setattr(solver, name, dict.fromkeys(kernels, None))


def _step_n(model, solver, n_steps, *, state_in=None):
    state_in = model.state() if state_in is None else state_in
    state_out = model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    for _ in range(n_steps):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    return state_in


def _tree_inputs(model, solver, art):
    return {
        "S": solver._propagation_joint_S_flat.numpy().astype(np.float64),
        "U": solver._propagation_tree_U.numpy().astype(np.float64),
        "Dinv": solver._propagation_tree_D_inv.numpy().astype(np.float64),
        "com": solver._propagation_body_com_rel.numpy().astype(np.float64),
        "jp": model.joint_parent.numpy(),
        "jc": model.joint_child.numpy(),
        "qd_start": model.joint_qd_start.numpy(),
        "j0": int(model.articulation_start.numpy()[art]),
        "j1": int(model.articulation_start.numpy()[art + 1]),
    }


def _xlt_wrench(w, e):
    return np.concatenate([w[:3], w[3:] + np.cross(e, w[:3])])


def _xlt_twist(v, e):
    return np.concatenate([v[:3] + np.cross(v[3:], e), v[3:]])


def _ref_propagate(ti, imp_by_body, v_out):
    """Mirror of kernels.propagate_tree_impulses_for_size (one articulation)."""
    S, U, Dinv, com = ti["S"], ti["U"], ti["Dinv"], ti["com"]
    jp, jc, qd_start = ti["jp"], ti["jc"], ti["qd_start"]
    j0, j1 = ti["j0"], ti["j1"]
    pA = {}
    u = np.zeros(S.shape[0])
    body_delta = {}
    for j in range(j0, j1):
        pA[int(jc[j])] = -np.asarray(imp_by_body.get(int(jc[j]), np.zeros(6)), dtype=np.float64)
        body_delta[int(jc[j])] = np.zeros(6)
    for j in range(j1 - 1, j0 - 1, -1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        for g in range(d0, d1):
            u[g] = -np.dot(S[g], pA[child])
        if parent >= 0:
            p = pA[child].copy()
            for a in range(d0, d1):
                coeff = sum(Dinv[j, a - d0, b - d0] * u[b] for b in range(d0, d1))
                p += U[a] * coeff
            pA[parent] += _xlt_wrench(p, com[child] - com[parent])
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        pd = np.zeros(6)
        if parent >= 0:
            pd = _xlt_twist(body_delta[parent], com[child] - com[parent])
        qdd = np.zeros(d1 - d0)
        for a in range(d0, d1):
            acc = 0.0
            for b in range(d0, d1):
                pt = np.dot(U[b], pd) if parent >= 0 else 0.0
                acc += Dinv[j, a - d0, b - d0] * (u[b] - pt)
            qdd[a - d0] = acc
            v_out[a] += acc
        body_delta[child] = pd + sum(S[a] * qdd[a - d0] for a in range(d0, d1))
    # third pass: recompute body twists from the updated v_out
    body_qd = {}
    twist = {}
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        val = np.zeros(6)
        if parent >= 0:
            val = _xlt_twist(twist[parent], com[child] - com[parent])
        for a in range(d0, d1):
            val = val + S[a] * v_out[a]
        twist[child] = val
        body_qd[child] = val
    return v_out, body_qd


def _ref_body_response(ti, body):
    """6x6 response of ``body``: column = body twist delta per unit basis wrench."""
    R = np.zeros((6, 6))
    for basis in range(6):
        imp = np.zeros(6)
        imp[basis] = 1.0
        # At zero initial generalized velocity the recomputed body twist for a
        # unit test wrench equals the body's velocity response column.
        _, body_qd = _ref_propagate(ti, {body: imp}, np.zeros(ti["S"].shape[0]))
        R[:, basis] = body_qd[body]
    return R


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationFreeRootWarp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        # contact-free model: floating chains tumbling in space
        cls.model_free = _build_free_root_chain_model(cls.device, base_z=2.0, contact_sphere=None, ground=False)

    def test_free_root_shape_detected(self):
        solver = _make_solver_propagation_free_root_warp(self.model_free, "propagation_colored")
        sizes = [int(s) for s in solver._size_groups if solver._n_arts_by_size[s] > 0]
        self.assertEqual(len(sizes), 1)
        size = sizes[0]
        self.assertFalse(solver._propagation_tree_single_dof_by_size[size])
        self.assertTrue(solver._propagation_tree_free_root_by_size[size])
        for name in _WARP_KERNEL_DICTS:
            kernels = getattr(solver, name, None)
            self.assertIsNotNone(kernels, name)
            self.assertIsNotNone(kernels.get(size), f"{name}[{size}] should hold a warp variant")

    def _paired_solvers(self, n_steps):
        """Two solvers advanced identically on a contact-free trajectory.

        Without contact rows the tree kernels do not influence the
        trajectory, and no atomically-ordered contact bookkeeping runs, so
        both solvers reach bitwise-identical states and tree factorizations
        are computed from identical inputs.
        """
        solver_w = _make_solver_propagation_free_root_warp(self.model_free, "propagation_colored")
        solver_s = _make_solver_propagation_free_root_warp(self.model_free, "propagation_colored")
        _force_serial_fallback(solver_s)
        state_w = _step_n(self.model_free, solver_w, n_steps)
        state_s = _step_n(self.model_free, solver_s, n_steps)
        np.testing.assert_array_equal(state_w.joint_q.numpy(), state_s.joint_q.numpy())
        return solver_w, solver_s

    def test_factor_matches_serial(self):
        model = self.model_free
        qd_start = model.joint_qd_start.numpy()
        n_joints = model.joint_count
        for n_steps in (1, 12, 25):
            with self.subTest(n_steps=n_steps):
                solver_w, solver_s = self._paired_solvers(n_steps)
                U_w = solver_w._propagation_tree_U.numpy()
                U_s = solver_s._propagation_tree_U.numpy()
                Dc_w = solver_w._propagation_tree_D_chol.numpy()
                Dc_s = solver_s._propagation_tree_D_chol.numpy()
                Di_w = solver_w._propagation_tree_D_inv.numpy()
                Di_s = solver_s._propagation_tree_D_inv.numpy()
                for j in range(n_joints):
                    dc = int(qd_start[j + 1] - qd_start[j])
                    if dc == 0:
                        continue
                    d0 = int(qd_start[j])
                    scale = max(1.0, float(np.max(np.abs(U_s[d0 : d0 + dc]))))
                    self.assertLess(float(np.max(np.abs(U_w[d0 : d0 + dc] - U_s[d0 : d0 + dc]))), 1e-5 * scale)
                    scale = max(1.0, float(np.max(np.abs(Di_s[j, :dc, :dc]))))
                    self.assertLess(float(np.max(np.abs(Di_w[j, :dc, :dc] - Di_s[j, :dc, :dc]))), 1e-5 * scale)
                    scale = max(1.0, float(np.max(np.abs(Dc_s[j, :dc, :dc]))))
                    self.assertLess(float(np.max(np.abs(Dc_w[j, :dc, :dc] - Dc_s[j, :dc, :dc]))), 1e-5 * scale)

    def test_propagate_matches_serial(self):
        model = self.model_free
        rng = np.random.default_rng(1234)
        solver_w, solver_s = self._paired_solvers(20)
        bodies_per_world = model.body_count // N_WORLDS
        patterns = [rng.normal(scale=0.5, size=(model.body_count, 6)).astype(np.float32)]
        for body in (0, 1, bodies_per_world - 1):  # root, first link, deepest link
            p = np.zeros((model.body_count, 6), dtype=np.float32)
            p[body] = rng.normal(scale=1.0, size=6)
            patterns.append(p)
        for pi, imps in enumerate(patterns):
            with self.subTest(pattern=pi):
                v0 = rng.normal(scale=0.1, size=solver_w._v_out.shape).astype(np.float32)
                for s in (solver_w, solver_s):
                    s._propagation_body_impulses.assign(imps)
                    s._v_out.assign(v0)
                    s._propagation_body_qd.zero_()
                solver_w._propagate_response()
                solver_s._propagate_response()
                wp.synchronize()
                dv = solver_w._v_out.numpy() - solver_s._v_out.numpy()
                dqd = solver_w._propagation_body_qd.numpy() - solver_s._propagation_body_qd.numpy()
                scale = max(1.0, float(np.max(np.abs(solver_s._v_out.numpy()))))
                self.assertLess(float(np.max(np.abs(dv))), 1e-5 * scale)
                self.assertLess(float(np.max(np.abs(dqd))), 1e-5 * scale)
                # deferred impulses must be consumed
                self.assertEqual(float(np.max(np.abs(solver_w._propagation_body_impulses.numpy()))), 0.0)

    def test_propagate_matches_numpy_reference(self):
        model = self.model_free
        rng = np.random.default_rng(99)
        solver_w, _ = self._paired_solvers(20)
        ti = _tree_inputs(model, solver_w, art=0)
        bodies_per_world = model.body_count // N_WORLDS
        imps = np.zeros((model.body_count, 6), dtype=np.float32)
        imps[:bodies_per_world] = rng.normal(scale=0.5, size=(bodies_per_world, 6))
        solver_w._propagation_body_impulses.assign(imps)
        solver_w._v_out.zero_()
        solver_w._propagation_body_qd.zero_()
        solver_w._propagate_response()
        wp.synchronize()
        n_dofs_w0 = int(model.joint_dof_count) // N_WORLDS
        ref_v, ref_body_qd = _ref_propagate(
            ti,
            {b: imps[b].astype(np.float64) for b in range(bodies_per_world)},
            np.zeros(ti["S"].shape[0]),
        )
        got_v = solver_w._v_out.numpy()[:n_dofs_w0]
        self.assertLess(float(np.max(np.abs(got_v - ref_v[:n_dofs_w0]))), 1e-4)
        got_qd = solver_w._propagation_body_qd.numpy()
        for b in range(bodies_per_world):
            self.assertLess(float(np.max(np.abs(got_qd[b] - ref_body_qd[b]))), 1e-4)

    def test_refresh_matches_serial(self):
        rng = np.random.default_rng(4321)
        solver_w, solver_s = self._paired_solvers(20)
        v0 = rng.normal(scale=0.5, size=solver_w._v_out.shape).astype(np.float32)
        for s in (solver_w, solver_s):
            s._v_out.assign(v0)
            s._propagation_body_qd.zero_()
        solver_w._refresh_propagation_body_qd_from_vout(force=True)
        solver_s._refresh_propagation_body_qd_from_vout(force=True)
        wp.synchronize()
        qd_w = solver_w._propagation_body_qd.numpy()
        qd_s = solver_s._propagation_body_qd.numpy()
        scale = max(1.0, float(np.max(np.abs(qd_s))))
        self.assertLess(float(np.max(np.abs(qd_w - qd_s))), 1e-5 * scale)

    def test_response_matches_numpy_reference(self):
        """Warp free-root response vs the serial tree-solve semantics.

        The serial fallback for free-root groups is the path-restricted
        generic kernel, which only fills rows for contact-active bodies; the
        warp variant fills every link (the single-DOF revolute contract). Both
        must equal the exact articulated-body response, so the warp output is
        gated against a float64 reference of the serial algorithm.
        """
        model = self.model_free
        solver_w, _ = self._paired_solvers(20)
        size = next(int(s) for s in solver_w._size_groups if solver_w._n_arts_by_size[s] > 0)
        self.assertIsNotNone(solver_w._response_tree_warp_kernels_by_size.get(size))
        ti = _tree_inputs(model, solver_w, art=0)
        resp = solver_w._propagation_body_response.numpy()
        bodies_per_world = model.body_count // N_WORLDS
        for body in range(bodies_per_world):
            ref = _ref_body_response(ti, body)
            scale = max(1.0, float(np.max(np.abs(ref))))
            self.assertLess(
                float(np.max(np.abs(resp[body] - ref))),
                1e-4 * scale,
                f"body {body} response mismatch",
            )

    def test_end_to_end_single_contact(self):
        """Full stepping over ground with one contact sphere per world.

        A single contact point keeps the contact pipeline's atomic slot
        allocation trivially deterministic, so the warp-vs-serial trajectory
        difference reflects only the tree-kernel float reassociation.
        """
        for contact_sphere, base_z in (("base", 0.115), ("deep", 0.12)):
            model = _build_free_root_chain_model(self.device, base_z=base_z, contact_sphere=contact_sphere, ground=True)
            for response in ("propagation_colored", "propagation"):
                with self.subTest(contact=contact_sphere, articulated_contact_response=response):
                    results = {}
                    for mode in ("warp", "serial", "serial2"):
                        solver = _make_solver_propagation_free_root_warp(model, response)
                        if mode != "warp":
                            _force_serial_fallback(solver)
                        state = _step_n(model, solver, 40)
                        results[mode] = {
                            "joint_q": state.joint_q.numpy().copy(),
                            "joint_qd": state.joint_qd.numpy().copy(),
                            "propagation_body_qd": solver._propagation_body_qd.numpy().copy(),
                            "rows": int(solver._propagation_constraint_count.numpy().sum()),
                        }
                    self.assertGreater(results["warp"]["rows"], 0, "no propagation contact rows produced")
                    # precondition: the scene is pipeline-deterministic
                    for key in ("joint_q", "joint_qd"):
                        np.testing.assert_array_equal(
                            results["serial"][key],
                            results["serial2"][key],
                            err_msg=f"serial trajectory not reproducible for {key}; scene invalidates the gate",
                        )
                    # The warp and serial paths differ only by float reassociation, but 40 steps of
                    # contact dynamics amplify that seed chaotically: measured joint_qd diffs are
                    # ~5e-7 at 10 steps, 1e-5..4e-5 at 20 and 4e-5..1.2e-4 at 40, with the 40-step
                    # value swinging by 3x across GPU architectures (8.1e-5 on sm_120 vs 1.2e-4 on
                    # sm_86 at the same commit) and across physically equivalent trajectory changes.
                    # The gate is set ~4x above the observed band to stay deterministic per
                    # hardware; structural kernel bugs produce diffs of 1e-2 or more and still trip
                    # it comfortably.
                    for key in ("joint_q", "joint_qd", "propagation_body_qd"):
                        diff = float(np.max(np.abs(results["warp"][key] - results["serial"][key])))
                        self.assertLessEqual(
                            diff,
                            5e-4,
                            f"{key} max abs diff {diff:.3e} exceeds 5e-4",
                        )


_VARIANT_ENV = (
    "FEATHER_PGS_COLORED_PREFETCH",
    "FEATHER_PGS_COLORED_STAGED",
    "FEATHER_PGS_COLORED_PACKED",
    "FEATHER_PGS_COLORED_STRAIGHT_LINE",
)


def _solver_colored_pipeline(model, env, regularization=0.0):
    """Build the colored solver with only ``env`` set among the kernel-variant switches."""
    with mock.patch.dict(os.environ, env):
        for key in _VARIANT_ENV:
            if key not in env:
                os.environ.pop(key, None)
        return newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            iterations=8,
            free_body_constraint_max=8192,
            articulation_constraint_max=64,
            contact_regularization=regularization,
        )


class TestFeatherPGSColoredStagingSelection(unittest.TestCase):
    def test_staging_requires_cp_async_and_few_worlds(self):
        """Staging emits cp.async (compute capability 8.0+) and is for a few worlds only."""
        self.assertFalse(_colored_staging_supported(75, 1, 1, 60), "SM75 cannot compile cp.async")
        self.assertTrue(_colored_staging_supported(80, 1, 1, 60))
        self.assertTrue(_colored_staging_supported(120, 1, 16, 60))
        self.assertFalse(_colored_staging_supported(120, 1, 17, 60), "many worlds keep the packed kernel")
        self.assertFalse(_colored_staging_supported(120, 2, 1, 60), "staging needs the lane-per-unit body")
        self.assertFalse(_colored_staging_supported(120, 1, 1, 400), "staging must fit static shared memory")


@unittest.skipUnless(wp.get_device().is_cuda, "propagation_colored requires CUDA")
class TestFeatherPGSColoredPipeline(unittest.TestCase):
    def test_pipelined_variants_match_the_reference_bitwise(self):
        """Prefetched and (where supported) staged sweeps match the reference unit body."""
        for regularization in (0.0, 0.01):
            with self.subTest(regularization=regularization):
                self._check_variants(regularization)

    def test_straight_line_units_match_the_row_loop(self):
        """Standard units solved straight-line agree with the generic row loop.

        Every variant shares the straight-line code, so the bitwise check above cannot see
        a mistake in it; this compares it with the row loop it replaces. The two differ
        only in summation order (pairwise dot products), so one step (eight sweeps) agrees
        to rounding, impulses included. Later steps are not compared: friction-patch
        decisions taken from those impulses may relayout the rows.
        """
        for regularization in (0.0, 0.01):
            with self.subTest(regularization=regularization):
                model, contacts, control, start = self._settled_heap(regularization)
                results = {}
                for name, env in (("row loop", {"FEATHER_PGS_COLORED_STRAIGHT_LINE": "0"}), ("straight line", {})):
                    solver = _solver_colored_pipeline(model, env, regularization)
                    results[name] = self._run(solver, model, contacts, control, start, 1)
                    if name == "straight line":
                        n = int(solver._color_world_offsets.numpy()[-1])
                        kinds = solver._color_unit_meta.numpy().reshape(-1, PROPAGATION_UNIT_META)[:n]
                        kinds = kinds[:, 6 + PROPAGATION_UNIT_RING]
                        self.assertGreater(int(np.sum(kinds == 1)), n // 2, "few units took the straight-line path")
                for label, got, want in zip(
                    ("body_q", "body_qd", "impulses"), results["straight line"], results["row loop"], strict=True
                ):
                    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6, err_msg=label)

    def _settled_heap(self, regularization):
        model, _ = _heap(nx=6, ny=6, nz=3)
        model.rigid_contact_max = 8192
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        warm = _solver_colored_pipeline(model, {}, regularization)
        for _ in range(8):  # settle into resting contact with patch history
            pipeline.collide(state_0, contacts)
            warm.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
        return model, contacts, control, start

    @staticmethod
    def _run(solver, model, contacts, control, start, steps):
        a, b = model.state(), model.state()
        for field, value in start.items():
            getattr(a, field).assign(value)
        solver.reset(a, flags=0)
        for _ in range(steps):
            solver.step(a, b, control, contacts, 1.0 / 240.0)
            a, b = b, a
        return a.body_q.numpy(), a.body_qd.numpy(), solver._propagation_impulses.numpy()

    def _check_variants(self, regularization):
        """Prefetched and shared-memory-staged sweeps reproduce the reference unit body.

        A settling heap of boxes with friction patches (the default) gives colors with
        many units, contact rows with friction pairs, and patch rings. Contacts are
        generated once so every solver sees identical input; the variants then run the
        same steps and must agree bit for bit on body state and row impulses. Without
        regularization the row-weight buffer is a one-element placeholder the kernels
        must not read; with it the weights are real rows.
        """
        model, _ = _heap(nx=6, ny=6, nz=3)
        model.rigid_contact_max = 8192
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        control = model.control()
        state_0, state_1 = model.state(), model.state()
        warm = _solver_colored_pipeline(model, {}, regularization)
        for _ in range(8):  # settle into resting contact with patch history
            pipeline.collide(state_0, contacts)
            warm.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        pipeline.collide(state_0, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 400)
        start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}

        variants = {
            "reference": ({"FEATHER_PGS_COLORED_PREFETCH": "0"}, "_pf0_st0"),
            "prefetch": ({"FEATHER_PGS_COLORED_PACKED": "0"}, "_pf1_st0_pk0"),
            "default": ({}, "_pf1_st0_pk1"),
            "staged": ({"FEATHER_PGS_COLORED_STAGED": "1"}, None),
        }
        results = {}
        for name, (env, expected) in variants.items():
            solver = _solver_colored_pipeline(model, env, regularization)
            kernel = solver._pgs_solve_propagation_colored_warp_kernel
            self.assertIsNotNone(kernel)
            tag = expected
            if tag is None:
                # Staging needs cp.async (SM80+); older devices take the unstaged packed
                # kernel, which must match as well.
                staged = _colored_staging_supported(
                    int(model.device.arch), 1, solver._world_count, int(solver._max_propagation_bodies)
                )
                tag = "_pf1_st1" if staged else "_pf1_st0_pk1"
            self.assertIn(tag, kernel.key, f"{name} did not build its kernel variant")
            a, b = model.state(), model.state()
            for field, value in start.items():
                getattr(a, field).assign(value)
            solver.reset(a, flags=0)
            for _ in range(5):
                solver.step(a, b, control, contacts, 1.0 / 240.0)
                a, b = b, a
            results[name] = (a.body_q.numpy(), a.body_qd.numpy(), solver._propagation_impulses.numpy())

        ref = results["reference"]
        self.assertTrue(all(np.isfinite(x).all() for x in ref))
        self.assertGreater(np.abs(ref[2]).max(), 0.0, "no contact impulses were solved")
        for name in ("prefetch", "default", "staged"):
            for label, got, want in zip(("body_q", "body_qd", "impulses"), results[name], ref, strict=True):
                np.testing.assert_array_equal(got, want, err_msg=f"{name} {label} differs from the reference")


def _heap(nx=6, ny=6, nz=3, pitch=0.042, half=0.02, kinematic_tray=False):
    """A dense heap of touching boxes on the ground, optionally inside a kinematic tray."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    tray = -1
    if kinematic_tray:
        # One kinematic body under the whole heap: every box rests on it, so it is a hub
        # touching every contact unit of the bottom layer.
        tray = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(tray, hx=nx * pitch, hy=ny * pitch, hz=0.01)
    z0 = 0.04 if kinematic_tray else 0.021
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                b = builder.add_body(
                    xform=wp.transform(
                        wp.vec3((ix - nx / 2) * pitch, (iy - ny / 2) * pitch, z0 + iz * (2 * half + 0.001)),
                        wp.quat_identity(),
                    )
                )
                builder.add_shape_box(b, hx=half, hy=half, hz=half)
    return builder.finalize(), tray


def _color_counts(solver):
    """Units per color for world 0; the last live entry is the serial overflow tail."""
    entries = PROPAGATION_COLOR_TAIL + 2
    offsets = solver._color_world_offsets.numpy()[:entries].astype(np.int64)
    counts = np.diff(offsets)
    return counts[:PROPAGATION_COLOR_TAIL], int(counts[PROPAGATION_COLOR_TAIL])


def _settle(model, steps=12, **solver_kwargs):
    model.rigid_contact_max = 8192  # sizes the solver's contact scratch; set before construction
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="propagation_colored",
        iterations=6,
        free_body_constraint_max=8192,
        articulation_constraint_max=64,
        **solver_kwargs,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return solver, state_0, int(contacts.rigid_contact_count.numpy()[0])


def _first_fit_reference(solver, world=0):
    """Reference coloring of world 0's units: sort by contact, first-fit, stable counting sort."""
    entries = PROPAGATION_COLOR_TAIL + 2
    stride = solver._propagation_max_constraints
    n = min(int(solver._color_world_unit_cursor.numpy()[world]), stride)
    base = world * stride
    contact = solver._color_unit_contact.numpy()[base : base + n]
    body_a = solver._color_unit_body_a.numpy()[base : base + n]
    body_b = solver._color_unit_body_b.numpy()[base : base + n]
    length = solver._color_unit_len.numpy()[base : base + n]
    order = np.argsort(contact, kind="stable")
    used = {}
    color = np.zeros(n, dtype=np.int64)
    for u in order:
        c = 0
        while c < PROPAGATION_COLOR_TAIL and any(x >= 0 and c in used.get(x, ()) for x in (body_a[u], body_b[u])):
            c += 1
        color[u] = c
        if c < PROPAGATION_COLOR_TAIL:
            for x in (body_a[u], body_b[u]):
                if x >= 0:
                    used.setdefault(x, set()).add(c)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(color, minlength=entries))])[:entries]
    cursor = offsets.copy()
    position_unit = np.zeros(n, dtype=np.int64)
    for u in order:
        position_unit[cursor[color[u]]] = u
        cursor[color[u]] += 1
    row_start = np.concatenate([[0], np.cumsum(length[position_unit])])[:n]
    return n, color, offsets, row_start, contact[position_unit]


@unittest.skipUnless(wp.get_device().is_cuda, "propagation_colored requires CUDA")
class TestFeatherPGSColoredScheduling(unittest.TestCase):
    def test_dense_heap_colors_without_a_serial_tail(self):
        """A dense heap must color into few colors with nothing left for the serial tail.

        Round-based ticket bidding admitted only the units holding the maximum ticket on
        both of their bodies, which on a heap is a handful per round: it exhausted the
        color budget and spilled most units into the ordered serial tail, making the
        sweep serial. First-fit greedy edge coloring is bounded by 2*degree-1 colors.
        """
        model, _ = _heap()
        solver, state, contacts = _settle(model)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertGreater(contacts, 400, "heap did not generate a dense contact set")

        counts, tail = _color_counts(solver)
        used = int(np.sum(counts > 0))
        self.assertEqual(tail, 0, f"{tail} units spilled into the serial overflow tail")
        self.assertLess(used, PROPAGATION_COLOR_TAIL // 2, f"coloring used {used} colors")
        self.assertGreater(
            counts[counts > 0].mean(), 8.0, "colors hold too few units each; the sweep is effectively serial"
        )

    def test_kinematic_hub_does_not_serialize_its_contacts(self):
        """One kinematic body under the whole heap must not multiply the color count.

        A prescribed body takes no velocity update from a row, so rows touching it cannot
        conflict with each other. Counting it as a conflicting body gives every contact
        against the tray its own color, so the count scales with the number of boxes
        resting on it; exempting it keeps the coloring close to the tray-free heap.
        """
        model_free, _ = _heap()
        solver_free, _, _ = _settle(model_free)
        counts_free, tail_free = _color_counts(solver_free)
        colors_free = int(np.sum(counts_free > 0))

        model_hub, tray = _heap(kinematic_tray=True)
        self.assertGreaterEqual(tray, 0)
        solver_hub, state, _ = _settle(model_hub)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        counts_hub, tail_hub = _color_counts(solver_hub)
        colors_hub = int(np.sum(counts_hub > 0))

        self.assertEqual(tail_free, 0)
        self.assertEqual(tail_hub, 0)
        self.assertLessEqual(
            colors_hub,
            1.5 * colors_free + 8,
            f"a kinematic hub raised the color count from {colors_free} to {colors_hub}",
        )

    def test_colored_result_matches_the_immediate_response(self):
        """The colored schedule must settle the heap like the reference row path."""
        model, _ = _heap(nx=4, ny=4, nz=2)
        _, colored, _ = _settle(model, steps=24)
        model_ref, _ = _heap(nx=4, ny=4, nz=2)
        model_ref.rigid_contact_max = 8192
        solver = newton.solvers.SolverFeatherPGS(
            model_ref,
            solve_mode="matrix_free",
            articulated_contact_response="immediate",
            iterations=6,
            free_body_constraint_max=8192,
            articulation_constraint_max=64,
            row_watermark=True,
        )
        pipeline = newton.CollisionPipeline(model_ref, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        s0, s1 = model_ref.state(), model_ref.state()
        control = model_ref.control()
        for _ in range(24):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0

        # An undersized reference drops rows in atomic order, which makes it nondeterministic.
        self.assertEqual(solver.constraint_row_watermarks()["free_body_dropped_contact_rows_high_water"], 0)
        a = colored.body_q.numpy()[:, :3]
        b = s0.body_q.numpy()[:, :3]
        self.assertTrue(np.isfinite(b).all())
        # Different sweep order, so not bitwise equal; the heap must still stand.
        self.assertLess(float(np.max(np.abs(a[:, 2] - b[:, 2]))), 0.01)

    def test_coloring_is_deterministic_for_a_fixed_contact_set(self):
        """Same state and contacts must give the same partition and the same velocities.

        The unit list is gathered with an atomic cursor, so its order differs between
        launches; the coloring must not inherit that order. The pre-build kernel sorts
        units by global contact index before the order-dependent greedy pass.
        """
        model, _ = _heap()
        _, state, _ = _settle(model)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        state.clear_forces()
        pipeline.collide(state, contacts)
        control = model.control()
        results = []
        for _ in range(3):
            state_in, state_out = model.state(), model.state()
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                getattr(state_in, name).assign(getattr(state, name))
            trial = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                articulated_contact_response="propagation_colored",
                iterations=6,
                contact_warmstart=False,
                free_body_constraint_max=8192,
                articulation_constraint_max=64,
            )
            trial.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            entries = PROPAGATION_COLOR_TAIL + 2
            offsets = trial._color_world_offsets.numpy()[:entries].copy()
            order = trial._color_unit_sorted.numpy()[: offsets[-1]].copy()
            results.append((offsets, order, state_out.body_qd.numpy().copy()))
        offsets, order, body_qd = results[0]
        self.assertGreater(int(offsets[-1]), 400)
        for other_offsets, other_order, other_qd in results[1:]:
            np.testing.assert_array_equal(other_offsets, offsets)
            np.testing.assert_array_equal(other_order, order)
            np.testing.assert_array_equal(other_qd, body_qd)

    def test_prescribed_mask_marks_only_kinematic_free_bodies(self):
        """Only a kinematic free rigid body has a zero response and may be shared by a color.

        The kinematic root of a multi-body articulation still carries the tree response of
        its joint dofs (the factorization does not read the kinematic flag), so it must
        remain a coloring conflict; the builder only allows kinematic bodies at roots.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        free_kinematic = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True
        )
        builder.add_shape_box(free_kinematic, hx=0.1, hy=0.1, hz=0.1)
        free_dynamic = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(free_dynamic, hx=0.1, hy=0.1, hz=0.1)
        # add_link, not add_body: add_body wraps each body in its own free-joint articulation.
        root = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        j_root = builder.add_joint_revolute(parent=-1, child=root, axis=wp.vec3(0.0, 1.0, 0.0))
        link = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.8), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        j_link = builder.add_joint_revolute(
            parent=root,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
        )
        builder.add_articulation([j_root, j_link], label="kinematic_root_chain")
        model = builder.finalize()
        model.rigid_contact_max = 256
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            free_body_constraint_max=256,
        )
        prescribed = solver._body_prescribed.numpy()
        self.assertEqual(int(prescribed[free_kinematic]), 1)
        self.assertEqual(int(prescribed[free_dynamic]), 0)
        self.assertEqual(int(prescribed[root]), 0)
        self.assertEqual(int(prescribed[link]), 0)

    def test_prebuild_matches_the_first_fit_reference(self):
        """The device coloring reproduces sort-by-contact first-fit exactly.

        The prebuild stages units in shared memory for its serial passes, sorts packed
        keys in shared memory when a world has at most 4096 units, and keeps the
        used-color masks in shared memory when the unit bodies span at most 256
        indices. The large heap spans more bodies than that and takes the global masks.
        """
        for shape in ((6, 6, 3), (7, 7, 6)):
            model, _ = _heap(*shape)
            solver, state, _ = _settle(model, steps=4)
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())
            n, color, offsets, row_start, sorted_contact = _first_fit_reference(solver)
            self.assertGreater(n, 400)
            entries = PROPAGATION_COLOR_TAIL + 2
            np.testing.assert_array_equal(solver._color_unit_color.numpy()[:n], color, err_msg=f"{shape} colors")
            np.testing.assert_array_equal(solver._color_world_offsets.numpy()[:entries], offsets, err_msg=f"{shape}")
            np.testing.assert_array_equal(solver._color_world_row_order.numpy()[:n], row_start, err_msg=f"{shape}")
            np.testing.assert_array_equal(solver._color_unit_sorted.numpy()[:n], sorted_contact, err_msg=f"{shape}")

    def test_colored_accepts_a_large_row_budget(self):
        """The coloring keeps per-unit state in global scratch, so no staging cap applies."""
        model, _ = _heap(nx=2, ny=2, nz=1)
        model.rigid_contact_max = 65536
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            free_body_constraint_max=32768,
            articulation_constraint_max=2048,
        )
        self.assertEqual(int(solver._propagation_max_constraints), 32768 + 2048)


def _add_arm_on_heap(world, *, heap=(5, 2, 2), tight_limits=False):
    """A fixed-base six-link arm drooping onto a box heap; spheres on links 0, 3 and 5 touch the boxes."""
    world.default_shape_cfg.density = 1000.0
    world.default_shape_cfg.mu = 0.7
    no_collision = world.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0
    limits = {"limit_lower": -0.15, "limit_upper": 0.15} if tight_limits else {}
    prev, joints = -1, []
    for i in range(6):
        link = world.add_link()
        world.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if i in (0, 3, 5):
            world.add_shape_sphere(link, radius=0.08)
        parent_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.32) if prev == -1 else wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()
        )
        joints.append(
            world.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                **limits,
            )
        )
        prev = link
    world.add_articulation(joints)
    nx, ny, nz = heap
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                body = world.add_body(
                    xform=wp.transform(
                        wp.vec3(0.15 + 0.216 * ix, 0.083 * (iy - (ny - 1) / 2), 0.041 + 0.082 * iz),
                        wp.quat_rpy(0.0, 0.0, 0.05 * (ix + iy + iz)),
                    )
                )
                world.add_shape_box(body, hx=0.04, hy=0.04, hz=0.04)


def _build_arm_heap_model(device: str, worlds: int = N_WORLDS, **world_kwargs) -> newton.Model:
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
    scene.rigid_gap = 0.005
    for _ in range(worlds):
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        world.rigid_gap = 0.005
        _add_arm_on_heap(world, **world_kwargs)
        scene.add_world(world)
    scene.add_ground_plane()
    model = scene.finalize(device=device)
    model.rigid_contact_max = 4096 * worlds
    return model


_STATE_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


def _colored_deterministic_solver(model, **kwargs):
    args = {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "propagation_colored",
        "deterministic": True,
        "iterations": 16,
        "contact_regularization": 0.01,
        "free_body_constraint_max": 1024,
        "articulation_constraint_max": 128,
        "enable_joint_limits": True,
        "row_watermark": True,
    }
    args.update(kwargs)
    return SolverFeatherPGS(model, **args)


class _Stepper:
    """One solver, one deterministic pipeline and a state pair advanced in place, so steps can be captured."""

    def __init__(self, model, **solver_kwargs):
        self.model = model
        self.solver = _colored_deterministic_solver(model, **solver_kwargs)
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=model.rigid_contact_max, deterministic=True)
        self.contacts = self.pipeline.contacts()
        self.state_in, self.state_out = model.state(), model.state()
        self.control = model.control()

    def step(self):
        self.state_in.clear_forces()
        self.pipeline.collide(self.state_in, self.contacts)
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, DT)
        for name in _STATE_FIELDS:
            wp.copy(getattr(self.state_in, name), getattr(self.state_out, name))

    def snapshot(self):
        return {name: getattr(self.state_in, name).numpy().copy() for name in _STATE_FIELDS}


def _world_slices(model, worlds):
    """Per-field index slices of each world's rows, for replicated worlds."""
    counts = {"body_q": model.body_count, "body_qd": model.body_count}
    counts.update(joint_q=model.joint_coord_count, joint_qd=model.joint_dof_count)
    return {name: [slice(w * n // worlds, (w + 1) * n // worlds) for w in range(worlds)] for name, n in counts.items()}


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestFeatherPGSColoredDeterministic(unittest.TestCase):
    """``deterministic=True`` on the colored response: repeatable within the mode; bitwise agreement with immediate is not guaranteed."""

    STEPS = 80

    def _rollout(self, model, steps, capture_at=None, **solver_kwargs):
        stepper = _Stepper(model, **solver_kwargs)
        graph = None
        history = []
        for k in range(steps):
            if k == capture_at:
                with wp.ScopedCapture() as capture:
                    stepper.step()
                graph = capture.graph
            elif graph is not None:
                wp.capture_launch(graph)
            else:
                stepper.step()
            history.append(stepper.snapshot())
        return history, stepper

    def _assert_bitwise(self, model, steps=STEPS, capture_at=None, **solver_kwargs):
        first, stepper = self._rollout(model, steps, capture_at=capture_at, **solver_kwargs)
        second, _ = self._rollout(model, steps, capture_at=capture_at, **solver_kwargs)
        slices = _world_slices(model, N_WORLDS)
        for k, (a, b) in enumerate(zip(first, second, strict=True)):
            for name in _STATE_FIELDS:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"run-to-run {name} at step {k}")
                for world in range(1, N_WORLDS):
                    np.testing.assert_array_equal(
                        a[name][slices[name][world]], a[name][slices[name][0]], err_msg=f"world {world} {name} step {k}"
                    )
        return stepper

    def test_gate_admits_colored_only(self):
        """Accept deterministic colored; reject plain and fused, and colored with contact warm start."""
        model = _build_arm_heap_model("cuda:0", worlds=1)
        _colored_deterministic_solver(model)
        for response in ("propagation", "propagation_fused"):
            with self.assertRaisesRegex(ValueError, "deterministic=True requires"):
                _colored_deterministic_solver(model, articulated_contact_response=response)
        with self.assertRaisesRegex(ValueError, "with contact_warmstart requires"):
            _colored_deterministic_solver(model, contact_warmstart=True)

    def test_arm_on_heap_is_bitwise_repeatable(self):
        """Repeat an arm pressing into a heap bitwise across runs and identical worlds."""
        stepper = self._assert_bitwise(_build_arm_heap_model("cuda:0"))
        self.assertGreaterEqual(int(stepper.solver._propagation_body_count.numpy().max()), 2)

    def test_arm_on_heap_is_bitwise_repeatable_uncached(self):
        """Repeat the arm-on-heap rollout bitwise with the cached response off."""
        self._assert_bitwise(_build_arm_heap_model("cuda:0"), propagation_cached_response=False)

    def test_overflow_with_limit_rows_is_bitwise_repeatable(self):
        """Repeat bitwise while dense joint-limit, velocity-limit and propagation rows overflow their capacity."""
        model = _build_arm_heap_model("cuda:0", tight_limits=True)
        wp.synchronize()
        capture = StdOutCapture()
        capture.begin()
        try:
            stepper = self._assert_bitwise(
                model,
                enable_joint_velocity_limits=True,
                velocity_limit_activation_fraction=0.0,
                free_body_constraint_max=160,
                articulation_constraint_max=24,
            )
        finally:
            try:
                wp.synchronize()
            finally:
                printed = capture.end()
        # Each of the two rollouts warns once per overflowing row family.
        self.assertEqual(printed.count("dense constraint-row overflow"), 2, printed)
        self.assertEqual(printed.count("propagation constraint-row overflow"), 2, printed)
        marks = stepper.solver.constraint_row_watermarks()
        self.assertGreater(marks["propagation_overflow_world_steps"], 0, "scene must overflow its propagation rows")

    def test_mixed_rows_are_bitwise_repeatable(self):
        """Repeat bitwise with joint-limit and velocity-limit rows active alongside colored contact rows."""
        stepper = self._assert_bitwise(
            _build_arm_heap_model("cuda:0", tight_limits=True),
            enable_joint_velocity_limits=True,
            velocity_limit_activation_fraction=0.0,
        )
        marks = stepper.solver.constraint_row_watermarks()
        self.assertGreater(marks["articulation_high_water"], 0, "scene must keep dense limit rows")
        self.assertEqual(marks["propagation_overflow_world_steps"], 0)

    def test_captured_rollout_is_bitwise_repeatable(self):
        """Repeat bitwise when most steps replay a captured graph."""
        self._assert_bitwise(_build_arm_heap_model("cuda:0"), capture_at=2)

    def test_cold_capture_replays_like_eager_steps(self):
        """Capture the first step of a fresh solver, then match an eager twin over repeated replays."""
        model = _build_arm_heap_model("cuda:0")
        for env in ({}, {"FEATHER_PGS_COLORED_PREFETCH": "0"}, {"FEATHER_PGS_COLORED_STAGED": "1"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                graph_stepper = _Stepper(model)
                eager = _Stepper(model)
                # Capture records the first step without running it; each replay then pairs with one eager step.
                with wp.ScopedCapture() as capture:
                    graph_stepper.step()
                for k in range(60):
                    wp.capture_launch(capture.graph)
                    eager.step()
                    a, b = graph_stepper.snapshot(), eager.snapshot()
                    for name in _STATE_FIELDS:
                        np.testing.assert_array_equal(a[name], b[name], err_msg=f"{env} replay {k} {name}")

    def test_moving_hub_cold_capture_replays_like_eager_steps(self):
        """Cold-capture a moving kinematic hub with both passes folding, then match an eager twin on each route."""
        for env in ({}, {"FEATHER_PGS_COLORED_LANES": "3"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                model, tray = _heap(nx=3, ny=3, nz=1, kinematic_tray=True)
                model.rigid_contact_max = 2048
                tray_joint = int(np.flatnonzero(model.joint_child.numpy() == tray)[0])
                tray_dof = int(model.joint_qd_start.numpy()[tray_joint])
                captured, eager = (
                    _Stepper(
                        model,
                        iterations=4,
                        velocity_iterations=2,
                        free_body_constraint_max=512,
                        articulation_constraint_max=64,
                    )
                    for _ in range(2)
                )
                for stepper in (captured, eager):
                    qd = stepper.state_in.joint_qd.numpy()
                    qd[tray_dof : tray_dof + 6] = (0.2, 0.0, 0.0, 0.0, 0.0, 0.3)
                    stepper.state_in.joint_qd.assign(qd)
                    newton.eval_fk(model, stepper.state_in.joint_q, stepper.state_in.joint_qd, stepper.state_in)
                with wp.ScopedCapture() as capture:
                    captured.step()
                # An eager step after recording must fold afresh rather than reuse the unexecuted capture's fold.
                captured.step()
                eager.step()
                for k in range(12):
                    wp.capture_launch(capture.graph)
                    eager.step()
                    a, b = captured.snapshot(), eager.snapshot()
                    for name in _STATE_FIELDS:
                        np.testing.assert_array_equal(a[name], b[name], err_msg=f"{env} replay {k} {name}")
                count = int(captured.solver._propagation_constraint_count.numpy()[0])
                self.assertGreater(count, 0)
                swept = np.concatenate(
                    [captured.solver._color_body_a.numpy()[0, :count], captured.solver._color_body_b.numpy()[0, :count]]
                )
                self.assertNotIn(tray, swept)

    def test_fold_buffers_exist_only_with_prescribed_bodies(self):
        """Allocate the folded sweep rows only for a model with a kinematic free body."""
        self.assertIsNone(_Stepper(_build_arm_heap_model("cuda:0")).solver._color_body_a)
        model, _ = _heap(nx=2, ny=2, nz=1, kinematic_tray=True)
        model.rigid_contact_max = 1024
        solver = _Stepper(model, free_body_constraint_max=256, articulation_constraint_max=64).solver
        self.assertEqual(solver._color_body_a.shape, (solver._world_count, solver._propagation_max_constraints))

    def test_partial_reset_keeps_synced_and_untouched_worlds_in_lockstep(self):
        """Copy world 0 into world 1, reset only those two, and keep every field bitwise in lockstep."""
        model = _build_arm_heap_model("cuda:0")
        slices = _world_slices(model, N_WORLDS)
        synced, reference = _Stepper(model), _Stepper(model)
        for stepper in (synced, reference):
            q = stepper.state_in.joint_q.numpy()
            q[slices["joint_q"][1]] += 0.05
            stepper.state_in.joint_q.assign(q)
            newton.eval_fk(model, stepper.state_in.joint_q, stepper.state_in.joint_qd, stepper.state_in)
        for _ in range(30):
            synced.step()
            reference.step()
        for name in _STATE_FIELDS:
            for state in (synced.state_in, synced.state_out):
                values = getattr(state, name).numpy()
                values[slices[name][1]] = values[slices[name][0]]
                getattr(state, name).assign(values)
        mask = np.zeros(N_WORLDS, dtype=bool)
        mask[:2] = True
        synced.solver.reset(synced.state_in, world_mask=wp.array(mask, dtype=wp.bool, device=model.device))
        synced.pipeline.reset_contact_matching(wp.array(np.append(mask, False), dtype=wp.bool, device=model.device))
        for k in range(60):
            synced.step()
            reference.step()
            a, b = synced.snapshot(), reference.snapshot()
            for name in _STATE_FIELDS:
                np.testing.assert_array_equal(a[name][slices[name][1]], a[name][slices[name][0]], err_msg=f"{name} {k}")
                for world in (2, 3):
                    np.testing.assert_array_equal(
                        a[name][slices[name][world]], b[name][slices[name][world]], err_msg=f"world {world} {name} {k}"
                    )

    def test_unit_gather_order_does_not_change_the_rollout(self):
        """Gather each world's units serially in reverse contact order and keep the rollout bitwise."""
        model = _build_arm_heap_model("cuda:0")
        reference, _ = self._rollout(model, 40)
        module = inspect.getmodule(SolverFeatherPGS)
        launch = wp.launch

        def adverse_launch(kernel, dim, *args, **kwargs):
            if kernel is module.collect_propagation_units:
                kernel, dim = _collect_propagation_units_reversed, model.world_count
            return launch(kernel, dim, *args, **kwargs)

        with patch.object(module.wp, "launch", adverse_launch):
            adverse, _ = self._rollout(model, 40)
        for k, (a, b) in enumerate(zip(reference, adverse, strict=True)):
            for name in _STATE_FIELDS:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} step {k}")

    def test_moving_kinematic_hub_is_never_written(self):
        """Keep the sweep off a moving kinematic tray shared by many units of one color; it still drives the heap, repeatably."""
        model, tray = _heap(nx=5, ny=5, nz=2, kinematic_tray=True)
        model.rigid_contact_max = 8192
        tray_joint = int(np.flatnonzero(model.joint_child.numpy() == tray)[0])
        tray_dof = int(model.joint_qd_start.numpy()[tray_joint])
        for velocity_iterations in (0, 2):
            with self.subTest(velocity_iterations=velocity_iterations):
                self._check_kinematic_hub(model, tray, tray_dof, velocity_iterations)

    def _check_kinematic_hub(self, model, tray, tray_dof, velocity_iterations):
        def rollout(response):
            solver = SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                articulated_contact_response=response,
                deterministic=True,
                velocity_iterations=velocity_iterations,
                iterations=16,
                free_body_constraint_max=8192,
                articulation_constraint_max=64,
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192, deterministic=True)
            contacts = pipeline.contacts()
            state_in, state_out = model.state(), model.state()
            qd = state_in.joint_qd.numpy()
            qd[tray_dof : tray_dof + 6] = (0.2, 0.0, 0.0, 0.0, 0.0, 0.3)
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            control = model.control()
            for _ in range(40):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
                state_in, state_out = state_out, state_in
            return solver, contacts, state_in.body_qd.numpy().copy(), state_in.body_q.numpy().copy()

        solver, contacts, body_qd, body_q = rollout("propagation_colored")
        _, _, repeat_qd, repeat_q = rollout("propagation_colored")
        np.testing.assert_array_equal(body_qd, repeat_qd)
        np.testing.assert_array_equal(body_q, repeat_q)
        immediate_qd = rollout("immediate")[2]
        shape_body = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()
        shape1 = contacts.rigid_contact_shape1.numpy()

        def touches_tray(contact):
            return any(s >= 0 and shape_body[s] == tray for s in (shape0[contact], shape1[contact]))

        entries = PROPAGATION_COLOR_TAIL + 2
        offsets = solver._color_world_offsets.numpy()[:entries]
        sorted_contacts = solver._color_unit_sorted.numpy()
        per_color = [
            sum(touches_tray(c) for c in sorted_contacts[offsets[i] : offsets[i + 1]]) for i in range(entries - 1)
        ]
        self.assertGreaterEqual(max(per_color), 8, "the tray must be shared by many units of one color")
        m = int(solver._propagation_constraint_count.numpy()[0])
        rows = np.concatenate([solver._propagation_body_a.numpy()[0, :m], solver._propagation_body_b.numpy()[0, :m]])
        swept = np.concatenate([solver._color_body_a.numpy()[0, :m], solver._color_body_b.numpy()[0, :m]])
        self.assertIn(tray, rows)
        self.assertNotIn(tray, swept, "the colored sweep still addresses the kinematic tray")
        # The tray's prescribed motion still reaches the boxes resting on it.
        bottom = [b for b in range(model.body_count) if b != tray and body_q[b, 2] < 0.065]
        self.assertGreater(len(bottom), 10)
        colored_vx = float(np.mean(body_qd[bottom, 0]))
        immediate_vx = float(np.mean(immediate_qd[bottom, 0]))
        self.assertGreater(colored_vx, 0.05)
        # Measured gate with margin across GPU architectures, not a derived bound.
        self.assertLess(abs(colored_vx - immediate_vx), 0.03)


@wp.kernel
def _collect_propagation_units_reversed(
    contact_count: wp.array[int],
    contact_path: wp.array[int],
    contact_world: wp.array[int],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    body_prescribed: wp.array[int],
    contact_slots_needed: wp.array[int],
    propagation_max_constraints: int,
    world_unit_cursor: wp.array[int],
    unit_contact: wp.array[int],
    unit_body_a: wp.array[int],
    unit_body_b: wp.array[int],
    unit_len: wp.array[int],
):
    """Gather like ``collect_propagation_units``, one thread per world walking contacts backwards."""
    world = wp.tid()
    for r in range(contact_count[0]):
        c = contact_count[0] - 1 - r
        if contact_path[c] != 2 or contact_world[c] != world:
            continue
        idx = world_unit_cursor[world]
        world_unit_cursor[world] = idx + 1
        if idx >= propagation_max_constraints:
            continue
        base = world * propagation_max_constraints
        body_a = int(-1)
        body_b = int(-1)
        if contact_shape0[c] >= 0:
            body_a = shape_body[contact_shape0[c]]
            if body_a >= 0 and body_prescribed[body_a] != 0:
                body_a = -1
        if contact_shape1[c] >= 0:
            body_b = shape_body[contact_shape1[c]]
            if body_b >= 0 and body_prescribed[body_b] != 0:
                body_b = -1
        unit_contact[base + idx] = c
        unit_body_a[base + idx] = body_a
        unit_body_b[base + idx] = body_b
        unit_len[base + idx] = contact_slots_needed[c]


RADIUS = 0.05


ROW_CAPACITY = 6


CONTACT_COUNT = 4


FRICTION_GAP_THRESHOLD = 0.0


def _build_mixed_gap_model(device: str) -> newton.Model:
    """Build four disjoint spheres whose ground contacts use one or three rows."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.6,
        margin=0.0,
        gap=0.015,
    )
    for index, gap in enumerate((-0.001, 0.004, 0.007, 0.010)):
        body = world.add_link(
            xform=wp.transform(wp.vec3(0.25 * index, 0.0, RADIUS + gap), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001),
            lock_inertia=True,
        )
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene.add_world(world)
    scene.add_ground_plane(cfg=cfg)
    return scene.finalize(device=device)


def _make_solver_colored_unit_capacity(model: newton.Model, response: str) -> SolverFeatherPGS:
    """Build a serial or colored propagation solver with six contact rows."""
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        iterations=8,
        contact_warmstart=False,
        propagation_cached_response=False,
        articulation_constraint_max=3,
        free_body_constraint_max=3,
        contact_friction_gap_max=FRICTION_GAP_THRESHOLD,
        contact_speculative_scale=0.0,
        use_parallel_streams=False,
    )
    solver._rigid_body_angular_damping.zero_()
    return solver


def _seed_state(model: newton.Model, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
    """Give every sphere the same downward speed and a distinct tangent speed."""
    joint_q = state.joint_q.numpy().copy()
    joint_qd = np.zeros(state.joint_qd.shape, dtype=np.float32)
    dof_starts = model.joint_qd_start.numpy()
    for index in range(CONTACT_COUNT):
        dof = int(dof_starts[index])
        joint_qd[dof] = 0.15 + 0.03 * index
        joint_qd[dof + 2] = -0.1
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return joint_q, joint_qd


def _unit_impulses(model: newton.Model, solver: SolverFeatherPGS, contacts: newton.Contacts):
    """Return propagation impulses keyed by the dynamic body of each contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    slots = solver._contact_slot.numpy()[:count]
    lengths = solver._contact_slots_needed.numpy()[:count]
    paths = solver._contact_path.numpy()[:count]
    rows = solver._propagation_impulses.numpy()[0]
    result = {}
    for contact in range(count):
        if int(paths[contact]) != 2:
            continue
        body_a = int(shape_body[int(shape0[contact])]) if shape0[contact] >= 0 else -1
        body_b = int(shape_body[int(shape1[contact])]) if shape1[contact] >= 0 else -1
        body = max(body_a, body_b)
        slot = int(slots[contact])
        length = int(lengths[contact])
        result[body] = rows[slot : slot + length].copy()
    return result


def _assert_mixed_topology(test: unittest.TestCase, solver: SolverFeatherPGS, contacts: newton.Contacts) -> None:
    """Keep all four mixed-length units in the first color and preserve their rows."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    paths = solver._contact_path.numpy()[:count]
    np.testing.assert_array_equal(paths, np.full(CONTACT_COUNT, 2, dtype=np.int32))
    lengths = solver._contact_slots_needed.numpy()[:count]
    test.assertEqual(sorted(lengths.tolist()), [1, 1, 1, 3])
    test.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), ROW_CAPACITY)
    test.assertEqual(int(solver._color_world_unit_cursor.numpy()[0]), CONTACT_COUNT)

    offsets = solver._color_world_offsets.numpy().reshape(solver._world_count, -1)[0]
    bucket_counts = np.diff(offsets)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)
    # Coloring now stages up to the row capacity, rather than row_capacity // 3.
    # All four disjoint units fit even though three only have a normal row.
    test.assertEqual(int(offsets[-2]), CONTACT_COUNT)
    test.assertEqual(int(bucket_counts[0]), CONTACT_COUNT)
    np.testing.assert_array_equal(bucket_counts[1:-1], np.zeros_like(bucket_counts[1:-1]))
    test.assertEqual(int(bucket_counts[-1]), 0)

    ordered = solver._color_unit_sorted.numpy()[:CONTACT_COUNT]
    test.assertEqual(sorted(ordered.tolist()), list(range(CONTACT_COUNT)))


@unittest.skipUnless(wp.is_cuda_available(), "propagation_colored requires CUDA")
class TestFeatherPGSColoredUnitCapacity(unittest.TestCase):
    def test_mixed_gap_matches_serial_propagation(self):
        """Match impulses, body velocity, and a short trajectory with mixed-length units."""
        device = "cuda:0"
        model = _build_mixed_gap_model(device)
        colored = _make_solver_colored_unit_capacity(model, "propagation_colored")
        serial = _make_solver_colored_unit_capacity(model, "propagation")
        colored_in, colored_out = model.state(), model.state()
        serial_in, serial_out = model.state(), model.state()
        _seed_state(model, colored_in)
        _seed_state(model, serial_in)
        colored_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        serial_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        colored_contacts = colored_pipeline.contacts()
        serial_contacts = serial_pipeline.contacts()
        control = model.control()

        for step in range(4):
            colored_in.clear_forces()
            serial_in.clear_forces()
            colored_pipeline.collide(colored_in, colored_contacts)
            serial_pipeline.collide(serial_in, serial_contacts)
            colored.step(colored_in, colored_out, control, colored_contacts, DT)
            serial.step(serial_in, serial_out, control, serial_contacts, DT)

            if step == 0:
                _assert_mixed_topology(self, colored, colored_contacts)
            colored_impulses = _unit_impulses(model, colored, colored_contacts)
            serial_impulses = _unit_impulses(model, serial, serial_contacts)
            self.assertEqual(sorted(colored_impulses), sorted(serial_impulses))
            for body, expected in serial_impulses.items():
                np.testing.assert_allclose(colored_impulses[body], expected, rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(
                colored._propagation_body_qd.numpy(),
                serial._propagation_body_qd.numpy(),
                rtol=2.0e-5,
                atol=2.0e-6,
            )
            np.testing.assert_allclose(
                colored_out.joint_qd.numpy(), serial_out.joint_qd.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            np.testing.assert_allclose(
                colored_out.joint_q.numpy(), serial_out.joint_q.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            colored_in, colored_out = colored_out, colored_in
            serial_in, serial_out = serial_out, serial_in

    def test_mixed_gap_cuda_graph_matches_eager(self):
        """Replay a captured mixed-unit colored step with the same result as eager execution."""
        device = wp.get_device("cuda:0")
        model = _build_mixed_gap_model(str(device))
        graph_solver = _make_solver_colored_unit_capacity(model, "propagation_colored")
        eager_solver = _make_solver_colored_unit_capacity(model, "propagation_colored")
        graph_state, graph_out = model.state(), model.state()
        eager_state, eager_out = model.state(), model.state()
        initial_q, initial_qd = _seed_state(model, graph_state)
        _seed_state(model, eager_state)
        graph_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        eager_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        graph_contacts = graph_pipeline.contacts()
        eager_contacts = eager_pipeline.contacts()
        graph_control = model.control()
        eager_control = model.control()

        def one_step(solver, pipeline, contacts, state_in, state_out, control):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)
        for state in (graph_state, eager_state):
            state.joint_q.assign(initial_q)
            state.joint_qd.assign(initial_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        with wp.ScopedCapture(device) as capture:
            one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        wp.capture_launch(capture.graph)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)

        _assert_mixed_topology(self, graph_solver, graph_contacts)
        np.testing.assert_allclose(graph_state.joint_q.numpy(), eager_state.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(graph_state.joint_qd.numpy(), eager_state.joint_qd.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            graph_solver._propagation_body_qd.numpy(),
            eager_solver._propagation_body_qd.numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )


def _build_model_leaf_publication(device="cpu", *, locked_d6=False, chain=False, leaves=7):
    """Use real joints, nontrivial anchors/COMs, and unrelated moving bodies."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))

    def body(index):
        return builder.add_link(
            mass=0.5 + 0.1 * index,
            com=wp.vec3(0.07, -0.03, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )

    base = body(0)
    root_kwargs = {
        "parent": -1,
        "child": base,
        "parent_xform": wp.transform(wp.vec3(2.0, -1.0, 0.6), wp.quat_rpy(0.3, -0.2, 0.4)),
        "child_xform": wp.transform(wp.vec3(0.2, -0.1, 0.05), wp.quat_rpy(-0.1, 0.2, 0.0)),
    }
    root = builder.add_joint_d6(**root_kwargs) if locked_d6 else builder.add_joint_fixed(**root_kwargs)
    joints = [root]
    leaf_bodies = []
    for index in range(leaves):
        child = body(index + 1)
        leaf_bodies.append(child)
        joints.append(
            builder.add_joint_prismatic(
                parent=(base if index == 0 else leaf_bodies[index - 1]) if chain else base,
                child=child,
                axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9),
                parent_xform=wp.transform(wp.vec3(0.1 * index, -0.05, 0.13), wp.quat_rpy(0.2, -0.1 * index, 0.3)),
                child_xform=wp.transform(wp.vec3(-0.09, 0.04, 0.08), wp.quat_rpy(0.4, 0.2, -0.1)),
                target_ke=3.0,
                target_kd=0.2,
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
    builder.add_articulation(joints)
    arm = body(leaves + 1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(-0.8, 0.1, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    free = body(leaves + 2)
    builder.add_articulation([builder.add_joint_free(free)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    q = model.joint_q.numpy()
    q[: leaves + 1] = np.linspace(-0.31, 0.27, leaves + 1)
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    # The specialized producer must not assume a unit axis.
    axes = model.joint_axis.numpy()
    axes[0] *= 1.4
    model.joint_axis.assign(axes)
    return model, joints, leaf_bodies


def _build_branched_model(device="cpu", *, floating_root=False):
    """Interleave terminal joints and internal parents with nonzero parent motion."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    bodies = [
        builder.add_link(
            mass=0.7 + 0.1 * index,
            com=wp.vec3(0.03, -0.04, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )
        for index in range(7)
    ]
    root_args = {"parent": -1, "child": bodies[0]}
    root = builder.add_joint_free(**root_args) if floating_root else builder.add_joint_revolute(**root_args)
    joints = [root]
    # J1 is terminal despite preceding the internal J2. A prefix cannot express
    # the retained traversal [J0, J2]; later leaves use both moving parents.
    for index, kind in enumerate(("revolute", "revolute", "prismatic", "ball", "d6", "fixed"), start=1):
        args = {
            "parent": bodies[2] if index in (4, 5) else bodies[0],
            "child": bodies[index],
            "parent_xform": wp.transform(wp.vec3(0.1 * index, -0.07, 0.13), wp.quat_rpy(0.2, -0.1, 0.3)),
            "child_xform": wp.transform(wp.vec3(-0.03, 0.04, 0.02), wp.quat_rpy(-0.1, 0.2, 0.1)),
        }
        if kind in ("revolute", "prismatic"):
            args["axis"] = wp.vec3(0.2, 0.5, 0.9)
        if kind == "d6":
            args["linear_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
            args["angular_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y)]
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    for joint in (1, 2, 3, 5):
        q[starts[joint] : starts[joint + 1]] = 0.07 * joint
    if floating_root:
        q[:7] = (0.4, -0.2, 0.6, *wp.quat_rpy(0.3, -0.2, 0.1))
    else:
        q[0] = 0.31
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.9, 0.8, model.joint_dof_count, dtype=np.float32))
    return model, joints, bodies


def _build_contact_model(device):
    """Load three terminal joints against the ground under a moving root."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=root,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.185), wp.quat_identity()),
        )
    ]
    for index, kind in enumerate(("revolute", "prismatic", "ball")):
        child = builder.add_link(mass=0.5, inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.03))
        args = {
            "parent": root,
            "child": child,
            "parent_xform": wp.transform(wp.vec3(0.5 * (index - 1), 0.3, 0.0), wp.quat_identity()),
        }
        if kind != "ball":
            args["axis"] = newton.Axis.Y if kind == "revolute" else newton.Axis.Z
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
        builder.add_shape_sphere(child, radius=0.2)
    builder.add_articulation(joints)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.linspace(-0.1, 0.15, model.joint_dof_count, dtype=np.float32))
    return model


def _solver_leaf_publication(model, *, enabled, mode="split", velocity_limits=False):
    with kernel_overrides({"pgs_kernel": "loop"}):
        solver = SolverFeatherPGS(
            model,
            solve_mode=mode,
            iterations=8,
            update_mass_matrix_interval=2,
            use_parallel_streams=False,
            enable_joint_velocity_limits=velocity_limits,
            parallel_tree=enabled,
        )
    if enabled and model.device.is_cuda:
        assert solver._tree_plan is not None
    if not enabled or model.device.is_cpu:
        assert solver._tree_plan is None
        assert solver._tree_net_wrenches == ()
    # Exercise the production cached Stage 7 on CPU, where it is normally off.
    if model.device.is_cpu:
        solver._fk_id_cache_enabled = True
    # CPU does not normally allocate the optional parallel-refresh terms.
    if model.device.is_cpu:
        solver._body_inertia_terms = wp.empty((model.body_count, 12), dtype=float, device=model.device)
    solver._prepare_augmented_state(model.state(), model.state(), model.control())
    return solver


def _fields(solver, state):
    cache = solver._fk_id_cache

    def dynamics(name):
        return getattr(solver, f"_{name}") if cache is None else getattr(cache, name)

    return {
        "body_q": state.body_q,
        "body_qd": state.body_qd,
        "body_q_com": dynamics("body_q_com"),
        "origin": dynamics("articulation_origin"),
        "S": dynamics("joint_S_s"),
        "v": dynamics("body_v_s"),
        "a": dynamics("body_a_s"),
        "I": dynamics("body_I_s"),
        "terms": dynamics("body_inertia_terms"),
        "f": dynamics("body_f_s"),
        "valid": solver._fk_id_cache_valid,
    }


class TestLeafPublication(unittest.TestCase):
    def test_cuda_actual_graph_publication(self):
        """Match poisoned-buffer graph replays for prismatic and mixed moving trees."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        for device in devices:
            self._check_graph_publication(_build_model_leaf_publication(device, locked_d6=True, leaves=108)[0])
            self._check_graph_publication(_build_branched_model(device, floating_root=True)[0])

    def _check_graph_publication(self, model):
        device = model.device
        solvers = [_solver_leaf_publication(model, enabled=value, mode="matrix_free") for value in (False, True)]
        inputs = {
            name: getattr(model, name)
            for name in (
                "joint_X_p",
                "joint_X_c",
                "joint_axis",
                "body_com",
                "body_mass",
                "body_inertia",
                "gravity",
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_q_start",
                "joint_qd_start",
            )
        }
        before = {name: array.numpy().copy() for name, array in inputs.items()}
        for step in (0, 1):
            eager = []
            for solver in solvers:
                state = model.state()
                solver._step = step
                fields = _fields(solver, state)
                q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()

                def poison(outputs=fields):
                    for name, array in outputs.items():
                        array.fill_(-123 if name == "valid" else -123.25)

                poison()
                solver._stage7_update_kinematics(state, solver)
                expected = {name: array.numpy().copy() for name, array in fields.items()}
                self.assertTrue(all(np.isfinite(value).all() for value in expected.values()))
                with wp.ScopedCapture(device=device) as capture:
                    solver._stage7_update_kinematics(state, solver)
                for _ in range(2):
                    poison()
                    wp.capture_launch(capture.graph)
                    for name, array in fields.items():
                        np.testing.assert_array_equal(array.numpy(), expected[name], err_msg=name)
                np.testing.assert_array_equal(state.joint_q.numpy(), q)
                np.testing.assert_array_equal(state.joint_qd.numpy(), qd)
                eager.append(expected)
            for name in eager[0]:
                np.testing.assert_allclose(eager[1][name], eager[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
        for name, array in inputs.items():
            np.testing.assert_array_equal(array.numpy(), before[name], err_msg=name)

    def test_interleaved_mixed_leaves_and_moving_parents(self):
        """Keep internal joints in order and preserve general moving-parent kinematics."""
        for floating in (False, True):
            model, _, bodies = _build_branched_model(floating_root=floating)
            solvers = [_solver_leaf_publication(model, enabled=value) for value in (False, True)]
            leaves = [bodies[index] for index in (1, 3, 4, 5, 6)]
            snapshots = []
            for solver in solvers:
                state = model.state()
                solver._step = 1
                for name, array in _fields(solver, state).items():
                    array.fill_(-123 if name == "valid" else -123.25)
                solver._stage7_update_kinematics(state, solver)
                snapshots.append({name: array.numpy().copy() for name, array in _fields(solver, state).items()})
            for name in snapshots[0]:
                np.testing.assert_allclose(snapshots[1][name], snapshots[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
            self.assertGreater(float(np.max(np.abs(snapshots[1]["a"][leaves]))), 1e-3)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshots[1]["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
            # Ordinary FPGS already differs from eval_fk for mixed D6
            # translation/angular lever arms; its complete fields match above.
            ordinary_velocity = model.joint_type.numpy() != int(newton.JointType.D6)
            checked_bodies = model.joint_child.numpy()[ordinary_velocity]
            np.testing.assert_allclose(
                snapshots[1]["body_qd"][checked_bodies],
                reference.body_qd.numpy()[checked_bodies],
                rtol=3e-6,
                atol=3e-6,
            )

    def test_complete_canonical_publication_and_held_inertia(self):
        """Match all canonical fields, preserving held I/terms on each cadence arm."""
        model, _, leaves = _build_model_leaf_publication(locked_d6=True, leaves=108)
        baseline = _solver_leaf_publication(model, enabled=False)
        candidate = _solver_leaf_publication(model, enabled=True)
        for step, compact in ((0, False), (1, False), (1, True)):
            snapshots = []
            for solver in (baseline, candidate):
                state = model.state()
                solver._step = step
                # Only the identity check is consumed by Stage 7, no stream is used.
                solver._global_inertia_stream = object() if compact else None
                for name, array in _fields(solver, state).items():
                    array.fill_(-123 if name == "valid" else -123.25)
                solver._stage7_update_kinematics(state, solver)
                snapshots.append({name: array.numpy().copy() for name, array in _fields(solver, state).items()})
                solver._global_inertia_stream = None
            for name in snapshots[0]:
                np.testing.assert_allclose(snapshots[1][name], snapshots[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
            np.testing.assert_array_equal(snapshots[1]["valid"], np.ones(model.articulation_count))
            np.testing.assert_array_equal(snapshots[1]["a"][leaves], 0.0)
            if step == 0 or compact:
                np.testing.assert_array_equal(snapshots[1]["I"][leaves], -123.25)
            if not compact:
                np.testing.assert_array_equal(snapshots[1]["terms"][leaves], -123.25)
            # Independent public FK computes COM velocities from q/qd, not our cache.
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshots[1]["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(snapshots[1]["body_qd"], reference.body_qd.numpy(), rtol=3e-6, atol=3e-6)

    def test_current_root_frames_and_inertial_notifications(self):
        """Use notified current root/leaf frames, axes, COMs, mass, inertia, and gravity."""
        self._check_current_frames_and_notifications("cpu")

    def _check_current_frames_and_notifications(self, device):
        model, _, _ = _build_model_leaf_publication(device)
        solvers = [_solver_leaf_publication(model, enabled=value) for value in (False, True)]
        before = []
        for solver in solvers:
            state = model.state()
            solver._step = 1
            solver._stage7_update_kinematics(state, solver)
            before.append(state.body_q.numpy().copy())
        frames = model.joint_X_p.numpy()
        frames[0, :3] += (0.3, -0.7, 0.5)
        frames[1, :3] += (0.2, 0.4, -0.1)
        model.joint_X_p.assign(frames)
        child_frames = model.joint_X_c.numpy()
        child_frames[1, :3] += (-0.1, 0.07, 0.09)
        model.joint_X_c.assign(child_frames)
        axes = model.joint_axis.numpy()
        axes[0] = (0.4, -0.7, 1.2)
        model.joint_axis.assign(axes)
        com = model.body_com.numpy()
        com += (0.01, 0.02, -0.03)
        model.body_com.assign(com)
        model.body_mass.assign(model.body_mass.numpy() * 1.3)
        model.body_inertia.assign(model.body_inertia.numpy() * 1.2)
        model.gravity.assign(np.array([[1.0, -2.0, -8.0]], dtype=np.float32))
        after = []
        for solver in solvers:
            solver.notify_model_changed(newton.ModelFlags.ALL)
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
            state = model.state()
            solver._stage7_update_kinematics(state, solver)
            after.append(
                {name: array.numpy().copy() for name, array in _fields(solver, state).items() if name != "terms"}
            )
            self.assertIs(solver._fk_id_cache_source_state, state)
            solver.reset(state)
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        self.assertFalse(np.array_equal(before[1], after[1]["body_q"]))
        for name in after[0]:
            np.testing.assert_allclose(after[1][name], after[0][name], rtol=3e-6, atol=3e-6, err_msg=name)

    def test_cached_next_step_dynamics_and_reset(self):
        """Feed publication back into original dynamics across refresh/reuse and reset."""
        self._check_cached_next_step_dynamics_and_reset("cpu", "split")
        self._check_cached_next_step_dynamics_and_reset("cpu", "split", loaded=True)

    def test_cuda_cached_next_step_dynamics_and_reset(self):
        """Preserve complete CUDA steps with matrix-free and snapshot-backed publication."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA full-step coverage requires CUDA")
        for device in devices:
            self._check_current_frames_and_notifications(device)
            for mode in ("matrix_free", "split"):
                with self.subTest(device=str(device), mode=mode):
                    self._check_cached_next_step_dynamics_and_reset(device, mode)
                    self._check_cached_next_step_dynamics_and_reset(device, mode, loaded=True)
            self._check_cached_next_step_dynamics_and_reset(device, "matrix_free", loaded=True, velocity_limits=True)

    def _check_cached_next_step_dynamics_and_reset(self, device, mode, *, loaded=False, velocity_limits=False):
        model = _build_contact_model(device) if loaded else _build_model_leaf_publication(device, leaves=4)[0]
        contact_capacity = 32 if loaded else 1
        model.rigid_contact_max = contact_capacity
        if velocity_limits:
            model.joint_velocity_limit.fill_(0.01)
        solvers = [
            _solver_leaf_publication(model, enabled=value, mode=mode, velocity_limits=velocity_limits)
            for value in (False, True)
        ]
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=contact_capacity)
        contacts = [pipeline.contacts() for _ in solvers]
        saw_contacts = False
        for pair in states:
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        for step in range(6):
            for index, solver in enumerate(solvers):
                current, following = states[index]
                if step == 3:
                    solver.reset(current)
                if loaded:
                    pipeline.collide(current, contacts[index])
                    saw_contacts |= int(contacts[index].rigid_contact_count.numpy()[0]) > 0
                forces = np.zeros((model.body_count, 6), dtype=np.float32)
                forces[1, :3] = (0.2 * step, -0.1, 0.3)
                current.body_f.assign(forces)
                controls[index].joint_f.assign(np.linspace(-0.1, 0.2, model.joint_dof_count, dtype=np.float32))
                solver.step(current, following, controls[index], contacts[index], 1.0 / 240.0)
                if velocity_limits and step == 0:
                    self.assertGreater(float(np.max(np.abs(solver._qd_work.numpy() - current.joint_qd.numpy()))), 0.01)
                states[index] = [following, current]
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(states[1][0], name).numpy(),
                    getattr(states[0][0], name).numpy(),
                    rtol=1e-5,
                    atol=2e-6,
                    err_msg=f"step {step} {name}",
                )
            np.testing.assert_allclose(solvers[1]._v_hat.numpy(), solvers[0]._v_hat.numpy(), rtol=1e-5, atol=2e-6)
            for size in solvers[0]._L_by_size:
                np.testing.assert_allclose(
                    solvers[1]._L_by_size[size].numpy(),
                    solvers[0]._L_by_size[size].numpy(),
                    rtol=1e-5,
                    atol=2e-6,
                )
        if loaded:
            self.assertTrue(saw_contacts, "The full-step fixture must exercise loaded contact rows")
        if loaded and model.device.is_cuda:
            graphs = []
            for index, solver in enumerate(solvers):
                pair = states[index]
                with wp.ScopedCapture(device=model.device) as capture:
                    for phase in (0, 1):
                        pipeline.collide(pair[phase], contacts[index])
                        solver.step(pair[phase], pair[1 - phase], controls[index], contacts[index], 1.0 / 240.0)
                graphs.append(capture.graph)
            for replay in range(2):
                if replay == 1:
                    model.body_mass.assign(model.body_mass.numpy() * 1.03)
                    model.body_inertia.assign(model.body_inertia.numpy() * 1.03)
                    for index, solver in enumerate(solvers):
                        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                        solver.reset(states[index][0])
                for graph in graphs:
                    wp.capture_launch(graph)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    np.testing.assert_allclose(
                        getattr(states[1][0], name).numpy(),
                        getattr(states[0][0], name).numpy(),
                        rtol=1e-5,
                        atol=2e-6,
                        err_msg=f"captured replay {replay}: {name}",
                    )


def _build_fingers(device="cpu", *, articulations=3, include_chain=False, ragged=False):
    """Build four four-joint fingers per moving palm, plus an optional serial tree."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    inertia = wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5)
    for articulation in range(articulations + int(include_chain)):
        root = builder.add_link(mass=1.0, inertia=inertia, com=wp.vec3(0.03, -0.02, 0.01))
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=root,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(2.0 * articulation, -0.3, 0.8), wp.quat_rpy(0.2, -0.1, 0.3)),
            )
        ]
        fingers = 1 if articulation == articulations else 4
        for finger in range(fingers):
            parent = root
            length = finger + 1 if ragged and articulation == 1 else 4
            for level in range(length):
                child = builder.add_link(mass=0.5, inertia=inertia, com=wp.vec3(0.02, 0.01, -0.03))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y if level % 2 else newton.Axis.X,
                        parent_xform=wp.transform(wp.vec3(0.05, 0.04 * finger, 0.08), wp.quat_rpy(0.1, -0.2, 0.05)),
                        child_xform=wp.transform(wp.vec3(-0.01, 0.02, 0.03), wp.quat_identity()),
                    )
                )
                parent = child
            if ragged and articulation == 1 and finger == 1:
                for branch in range(2):
                    child = builder.add_link(mass=0.6, inertia=inertia)
                    joints.append(
                        builder.add_joint_revolute(
                            parent=parent,
                            child=child,
                            axis=newton.Axis.X,
                            parent_xform=wp.transform(wp.vec3(0.1, 0.1 * branch, 0.2), wp.quat_identity()),
                        )
                    )
        if ragged and articulation == 2:
            parent = -1
            for _level in range(3):
                child = builder.add_link(mass=0.8, inertia=inertia, com=wp.vec3(0.02, -0.03, 0.04))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y,
                        parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_identity()),
                    )
                )
                parent = child
        builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.3, 0.4, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


class TestFeatherPGSTreePlan(unittest.TestCase):
    def test_parallel_tree_is_keyword_only_and_defaults_to_serial(self):
        """Require an explicit construction-only opt-in without shifting positional arguments."""
        parameters = inspect.signature(SolverFeatherPGS).parameters
        self.assertIn("parallel_tree", parameters)
        self.assertIs(parameters["parallel_tree"].default, False)
        self.assertEqual(parameters["parallel_tree"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_serial_selection_skips_tree_setup(self):
        """Keep default and explicit serial execution free of tree plans and scratch."""
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            model = _build_fingers(device, articulations=1)
            states = []
            for options in ({}, {"parallel_tree": False}):
                with self.subTest(device=str(device), options=options):
                    with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                        solver = SolverFeatherPGS(model, use_parallel_streams=False, **options)
                    build.assert_not_called()
                    self.assertIsNone(solver._tree_plan)
                    self.assertEqual(solver._tree_net_wrenches, ())
                    state = model.state()
                    solver._prepare_augmented_state(state, model.state(), model.control())
                    solver._stage7_update_kinematics(state, solver)
                    states.append(state)
            for name in ("body_q", "body_qd"):
                np.testing.assert_array_equal(getattr(states[0], name).numpy(), getattr(states[1], name).numpy())

    def test_explicit_parallel_preserves_eligibility_fallback(self):
        """Admit CUDA branches while retaining CPU and serial-tree fallback."""
        for device in (wp.get_device("cpu"), *wp.get_cuda_devices()):
            with self.subTest(device=str(device), topology="branched"):
                model = _build_fingers(device, articulations=1)
                with patch.object(_FeatherPGSTreePlan, "build", wraps=_FeatherPGSTreePlan.build) as build:
                    solver = SolverFeatherPGS(model, parallel_tree=True, use_parallel_streams=False)
                if device.is_cuda:
                    build.assert_called_once()
                    self.assertIsNotNone(solver._tree_plan)
                    self.assertTrue(any(group.lanes > 1 for group in solver._tree_plan.groups))
                    self.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
                else:
                    build.assert_not_called()
                    self.assertIsNone(solver._tree_plan)
                    self.assertEqual(solver._tree_net_wrenches, ())
            with self.subTest(device=str(device), topology="chain"):
                model = _build_model_leaf_publication(device, chain=True)[0]
                solver = SolverFeatherPGS(model, parallel_tree=True, use_parallel_streams=False)
                self.assertIsNone(solver._tree_plan)
                self.assertEqual(solver._tree_net_wrenches, ())

    def test_complete_finger_levels_and_serial_group(self):
        """Schedule every finger segment and retain serial trees as one-lane groups."""
        model = _build_fingers(include_chain=True)
        solver = SolverFeatherPGS(model, solve_mode="split", use_parallel_streams=False)
        plan = _FeatherPGSTreePlan.build(model, solver._articulation_joint_end)
        self.assertIsNotNone(plan)
        self.assertTrue(any(group.lanes > 1 for group in plan.groups))
        self.assertTrue(any(group.lanes == 1 for group in plan.groups))
        seen = []
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        starts = model.articulation_start.numpy()
        for group in plan.groups:
            offsets = group.level_offsets.numpy()
            segment_offsets = group.segment_offsets.numpy()
            joints = group.segment_joints.numpy()
            child_offsets = group.child_offsets.numpy()
            child_segments = group.child_segments.numpy()
            for row, articulation in enumerate(group.articulations.numpy()):
                visited = set()
                for level in range(group.max_levels):
                    previous_levels = visited.copy()
                    for segment in range(offsets[row, level], offsets[row, level + 1]):
                        segment_joints = joints[segment_offsets[segment] : segment_offsets[segment + 1]]
                        first_parent = int(parents[segment_joints[0]])
                        if first_parent >= 0:
                            self.assertIn(first_parent, previous_levels)
                        for joint in segment_joints:
                            if parents[joint] >= 0:
                                self.assertIn(int(parents[joint]), visited)
                            visited.add(int(children[joint]))
                            seen.append(int(joint))
                        child_ids = child_segments[child_offsets[segment] : child_offsets[segment + 1]]
                        child_starts = [int(joints[segment_offsets[child]]) for child in child_ids]
                        self.assertEqual(child_starts, sorted(child_starts, reverse=True))
                        for child in child_starts:
                            self.assertEqual(int(parents[child]), int(children[segment_joints[-1]]))
                if articulation < 3:
                    self.assertEqual([int(offsets[row, level + 1] - offsets[row, level]) for level in range(2)], [1, 4])
                    first, last = offsets[row, 0], offsets[row, 2]
                    np.testing.assert_array_equal(np.diff(segment_offsets[first : last + 1]), [1, 4, 4, 4, 4])
                    self.assertEqual(len(visited), int(starts[articulation + 1] - starts[articulation]))
        self.assertEqual(sorted(seen), list(range(model.joint_count)))

    def test_comb_retains_joint_levels_when_chain_compression_adds_rounds(self):
        """Keep overlapping comb teeth parallel when compressed chains increase span."""
        builder = newton.ModelBuilder()
        joints = []
        parent = -1
        for _ in range(4):
            spine = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
            joints.append(builder.add_joint_revolute(parent=parent, child=spine, axis=newton.Axis.Z))
            tooth_parent = spine
            for _ in range(4):
                tooth = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                joints.append(builder.add_joint_revolute(parent=tooth_parent, child=tooth, axis=newton.Axis.Y))
                tooth_parent = tooth
            parent = spine
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")
        ends = wp.array(model.articulation_start.numpy()[1:], dtype=wp.int32, device=model.device)

        # Compression delays the next fork until the whole preceding tooth ends.
        compressed_lengths = [[1], [4, 1], [4, 1], [4, 5]]
        joint_widths = [1, 2, 3, 4, 4, 3, 2, 1]
        compressed_rounds = sum(max(level) for level in compressed_lengths)
        joint_rounds = sum((width + 3) // 4 for width in joint_widths)
        self.assertEqual((compressed_rounds, joint_rounds), (14, 8))

        plan = _FeatherPGSTreePlan.build(model, ends)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.groups), 1)
        group = plan.groups[0]
        self.assertEqual(group.lanes, 4)
        self.assertEqual(group.max_levels, len(joint_widths))
        np.testing.assert_array_equal(np.diff(group.level_offsets.numpy()[0]), joint_widths)
        np.testing.assert_array_equal(
            np.diff(group.segment_offsets.numpy()), np.ones(model.joint_count, dtype=np.int32)
        )
        self.assertEqual(sorted(group.segment_joints.numpy().tolist()), list(range(model.joint_count)))

    def test_serial_and_unsafe_topologies_fall_back(self):
        """Retain wholly serial, aliased, cross-articulation and loop topologies."""
        model, _, _ = _build_model_leaf_publication(chain=True)
        solver = SolverFeatherPGS(model, solve_mode="split", use_parallel_streams=False)
        self.assertIsNone(_FeatherPGSTreePlan.build(model, solver._articulation_joint_end))
        for unsafe in ("alias", "foreign_parent", "loop"):
            with self.subTest(unsafe=unsafe):
                model = _build_fingers(articulations=2)
                solver = SolverFeatherPGS(model, solve_mode="split", use_parallel_streams=False)
                ends = solver._articulation_joint_end.numpy().copy()
                if unsafe == "alias":
                    children = model.joint_child.numpy().copy()
                    children[2] = children[1]
                    model.joint_child.assign(children)
                elif unsafe == "foreign_parent":
                    parents = model.joint_parent.numpy().copy()
                    parents[1] = model.joint_child.numpy()[model.articulation_start.numpy()[1]]
                    model.joint_parent.assign(parents)
                else:
                    ends[0] -= 1
                plan = _FeatherPGSTreePlan.build(model, wp.array(ends, dtype=wp.int32, device=model.device))
                self.assertIsNone(plan)


class TestFeatherPGSTreeExecution(unittest.TestCase):
    def test_cuda_tree_and_device_torsion_initialize_together(self):
        """Retain both opt-in initializers and torsion capture preparation after merging."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Device torsion and parallel trees require CUDA")
        for device in devices:
            with self.subTest(device=str(device)):
                model = _build_contact_model(device)
                set_torsion_radius(model, 0.01)
                solver = SolverFeatherPGS(
                    model,
                    solve_mode="matrix_free",
                    parallel_tree=True,
                    contact_torsion_device=True,
                    friction_anchor_beta=0.0,
                    use_parallel_streams=False,
                )
                self.assertIsNotNone(solver._tree_plan)
                self.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
                self.assertIsNotNone(getattr(solver, "_device_torsion", None))
                solver.check_constraint_capacity()

    def test_cuda_backward_torques_with_live_forces_and_passive_terms(self):
        """Preserve child reductions, additive torque and original-velocity damping."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel backward tree execution requires CUDA")
        for device in devices:
            for velocity_limits in (False, True):
                with self.subTest(device=str(device), velocity_limits=velocity_limits):
                    model = _build_fingers(device, ragged=True)
                    model.joint_velocity_limit.fill_(0.05)
                    damping = np.linspace(0.1, 0.4, model.joint_dof_count, dtype=np.float32)
                    model.joint_damping.assign(damping)
                    model.joint_spring_stiffness.assign(np.linspace(0.3, 0.8, model.joint_dof_count, dtype=np.float32))
                    model.joint_spring_ref.fill_(0.1)
                    solvers = [
                        _solver_leaf_publication(
                            model, enabled=value, mode="matrix_free", velocity_limits=velocity_limits
                        )
                        for value in (False, True)
                    ]
                    states = [model.state() for _ in solvers]
                    following = [model.state() for _ in solvers]
                    controls = [model.control() for _ in solvers]
                    seed = np.linspace(-0.4, 0.3, model.joint_dof_count, dtype=np.float32)
                    previous_tau = None
                    for force_phase in range(2):
                        overwritten = None
                        for additive in (False, True):
                            for index, solver in enumerate(solvers):
                                state = states[index]
                                if force_phase == 0 and not additive:
                                    solver._stage1_fk_id(state, solver, following[index])
                                elif force_phase == 1 and not additive:
                                    solver._fk_id_cache_valid.fill_(1)
                                    solver._stage1_fk_id(state, solver, following[index])
                                wrench = np.arange(model.body_count * 6, dtype=np.float32).reshape(-1, 6)
                                wrench = (wrench % 11 - 5) * (0.03 + force_phase * 0.04)
                                state.body_f.assign(wrench)
                                controls[index].joint_f.assign(seed * (1.0 + force_phase))
                                solver._joint_tau.assign(seed)
                                solver._body_ft_s.fill_(-123.25)
                                for scratch in solver._tree_net_wrenches:
                                    scratch.fill_(-123.25)
                                solver._launch_rigid_tau(state, solver, controls[index], add_to_existing=additive)
                            for name in ("_body_ft_s", "_joint_tau"):
                                np.testing.assert_allclose(
                                    getattr(solvers[1], name).numpy(),
                                    getattr(solvers[0], name).numpy(),
                                    rtol=3e-6,
                                    atol=3e-6,
                                    err_msg=f"forces {force_phase}, additive {additive}: {name}",
                                )
                            tau = solvers[1]._joint_tau.numpy().copy()
                            if additive:
                                np.testing.assert_allclose(tau - overwritten, seed, rtol=3e-6, atol=3e-6)
                            else:
                                overwritten = tau
                        if previous_tau is not None:
                            self.assertGreater(float(np.max(np.abs(overwritten - previous_tau))), 0.01)
                        previous_tau = overwritten
                    model.joint_damping.zero_()
                    for index, solver in enumerate(solvers):
                        solver._refresh_passive_joint_damping()
                        solver._launch_rigid_tau(states[index], solver, controls[index])
                        np.testing.assert_allclose(
                            previous_tau - solver._joint_tau.numpy(),
                            -damping * states[index].joint_qd.numpy(),
                            rtol=3e-5,
                            atol=3e-6,
                        )

    def test_cuda_stage1_cold_mixed_cache_and_velocity_prescale(self):
        """Match every canonical dynamics field across partial warps and active prescaling."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree execution requires CUDA")
        for device in devices:
            for velocity_limits in (False, True):
                with self.subTest(device=str(device), velocity_limits=velocity_limits):
                    model = _build_fingers(device, ragged=True)
                    model.joint_velocity_limit.fill_(0.05)
                    solvers = []
                    states = []
                    outputs = []
                    for parallel in (False, True):
                        solver = SolverFeatherPGS(
                            model,
                            solve_mode="matrix_free",
                            iterations=8,
                            use_parallel_streams=False,
                            enable_joint_velocity_limits=velocity_limits,
                            parallel_tree=parallel,
                        )
                        if parallel:
                            self.assertIsNotNone(solver._tree_plan)
                        else:
                            self.assertIsNone(solver._tree_plan)
                        state, following = model.state(), model.state()
                        solver._prepare_augmented_state(state, following, model.control())
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        for name, array in _fields(solver, state).items():
                            if name not in ("body_q", "body_qd", "valid"):
                                array.fill_(-123.25)
                        solver._fk_id_cache_valid.zero_()
                        solvers.append(solver)
                        states.append(state)
                        outputs.append(following)
                    original_qd = model.joint_qd.numpy().copy()
                    for phase in ("cold", "mixed", "warm"):
                        for solver, state, following in zip(solvers, states, outputs, strict=True):
                            if phase == "mixed":
                                q = state.joint_q.numpy().copy()
                                start, end = model.articulation_start.numpy()[1:3]
                                q0, q1 = model.joint_q_start.numpy()[[start, end]]
                                q[q0:q1] += 0.1
                                state.joint_q.assign(q)
                                solver._fk_id_cache_valid.fill_(1)
                                solver.notify_state_changed(wp.array([1], dtype=wp.int32, device=device))
                                np.testing.assert_array_equal(
                                    solver._fk_id_cache_valid.numpy(),
                                    [1, 0, 1] if solver._fk_id_cache_enabled else [1, 1, 1],
                                )
                            elif phase == "warm":
                                solver._fk_id_cache_valid.fill_(1)
                            _, predictor_qd = solver._stage1_fk_id(state, solver, following)
                            np.testing.assert_array_equal(state.joint_qd.numpy(), original_qd)
                            if velocity_limits:
                                self.assertGreater(float(np.max(np.abs(predictor_qd.numpy() - original_qd))), 0.01)
                        for name, actual in _fields(solvers[1], states[1]).items():
                            np.testing.assert_allclose(
                                actual.numpy(),
                                _fields(solvers[0], states[0])[name].numpy(),
                                rtol=3e-6,
                                atol=3e-6,
                                err_msg=f"{phase}: {name}",
                            )
                        if velocity_limits:
                            np.testing.assert_array_equal(solvers[1]._qd_work.numpy(), solvers[0]._qd_work.numpy())

    def test_cuda_uncached_publication_matches_public_fk(self):
        """Publish uncached FREE and mixed D6 states using the public FK convention."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree publication requires CUDA")
        for device in devices:
            model, _, _ = _build_branched_model(device, floating_root=True)
            solver = _solver_leaf_publication(model, enabled=True, mode="matrix_free")
            solver._fk_id_cache_enabled = False
            state = model.state()
            state.body_q.fill_(-123.25)
            state.body_qd.fill_(-123.25)
            solver._stage7_update_kinematics(state, solver)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            for name in ("body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6, err_msg=name
                )


if __name__ == "__main__":
    unittest.main()
