# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS patch-wrench contact blocks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_PATCH_MOMENT,
    PGS_CONSTRAINT_TYPE_PATCH_TORSION,
    _patch_member_mask,
    gather_mf_warmstart,
)
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


@wp.kernel
def _eval_patch_member_mask(
    leader: int,
    contact_count: int,
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    mask_out: wp.array[int],
):
    mask, _phi_min = _patch_member_mask(
        leader,
        contact_count,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_thickness0,
        contact_thickness1,
        shape_body,
        body_q,
    )
    mask_out[0] = mask


def test_patch_wrench_tilted_box_falls_flat(test: unittest.TestCase, device):
    """Verify a box dropped on an edge or vertex tips over and settles flat.

    Runs in both matrix_free and split modes: the negative-mu moment clamp
    is inlined separately in the fused MF-GS kernel and the standalone MF
    kernel, so both code paths need the physics check.

    The patch-wrench moment clamp must not grant tipping capacity a
    degenerate (line/point) support does not have: the wrench center must
    be the support centroid (a body-anchored center zeroes the gravity
    torque in the constraint frame, so unbalanced boxes hang mid-tilt) and
    the clamp basis must align with the support's principal axes for
    anisotropic supports (an arbitrary basis circumscribes a diagonal edge,
    creating cross-moment capacity out of nothing).
    """
    with wp.ScopedDevice(device):
        cases = [
            (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(25.0))), "edge"),
            (
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(25.0)))
                * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.radians(45.0))),
                "diagonal edge",
            ),
            (wp.quat_from_axis_angle(wp.vec3(0.7071, 0.7071, 0.0), float(np.radians(30.0))), "vertex"),
        ]
        for q0, label in [(q, lb + f" ({mode})") for q, lb in cases for mode in ("matrix_free", "split")]:
            mode = label.rsplit("(", 1)[1].rstrip(")")
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), q0))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=256,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode=mode,
                pgs_iterations=8,
                contact_patch_wrench=True,
                mf_warmstart=True,
                # velocity iterations are matrix_free-only; split exercises the
                # standalone MF kernel's moment-clamp path instead
                pgs_velocity_iterations=2 if mode == "matrix_free" else 0,
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            for _k in range(300):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, 1.0 / 60.0)
                s0, s1 = s1, s0
            z = float(s0.body_q.numpy()[body][2])
            test.assertLess(abs(z - 0.05), 0.01, f"{label}: box stuck tilted at z={z:.3f}, expected flat at 0.05")


def test_patch_wrench_block_owner_assignment(test: unittest.TestCase, device):
    """A multi-contact planar pair forms one owner-led patch; a single-contact
    pair falls back to classic per-contact rows (owner -1, own slot)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0505), wp.quat_identity()))
        box_shape = builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        sphere = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0505), wp.quat_identity()))
        sphere_shape = builder.add_shape_sphere(sphere, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=256,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            contact_patch_wrench=True,
            mf_warmstart=True,
        )
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, 1.0 / 60.0)

        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        owner = solver.contact_block_owner.numpy()[:count]
        slot = solver.contact_slot.numpy()[:count]
        path = solver.contact_path.numpy()[:count]

        box_idx = [i for i in range(count) if box_shape in (shape0[i], shape1[i]) and path[i] == 1]
        sphere_idx = [i for i in range(count) if sphere_shape in (shape0[i], shape1[i]) and path[i] == 1]
        test.assertGreaterEqual(len(box_idx), 2, "expected a multi-contact box-ground manifold")
        test.assertEqual(len(sphere_idx), 1, "expected a single sphere-ground contact")

        leader = min(box_idx)
        for i in box_idx:
            test.assertEqual(int(owner[i]), leader, f"box contact {i} not owned by leader {leader}")
        test.assertGreaterEqual(int(slot[leader]), 0, "patch leader must own a row block")
        for i in box_idx:
            if i != leader:
                test.assertEqual(int(slot[i]), -1, f"patch follower {i} must own no rows")

        s_i = sphere_idx[0]
        test.assertEqual(int(owner[s_i]), -1, "single-contact pair must stay classic")
        test.assertGreaterEqual(int(slot[s_i]), 0, "classic contact must own its rows")


def test_patch_wrench_warmstart_gather_ownership(test: unittest.TestCase, device):
    """Warm-start block recovery and isolation contracts, on synthetic arrays:

    - a patch leader recovers its block through a matched follower even when
      the leader contact itself is unmatched;
    - a classic contact matched into a previous patch block stays cold (the
      block impulse is the whole patch's, not that contact's share);
    - a contact whose previous block had fewer rows must not seed from the
      next block's rows (previous-parent validation).
    """
    with wp.ScopedDevice(device):
        n_contacts = 6
        mf_max_c = 16
        CT = PGS_CONSTRAINT_TYPE_CONTACT
        FR = PGS_CONSTRAINT_TYPE_FRICTION

        contact_count = wp.array([n_contacts], dtype=wp.int32)
        contact_path = wp.array([1] * n_contacts, dtype=wp.int32)
        # contacts 0..3: patch (leader 0, slot 0); 4: classic slot 6; 5: classic slot 9
        contact_slot = wp.array([0, -1, -1, -1, 6, 9], dtype=wp.int32)
        contact_world = wp.zeros(n_contacts, dtype=wp.int32)
        contact_block_owner = wp.array([0, 0, 0, 0, -1, -1], dtype=wp.int32)
        # leader unmatched; follower 1 matched -> prev sorted idx 2;
        # contact 4 matched -> prev idx 5 (a prev patch member);
        # contact 5 matched -> prev idx 7 (a prev 1-row classic contact)
        match_index = wp.array([-1, 2, -1, -1, 5, 7], dtype=wp.int32)
        prev_slot_sorted_np = np.full(n_contacts + 4, -1, dtype=np.int32)
        prev_slot_sorted_np[2] = -2 - 4  # prev patch block base slot 4
        prev_slot_sorted_np[5] = -2 - 4
        prev_slot_sorted_np[7] = 10  # prev classic base slot 10
        prev_slot_sorted = wp.array(prev_slot_sorted_np, dtype=wp.int32)

        prev_type = np.zeros((1, mf_max_c), dtype=np.int32)
        prev_parent = np.full((1, mf_max_c), -1, dtype=np.int32)
        prev_imp = np.zeros((1, mf_max_c), dtype=np.float32)
        # prev patch block rows 4..9
        prev_type[0, 4] = CT
        prev_type[0, 5:10] = FR
        prev_parent[0, 5:10] = 4
        prev_imp[0, 4:10] = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        # prev classic 1-row at 10, then an unrelated 3-row block at 11..13
        prev_type[0, 10] = CT
        prev_imp[0, 10] = 7.0
        prev_type[0, 11] = CT
        prev_type[0, 12] = FR
        prev_type[0, 13] = FR
        prev_parent[0, 12:14] = 11
        prev_imp[0, 11:14] = [8.0, 9.0, 9.5]

        cur_type = np.zeros((1, mf_max_c), dtype=np.int32)
        cur_parent = np.full((1, mf_max_c), -1, dtype=np.int32)
        # current patch block rows 0..5
        cur_type[0, 0] = CT
        cur_type[0, 1:6] = FR
        cur_parent[0, 1:6] = 0
        # current classic 3-row blocks at 6..8 and 9..11
        for base in (6, 9):
            cur_type[0, base] = CT
            cur_type[0, base + 1] = FR
            cur_type[0, base + 2] = FR
            cur_parent[0, base + 1] = base
            cur_parent[0, base + 2] = base

        up = [0.0, 0.0, 1.0]
        tx = [1.0, 0.0, 0.0]
        rot_t = [0.9238795, 0.3826834, 0.0]  # t0 yawed 22.5 deg
        rot_n = [0.3826834, 0.0, 0.9238795]  # normal tilted 22.5 deg

        def run(prev_n_base, prev_t0_base):
            basis_n = np.zeros((1, mf_max_c, 3), dtype=np.float32)
            basis_t0 = np.zeros((1, mf_max_c, 3), dtype=np.float32)
            basis_n[0, 0] = up
            basis_t0[0, 0] = tx
            prev_basis_n = np.zeros((1, mf_max_c, 3), dtype=np.float32)
            prev_basis_t0 = np.zeros((1, mf_max_c, 3), dtype=np.float32)
            prev_basis_n[0, 4] = prev_n_base
            prev_basis_t0[0, 4] = prev_t0_base
            mf_impulses = wp.zeros((1, mf_max_c), dtype=wp.float32)
            wp.launch(
                gather_mf_warmstart,
                dim=n_contacts,
                inputs=[
                    contact_count,
                    contact_path,
                    contact_slot,
                    contact_world,
                    contact_block_owner,
                    match_index,
                    prev_slot_sorted,
                    wp.array(prev_imp, dtype=wp.float32),
                    wp.array(prev_type, dtype=wp.int32),
                    wp.array(prev_parent, dtype=wp.int32),
                    wp.array(prev_basis_n, dtype=wp.vec3),
                    wp.array(prev_basis_t0, dtype=wp.vec3),
                    wp.array(cur_type, dtype=wp.int32),
                    wp.array(cur_parent, dtype=wp.int32),
                    wp.array(basis_n, dtype=wp.vec3),
                    wp.array(basis_t0, dtype=wp.vec3),
                    1.0,  # decay
                    1.0,  # dt_scale
                    mf_max_c,
                ],
                outputs=[mf_impulses],
            )
            return mf_impulses.numpy()[0]

        out = run(up, tx)
        np.testing.assert_allclose(out[0:6], [10.0, 1.0, 2.0, 3.0, 4.0, 5.0], err_msg="block not recovered via follower")
        np.testing.assert_allclose(out[6:9], 0.0, err_msg="classic contact seeded from a previous patch block")
        test.assertAlmostEqual(float(out[9]), 7.0, msg="classic normal carry lost")
        np.testing.assert_allclose(out[10:12], 0.0, err_msg="seeded from the next block's rows")

        out = run(up, rot_t)
        test.assertAlmostEqual(float(out[0]), 10.0, msg="F carry lost on tangent-frame rotation")
        np.testing.assert_allclose(out[1:6], 0.0, err_msg="offset rows carried across a rotated tangent frame")

        out = run(rot_n, tx)
        np.testing.assert_allclose(out[0:6], 0.0, err_msg="block carried across a rotated normal")


def test_patch_wrench_membership_eligibility(test: unittest.TestCase, device):
    """Patch membership contract: >=2 coplanar members required, out-of-plane
    and normal-disagreeing contacts excluded, other pairs never joined."""
    with wp.ScopedDevice(device):

        def run_mask(leader, points, normals, pairs):
            n = len(points)
            # contact_normal is stored A-to-B; the kernel negates it.
            args = dict(
                contact_shape0=wp.array([p[0] for p in pairs], dtype=wp.int32),
                contact_shape1=wp.array([p[1] for p in pairs], dtype=wp.int32),
                contact_point0=wp.array([wp.vec3(*p) for p in points], dtype=wp.vec3),
                contact_point1=wp.array([wp.vec3(p[0], p[1], 0.0) for p in points], dtype=wp.vec3),
                contact_normal=wp.array([wp.vec3(*nrm) for nrm in normals], dtype=wp.vec3),
                contact_thickness0=wp.zeros(n, dtype=wp.float32),
                contact_thickness1=wp.zeros(n, dtype=wp.float32),
                shape_body=wp.array([-1, -1], dtype=wp.int32),
                body_q=wp.zeros(1, dtype=wp.transform),
            )
            mask = wp.zeros(1, dtype=wp.int32)
            wp.launch(
                _eval_patch_member_mask,
                dim=1,
                inputs=[leader, n, *args.values()],
                outputs=[mask],
            )
            return int(mask.numpy()[0])

        down = (0.0, 0.0, -1.0)  # A-to-B normal; kernel-internal normal is +z
        # 4 coplanar corners: full patch
        square = [(0.05, 0.05, 0.0), (-0.05, 0.05, 0.0), (-0.05, -0.05, 0.0), (0.05, -0.05, 0.0)]
        mask = run_mask(0, square, [down] * 4, [(0, 1)] * 4)
        test.assertEqual(mask, 0b1111, "coplanar square must form a full patch")

        # one corner lifted far out of plane: excluded, others kept
        lifted = [square[0], square[1], (-0.05, -0.05, 0.02), square[3]]
        mask = run_mask(0, lifted, [down] * 4, [(0, 1)] * 4)
        test.assertEqual(mask, 0b1011, "out-of-plane contact must fall back to classic rows")

        # disagreeing normal: excluded
        tilted_n = (0.2588, 0.0, -0.9659)  # 15 degrees off
        mask = run_mask(0, square, [down, down, tilted_n, down], [(0, 1)] * 4)
        test.assertEqual(mask, 0b1011, "normal-disagreeing contact must fall back to classic rows")

        # single contact: no patch
        mask = run_mask(0, square[:1], [down], [(0, 1)])
        test.assertEqual(mask, 0, "a single contact must not form a patch")

        # run ends at a different shape pair
        mask = run_mask(0, square, [down] * 4, [(0, 1), (0, 1), (1, 0), (1, 0)])
        test.assertEqual(mask, 0b0011, "membership must stop at the pair boundary")


def test_patch_wrench_row_layout(test: unittest.TestCase, device):
    """Typed patch rows honor the friction controls: 6 rows
    [F, f0, f1, Mx, My, Mn] with active friction, 3 rows [F, Mx, My] when
    friction is disabled — the torsion row must disappear with the pair,
    while the tipping moments (distributed normal pressure) stay."""
    CT = PGS_CONSTRAINT_TYPE_CONTACT
    FR = PGS_CONSTRAINT_TYPE_FRICTION
    PM = PGS_CONSTRAINT_TYPE_PATCH_MOMENT
    PT = PGS_CONSTRAINT_TYPE_PATCH_TORSION
    for friction_on, want in ((True, [CT, FR, FR, PM, PM, PT]), (False, [CT, PM, PM])):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
            box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0505), wp.quat_identity()))
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=64,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=8,
                contact_patch_wrench=True,
                mf_warmstart=True,
                enable_contact_friction=friction_on,
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 1.0 / 60.0)

            count = int(contacts.rigid_contact_count.numpy()[0])
            owner = solver.contact_block_owner.numpy()[:count]
            slot = solver.contact_slot.numpy()[:count]
            leaders = [i for i in range(count) if owner[i] == i]
            test.assertEqual(len(leaders), 1, "expected exactly one patch leader")
            base = int(slot[leaders[0]])
            row_type = solver.mf_row_type.numpy()[0]
            row_parent = solver.mf_row_parent.numpy()[0]
            got = [int(row_type[base + r]) for r in range(len(want))]
            test.assertEqual(got, want, f"friction_on={friction_on}: row layout mismatch")
            for r in range(1, len(want)):
                test.assertEqual(int(row_parent[base + r]), base, f"row {r} not parented to block base")
            mu_row = solver.mf_row_mu.numpy()[0]
            for r in range(len(want)):
                test.assertGreaterEqual(float(mu_row[base + r]), 0.0, "negative-mu sentinel must be gone")


def test_patch_wrench_moving_kinematic_surface(test: unittest.TestCase, device):
    """Patch rows must measure against prescribed surface motion: a box
    resting on a translating kinematic slab is carried with it, and a box
    on a yawing slab is spun with it (torque-row targets are the relative
    prescribed angular velocity, not the point-velocity Jacobian)."""
    for case in ("translate", "rotate"):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
            slab = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                is_kinematic=True,
            )
            builder.add_shape_box(slab, hx=1.0, hy=1.0, hz=0.1, cfg=cfg)
            box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2505), wp.quat_identity()))
            builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=64,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=8,
                contact_patch_wrench=True,
                mf_warmstart=True,
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
            if case == "translate":
                joint_qd[0] = 1.0  # slab vx
            else:
                joint_qd[5] = 1.0  # slab yaw rate
            s0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
            for _k in range(40):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, 1.0 / 60.0)
                s0, s1 = s1, s0
            qd = s0.body_qd.numpy()[box]
            if case == "translate":
                test.assertAlmostEqual(
                    float(qd[0]), 1.0, delta=0.05, msg="box not carried by the translating surface"
                )
            else:
                test.assertAlmostEqual(
                    float(qd[5]), 1.0, delta=0.1, msg="box not spun with the yawing surface"
                )
            z = float(s0.body_q.numpy()[box][2])
            test.assertLess(abs(z - 0.2505), 0.01, f"{case}: box lost the surface (z={z:.4f})")


class TestFeatherPGSPatchWrench(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_tilted_box_falls_flat",
    test_patch_wrench_tilted_box_falls_flat,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_block_owner_assignment",
    test_patch_wrench_block_owner_assignment,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_warmstart_gather_ownership",
    test_patch_wrench_warmstart_gather_ownership,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_membership_eligibility",
    test_patch_wrench_membership_eligibility,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_row_layout",
    test_patch_wrench_row_layout,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_moving_kinematic_surface",
    test_patch_wrench_moving_kinematic_surface,
    devices=get_selected_cuda_test_devices(),
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
