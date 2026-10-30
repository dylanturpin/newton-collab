# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block-per-world fused matrix-free GS solve: one warp per group of independent islands.

Rows of articulations that share no row form independent Gauss-Seidel chains, so each chain may sweep on
its own warp. Within a chain the rows keep their original order, so the result matches the one-warp sweep.
"""

import re

import numpy as np
import warp as wp

ISLAND_MAX_COMPONENTS = 32
ISLAND_MF_LIST_MAX = 1024


def world_dof_components(world_dof_indices: np.ndarray, dof_articulation: np.ndarray) -> np.ndarray:
    """Return each world-local response DOF's articulation rank within its world, capped at 31."""
    world_count, max_dofs = world_dof_indices.shape
    components = np.zeros((world_count, max_dofs), dtype=np.int32)
    for world in range(world_count):
        dofs = world_dof_indices[world]
        valid = dofs >= 0
        articulations = dof_articulation[dofs[valid]]
        _, rank = np.unique(articulations, return_inverse=True)
        # Merging articulations past the cap only joins islands, which keeps the partition valid.
        components[world, valid] = np.minimum(rank, ISLAND_MAX_COMPONENTS - 1)
    return components


def island_shared_bytes(max_constraints: int, mf_max_constraints: int, dofs: int, warps: int) -> int:
    """Return the static shared memory [B] the island sweep adds to the one-warp MF-GS kernel."""
    index_bytes = 2 if max(max_constraints, mf_max_constraints) <= 32767 else 4
    lists = index_bytes * (max_constraints + min(mf_max_constraints, ISLAND_MF_LIST_MAX))
    return 2 * dofs + max_constraints + lists + 4 * warps * dofs + 32 * 14 + 8


def island_warp_count(components: np.ndarray, max_warps: int) -> int:
    """Return the warps per world: one per articulation of the largest world, capped at ``max_warps``."""
    if components.size == 0:
        return 1
    return int(min(max_warps, components.max() + 1))


def islands_snippet(snippet: str, *, max_constraints: int, mf_max_constraints: int, dofs: int, warps: int) -> str:
    """Rewrite the one-warp MF-GS sweep so each of ``warps`` warps sweeps its own islands' rows."""
    M_D, D = max_constraints, dofs
    mf_list = min(mf_max_constraints, ISLAND_MF_LIST_MAX)
    index_type = "short" if max(M_D, mf_list) <= 32767 else "int"

    def once(source: str, old: str, new: str) -> str:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one generated-source fragment: {old.strip()!r}")
        return source.replace(old, new)

    def region(source: str, start: str, end: str, edit) -> str:
        a = source.index(start)
        b = source.index(end, a)
        return source[:a] + edit(source[a:b]) + source[b:]

    def block_stride(text: str) -> str:
        return re.sub(
            r"for \(int (\w+) = (\w+ \+ )?lane; \1 < (\w+); \1 \+= 32\)",
            r"for (int \1 = \2threadIdx.x; \1 < \3; \1 += blockDim.x)",
            text,
        )

    if re.search(r"[(,]\s*s_v\s*[,)]", snippet):
        raise ValueError("the island sweep cannot guard velocity writes made through a helper's pointer")
    s = once(
        snippet,
        "    int lane = threadIdx.x;\n",
        "    int lane = threadIdx.x & 31;\n    const int warp_id = threadIdx.x >> 5;\n",
    )
    s = region(s, "// LOAD PHASE", "// SOLVE PHASE", block_stride)
    s = region(s, "// STORE PHASE", "#endif", block_stride)
    load_end = s.index("// SOLVE PHASE")
    sync = s.rindex("    __syncwarp();\n", 0, load_end)
    s = s[:sync] + "    __syncthreads();\n" + s[sync + len("    __syncwarp();\n") :]
    store = s.rindex("\n", 0, s.rindex("\n", 0, s.index("// STORE PHASE"))) + 1
    s = s[:store] + "    __syncthreads();\n" + s[store:]

    if "int torsion_head = -1;" in s:
        s = once(
            s,
            "    if (row_phase == 0) {\n        if (lane == 0) {",
            "    if (row_phase == 0) {\n        if (threadIdx.x == 0) {",
        )
        s = once(
            s,
            "        torsion_head = __shfl_sync(MASK, torsion_head, 0);\n        __syncwarp();\n",
            "        __shared__ int s_isl_torsion_head;\n"
            "        if (threadIdx.x == 0) s_isl_torsion_head = torsion_head;\n"
            "        __syncthreads();\n"
            "        torsion_head = s_isl_torsion_head;\n",
        )

    # The torsion group array is a (1, 1) placeholder unless the torsion sweep is compiled in.
    torsion = "torsion_row_radius.data" in s
    s = once(
        s,
        "    // Dense pipeline registers\n",
        _prologue(M_D, D, mf_list, warps, index_type, torsion) + "    // Dense pipeline registers\n",
    )

    def dense_phase(text: str) -> str:
        text = region(
            text,
            "// Prefetch the first constraint",
            "for (int i = dense_lo; i < dense_hi; i++) {",
            lambda t: re.sub(r"\bdense_lo\b", "isl_dense_first", t),
        )
        text = once(
            text,
            "for (int i = dense_lo; i < dense_hi; i++) {\n",
            "for (int i = isl_dense_first, i_next = 0; i < dense_hi; i = i_next) {\n"
            "            i_next = s_isl_next_dense[i];\n",
        )
        return region(text, "// Prefetch constraint i+1", "int row_type", lambda t: t.replace("i + 1", "i_next"))

    s = region(s, "// ── Phase 1: Dense constraints", "// ── Phase 2: MF constraints", dense_phase)

    def mf_prefetch(text: str) -> str:
        text = once(text, "if (mf_contact_end > 0) {", "if (isl_mf_first < mf_contact_end) {")
        text = once(text, "&mf_meta.data[off_meta]", "&mf_meta.data[off_meta + isl_mf_first * 4]")
        return text.replace("mf6_base + lane", "mf6_base + isl_mf_first * 6 + lane")

    s = region(s, "// Prefetch constraint 0 (contact/friction rows", "for (int i = 0, next_i = 0;", mf_prefetch)
    s = once(
        s,
        "for (int i = 0, next_i = 0; i < mf_contact_end; i = next_i) {",
        "for (int i = isl_mf_first, next_i = 0; i < mf_contact_end; i = next_i) {",
    )
    s = once(s, "next_i = i + 1;", "next_i = isl_linear_mf ? i + 1 : s_isl_next_mf[i];")
    s = s.replace("next_i = i + 2;", "next_i = isl_linear_mf ? i + 2 : s_isl_next_mf[i + 1];")
    s = once(
        s,
        "if (next_i == i + 2) s_lam_mf[i + 1] = 0.0f;",
        "if (next_i != (isl_linear_mf ? i + 1 : s_isl_next_mf[i])) s_lam_mf[i + 1] = 0.0f;",
    )

    dense_owner = "if (s_isl_dense_bin[{}] != warp_id) continue;"
    for header, row in (
        ("for (int i = vlim_lo; i < vlim_hi; i++) {", "i"),
        ("for (int i = 0; i < drive_hi_fvl; i++) {", "i"),
        ("for (int spin = torsion_head; spin >= 0; spin = s_torsion_next[spin]) {", "spin"),
        ("for (int spin = 0; spin < m_dense; ++spin) {", "spin"),
    ):
        s = s.replace(header, header + " " + dense_owner.format(row))
    mf_vlim = (
        "if (mf_rt != 4) continue;\n\n"
        "                int packed_dofs = meta.x;\n"
        "                int dof_a = packed_dofs >> 16;\n"
        "                int dof_b = (packed_dofs << 16) >> 16;\n"
    )
    s = once(
        s,
        mf_vlim,
        mf_vlim + "                if (s_isl_dof_bin[dof_a >= 0 ? dof_a : max(dof_b, 0)] != warp_id) continue;\n",
    )

    # Each warp sweeps a private copy of the world velocity; other islands' entries in it never change, and every
    # DOF takes its owning warp's value before the store.
    s = region(s, "// SOLVE PHASE", "// STORE PHASE", lambda t: re.sub(r"\bs_v\[", "s_isl_v[warp_id][", t))
    store = s.rindex("\n", 0, s.rindex("\n", 0, s.index("// STORE PHASE"))) + 1
    s = (
        s[:store]
        + f"    for (int d = threadIdx.x; d < {D}; d += blockDim.x) s_v[d] = s_isl_v[s_isl_dof_bin[d]][d];\n"
        + "    __syncthreads();\n"
        + s[store:]
    )

    # The row-type and friction-pair caches are built once by warp 0; the prologue's barriers publish them.
    for opening in ("    {\n        const int* row_type_source", "    {\n        int pairs = 0;"):
        if opening in s:
            s = once(s, opening, "    if (warp_id == 0) " + opening.lstrip(" "))
    if re.search(r"\blane == 0\) \{\n\s+for \(int n = m_dense", s):
        raise RuntimeError("Island sweep left a per-warp shared-list build")
    return s


def _prologue(M_D: int, D: int, mf_list: int, warps: int, index_type: str, torsion: bool) -> str:
    torsion_union = (
        """        const int group = world_torsion_group.data[off_dense + r];
        if (group >= 0 && group < m_dense) mask |= isl_row_mask(group);
"""
        if torsion
        else ""
    )
    return f"""
    // ── Island partition: union the articulations each row touches, then give each warp whole islands ──
    __shared__ unsigned s_isl_reach[32];
    __shared__ int s_isl_rows[32];
    __shared__ int s_isl_island_rows[32];
    __shared__ unsigned char s_isl_bin_of_rep[32];
    __shared__ unsigned char s_isl_comp_bin[32];
    __shared__ unsigned char s_isl_dof_comp[{D}];
    __shared__ unsigned char s_isl_dof_bin[{D}];
    __shared__ unsigned char s_isl_dense_bin[{M_D}];
    __shared__ {index_type} s_isl_next_dense[{M_D}];
    __shared__ {index_type} s_isl_next_mf[{mf_list}];
    __shared__ int s_isl_single;
    __shared__ float s_isl_v[{warps}][{D}];
    if (threadIdx.x < 32) {{
        s_isl_reach[threadIdx.x] = 1u << threadIdx.x;
        s_isl_rows[threadIdx.x] = 0;
        s_isl_bin_of_rep[threadIdx.x] = 0;
    }}
    for (int d = threadIdx.x; d < {D}; d += blockDim.x)
        s_isl_dof_comp[d] = (unsigned char)island_dof_component.data[world * {D} + d];
    __syncthreads();
    auto isl_row_mask = [&](int r) {{
        const float* isl_row = &J_world.data[jy_world_base + r * {D}];
        unsigned row_mask = 0u;
        for (int d = 0; d < {D}; ++d)
            if (isl_row[d] != 0.0f) row_mask |= 1u << s_isl_dof_comp[d];
        return row_mask;
    }};
    for (int r = threadIdx.x; r < m_dense; r += blockDim.x) {{
        unsigned mask = isl_row_mask(r);
        // A contact's rows share load through parents and torsion groups, so they join even where one row's J
        // misses an articulation.
        const int parent = world_row_parent.data[off_dense + r];
        if (parent >= 0 && parent < m_dense) mask |= isl_row_mask(parent);
{torsion_union}        if (mask == 0u) mask = 1u;
        s_isl_dense_bin[r] = (unsigned char)(__ffs(mask) - 1);
        atomicAdd(&s_isl_rows[__ffs(mask) - 1], 1);
        if (mask & (mask - 1u))
            for (unsigned bits = mask; bits; bits &= bits - 1u) atomicOr(&s_isl_reach[__ffs(bits) - 1], mask);
    }}
    for (int r = threadIdx.x; r < m_mf; r += blockDim.x) {{
        const int packed = mf_meta.data[off_meta + r * 4];
        const int da = packed >> 16, db = (packed << 16) >> 16;
        unsigned mask = (da >= 0 ? 1u << s_isl_dof_comp[da] : 0u) | (db >= 0 ? 1u << s_isl_dof_comp[db] : 0u);
        if (mask == 0u) mask = 1u;
        atomicAdd(&s_isl_rows[__ffs(mask) - 1], 1);
        if (mask & (mask - 1u))
            for (unsigned bits = mask; bits; bits &= bits - 1u) atomicOr(&s_isl_reach[__ffs(bits) - 1], mask);
    }}
    __syncthreads();
    if (threadIdx.x < 32) {{
        // Transitive closure by repeated squaring: five rounds cover any path over 32 components.
        unsigned reach = s_isl_reach[lane];
        for (int round = 0; round < 5; ++round) {{
            unsigned grown = reach;
            for (int b = 0; b < 32; ++b) {{
                const unsigned other = __shfl_sync(MASK, reach, b);
                if (reach & (1u << b)) grown |= other;
            }}
            if (__ballot_sync(MASK, grown != reach) == 0u) break;
            reach = grown;
        }}
        const int rep = __ffs(reach) - 1;
        const int rows = s_isl_rows[lane];
        int island_rows = 0;
        for (int b = 0; b < 32; ++b) {{
            const int other = __shfl_sync(MASK, rows, b);
            if (reach & (1u << b)) island_rows += other;
        }}
        s_isl_island_rows[lane] = island_rows;
        const unsigned reps = __ballot_sync(MASK, rep == lane && island_rows > 0);
        __syncwarp();
        if (lane == 0) {{
            // Longest island first onto the least-loaded warp; ties keep the lower index.
            int load[{warps}];
            for (int w = 0; w < {warps}; ++w) load[w] = 0;
            for (unsigned left = reps; left;) {{
                int best = __ffs(left) - 1;
                for (unsigned bits = left; bits; bits &= bits - 1u) {{
                    const int c = __ffs(bits) - 1;
                    if (s_isl_island_rows[c] > s_isl_island_rows[best]) best = c;
                }}
                left &= ~(1u << best);
                int bin = 0;
                for (int w = 1; w < {warps}; ++w)
                    if (load[w] < load[bin]) bin = w;
                load[bin] += s_isl_island_rows[best];
                s_isl_bin_of_rep[best] = (unsigned char)bin;
            }}
            s_isl_single = mf_contact_end > {mf_list};
        }}
        __syncwarp();
        s_isl_comp_bin[lane] = s_isl_single ? 0 : s_isl_bin_of_rep[rep];
    }}
    __syncthreads();
    for (int r = threadIdx.x; r < m_dense; r += blockDim.x) s_isl_dense_bin[r] = s_isl_comp_bin[s_isl_dense_bin[r]];
    for (int d = threadIdx.x; d < {D}; d += blockDim.x) s_isl_dof_bin[d] = s_isl_comp_bin[s_isl_dof_comp[d]];
    __syncthreads();
    // Each warp links its own rows in ascending order; the last row points at the range end.
    int isl_dense_first = dense_hi;
    {{
        int last = -1;
        for (int base = dense_lo; base < dense_hi; base += 32) {{
            const int r = base + lane;
            const bool mine = r < dense_hi && s_isl_dense_bin[r] == warp_id;
            const unsigned bal = __ballot_sync(MASK, mine);
            if (bal == 0u) continue;
            const int lo = base + __ffs(bal) - 1;
            if (last < 0) isl_dense_first = lo;
            else if (lane == 0) s_isl_next_dense[last] = ({index_type})lo;
            if (mine) {{
                const unsigned above = bal & ~((2u << lane) - 1u);
                s_isl_next_dense[r] = ({index_type})(above ? base + __ffs(above) - 1 : dense_hi);
            }}
            last = base + 31 - __clz(bal);
            __syncwarp();
        }}
    }}
    const int isl_linear_mf = s_isl_single;
    int isl_mf_first = isl_linear_mf && warp_id == 0 ? 0 : mf_contact_end;
    if (!isl_linear_mf) {{
        int last = -1;
        for (int base = 0; base < mf_contact_end; base += 32) {{
            const int r = base + lane;
            bool mine = false;
            if (r < mf_contact_end) {{
                const int packed = mf_meta.data[off_meta + r * 4];
                const int da = packed >> 16, db = (packed << 16) >> 16;
                mine = s_isl_dof_bin[da >= 0 ? da : max(db, 0)] == warp_id;
            }}
            const unsigned bal = __ballot_sync(MASK, mine);
            if (bal == 0u) continue;
            const int lo = base + __ffs(bal) - 1;
            if (last < 0) isl_mf_first = lo;
            else if (lane == 0) s_isl_next_mf[last] = ({index_type})lo;
            if (mine) {{
                const unsigned above = bal & ~((2u << lane) - 1u);
                s_isl_next_mf[r] = ({index_type})(above ? base + __ffs(above) - 1 : mf_contact_end);
            }}
            last = base + 31 - __clz(bal);
            __syncwarp();
        }}
    }}
    for (int d = lane; d < {D}; d += 32) s_isl_v[warp_id][d] = (s_v)[d];
    __syncwarp();

"""


def islands_kernel(snippet: str, name: str) -> "wp.Kernel":
    """Wrap an island-rewritten MF-GS snippet; the signature adds ``island_dof_component`` after the dense inputs."""

    @wp.func_native(snippet)
    def pgs_solve_mf_gs_islands_native(
        world: int,
        world_constraint_count: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        local_solve_owner: wp.array[int],
        world_dof_indices: wp.array2d[int],
        island_dof_component: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_row_w: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_torsion_group: wp.array2d[int],
        torsion_row_radius: wp.array2d[float],
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        world_drive_vel_limit: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        compliance_dense_gamma: wp.array2d[float],
        compliance_mf_gamma: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ): ...

    def pgs_solve_mf_gs_islands_template(  # pragma: no cover - compiled by wp.kernel
        world_constraint_count: wp.array[int],
        dense_phase_bounds: wp.array2d[int],
        local_solve_owner: wp.array[int],
        world_dof_indices: wp.array2d[int],
        island_dof_component: wp.array2d[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_row_w: wp.array2d[float],
        world_impulses: wp.array2d[float],
        J_world: wp.array3d[float],
        Y_world: wp.array3d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        world_torsion_group: wp.array2d[int],
        torsion_row_radius: wp.array2d[float],
        world_drive_target_vel_bias: wp.array2d[float],
        world_drive_vel_multiplier: wp.array2d[float],
        world_drive_impulse_multiplier: wp.array2d[float],
        world_drive_max_impulse: wp.array2d[float],
        world_drive_vel_limit: wp.array2d[float],
        mf_constraint_count: wp.array[int],
        mf_contact_rows_end: wp.array[int],
        mf_meta: wp.array2d[int],
        mf_impulses: wp.array2d[float],
        mf_J_a: wp.array3d[float],
        mf_J_b: wp.array3d[float],
        mf_MiJt_a: wp.array3d[float],
        mf_MiJt_b: wp.array3d[float],
        mf_row_mu: wp.array2d[float],
        mf_row_w: wp.array2d[float],
        compliance_dense_gamma: wp.array2d[float],
        compliance_mf_gamma: wp.array2d[float],
        iterations: int,
        omega: float,
        regularize: int,
        row_phase: int,
        friction_start_iteration: int,
        iteration_offset: int,
        freeze_drive_rows: int,
        v_out: wp.array[float],
    ):
        world, _thread = wp.tid()
        pgs_solve_mf_gs_islands_native(
            world,
            world_constraint_count,
            dense_phase_bounds,
            local_solve_owner,
            world_dof_indices,
            island_dof_component,
            rhs_bias,
            world_diag,
            world_row_w,
            world_impulses,
            J_world,
            Y_world,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_torsion_group,
            torsion_row_radius,
            world_drive_target_vel_bias,
            world_drive_vel_multiplier,
            world_drive_impulse_multiplier,
            world_drive_max_impulse,
            world_drive_vel_limit,
            mf_constraint_count,
            mf_contact_rows_end,
            mf_meta,
            mf_impulses,
            mf_J_a,
            mf_J_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_row_mu,
            mf_row_w,
            compliance_dense_gamma,
            compliance_mf_gamma,
            iterations,
            omega,
            regularize,
            row_phase,
            friction_start_iteration,
            iteration_offset,
            freeze_drive_rows,
            v_out,
        )

    pgs_solve_mf_gs_islands_template.__name__ = name
    pgs_solve_mf_gs_islands_template.__qualname__ = name
    return wp.kernel(enable_backward=False, module="unique")(pgs_solve_mf_gs_islands_template)
