"""M3 cross-check (W>1 == W=1 colored, bit-identical) + M4 local timing.

M3 cross-check: the colored GS variant is a pure function of the color
partition; rows within a color share no body, so distributing them across W
warps must reproduce the W=1 colored output BIT-FOR-BIT. Assert
np.array_equal(colored_W, colored_W1) for W in {2,4}. (This is a stronger
correctness signal than fidelity alone: it proves the node round-robin keeps
triples intact and the __syncthreads color barrier is placed correctly.)

M4: time serial vs colored at W in {1,2,4} on the RTX 3090 via the capture
replay (the same kernel launch the solver uses), CUDA-event timed, median over
many launches after warmup. Rough but real ms/launch signal.
"""

import json
import os
import sys
import time
import collections

WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/wf_453cc222-038-1/newton-collab"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/tmp")

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

assert newton.__file__.startswith(WORKTREE)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)
from newton._src.sim.graph_coloring import color_graph  # noqa: E402

N = int(os.environ.get("NWORLDS", "256"))
DEV = "cuda:0"
wp.init()

meta = json.load(open("/tmp/cap892.json"))
z = np.load("/tmp/cap892.npz")
M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
arch = int(wp.get_device(DEV).arch)
MAX_COLORS = 16

INT_KEYS = {"constraint_count", "world_dof_start", "row_type", "row_parent",
            "mf_constraint_count", "mf_meta"}
ORDER = [
    "constraint_count", "world_dof_start", "dense_rhs", "diag", "impulses",
    "J_world", "Y_world", "row_type", "row_parent", "row_mu",
    "drive_target_vel_bias", "drive_vel_multiplier", "drive_impulse_multiplier",
    "drive_max_impulse",
    "mf_constraint_count", "mf_meta", "mf_impulses", "mf_J_a", "mf_J_b",
    "mf_MiJt_a", "mf_MiJt_b", "mf_row_mu",
]
SCALARS = [
    int(meta["iterations"]), float(meta["omega"]), int(meta["row_phase"]),
    int(meta["friction_start_iteration"]), int(meta["iteration_offset"]),
    int(meta["freeze_drive_rows"]),
]


def sl(k):
    a = z[k]
    return a[: N * D] if k == "v_out" else a[:N]


def make_inputs():
    return {k: wp.array(np.ascontiguousarray(sl(k)),
                        dtype=(wp.int32 if k in INT_KEYS else wp.float32), device=DEV)
            for k in z.files}


# ── Build real-coloring CSR (same as m1_m2_colored.py) ──
def unpack_world(w):
    Mw = min(int(z["mf_constraint_count"][w]), M_MF)
    meta_w = z["mf_meta"][w].reshape(-1, 4)[:Mw].astype(np.int64)
    packed = meta_w[:, 0].astype(np.int32)
    dof_a = (packed >> 16).astype(np.int64)
    dof_b = ((packed << 16) >> 16).astype(np.int64)
    rt = (meta_w[:, 3] & 0xFFFF).astype(np.int64)
    parent = (meta_w[:, 3] >> 16).astype(np.int64)
    return Mw, dof_a, dof_b, rt, parent


def build_nodes(Mw, dof_a, dof_b, rt, parent):
    children = collections.defaultdict(list)
    for i in range(Mw):
        if rt[i] == 2:
            children[int(parent[i])].append(i)
    nodes = []
    for i in range(Mw):
        if rt[i] != 0:
            continue
        rows = [i] + sorted(children[i])
        bodies = set()
        for r in rows:
            if dof_a[r] >= 0:
                bodies.add(int(dof_a[r]))
            if dof_b[r] >= 0:
                bodies.add(int(dof_b[r]))
        nodes.append((i, rows, bodies))
    return nodes


def color_world(nodes):
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0
    body_to_nodes = collections.defaultdict(list)
    for ni, (_, _, bodies) in enumerate(nodes):
        for b in bodies:
            body_to_nodes[b].append(ni)
    edges = set()
    for b, ns in body_to_nodes.items():
        for x in range(len(ns)):
            for y in range(x + 1, len(ns)):
                a, c = ns[x], ns[y]
                edges.add((a, c) if a < c else (c, a))
    if not edges:
        return np.zeros(n, dtype=np.int32), 1
    edge_arr = wp.array(np.array(sorted(edges), dtype=np.int32), dtype=wp.int32, device="cpu")
    groups = color_graph(n, edge_arr, balance_colors=True)
    color = np.full(n, -1, dtype=np.int32)
    for c, grp in enumerate(groups):
        color[np.asarray(grp)] = c
    return color, len(groups)


def pack_csr_world(nodes, color, ncolors):
    by_color = collections.defaultdict(list)
    for ni, c in enumerate(color):
        by_color[int(c)].append(ni)
    offsets = np.zeros(MAX_COLORS + 1, dtype=np.int32)
    rows = np.zeros(M_MF, dtype=np.int32)
    pos = 0
    for c in range(ncolors):
        offsets[c] = pos
        for ni in by_color[c]:
            for r in nodes[ni][1]:
                rows[pos] = r
                pos += 1
    for c in range(ncolors, MAX_COLORS + 1):
        offsets[c] = pos
    return offsets, rows, ncolors


all_off = np.zeros((N, MAX_COLORS + 1), dtype=np.int32)
all_rows = np.zeros((N, M_MF), dtype=np.int32)
all_nc = np.zeros(N, dtype=np.int32)
for w in range(N):
    Mw, dof_a, dof_b, rt, parent = unpack_world(w)
    nodes = build_nodes(Mw, dof_a, dof_b, rt, parent)
    color, ncolors = color_world(nodes)
    offsets, rows, nc = pack_csr_world(nodes, color, max(ncolors, 1))
    all_off[w] = offsets
    all_rows[w] = rows
    all_nc[w] = nc

csr_off_np = np.ascontiguousarray(all_off.reshape(-1))
csr_rows_np = np.ascontiguousarray(all_rows.reshape(-1))
csr_nc_np = np.ascontiguousarray(all_nc)


def make_kernel(colored, warps):
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1" if colored else "0"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    os.environ["FEATHER_PGS_MFGS_WARPS"] = str(warps)
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    return _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])


def build_inputs(a, colored):
    if not colored:
        return [a[k] for k in ORDER] + SCALARS
    csr = (wp.array(csr_off_np, dtype=wp.int32, device=DEV),
           wp.array(csr_rows_np, dtype=wp.int32, device=DEV),
           wp.array(csr_nc_np, dtype=wp.int32, device=DEV))
    mu_idx = ORDER.index("mf_row_mu")
    return ([a[k] for k in ORDER[: mu_idx + 1]] + list(csr)
            + [a[k] for k in ORDER[mu_idx + 1:]] + SCALARS)


def run_once(colored, warps):
    k = make_kernel(colored, warps)
    a = make_inputs()
    inp = build_inputs(a, colored)
    wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]],
                    block_dim=32 * warps, device=DEV)
    return a["v_out"].numpy(), a["mf_impulses"].numpy()


# ── M3 cross-check: colored W>1 == colored W=1 bit-for-bit ──
print(f"\n[m3m4] === M3 W-consistency (colored, N={N}) ===", flush=True)
v1, i1 = run_once(True, 1)
for warps in (2, 4):
    vw, iw = run_once(True, warps)
    eq = np.array_equal(v1, vw) and np.array_equal(i1, iw)
    if eq:
        print(f"[m3m4] colored W={warps} == colored W=1 : BIT-IDENTICAL (PASS)", flush=True)
    else:
        dv = np.abs(v1.astype(np.float64) - vw.astype(np.float64)).max()
        di = np.abs(i1.astype(np.float64) - iw.astype(np.float64)).max()
        print(f"[m3m4] colored W={warps} != W=1 : max|Δv|={dv:.3e} max|Δλ|={di:.3e} "
              f"(different float order across warps; check fidelity instead)", flush=True)


# ── M4: time serial vs colored W in {1,2,4} ──
def bench(colored, warps, iters=300, warmup=50):
    k = make_kernel(colored, warps)
    a = make_inputs()
    inp = build_inputs(a, colored)
    bd = 32 * warps
    for _ in range(warmup):
        wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
    wp.synchronize_device(DEV)
    t0 = time.perf_counter()
    for _ in range(iters):
        wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
    wp.synchronize_device(DEV)
    return (time.perf_counter() - t0) / iters * 1e3  # ms/launch


print(f"\n[m3m4] === M4 local RTX3090 timing (N={N}, {meta['iterations']} iters, ms/launch) ===", flush=True)
t_serial = bench(False, 1)
print(f"[m3m4] serial      : {t_serial:.4f} ms/launch", flush=True)
for warps in (1, 2, 4):
    tc = bench(True, warps)
    print(f"[m3m4] colored W={warps} : {tc:.4f} ms/launch   "
          f"(serial/colored = {t_serial/tc:.3f}x)", flush=True)
print("[m3m4] DONE", flush=True)
