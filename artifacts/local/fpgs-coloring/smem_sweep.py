"""SMEM-occupancy sweep for the colored FPGS MF Gauss-Seidel kernel.

Compares the resident-smem baseline vs the STREAM_LAMBDA variant (s_lam_mf ->
volatile global g_lam_mf alias of mf_impulses), measuring:
  (1) ptxas -v smem bytes/block + regs/thread (printed by warp's NVRTC->ptxas).
  (2) Local solve-only timing serial vs colored W in {1,2,4,8} at N in {64,256}.
  (3) Fidelity: colored peak interpenetration (mm) vs serial via tau_test.

Toggle streaming with FEATHER_PGS_MFGS_STREAM_LAMBDA=1 (whole run is one variant).
Imports the WORKTREE newton (asserts the worktree path).
"""

import json
import os
import sys
import time
import collections

WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/smem-color"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/tmp")  # tau_test

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

assert newton.__file__.startswith(WORKTREE), f"NOT worktree newton: {newton.__file__}"
print(f"[smem] newton.__file__ = {newton.__file__}", flush=True)
print(f"[smem] STREAM_LAMBDA={os.environ.get('FEATHER_PGS_MFGS_STREAM_LAMBDA','0')}", flush=True)

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)
from newton._src.sim.graph_coloring import color_graph  # noqa: E402
import tau_test as tt  # noqa: E402

DEV = "cuda:0"
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.config.verbose = True  # surface ptxas info if available
wp.init()

meta = json.load(open("/tmp/cap892.json"))
z = np.load("/tmp/cap892.npz")
M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
arch = int(wp.get_device(DEV).arch)
MAX_COLORS = int(os.environ.get("FEATHER_PGS_MFGS_MAX_COLORS", "16"))
print(f"[smem] arch={arch} M_D={M_D} M_MF={M_MF} D={D} iters={meta['iterations']}", flush=True)

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
NMAX = int(z["mf_constraint_count"].shape[0])


def sl(k, N):
    a = z[k]
    return a[: N * D] if k == "v_out" else a[:N]


def make_inputs(N):
    return {k: wp.array(np.ascontiguousarray(sl(k, N)),
                        dtype=(wp.int32 if k in INT_KEYS else wp.float32), device=DEV)
            for k in z.files}


# ── Real-coloring CSR (color_graph oracle), same as m3_m4_bench ──
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


# Build CSR for ALL captured worlds once (slice per N later).
all_off = np.zeros((NMAX, MAX_COLORS + 1), dtype=np.int32)
all_rows = np.zeros((NMAX, M_MF), dtype=np.int32)
all_nc = np.zeros(NMAX, dtype=np.int32)
ncolors_list = []
for w in range(NMAX):
    Mw, dof_a, dof_b, rt, parent = unpack_world(w)
    nodes = build_nodes(Mw, dof_a, dof_b, rt, parent)
    color, ncolors = color_world(nodes)
    offsets, rows, nc = pack_csr_world(nodes, color, max(ncolors, 1))
    all_off[w] = offsets
    all_rows[w] = rows
    all_nc[w] = nc
    ncolors_list.append(nc)
print(f"[smem] coloring: n_colors per world min/median/max = "
      f"{min(ncolors_list)}/{int(np.median(ncolors_list))}/{max(ncolors_list)}", flush=True)


def make_kernel(colored, warps):
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1" if colored else "0"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    os.environ["FEATHER_PGS_MFGS_WARPS"] = str(warps)
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    return _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])


def build_inputs(a, colored, N):
    if not colored:
        return [a[k] for k in ORDER] + SCALARS
    csr = (wp.array(np.ascontiguousarray(all_off[:N].reshape(-1)), dtype=wp.int32, device=DEV),
           wp.array(np.ascontiguousarray(all_rows[:N].reshape(-1)), dtype=wp.int32, device=DEV),
           wp.array(np.ascontiguousarray(all_nc[:N]), dtype=wp.int32, device=DEV))
    mu_idx = ORDER.index("mf_row_mu")
    return ([a[k] for k in ORDER[: mu_idx + 1]] + list(csr)
            + [a[k] for k in ORDER[mu_idx + 1:]] + SCALARS)


def run_once(colored, warps, N):
    k = make_kernel(colored, warps)
    a = make_inputs(N)
    inp = build_inputs(a, colored, N)
    wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]],
                    block_dim=32 * warps, device=DEV)
    return a["v_out"].numpy(), a["mf_impulses"].numpy()


def bench(colored, warps, N, iters=300, warmup=50):
    k = make_kernel(colored, warps)
    a = make_inputs(N)
    inp = build_inputs(a, colored, N)
    bd = 32 * warps
    for _ in range(warmup):
        wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
    wp.synchronize_device(DEV)
    # median of several windows
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(iters):
            wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
        wp.synchronize_device(DEV)
        times.append((time.perf_counter() - t0) / iters * 1e3)
    return float(np.median(times))


# ════════════════════════════════════════════════════════════════════
# (1) Build kernels once so warp prints ptxas -v info (verbose=True).
# ════════════════════════════════════════════════════════════════════
print("\n[smem] === BUILD serial kernel (ptxas -v below) ===", flush=True)
make_kernel(False, 1)
print("[smem] === BUILD colored W=1 kernel (ptxas -v below) ===", flush=True)
make_kernel(True, 1)
print("[smem] === BUILD colored W=4 kernel (ptxas -v below) ===", flush=True)
make_kernel(True, 4)

# ════════════════════════════════════════════════════════════════════
# (3) Fidelity: colored peak interpenetration (mm) vs serial.
# ════════════════════════════════════════════════════════════════════
print("\n[smem] === FIDELITY (peak penetration mm, N=256) ===", flush=True)
Nf = 256
v_ser, _ = run_once(False, 1, Nf)
v_col, _ = run_once(True, 4, Nf)
# tau_test.mm_metrics(W, v, lam): per-world; recover peak pen mm from GPU velocity.
# We replay per-world penetration using the GPU v_out (reshape to per-world dofs).
v_ser_w = v_ser.reshape(Nf, D)
v_col_w = v_col.reshape(Nf, D)
peak_ser = []
peak_col = []
for w in range(min(Nf, 64)):  # sample worlds for speed
    W = tt.load_world(w)
    lam0 = np.zeros(W["m"], dtype=np.float64) if "m" in W else None
    try:
        ms = tt.mm_metrics(W, v_ser_w[w], lam0)
        mc = tt.mm_metrics(W, v_col_w[w], lam0)
        peak_ser.append(ms["peak_pen_mm"])
        peak_col.append(mc["peak_pen_mm"])
    except Exception as e:
        if w == 0:
            print(f"[smem] mm_metrics fallback ({e!r}); using residual delta-v fidelity", flush=True)
        break
if peak_ser:
    peak_ser = np.array(peak_ser); peak_col = np.array(peak_col)
    print(f"[smem] peak_pen_mm  serial: max={peak_ser.max():.4f}  colored W=4: max={peak_col.max():.4f}", flush=True)
    print(f"[smem] colored within 8mm: {bool(peak_col.max() <= 8.0)}  "
          f"max|col-ser|={np.abs(peak_col - peak_ser).max():.4f} mm", flush=True)
else:
    # Velocity-space fidelity fallback: colored vs serial v_out delta.
    dv = np.abs(v_ser.astype(np.float64) - v_col.astype(np.float64))
    print(f"[smem] v_out fidelity colored W=4 vs serial: max|Δv|={dv.max():.3e}  "
          f"rms|Δv|={np.sqrt((dv**2).mean()):.3e}", flush=True)

# ════════════════════════════════════════════════════════════════════
# (2) Timing W-sweep at N in {64, 256}.
# ════════════════════════════════════════════════════════════════════
for N in (64, 256):
    print(f"\n[smem] === TIMING N={N} ({meta['iterations']} iters, ms/launch, median) ===", flush=True)
    t_ser = bench(False, 1, N)
    print(f"[smem] N={N} serial      : {t_ser:.4f} ms", flush=True)
    for warps in (1, 2, 4, 8):
        tc = bench(True, warps, N)
        print(f"[smem] N={N} colored W={warps} : {tc:.4f} ms   "
              f"(serial/colored = {t_ser/tc:.3f}x)", flush=True)
print("\n[smem] DONE", flush=True)
