"""CUDA-event-timed W-sweep: baseline (resident smem) vs streaming, same process.

Wall-clock perf_counter over async launches is dominated by launch/dispatch noise.
This times the kernel itself with wp.ScopedTimer(use_nvtx=False, cuda_filter=...)
-> CUDA events, median over many launches, for a fair baseline-vs-streaming compare.

Builds BOTH variants in one process by toggling the streaming env BEFORE each
kernel build (the kernel name encodes the streaming tag so caches don't collide).
"""
import json
import os
import sys
import collections

WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/smem-color"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/tmp")

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

assert newton.__file__.startswith(WORKTREE), newton.__file__
from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)
from newton._src.sim.graph_coloring import color_graph  # noqa: E402

DEV = "cuda:0"
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.init()

meta = json.load(open("/tmp/cap892.json"))
z = np.load("/tmp/cap892.npz")
M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
arch = int(wp.get_device(DEV).arch)
MAX_COLORS = 16
NMAX = int(z["mf_constraint_count"].shape[0])

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
SCALARS = [int(meta["iterations"]), float(meta["omega"]), int(meta["row_phase"]),
           int(meta["friction_start_iteration"]), int(meta["iteration_offset"]),
           int(meta["freeze_drive_rows"])]


def sl(k, N):
    a = z[k]
    return a[: N * D] if k == "v_out" else a[:N]


def make_inputs(N):
    return {k: wp.array(np.ascontiguousarray(sl(k, N)),
                        dtype=(wp.int32 if k in INT_KEYS else wp.float32), device=DEV)
            for k in z.files}


def unpack_world(w):
    Mw = min(int(z["mf_constraint_count"][w]), M_MF)
    mw = z["mf_meta"][w].reshape(-1, 4)[:Mw].astype(np.int64)
    packed = mw[:, 0].astype(np.int32)
    return (Mw, (packed >> 16).astype(np.int64), ((packed << 16) >> 16).astype(np.int64),
            (mw[:, 3] & 0xFFFF).astype(np.int64), (mw[:, 3] >> 16).astype(np.int64))


def build_nodes(Mw, da, db, rt, par):
    ch = collections.defaultdict(list)
    for i in range(Mw):
        if rt[i] == 2:
            ch[int(par[i])].append(i)
    nodes = []
    for i in range(Mw):
        if rt[i] != 0:
            continue
        rows = [i] + sorted(ch[i])
        bodies = set()
        for r in rows:
            if da[r] >= 0:
                bodies.add(int(da[r]))
            if db[r] >= 0:
                bodies.add(int(db[r]))
        nodes.append((i, rows, bodies))
    return nodes


def color_world(nodes):
    n = len(nodes)
    if n == 0:
        return np.zeros(0, np.int32), 0
    b2n = collections.defaultdict(list)
    for ni, (_, _, bs) in enumerate(nodes):
        for b in bs:
            b2n[b].append(ni)
    edges = set()
    for _, ns in b2n.items():
        for x in range(len(ns)):
            for y in range(x + 1, len(ns)):
                a, c = ns[x], ns[y]
                edges.add((a, c) if a < c else (c, a))
    if not edges:
        return np.zeros(n, np.int32), 1
    ea = wp.array(np.array(sorted(edges), np.int32), dtype=wp.int32, device="cpu")
    groups = color_graph(n, ea, balance_colors=True)
    color = np.full(n, -1, np.int32)
    for c, g in enumerate(groups):
        color[np.asarray(g)] = c
    return color, len(groups)


def pack_csr(nodes, color, nc):
    bc = collections.defaultdict(list)
    for ni, c in enumerate(color):
        bc[int(c)].append(ni)
    off = np.zeros(MAX_COLORS + 1, np.int32)
    rows = np.zeros(M_MF, np.int32)
    pos = 0
    for c in range(nc):
        off[c] = pos
        for ni in bc[c]:
            for r in nodes[ni][1]:
                rows[pos] = r
                pos += 1
    for c in range(nc, MAX_COLORS + 1):
        off[c] = pos
    return off, rows, nc


all_off = np.zeros((NMAX, MAX_COLORS + 1), np.int32)
all_rows = np.zeros((NMAX, M_MF), np.int32)
all_nc = np.zeros(NMAX, np.int32)
for w in range(NMAX):
    Mw, da, db, rt, par = unpack_world(w)
    nodes = build_nodes(Mw, da, db, rt, par)
    color, nc = color_world(nodes)
    off, rows, nc = pack_csr(nodes, color, max(nc, 1))
    all_off[w] = off
    all_rows[w] = rows
    all_nc[w] = nc


def make_kernel(colored, warps, stream):
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1" if colored else "0"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    os.environ["FEATHER_PGS_MFGS_WARPS"] = str(warps)
    os.environ["FEATHER_PGS_MFGS_STREAM_LAMBDA"] = "1" if stream else "0"
    os.environ["FEATHER_PGS_MFGS_STREAM_DRIVE"] = "1" if stream else "0"
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    return _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])


def build_inputs(a, colored, N):
    if not colored:
        return [a[k] for k in ORDER] + SCALARS
    csr = (wp.array(np.ascontiguousarray(all_off[:N].reshape(-1)), dtype=wp.int32, device=DEV),
           wp.array(np.ascontiguousarray(all_rows[:N].reshape(-1)), dtype=wp.int32, device=DEV),
           wp.array(np.ascontiguousarray(all_nc[:N]), dtype=wp.int32, device=DEV))
    mu = ORDER.index("mf_row_mu")
    return [a[k] for k in ORDER[:mu + 1]] + list(csr) + [a[k] for k in ORDER[mu + 1:]] + SCALARS


def evt_time(colored, warps, stream, N, iters=200, warmup=30, reps=9):
    k = make_kernel(colored, warps, stream)
    a = make_inputs(N)
    inp = build_inputs(a, colored, N)
    bd = 32 * warps
    for _ in range(warmup):
        wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
    wp.synchronize_device(DEV)
    samples = []
    for _ in range(reps):
        e0 = wp.Event(enable_timing=True)
        e1 = wp.Event(enable_timing=True)
        wp.record_event(e0)
        for _ in range(iters):
            wp.launch_tiled(k, dim=[N], inputs=inp, outputs=[a["v_out"]], block_dim=bd, device=DEV)
        wp.record_event(e1)
        wp.synchronize_event(e1)
        samples.append(wp.get_event_elapsed_time(e0, e1) / iters)  # ms/launch
    return float(np.median(samples))


for N in (64, 256):
    print(f"\n=== N={N}  CUDA-event ms/launch (median of 9x200) ===", flush=True)
    t_ser_base = evt_time(False, 1, False, N)
    t_ser_strm = evt_time(False, 1, True, N)
    print(f"serial   baseline={t_ser_base:.4f}  streaming={t_ser_strm:.4f}  "
          f"(strm/base={t_ser_strm/t_ser_base:.3f})", flush=True)
    print(f"  {'W':>2} | {'base ms':>9} {'strm ms':>9} | {'base x(ser/col)':>16} {'strm x(ser/col)':>16} | strm-vs-base", flush=True)
    for w in (1, 2, 4, 8):
        tb = evt_time(True, w, False, N)
        ts = evt_time(True, w, True, N)
        # speedup uses each variant's OWN serial as the reference
        print(f"  {w:>2} | {tb:9.4f} {ts:9.4f} | "
              f"{t_ser_base/tb:16.3f} {t_ser_strm/ts:16.3f} | {tb/ts:.3f}x faster", flush=True)
print("\nEVT_DONE", flush=True)
