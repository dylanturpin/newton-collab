"""Blackwell tpw-vs-warp FPGS solve-kernel bench (self-contained, node-side).

This is the CORRECT thread-per-world (tpw) benchmark: it calls the tpw solve
kernel DIRECTLY via the standalone replay path (wp.launch on the tpw kernel),
NOT through step() (which on this branch is still the warp kernel). It therefore
exercises the ACTUAL tpw kernel.

It:
  * Forces the local (cloned-branch) `newton` ahead of any installed newton.
  * Loads the cap892 capture (256 real worlds) from $TPW_CAP_NPZ / $TPW_CAP_JSON
    (the repo ships a savez_compressed copy; this script transparently handles
    both compressed and uncompressed npz).
  * Tiles the 256 worlds up to arbitrary N so we can sweep N far beyond 256.
  * PROVES which kernel ran: prints the compiled kernel `.key` (must contain
    `_tpw_` for tpw, and `_current` WITHOUT `tpw` for warp) and module name, plus
    the GPU arch (sm_XX) and device name.
  * CUDA-event times the solve kernel (wp.TIMING_KERNEL) for warp-per-world and
    thread-per-world across the N-sweep.
  * Reports fidelity: peak interpenetration (mm) for tpw vs warp at N=256.

Env:
  TPW_REPO       path to the cloned il-newton-dev checkout (contains newton/)
  TPW_CAP_NPZ    path to cap892 npz (compressed or not)
  TPW_CAP_JSON   path to cap892 json meta
  TPW_NS         comma-sep N sweep (default 256,1024,4096,8192,16384)
  TPW_BD         tpw block_dim (default 256)
  WARP_CACHE_PATH optional warp kernel cache dir
"""
import json
import os
import sys

REPO = os.environ.get("TPW_REPO", "/home/dturpin/repos/il-newton-dev/.claude/worktrees/tpw")
sys.path.insert(0, REPO)

import numpy as np
import warp as wp
import newton as _n

assert REPO in _n.__file__, f"wrong newton: {_n.__file__} (expected under {REPO})"

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_mf_gs_kernel_tpw,
)

DEV = "cuda:0"
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.init()

CAP_JSON = os.environ.get("TPW_CAP_JSON", "/tmp/cap892.json")
CAP_NPZ = os.environ.get("TPW_CAP_NPZ", "/tmp/cap892.npz")
meta = json.load(open(CAP_JSON))
z = np.load(CAP_NPZ)

M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
dev = wp.get_device(DEV)
arch = int(dev.arch)
ITERS = int(meta["iterations"])
OMEGA = float(meta["omega"])
FRIC_START = int(meta["friction_start_iteration"])
ITER_OFF = int(meta["iteration_offset"])
FREEZE = int(meta["freeze_drive_rows"])
ROW_PHASE = int(meta["row_phase"])
BD_WARP = int(meta["block_dim"])
WBASE = z["mf_meta"].shape[0]  # 256
assert ROW_PHASE == 0, "tpw MVP supports row_phase 0 only"
assert meta["friction_mode"] == "current"

INT_KEYS = {"constraint_count", "world_dof_start", "row_type", "row_parent",
            "mf_constraint_count", "mf_meta"}
ORDER = ["constraint_count", "world_dof_start", "dense_rhs", "diag", "impulses",
         "J_world", "Y_world", "row_type", "row_parent", "row_mu",
         "drive_target_vel_bias", "drive_vel_multiplier", "drive_impulse_multiplier", "drive_max_impulse",
         "mf_constraint_count", "mf_meta", "mf_impulses", "mf_J_a", "mf_J_b",
         "mf_MiJt_a", "mf_MiJt_b", "mf_row_mu"]
SCALARS = [ITERS, OMEGA, ROW_PHASE, FRIC_START, ITER_OFF, FREEZE]

warp_kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode="current")


def kname(k):
    return getattr(k, "key", None) or getattr(k, "__name__", "?")


def kmod(k):
    m = getattr(k, "module", None)
    return getattr(m, "name", None)


def tile_idx(N):
    return (np.arange(N) % WBASE)


def build_warp(N):
    idx = tile_idx(N)
    a = {}
    for k in z.files:
        if k == "v_out":
            starts = z["world_dof_start"]
            chunks = [z["v_out"][int(starts[i]):int(starts[i]) + D] for i in idx]
            arr = np.ascontiguousarray(np.concatenate(chunks).astype(np.float32))
            a[k] = wp.array(arr, dtype=wp.float32, device=DEV)
        elif k == "world_dof_start":
            a[k] = wp.array(np.ascontiguousarray((np.arange(N) * D).astype(np.int32)), dtype=wp.int32, device=DEV)
        else:
            arr = z[k][idx]
            a[k] = wp.array(np.ascontiguousarray(arr),
                            dtype=(wp.int32 if k in INT_KEYS else wp.float32), device=DEV)
    return a


def build_tpw(N):
    idx = tile_idx(N)
    m_dense = z["constraint_count"][idx].astype(np.int32)
    m_mf = z["mf_constraint_count"][idx].astype(np.int32)
    w_dof_start = np.zeros(N, dtype=np.int32)

    def dT(key, dt):
        return np.ascontiguousarray(z[key][idx].T.astype(dt)).reshape(-1)
    J = np.ascontiguousarray(np.transpose(z["J_world"][idx], (1, 2, 0)).astype(np.float32)).reshape(-1)
    Y = np.ascontiguousarray(np.transpose(z["Y_world"][idx], (1, 2, 0)).astype(np.float32)).reshape(-1)
    meta4 = z["mf_meta"][idx].reshape(N, M_MF, 4)
    mf_meta = np.ascontiguousarray(np.transpose(meta4, (1, 2, 0)).astype(np.int32)).reshape(-1)

    def mf6(key):
        return np.ascontiguousarray(np.transpose(z[key][idx], (1, 2, 0)).astype(np.float32)).reshape(-1)
    starts = z["world_dof_start"]
    v0 = np.stack([z["v_out"][int(starts[i]):int(starts[i]) + D] for i in idx]).astype(np.float32)
    v_g = np.ascontiguousarray(v0.T).reshape(-1)

    g = lambda a, dt: wp.array(a, dtype=dt, device=DEV)
    arrs = [
        g(m_dense, wp.int32), g(m_mf, wp.int32), g(w_dof_start, wp.int32),
        g(dT("impulses", np.float32), wp.float32), g(dT("dense_rhs", np.float32), wp.float32),
        g(dT("diag", np.float32), wp.float32), g(dT("row_type", np.int32), wp.int32),
        g(dT("row_parent", np.int32), wp.int32), g(dT("row_mu", np.float32), wp.float32),
        g(dT("drive_target_vel_bias", np.float32), wp.float32), g(dT("drive_vel_multiplier", np.float32), wp.float32),
        g(dT("drive_impulse_multiplier", np.float32), wp.float32), g(dT("drive_max_impulse", np.float32), wp.float32),
        g(J, wp.float32), g(Y, wp.float32),
        g(mf_meta, wp.int32), g(dT("mf_impulses", np.float32), wp.float32),
        g(mf6("mf_J_a"), wp.float32), g(mf6("mf_J_b"), wp.float32),
        g(mf6("mf_MiJt_a"), wp.float32), g(mf6("mf_MiJt_b"), wp.float32),
        g(dT("mf_row_mu", np.float32), wp.float32),
        g(v_g, wp.float32),
    ]
    return arrs + [ITERS, OMEGA, FRIC_START, ITER_OFF, FREEZE], arrs[-1]  # also return v_g array for fidelity


def median_ms(launch, reps=20, warmup=5):
    for _ in range(warmup):
        launch()
    wp.synchronize_device()
    ts = []
    for _ in range(reps):
        with wp.ScopedTimer("k", cuda_filter=wp.TIMING_KERNEL, print=False, synchronize=True) as t:
            launch()
        ms = sum(r.elapsed for r in t.timing_results) if t.timing_results else float("nan")
        ts.append(ms)
    ts = [x for x in ts if x == x]
    ts.sort()
    return ts[len(ts) // 2]


def peak_mm_from_v(v_nd, N):
    BETA = 0.2
    DT = 0.005
    peak = 0.0
    for w in range(N):
        wi = w % WBASE
        M = int(z["mf_constraint_count"][wi])
        meta_w = z["mf_meta"][wi].reshape(-1, 4)[:M].astype(np.int64)
        packed = meta_w[:, 0].astype(np.int32)
        dof_a = (packed >> 16).astype(np.int64)
        dof_b = ((packed << 16) >> 16).astype(np.int64)
        bias = meta_w[:, 2].astype(np.int32).view(np.float32).astype(np.float64)
        rt = (meta_w[:, 3] & 0xFFFF).astype(np.int64)
        Ja = z["mf_J_a"][wi, :M].astype(np.float64)
        Jb = z["mf_J_b"][wi, :M].astype(np.float64)
        v = v_nd[w].astype(np.float64)
        for i in range(M):
            if rt[i] != 0:
                continue
            da, db = dof_a[i], dof_b[i]
            jv = 0.0
            if da >= 0:
                jv += np.dot(Ja[i], v[da:da + 6])
            if db >= 0:
                jv += np.dot(Jb[i], v[db:db + 6])
            residual = jv + bias[i]
            v_res = max(0.0, -residual)
            phi_now = bias[i] * DT / BETA
            phi_next = phi_now - v_res * DT
            peak = max(peak, max(0.0, -phi_next) * 1000.0)
    return peak


def fidelity(N=256):
    # warp
    aw = build_warp(N)
    win = [aw[k] for k in ORDER] + SCALARS
    wp.launch_tiled(warp_kernel, dim=[N], inputs=win, outputs=[aw["v_out"]], block_dim=BD_WARP, device=DEV)
    v_warp_flat = aw["v_out"].numpy().copy()
    starts = (np.arange(N) * D).astype(np.int64)
    v_warp_nd = np.stack([v_warp_flat[int(s):int(s) + D] for s in starts])
    del aw, win

    # tpw
    tpw_kernel = _get_pgs_solve_mf_gs_kernel_tpw(M_D, M_MF, D, N, arch, friction_mode="current")
    tin, v_g_arr = build_tpw(N)
    wp.launch(tpw_kernel, dim=N, inputs=tin, block_dim=int(os.environ.get("TPW_BD", "256")), device=DEV)
    v_tpw_nd = v_g_arr.numpy().reshape(D, N).T

    diff = np.abs(v_tpw_nd.astype(np.float64) - v_warp_nd.astype(np.float64))
    max_abs = float(diff.max())
    pm_warp = peak_mm_from_v(v_warp_nd, N)
    pm_tpw = peak_mm_from_v(v_tpw_nd, N)
    del tin
    return max_abs, pm_warp, pm_tpw, kname(tpw_kernel), kmod(tpw_kernel)


if __name__ == "__main__":
    Ns = [int(x) for x in os.environ.get("TPW_NS", "256,1024,4096,8192,16384").split(",")]
    bd = int(os.environ.get("TPW_BD", "256"))

    print("==================== TPW BLACKWELL BENCH ====================", flush=True)
    print(f"newton.__file__ = {_n.__file__}", flush=True)
    print(f"GPU device      = {dev.name}", flush=True)
    print(f"GPU arch        = sm_{arch}", flush=True)
    print(f"WARP kernel key = {kname(warp_kernel)}   module={kmod(warp_kernel)}", flush=True)
    print(f"cap meta        = M_D={M_D} M_MF={M_MF} D={D} iters={ITERS} omega={OMEGA} "
          f"row_phase={ROW_PHASE} warp_bd={BD_WARP} tpw_bd={bd} WBASE={WBASE}", flush=True)
    print(f"cap device_arch (capture) = {meta.get('device_arch')}", flush=True)

    # prove tpw kernel name BEFORE the sweep
    probe_tpw = _get_pgs_solve_mf_gs_kernel_tpw(M_D, M_MF, D, 256, arch, friction_mode="current")
    print(f"TPW kernel key  = {kname(probe_tpw)}   module={kmod(probe_tpw)}", flush=True)
    assert "_tpw_" in kname(probe_tpw), "TPW kernel name does not contain _tpw_ -- WRONG KERNEL"
    assert "_tpw_" not in kname(warp_kernel), "WARP kernel name contains _tpw_ -- mislabeled"
    print("PROOF: tpw kernel name contains '_tpw_'; warp kernel name does not.", flush=True)
    print("============================================================", flush=True)

    print(f"{'N':>7} {'warp_ms':>10} {'tpw_ms':>10} {'warp/wld_us':>12} {'tpw/wld_us':>11} {'tpw_speedup':>12}", flush=True)
    for N in Ns:
        aw = build_warp(N)
        win = [aw[k] for k in ORDER] + SCALARS
        warp_launch = lambda: wp.launch_tiled(warp_kernel, dim=[N], inputs=win, outputs=[aw["v_out"]], block_dim=BD_WARP, device=DEV)
        tw = median_ms(warp_launch)
        del aw, win

        tpw_kernel = _get_pgs_solve_mf_gs_kernel_tpw(M_D, M_MF, D, N, arch, friction_mode="current")
        tin, _ = build_tpw(N)
        tpw_launch = lambda: wp.launch(tpw_kernel, dim=N, inputs=tin, block_dim=bd, device=DEV)
        tt = median_ms(tpw_launch)
        del tin

        print(f"{N:>7} {tw:>10.4f} {tt:>10.4f} {tw / N * 1e3:>12.3f} {tt / N * 1e3:>11.3f} {tw / tt:>12.3f}", flush=True)

    print("------------------------------------------------------------", flush=True)
    ma, pmw, pmt, tk, tm = fidelity(256)
    print(f"[fidelity N=256] max|tpw-warp|={ma:.3e}  peak_pen_mm warp={pmw:.4f}  tpw={pmt:.4f}", flush=True)
    print(f"[fidelity] tpw kernel ran = {tk}  module={tm}", flush=True)
    print("==================== DONE ====================", flush=True)
