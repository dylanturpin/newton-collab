"""NCU driver: profile ONLY the SERIAL warp-per-world FPGS contact-solve kernel
(pgs_solve_mf_gs, the default/non-tpw path) at a SATURATED tiled world count.

This script launches NOTHING but the serial kernel `_get_pgs_solve_mf_gs_kernel`
via wp.launch_tiled. It does:
  1. warmup launches (NWARM) so the steady kernel is compiled + caches warm
  2. exactly NPROF profiled launches (default a few) of the same serial kernel

So under ncu, with --kernel-name regex:pgs_solve_mf_gs and a --launch-skip equal
to NWARM, ncu captures a WARM steady launch (not the cold first one). It also
prints the kernel .key (must contain `_current` and NOT `_tpw_`), module name,
GPU device name and sm arch -- proving the SERIAL kernel on sm_120 Blackwell.

Env:
  TPW_REPO        cloned il-newton-dev checkout (contains newton/)
  TPW_CAP_NPZ     cap892 npz (compressed or not)
  TPW_CAP_JSON    cap892 json meta
  NCU_N           tiled world count (default 8192 -- saturated)
  NCU_NWARM       warmup launches before the profiled launch (default 6)
  NCU_NPROF       profiled launches (default 3)
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


if __name__ == "__main__":
    N = int(os.environ.get("NCU_N", "8192"))
    NWARM = int(os.environ.get("NCU_NWARM", "6"))
    NPROF = int(os.environ.get("NCU_NPROF", "3"))

    print("==================== NCU SERIAL FPGS REPLAY ====================", flush=True)
    print(f"newton.__file__ = {_n.__file__}", flush=True)
    print(f"GPU device      = {dev.name}", flush=True)
    print(f"GPU arch        = sm_{arch}", flush=True)
    print(f"SERIAL kernel key = {kname(warp_kernel)}   module={kmod(warp_kernel)}", flush=True)
    assert "_tpw_" not in kname(warp_kernel), "kernel name contains _tpw_ -- WRONG (this is the tpw kernel)"
    print("PROOF: serial kernel name does NOT contain '_tpw_' (warp-per-world / serial baseline).", flush=True)
    print(f"cap meta        = M_D={M_D} M_MF={M_MF} D={D} iters={ITERS} omega={OMEGA} "
          f"row_phase={ROW_PHASE} warp_bd={BD_WARP} WBASE={WBASE}", flush=True)
    print(f"PROFILE N (tiled worlds) = {N}   NWARM={NWARM}  NPROF={NPROF}", flush=True)
    print("===============================================================", flush=True)

    aw = build_warp(N)
    win = [aw[k] for k in ORDER] + SCALARS

    def launch():
        wp.launch_tiled(warp_kernel, dim=[N], inputs=win, outputs=[aw["v_out"]],
                        block_dim=BD_WARP, device=DEV)

    # Warmup launches (ncu --launch-skip NWARM skips these; the profiled one is warm/steady).
    for i in range(NWARM):
        launch()
    wp.synchronize_device()
    print(f"[warmup done] {NWARM} launches at N={N}", flush=True)

    # Profiled launches.
    for i in range(NPROF):
        launch()
    wp.synchronize_device()
    print(f"[profiled done] {NPROF} launches at N={N}", flush=True)
    print("==================== NCU REPLAY DONE ====================", flush=True)
