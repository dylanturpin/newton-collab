"""Phase-1 bit-identity bitcheck for FPGS colored MF Gauss-Seidel.

Runs the capture /tmp/cap892.npz through BOTH the serial MF kernel and the
colored CSR kernel at the degenerate setting n_colors=1, W=1 (single color
listing every row in original slot order), then compares v_out / impulses /
mf_impulses via np.array_equal. PASS == all three bit-identical.

Imports the WORKTREE solver (sys.path.insert ahead of the installed newton);
prints newton.__file__ to prove it. warp comes from the active venv.
"""

import json
import os
import sys

# ── Force the WORKTREE newton ahead of any installed/editable newton. ──
WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/wf_453cc222-038-1/newton-collab"
sys.path.insert(0, WORKTREE)

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

print(f"[bitcheck] newton.__file__ = {newton.__file__}", flush=True)
assert newton.__file__.startswith(WORKTREE), f"NOT importing worktree newton: {newton.__file__}"

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)

N = int(os.environ.get("NWORLDS", "16"))
DEV = "cuda:0"
if os.environ.get("WARP_CACHE_PATH"):
    wp.config.kernel_cache_dir = os.environ["WARP_CACHE_PATH"]
wp.init()

meta = json.load(open("/tmp/cap892.json"))
z = np.load("/tmp/cap892.npz")
M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
arch = int(wp.get_device(DEV).arch)  # local arch so it compiles here

INT_KEYS = {"constraint_count", "world_dof_start", "row_type", "row_parent", "mf_constraint_count", "mf_meta"}
ORDER = [
    "constraint_count", "world_dof_start", "dense_rhs", "diag", "impulses",
    "J_world", "Y_world", "row_type", "row_parent", "row_mu",
    "drive_target_vel_bias", "drive_vel_multiplier", "drive_impulse_multiplier", "drive_max_impulse",
    "mf_constraint_count", "mf_meta", "mf_impulses", "mf_J_a", "mf_J_b",
    "mf_MiJt_a", "mf_MiJt_b", "mf_row_mu",
]
SCALARS = [
    int(meta["iterations"]), float(meta["omega"]), int(meta["row_phase"]),
    int(meta["friction_start_iteration"]), int(meta["iteration_offset"]), int(meta["freeze_drive_rows"]),
]
MAX_COLORS = int(os.environ.get("FEATHER_PGS_MFGS_MAX_COLORS", "16"))


def sl(k):
    a = z[k]
    return a[: N * D] if k == "v_out" else a[:N]


def make_inputs():
    """Fresh device arrays from the capture (so each run starts from identical state)."""
    return {
        k: wp.array(
            np.ascontiguousarray(sl(k)),
            dtype=(wp.int32 if k in INT_KEYS else wp.float32),
            device=DEV,
        )
        for k in z.files
    }


def build_degenerate_csr(mf_constraint_count_host):
    """Single color per world, listing rows 0..count-1 in slot order.

    n_colors=1; color_offsets[world] = [0, count, 0, 0, ...] width MAX_COLORS+1;
    color_rows[world] = [0,1,...,M_MF-1] (only the first `count` are ever read).
    `count` is clamped to M_MF exactly as the kernel clamps m_mf, so the colored
    loop visits the identical row sequence as the serial `for i in 0..m_mf`.
    """
    counts = np.minimum(mf_constraint_count_host[:N].astype(np.int32), M_MF)
    n_colors = np.ones(N, dtype=np.int32)
    offsets = np.zeros((N, MAX_COLORS + 1), dtype=np.int32)
    offsets[:, 1:] = counts[:, None]  # offsets[w] = [0, count, count, count, ...]
    rows = np.tile(np.arange(M_MF, dtype=np.int32), (N, 1))  # [w] = 0..M_MF-1
    # Kernel declares these as flat 1-D arrays indexed by off_co = world*(MAX_COLORS+1)
    # and off_cr = world*M_MF, so flatten the per-world rows here.
    return (
        wp.array(np.ascontiguousarray(offsets.reshape(-1)), dtype=wp.int32, device=DEV),
        wp.array(np.ascontiguousarray(rows.reshape(-1)), dtype=wp.int32, device=DEV),
        wp.array(np.ascontiguousarray(n_colors), dtype=wp.int32, device=DEV),
    )


def run_serial():
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "0"
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])
    a = make_inputs()
    inputs = [a[k] for k in ORDER] + SCALARS
    wp.launch_tiled(kernel, dim=[N], inputs=inputs, outputs=[a["v_out"]],
                    block_dim=int(meta["block_dim"]), device=DEV)
    return {
        "v_out": a["v_out"].numpy(),
        "impulses": a["impulses"].numpy(),
        "mf_impulses": a["mf_impulses"].numpy(),
    }


def run_colored():
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    # _get_pgs_solve_mf_gs_kernel is @cache'd on its args (which exclude the env
    # flag), so clear it to force a rebuild that reads the now-ON env flag.
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])
    a = make_inputs()
    csr_off, csr_rows, csr_nc = build_degenerate_csr(z["mf_constraint_count"])
    # colored signature inserts the 3 CSR arrays right after mf_row_mu.
    mf_row_mu_idx = ORDER.index("mf_row_mu")
    inputs = [a[k] for k in ORDER[: mf_row_mu_idx + 1]]
    inputs += [csr_off, csr_rows, csr_nc]
    inputs += [a[k] for k in ORDER[mf_row_mu_idx + 1:]]
    inputs += SCALARS
    wp.launch_tiled(kernel, dim=[N], inputs=inputs, outputs=[a["v_out"]],
                    block_dim=int(meta["block_dim"]), device=DEV)
    return {
        "v_out": a["v_out"].numpy(),
        "impulses": a["impulses"].numpy(),
        "mf_impulses": a["mf_impulses"].numpy(),
    }


serial = run_serial()
colored = run_colored()

print(f"[bitcheck] N={N} arch={arch} MAX_COLORS={MAX_COLORS} (colored n_colors=1, W=1)", flush=True)
all_eq = True
for key in ("v_out", "impulses", "mf_impulses"):
    s, c = serial[key], colored[key]
    eq = np.array_equal(s, c)
    all_eq &= eq
    if eq:
        print(f"[bitcheck] {key:12s}: np.array_equal = True", flush=True)
    else:
        diff = np.abs(s.astype(np.float64) - c.astype(np.float64))
        nbad = int((s != c).sum())
        print(
            f"[bitcheck] {key:12s}: np.array_equal = False  max_abs_diff={diff.max():.6e}  "
            f"n_differing={nbad}/{s.size}  first_bad={np.argwhere(s != c)[:3].ravel().tolist()}",
            flush=True,
        )

print(f"[bitcheck] BIT_IDENTICAL_ALL_THREE = {all_eq}", flush=True)
sys.exit(0 if all_eq else 1)
