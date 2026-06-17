"""M1 (real coloring) + M2 (W=1 fidelity) for FPGS colored MF Gauss-Seidel.

M1: Build the contact-triple conflict graph per world from the LIVE mf rows
    (mf_meta packed dof bases + mf_row_parent rt grouping). Color it with
    Newton's host color_graph (MCS+balance) as the Phase-0 oracle. Pack the
    CSR (mf_color_offsets / mf_color_rows / mf_n_colors) keeping each contact
    triple's 3 rows CONSECUTIVE (normal,t1,t2) inside ONE color. Assert
    COLORING VALIDITY in numpy: no color holds two nodes sharing a body.

M2: Run the colored kernel (multi-color, W=1) on /tmp/cap892.npz. Compare to
    the serial kernel. Colored order is a different valid GS variant (NOT
    bit-identical) -> measure PEAK INTERPENETRATION (mm) of colored vs serial
    via tau_test's projection, confirm colored <= 8mm and close to serial.

W (warps/world) read from MF_COLOR_WARPS (default 1). block_dim = 32*W.

Imports the WORKTREE newton (asserts newton.__file__).
"""

import json
import os
import sys
import collections

WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/wf_453cc222-038-1/newton-collab"
sys.path.insert(0, WORKTREE)
sys.path.insert(0, "/tmp")  # for tau_test projection

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import newton  # noqa: E402

assert newton.__file__.startswith(WORKTREE), f"NOT worktree newton: {newton.__file__}"
print(f"[m1m2] newton.__file__ = {newton.__file__}", flush=True)

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)
from newton._src.sim.graph_coloring import color_graph  # noqa: E402
import tau_test as tt  # noqa: E402  (numpy replay + mm_metrics + sweep)

N = int(os.environ.get("NWORLDS", "256"))
W = int(os.environ.get("MF_COLOR_WARPS", "1"))
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
MAX_COLORS = int(os.environ.get("FEATHER_PGS_MFGS_MAX_COLORS", "16"))

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
    return {
        k: wp.array(np.ascontiguousarray(sl(k)),
                    dtype=(wp.int32 if k in INT_KEYS else wp.float32), device=DEV)
        for k in z.files
    }


# ════════════════════════════════════════════════════════════════════
# M1: per-world contact-triple conflict graph + color_graph oracle CSR.
# ════════════════════════════════════════════════════════════════════
def unpack_world(w):
    """Decode mf_meta for world w into per-row dof_a, dof_b, rt, parent."""
    Mw = int(z["mf_constraint_count"][w])
    Mw = min(Mw, M_MF)
    meta_w = z["mf_meta"][w].reshape(-1, 4)[:Mw].astype(np.int64)
    packed = meta_w[:, 0].astype(np.int32)
    dof_a = (packed >> 16).astype(np.int64)
    dof_b = ((packed << 16) >> 16).astype(np.int64)  # sign-extended low16
    rt = (meta_w[:, 3] & 0xFFFF).astype(np.int64)
    parent = (meta_w[:, 3] >> 16).astype(np.int64)
    return Mw, dof_a, dof_b, rt, parent


def build_nodes(Mw, dof_a, dof_b, rt, parent):
    """A node = a contact triple keyed by normal rt0 row + its rt2 children,
    OR a standalone rt0 normal. rt4 excluded. Returns:
      nodes: list of (rep_row, sorted member_rows, set_of_body_dofbases)
    member_rows kept in (normal, t1, t2) order = consecutive slot order.
    """
    children = collections.defaultdict(list)
    for i in range(Mw):
        if rt[i] == 2:
            children[int(parent[i])].append(i)
    nodes = []
    for i in range(Mw):
        if rt[i] != 0:
            continue
        rows = [i] + sorted(children[i])  # normal first, frictions ascending
        bodies = set()
        for r in rows:
            if dof_a[r] >= 0:
                bodies.add(int(dof_a[r]))
            if dof_b[r] >= 0:
                bodies.add(int(dof_b[r]))
        nodes.append((i, rows, bodies))
    return nodes


def build_edges(nodes):
    """Two nodes conflict iff they share a body dof-base."""
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
    return edges


def color_world(nodes):
    """Use Newton host color_graph (MCS + balance). Returns color-per-node."""
    n = len(nodes)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0
    edges = build_edges(nodes)
    if not edges:
        # No conflicts: color_graph needs >=1 edge to run sensibly; all color 0.
        return np.zeros(n, dtype=np.int32), 1
    edge_arr = wp.array(np.array(sorted(edges), dtype=np.int32), dtype=wp.int32, device="cpu")
    groups = color_graph(n, edge_arr, balance_colors=True)
    color = np.full(n, -1, dtype=np.int32)
    for c, grp in enumerate(groups):
        g = np.asarray(grp)
        color[g] = c
    assert (color >= 0).all(), "color_graph left an uncolored node"
    return color, len(groups)


def pack_csr_world(nodes, color, ncolors):
    """Build per-world CSR row list grouped by color, triples consecutive.
    Returns (offsets length MAX_COLORS+1, rows length M_MF, n_colors)."""
    by_color = collections.defaultdict(list)
    for ni, c in enumerate(color):
        by_color[int(c)].append(ni)
    offsets = np.zeros(MAX_COLORS + 1, dtype=np.int32)
    rows = np.zeros(M_MF, dtype=np.int32)
    pos = 0
    nc = 0
    for c in range(ncolors):
        offsets[c] = pos
        for ni in by_color[c]:
            for r in nodes[ni][1]:  # member rows, normal first
                rows[pos] = r
                pos += 1
        nc += 1
    # fill remaining offsets to point at pos (empty colors)
    for c in range(nc, MAX_COLORS + 1):
        offsets[c] = pos
    return offsets, rows, nc


def assert_coloring_valid(nodes, color):
    """COLORING VALIDITY: no color holds two nodes sharing a body dof-base."""
    by_color = collections.defaultdict(list)
    for ni, c in enumerate(color):
        by_color[int(c)].append(ni)
    for c, members in by_color.items():
        seen = {}
        for ni in members:
            for b in nodes[ni][2]:
                if b in seen:
                    return False, (c, seen[b], ni, b)
                seen[b] = ni
    return True, None


def assert_triple_integrity(nodes, color, offsets, rows, nc, Mw, rt):
    """Each triple's rows consecutive in (normal,t1,t2) order inside one color."""
    # Reconstruct emitted order and check each node's rows are a consecutive run.
    # Build a map row -> node membership index.
    for ni, (rep, member_rows, _) in enumerate(nodes):
        # find rep in rows within its color span
        c = int(color[ni])
        span = rows[offsets[c]:offsets[c + 1]]
        # locate the block equal to member_rows
        idx = np.where(span == rep)[0]
        ok = False
        for j in idx:
            if j + len(member_rows) <= len(span) and list(span[j:j + len(member_rows)]) == member_rows:
                ok = True
                break
        if not ok:
            return False, (ni, rep, member_rows)
    return True, None


# Build CSR for all N worlds + run M1 assertions.
all_off = np.zeros((N, MAX_COLORS + 1), dtype=np.int32)
all_rows = np.zeros((N, M_MF), dtype=np.int32)
all_nc = np.zeros(N, dtype=np.int32)
m_over_c = []
ncolors_list = []
valid_all = True
triple_all = True
node_counts = []
for w in range(N):
    Mw, dof_a, dof_b, rt, parent = unpack_world(w)
    nodes = build_nodes(Mw, dof_a, dof_b, rt, parent)
    color, ncolors = color_world(nodes)
    if ncolors > MAX_COLORS:
        print(f"[m1m2] world {w}: ncolors {ncolors} > MAX_COLORS {MAX_COLORS} -- "
              f"would fall back to serial in production", flush=True)
    offsets, rows, nc = pack_csr_world(nodes, color, ncolors)
    ok, info = assert_coloring_valid(nodes, color)
    if not ok:
        valid_all = False
        print(f"[m1m2] world {w}: INVALID coloring {info}", flush=True)
    tok, tinfo = assert_triple_integrity(nodes, color, offsets, rows, nc, Mw, rt)
    if not tok:
        triple_all = False
        print(f"[m1m2] world {w}: TRIPLE INTEGRITY violated {tinfo}", flush=True)
    all_off[w] = offsets
    all_rows[w] = rows
    all_nc[w] = nc
    node_counts.append(len(nodes))
    ncolors_list.append(ncolors if ncolors > 0 else 1)
    if ncolors > 0:
        m_over_c.append(len(nodes) / ncolors)

print(f"\n[m1m2] === M1 REAL COLORING ===", flush=True)
print(f"[m1m2] N={N} worlds  nodes(min/med/max)="
      f"{min(node_counts)}/{int(np.median(node_counts))}/{max(node_counts)}", flush=True)
print(f"[m1m2] n_colors(min/med/max)="
      f"{min(ncolors_list)}/{int(np.median(ncolors_list))}/{max(ncolors_list)}", flush=True)
if m_over_c:
    print(f"[m1m2] m/c (avg parallel width) mean={np.mean(m_over_c):.1f} "
          f"min={min(m_over_c):.1f} max={max(m_over_c):.1f}", flush=True)
print(f"[m1m2] COLORING_VALIDITY (no color shares a body): {'PASS' if valid_all else 'FAIL'}", flush=True)
print(f"[m1m2] TRIPLE_INTEGRITY (3 rows consecutive, normal-first): {'PASS' if triple_all else 'FAIL'}", flush=True)


def build_trivial_csr():
    counts = np.minimum(z["mf_constraint_count"][:N].astype(np.int32), M_MF)
    n_colors = np.ones(N, dtype=np.int32)
    offsets = np.zeros((N, MAX_COLORS + 1), dtype=np.int32)
    offsets[:, 1:] = counts[:, None]
    rows = np.tile(np.arange(M_MF, dtype=np.int32), (N, 1))
    return offsets, rows, n_colors


def to_dev(off, rows, nc):
    return (
        wp.array(np.ascontiguousarray(off.reshape(-1)), dtype=wp.int32, device=DEV),
        wp.array(np.ascontiguousarray(rows.reshape(-1)), dtype=wp.int32, device=DEV),
        wp.array(np.ascontiguousarray(nc), dtype=wp.int32, device=DEV),
    )


def run_serial():
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "0"
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])
    a = make_inputs()
    inputs = [a[k] for k in ORDER] + SCALARS
    wp.launch_tiled(kernel, dim=[N], inputs=inputs, outputs=[a["v_out"]],
                    block_dim=32, device=DEV)
    return a["v_out"].numpy(), a["mf_impulses"].numpy()


def run_colored(off, rows, nc, warps):
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    os.environ["FEATHER_PGS_MFGS_WARPS"] = str(warps)
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=meta["friction_mode"])
    a = make_inputs()
    csr_off, csr_rows, csr_nc = to_dev(off, rows, nc)
    mu_idx = ORDER.index("mf_row_mu")
    inputs = [a[k] for k in ORDER[: mu_idx + 1]]
    inputs += [csr_off, csr_rows, csr_nc]
    inputs += [a[k] for k in ORDER[mu_idx + 1:]]
    inputs += SCALARS
    wp.launch_tiled(kernel, dim=[N], inputs=inputs, outputs=[a["v_out"]],
                    block_dim=32 * warps, device=DEV)
    return a["v_out"].numpy(), a["mf_impulses"].numpy()


# ════════════════════════════════════════════════════════════════════
# M2: fidelity. Peak interpenetration (mm) via tau_test projection on the
# kernel's v_out (and mf_impulses as lambda). Compare serial vs colored.
# ════════════════════════════════════════════════════════════════════
def peak_pen_from_vout(v_out_flat, mf_imp):
    """For each world, use tau_test.mm_metrics on the kernel's solved v_out
    and lambda. v_out_flat is the flat [N*D] array; mf_imp is [N, M_MF]."""
    peaks = []
    rms = []
    for w in range(N):
        Wd = tt.load_world(w)  # loads cap, gives M, dof_a/b, diag, bias, rt, ...
        s = int(z["world_dof_start"][w])
        v = v_out_flat[s:s + D].astype(np.float64).copy()
        lam = mf_imp[w, :Wd["M"]].astype(np.float64).copy()
        m = tt.mm_metrics(Wd, v, lam)
        peaks.append(m["peak_pen_mm"])
        rms.append(m["rms_pen_mm"])
    return np.array(peaks), np.array(rms)


print(f"\n[m1m2] === M2 W=1 FIDELITY (W={W}) ===", flush=True)
v_serial, imp_serial = run_serial()
v_colored, imp_colored = run_colored(all_off, all_rows, all_nc, W)

# sanity: trivial CSR colored at W=1 must equal serial bit-for-bit
if W == 1:
    t_off, t_rows, t_nc = build_trivial_csr()
    v_triv, imp_triv = run_colored(t_off, t_rows, t_nc, 1)
    bit_ok = np.array_equal(v_serial, v_triv) and np.array_equal(imp_serial, imp_triv)
    print(f"[m1m2] trivial-CSR W=1 bit-identical to serial: {bit_ok}", flush=True)

pk_s, rms_s = peak_pen_from_vout(v_serial, imp_serial)
pk_c, rms_c = peak_pen_from_vout(v_colored, imp_colored)

print(f"[m1m2] SERIAL  peak_pen_mm: max={pk_s.max():.4f} mean={pk_s.mean():.4f} "
      f"rms_pen_mm.mean={rms_s.mean():.4f}", flush=True)
print(f"[m1m2] COLORED peak_pen_mm: max={pk_c.max():.4f} mean={pk_c.mean():.4f} "
      f"rms_pen_mm.mean={rms_c.mean():.4f}", flush=True)
print(f"[m1m2] colored-vs-serial peak_pen_mm diff: max|Δ|={np.abs(pk_c-pk_s).max():.4f}", flush=True)
holds = bool(pk_c.max() <= 8.0)
print(f"[m1m2] COLORED holds <=8mm: {'PASS' if holds else 'FAIL'} "
      f"(colored peak={pk_c.max():.4f} mm)", flush=True)
print(f"[m1m2] DONE", flush=True)
