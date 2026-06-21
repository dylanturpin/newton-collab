"""Blackwell kpw-vs-serial DECISIVE bench.

CUDA-event-timed solve-kernel comparison of the SERIAL warp-per-world FPGS
contact-solve kernel against the K-threads-per-world (kpw) colored kernel at
K in {1,4,8}, for tiled world counts N in {4096,8192,16384}, on a single
Blackwell GPU (sm_120). Build + per-world contact-coloring CSR are constructed
OUTSIDE the timed region; only kernel launches are timed (CUDA-event median of
~20 launches, 5 warmups). Fidelity is the peak interpenetration (mm) over all
rt0 contact rows from the solved velocity field (must stay <= 8 mm and match
serial within ~0.01 mm).

Env:
  TPW_REPO        cloned newton-collab checkout (contains newton/)
  TPW_CAP_NPZ     cap892 npz (compressed cap892_c.npz works -- numpy reads it)
  TPW_CAP_JSON    cap892 json meta
  KPW_NS          comma N list (default 4096,8192,16384)
  KPW_KS          comma K list (default 1,4,8)
  KPW_REPS        timed launches (default 20)
  WARP_CACHE_PATH warp kernel cache dir
"""
import json
import os
import sys

REPO = os.environ.get("TPW_REPO", "/home/dturpin/repos/il-newton-dev/.claude/worktrees/kpw")
sys.path.insert(0, REPO)

import numpy as np
import warp as wp
import newton as _n

assert REPO in _n.__file__, f"wrong newton: {_n.__file__} (expected under {REPO})"

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_mf_gs_kernel_kpw,
)
from newton._src.sim.graph_coloring import color_graph  # noqa: E402

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
MAX_COLORS = 64
assert meta["friction_mode"] == "current"

INT_KEYS = {"constraint_count", "world_dof_start", "row_type", "row_parent",
            "mf_constraint_count", "mf_meta"}
ORDER = ["constraint_count", "world_dof_start", "dense_rhs", "diag", "impulses",
         "J_world", "Y_world", "row_type", "row_parent", "row_mu",
         "drive_target_vel_bias", "drive_vel_multiplier", "drive_impulse_multiplier", "drive_max_impulse",
         "mf_constraint_count", "mf_meta", "mf_impulses", "mf_J_a", "mf_J_b",
         "mf_MiJt_a", "mf_MiJt_b", "mf_row_mu"]
SCALARS = [ITERS, OMEGA, ROW_PHASE, FRIC_START, ITER_OFF, FREEZE]

BETA = 0.2
DT = 0.005


def kname(k):
    return getattr(k, "key", None) or getattr(k, "__name__", "?")


def tile_idx(N):
    return (np.arange(N) % WBASE)


# ── SERIAL (natural per-world) input build ──────────────────────────────────
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


# ── KPW (world-innermost) input build ───────────────────────────────────────
def build_world_innermost(N):
    idx = tile_idx(N)
    out = {}
    out["m_dense"] = z["constraint_count"][idx].astype(np.int32)
    out["m_mf"] = z["mf_constraint_count"][idx].astype(np.int32)
    out["w_dof_start"] = np.zeros(N, dtype=np.int32)

    def dT(key, dt):
        return np.ascontiguousarray(z[key][idx].T.astype(dt)).reshape(-1)
    out["lam"] = dT("impulses", np.float32)
    out["rhs"] = dT("dense_rhs", np.float32)
    out["diag"] = dT("diag", np.float32)
    out["rtype"] = dT("row_type", np.int32)
    out["parent"] = dT("row_parent", np.int32)
    out["mu"] = dT("row_mu", np.float32)
    out["dtarget"] = dT("drive_target_vel_bias", np.float32)
    out["dvmul"] = dT("drive_vel_multiplier", np.float32)
    out["dimul"] = dT("drive_impulse_multiplier", np.float32)
    out["dmaximp"] = dT("drive_max_impulse", np.float32)
    # J/Y are 2-D (M_D*D, N): flat length M_D*D*N exceeds Warp's 2^31 per-dim
    # array limit for N >= ~8192. Row-major (M_D*D, N) keeps .data[r*N+world]
    # (== the kernel's (i*D+d)*W+world) identical while no single dim > 2^31.
    out["J"] = np.ascontiguousarray(np.transpose(z["J_world"][idx], (1, 2, 0)).astype(np.float32)).reshape(M_D * D, N)
    out["Y"] = np.ascontiguousarray(np.transpose(z["Y_world"][idx], (1, 2, 0)).astype(np.float32)).reshape(M_D * D, N)
    meta4 = z["mf_meta"][idx].reshape(N, M_MF, 4)
    out["mf_meta"] = np.ascontiguousarray(np.transpose(meta4, (1, 2, 0)).astype(np.int32)).reshape(-1)

    def mf6(key):
        return np.ascontiguousarray(np.transpose(z[key][idx], (1, 2, 0)).astype(np.float32)).reshape(-1)
    out["mf_lam"] = dT("mf_impulses", np.float32)
    out["mf_J_a"] = mf6("mf_J_a"); out["mf_J_b"] = mf6("mf_J_b")
    out["mf_MiJt_a"] = mf6("mf_MiJt_a"); out["mf_MiJt_b"] = mf6("mf_MiJt_b")
    out["mf_row_mu"] = dT("mf_row_mu", np.float32)
    starts = z["world_dof_start"]
    v0 = np.stack([z["v_out"][int(starts[i]):int(starts[i]) + D] for i in idx]).astype(np.float32)  # (N,D)
    out["v_g"] = np.ascontiguousarray(v0.T).reshape(-1)
    out["meta4_raw"] = meta4
    out["N"] = N
    return out


def build_coloring(wi, max_colors=MAX_COLORS):
    N = wi["N"]
    counts = wi["m_mf"]
    meta = wi["meta4_raw"].astype(np.int64)
    offsets_host = np.zeros((N, max_colors + 1), dtype=np.int32)
    rows_host = np.zeros((N, M_MF), dtype=np.int32)
    ncolors_host = np.ones((N,), dtype=np.int32)
    per_world = []

    for w in range(N):
        Mw = int(min(int(counts[w]), M_MF))
        if Mw <= 0:
            per_world.append(([], [], np.zeros(0, dtype=np.int32)))
            continue
        meta_w = meta[w, :Mw]
        packed = meta_w[:, 0].astype(np.int32)
        dof_a = (packed >> 16).astype(np.int64)
        dof_b = ((packed << 16) >> 16).astype(np.int64)
        rt = (meta_w[:, 3] & 0xFFFF).astype(np.int64)
        parent = (meta_w[:, 3] >> 16).astype(np.int64)

        children = {}
        for i in range(Mw):
            if rt[i] == 2:
                children.setdefault(int(parent[i]), []).append(i)

        node_rows = []
        node_bodies = []
        for i in range(Mw):
            if rt[i] != 0:
                continue
            rows = [i] + sorted(children.get(i, []))
            bodies = set()
            for r in rows:
                if dof_a[r] >= 0:
                    bodies.add(int(dof_a[r]))
                if dof_b[r] >= 0:
                    bodies.add(int(dof_b[r]))
            node_rows.append(rows)
            node_bodies.append(bodies)

        n = len(node_rows)
        if n == 0:
            per_world.append(([], [], np.zeros(0, dtype=np.int32)))
            continue

        body_to_nodes = {}
        for ni, bodies in enumerate(node_bodies):
            for b in bodies:
                body_to_nodes.setdefault(b, []).append(ni)
        edges = set()
        for ns in body_to_nodes.values():
            for x in range(len(ns)):
                for y in range(x + 1, len(ns)):
                    aa, cc = ns[x], ns[y]
                    edges.add((aa, cc) if aa < cc else (cc, aa))

        if not edges:
            color = np.zeros(n, dtype=np.int32)
            ncolors = 1
        else:
            edge_arr = wp.array(np.array(sorted(edges), dtype=np.int32), dtype=wp.int32, device="cpu")
            groups = color_graph(n, edge_arr, balance_colors=True)
            color = np.full(n, -1, dtype=np.int32)
            for c, grp in enumerate(groups):
                color[np.asarray(grp)] = c
            ncolors = len(groups)

        if ncolors > max_colors:
            rows_host[w, :Mw] = np.arange(Mw, dtype=np.int32)
            offsets_host[w, 1:] = Mw
            ncolors_host[w] = 1
            per_world.append((node_rows, node_bodies, np.zeros(n, dtype=np.int32)))
            continue

        by_color = {}
        for ni, c in enumerate(color):
            by_color.setdefault(int(c), []).append(ni)
        pos = 0
        for c in range(ncolors):
            offsets_host[w, c] = pos
            for ni in by_color.get(c, []):
                for r in node_rows[ni]:
                    rows_host[w, pos] = r
                    pos += 1
        for c in range(ncolors, max_colors + 1):
            offsets_host[w, c] = pos
        ncolors_host[w] = ncolors
        per_world.append((node_rows, node_bodies, color))

    return (np.ascontiguousarray(offsets_host.reshape(-1)),
            np.ascontiguousarray(rows_host.reshape(-1)),
            np.ascontiguousarray(ncolors_host),
            per_world)


def make_warp_inputs(wi):
    g = lambda a, dt: wp.array(a, dtype=dt, device=DEV)
    return dict(
        m_dense=g(wi["m_dense"], wp.int32), m_mf=g(wi["m_mf"], wp.int32),
        w_dof_start=g(wi["w_dof_start"], wp.int32),
        lam=g(wi["lam"], wp.float32), rhs=g(wi["rhs"], wp.float32), diag=g(wi["diag"], wp.float32),
        rtype=g(wi["rtype"], wp.int32), parent=g(wi["parent"], wp.int32), mu=g(wi["mu"], wp.float32),
        dtarget=g(wi["dtarget"], wp.float32), dvmul=g(wi["dvmul"], wp.float32),
        dimul=g(wi["dimul"], wp.float32), dmaximp=g(wi["dmaximp"], wp.float32),
        J=g(wi["J"], wp.float32), Y=g(wi["Y"], wp.float32),
        mf_meta=g(wi["mf_meta"], wp.int32), mf_lam=g(wi["mf_lam"], wp.float32),
        mf_J_a=g(wi["mf_J_a"], wp.float32), mf_J_b=g(wi["mf_J_b"], wp.float32),
        mf_MiJt_a=g(wi["mf_MiJt_a"], wp.float32), mf_MiJt_b=g(wi["mf_MiJt_b"], wp.float32),
        mf_row_mu=g(wi["mf_row_mu"], wp.float32), v_g=g(wi["v_g"], wp.float32),
    )


def kpw_input_list(d, co, cr, nc):
    return [d["m_dense"], d["m_mf"], d["w_dof_start"],
            d["lam"], d["rhs"], d["diag"], d["rtype"], d["parent"], d["mu"],
            d["dtarget"], d["dvmul"], d["dimul"], d["dmaximp"],
            d["J"], d["Y"],
            d["mf_meta"], d["mf_lam"], d["mf_J_a"], d["mf_J_b"], d["mf_MiJt_a"], d["mf_MiJt_b"], d["mf_row_mu"],
            co, cr, nc,
            d["v_g"],
            ITERS, OMEGA, FRIC_START, ITER_OFF, FREEZE]


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


def v_world_innermost_to_NxD(v_flat, N):
    return v_flat.reshape(D, N).T.copy()


def peak_pen_mm_from_solution(v_NxD, N):
    idx = tile_idx(N)
    peak = 0.0
    pens_all = []
    # Compute over the unique tile sources only (N is a multiple of WBASE here,
    # so every distinct source world appears; peak/rms over distinct sources ==
    # peak/rms over all tiled copies since the velocity solution is identical
    # per source).  We still index v_NxD per tiled world for correctness.
    sample = idx if N <= 4096 else np.arange(WBASE)  # distinct sources cover all
    src_to_local = {}
    for li, s in enumerate(idx):
        src_to_local.setdefault(int(s), li)
    for src in (sample if N > 4096 else range(N)):
        if N > 4096:
            wi_local = src_to_local[int(src)]
            ssrc = int(src)
        else:
            wi_local = int(src)  # range(N): wi_local == world index
            ssrc = int(idx[wi_local])
        M = int(z["mf_constraint_count"][ssrc])
        m = z["mf_meta"][ssrc].reshape(-1, 4)[:M].astype(np.int64)
        packed = m[:, 0].astype(np.int32)
        dof_a = (packed >> 16).astype(np.int64)
        dof_b = ((packed << 16) >> 16).astype(np.int64)
        bias = m[:, 2].astype(np.int32).view(np.float32).astype(np.float64)
        rt = (m[:, 3] & 0xFFFF).astype(np.int64)
        Ja = z["mf_J_a"][ssrc, :M].astype(np.float64)
        Jb = z["mf_J_b"][ssrc, :M].astype(np.float64)
        v = v_NxD[wi_local].astype(np.float64)
        rows0 = np.where(rt == 0)[0]
        for i in rows0:
            da, db = int(dof_a[i]), int(dof_b[i])
            jv = 0.0
            if da >= 0:
                jv += float(np.dot(Ja[i], v[da:da + 6]))
            if db >= 0:
                jv += float(np.dot(Jb[i], v[db:db + 6]))
            residual = jv + bias[i]
            phi_now = bias[i] * DT / BETA
            v_res = max(0.0, -residual)
            phi_next = phi_now - v_res * DT
            pen = max(0.0, -phi_next) * 1000.0
            pens_all.append(pen)
            if pen > peak:
                peak = pen
    pens_all = np.array(pens_all)
    return peak, float(np.sqrt((pens_all ** 2).mean())) if len(pens_all) else 0.0


def occupancy_estimate(N, K):
    """Cheap device-fill estimate. kpw launches dim=N*K threads, block_dim=256.
    Report active-blocks-per-SM theoretical fill vs SM count."""
    try:
        props = wp.context.runtime.core.cuda_device_get_attribute
    except Exception:
        props = None
    nsm = getattr(dev, "sm_count", None)
    return nsm


def run():
    Ns = [int(x) for x in os.environ.get("KPW_NS", "4096,8192,16384").split(",")]
    Ks = [int(x) for x in os.environ.get("KPW_KS", "1,4,8").split(",")]
    reps = int(os.environ.get("KPW_REPS", "20"))

    serial_kernel = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode="current")
    print("==================== KPW vs SERIAL BLACKWELL BENCH ====================", flush=True)
    print(f"newton.__file__ = {_n.__file__}", flush=True)
    print(f"GPU device      = {dev.name}", flush=True)
    print(f"GPU arch        = sm_{arch}", flush=True)
    print(f"GPU SM count    = {getattr(dev,'sm_count','?')}", flush=True)
    print(f"SERIAL kernel key = {kname(serial_kernel)}", flush=True)
    assert "_kpw_" not in kname(serial_kernel) and "_tpw_" not in kname(serial_kernel)
    print(f"cap meta = M_D={M_D} M_MF={M_MF} D={D} iters={ITERS} omega={OMEGA} "
          f"row_phase={ROW_PHASE} warp_bd={BD_WARP} WBASE={WBASE}", flush=True)
    print(f"Ns={Ns} Ks={Ks} reps={reps}", flush=True)
    print("=" * 70, flush=True)

    rows = []
    fidelity = {}  # (N, label) -> peak_mm
    import gc
    for N in Ns:
        print(f"\n##### N={N} #####", flush=True)
        wi = build_world_innermost(N)
        co_np, cr_np, nc_np, per_world = build_coloring(wi)
        ncs = nc_np
        print(f"  coloring: n_colors min/med/max = {int(ncs.min())}/{int(np.median(ncs))}/{int(ncs.max())}", flush=True)
        co = wp.array(co_np, dtype=wp.int32, device=DEV)
        cr = wp.array(cr_np, dtype=wp.int32, device=DEV)
        nc = wp.array(nc_np, dtype=wp.int32, device=DEV)

        # ── SERIAL ──
        t_serial = float("nan")
        peak_serial = float("nan")
        try:
            aw = build_warp(N)
            win = [aw[k] for k in ORDER] + SCALARS
            # fidelity solve (one launch into v_out)
            wp.launch_tiled(serial_kernel, dim=[N], inputs=win, outputs=[aw["v_out"]], block_dim=BD_WARP, device=DEV)
            v_serial = aw["v_out"].numpy().reshape(N, D)
            peak_serial, rms_serial = peak_pen_mm_from_solution(v_serial, N)
            print(f"  [serial] launched OK at N={N}; peak_interpen={peak_serial:.4f} mm rms={rms_serial:.4f}", flush=True)
            # reset v_out for timing (re-build to get fresh v0; serial mutates in place)
            warp_launch = lambda: wp.launch_tiled(serial_kernel, dim=[N], inputs=win, outputs=[aw["v_out"]], block_dim=BD_WARP, device=DEV)
            t_serial = median_ms(warp_launch, reps=reps)
            del aw, win, v_serial
            gc.collect()
            print(f"  [serial] median = {t_serial:.4f} ms", flush=True)
        except Exception as e:
            print(f"  [serial] FAILED at N={N}: {type(e).__name__}: {e}", flush=True)
            v_serial = None
            gc.collect()

        # need a serial v reference for fidelity diff; recompute lightweight if needed
        # (kept above when it succeeded)
        fidelity[(N, "serial")] = peak_serial

        # ── KPW K ──
        kpw_ms = {}
        for K in Ks:
            try:
                d2 = make_warp_inputs(wi)
                _layout = os.environ.get("KPW_LAYOUT", "interleaved")
                kpw_kernel = _get_pgs_solve_mf_gs_kernel_kpw(M_D, M_MF, D, N, MAX_COLORS, arch, k_threads=K, friction_mode="current", lane_layout=_layout)
                if N == Ns[0] and K == Ks[0]:
                    print(f"  KPW kernel key = {kname(kpw_kernel)}", flush=True)
                assert "_kpw_" in kname(kpw_kernel)
                kin = kpw_input_list(d2, co, cr, nc)
                # fidelity solve
                wp.launch(kpw_kernel, dim=N * K, inputs=kin, block_dim=256, device=DEV)
                v_kpw = v_world_innermost_to_NxD(d2["v_g"].numpy(), N)
                peak_k, rms_k = peak_pen_mm_from_solution(v_kpw, N)
                fidelity[(N, f"kpw{K}")] = peak_k
                # re-make for clean timing
                del d2, kin, v_kpw
                gc.collect()
                d3 = make_warp_inputs(wi)
                kin3 = kpw_input_list(d3, co, cr, nc)
                t = median_ms(lambda: wp.launch(kpw_kernel, dim=N * K, inputs=kin3, block_dim=256, device=DEV), reps=reps)
                kpw_ms[K] = t
                print(f"  [kpw K={K}] median = {t:.4f} ms  peak_interpen={peak_k:.4f} mm", flush=True)
                del d3, kin3
                gc.collect()
            except Exception as e:
                print(f"  [kpw K={K}] FAILED at N={N}: {type(e).__name__}: {e}", flush=True)
                kpw_ms[K] = float("nan")
                gc.collect()

        del co, cr, nc, wi, co_np, cr_np, nc_np, per_world
        gc.collect()
        rows.append((N, t_serial, kpw_ms))

    # ── TABLE ──
    print("\n\n==================== RESULT TABLE (ms) ====================", flush=True)
    hdr = f"| {'N':>6} | {'serial':>9} |" + "".join(f" {'kpw K='+str(k):>9} |" for k in Ks)
    print(hdr, flush=True)
    print("|" + "-" * 8 + "|" + "-" * 11 + "|" + ("-" * 12 + "|") * len(Ks), flush=True)
    for (N, t_serial, kpw_ms) in rows:
        cells = f"| {N:>6} | {t_serial:>9.4f} |" + "".join(f" {kpw_ms.get(k, float('nan')):>9.4f} |" for k in Ks)
        print(cells, flush=True)

    print("\n==================== SPEEDUP vs SERIAL ====================", flush=True)
    for (N, t_serial, kpw_ms) in rows:
        parts = []
        for k in Ks:
            mk = kpw_ms.get(k, float("nan"))
            sp = (t_serial / mk) if (mk == mk and t_serial == t_serial and mk > 0) else float("nan")
            parts.append(f"K={k}:{sp:.3f}x")
        valid = {k: kpw_ms[k] for k in Ks if kpw_ms.get(k, float('nan')) == kpw_ms.get(k, float('nan'))}
        best = min(valid, key=valid.get) if valid else None
        bestsp = (t_serial / valid[best]) if best is not None and t_serial == t_serial else float("nan")
        print(f"  N={N}: " + "  ".join(parts) + f"   | best K={best} -> {bestsp:.3f}x vs serial", flush=True)

    print("\n==================== FIDELITY (peak interpen mm) ====================", flush=True)
    for (N, t_serial, kpw_ms) in rows:
        line = f"  N={N}: serial={fidelity.get((N,'serial'),float('nan')):.4f}"
        for k in Ks:
            line += f"  kpw{k}={fidelity.get((N,f'kpw{k}'),float('nan')):.4f}"
        print(line + "  mm  (limit 8mm; match serial ~0.01mm)", flush=True)

    print("\n==================== KPW K=8 CROSSOVER ====================", flush=True)
    for (N, t_serial, kpw_ms) in rows:
        m8 = kpw_ms.get(8, float("nan"))
        if m8 == m8 and t_serial == t_serial:
            verdict = "BEATS serial" if m8 < t_serial else "slower than serial"
            print(f"  N={N}: kpw K=8 {m8:.4f} ms vs serial {t_serial:.4f} ms -> {verdict} ({t_serial/m8:.3f}x)", flush=True)
        else:
            print(f"  N={N}: kpw K=8 or serial unavailable", flush=True)

    # copy nothing; stdout is the artifact. Also drop a json for the record.
    out = {
        "device": dev.name, "arch": f"sm_{arch}", "sm_count": getattr(dev, "sm_count", None),
        "serial_key": kname(serial_kernel),
        "Ns": Ns, "Ks": Ks, "reps": reps,
        "rows": [{"N": N, "serial_ms": ts, "kpw_ms": {str(k): kpw_ms.get(k) for k in Ks}} for (N, ts, kpw_ms) in rows],
        "fidelity_mm": {f"{N}:{lbl}": v for (N, lbl), v in fidelity.items()},
    }
    RUN_DIR = os.environ.get("RUN_DIR", "/workspace/slurm-run")
    for path in (os.path.join(RUN_DIR, "kpw_bench_result.json"), "kpw_bench_result.json"):
        try:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n[wrote] {path}", flush=True)
        except Exception as e:
            print(f"[warn] could not write {path}: {e}", flush=True)
    print("==================== KPW BENCH DONE ====================", flush=True)


if __name__ == "__main__":
    run()
