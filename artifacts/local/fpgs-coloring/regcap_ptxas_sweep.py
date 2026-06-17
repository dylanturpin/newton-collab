"""ptxas occupancy sweep: streaming + register-cap (launch_bounds) for colored FPGS MF-GS.

For the colored MF Gauss-Seidel kernel at W in {1,2,4,8}, builds each
(streaming, min_blocks-regcap) config through warp (which bakes __launch_bounds__
into the emitted .sm86.ptx), then runs `ptxas -arch=sm_86 -v` on that PTX to read
the AUTHORITATIVE regs/thread, smem bytes/block, and spill bytes that HONOR the
launch_bounds. Computes the sm_86 theoretical blocks/SM (occupancy).

Register cap = FEATHER_PGS_MFGS_MIN_BLOCKS=n -> wp.kernel(launch_bounds=(32*W, n))
-> __launch_bounds__(32*W, n); ptxas trims regs toward fitting n resident
blocks/SM (soft; spills if too aggressive).

The system ptxas is too old for warp's PTX ISA 8.8; we use a 12.9 ptxas.

Run: .venv/bin/python artifacts/local/fpgs-coloring/regcap_ptxas_sweep.py
"""
import glob
import json
import os
import re
import subprocess
import sys

WORKTREE = "/home/dturpin/repos/il-newton-dev/.claude/worktrees/smem-color"
sys.path.insert(0, WORKTREE)

import warp as wp  # noqa: E402
import newton  # noqa: E402

assert newton.__file__.startswith(WORKTREE), newton.__file__
from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
)

PTXAS = "/tmp/ptxas_dl/extracted929/nvidia/cuda_nvcc/bin/ptxas"
assert os.path.exists(PTXAS), PTXAS
DEV = "cuda:0"
CACHE = os.environ.get("WARP_CACHE_PATH", "/tmp/warp_cache_smemreg")
wp.config.kernel_cache_dir = CACHE
wp.init()

import numpy as np  # noqa: E402

meta = json.load(open("/tmp/cap892.json"))
z = np.load("/tmp/cap892.npz")
M_D = int(meta["dense_max_constraints"])
M_MF = int(meta["mf_max_constraints"])
D = int(meta["max_world_dofs"])
arch = int(wp.get_device(DEV).arch)
MAX_COLORS = 16
FRICTION = meta["friction_mode"]
print(f"[ptxas] newton={newton.__file__}", flush=True)
print(f"[ptxas] ptxas={subprocess.run([PTXAS,'--version'],capture_output=True,text=True).stdout.splitlines()[-1].strip()}", flush=True)
print(f"[ptxas] arch={arch} M_D={M_D} M_MF={M_MF} D={D} friction={FRICTION}", flush=True)

# ── sm_86 (RTX 3090 / GA10x) hardware limits ────────────────────────────────
SM_MAX_WARPS = 48          # 1536 threads / 32
SM_MAX_BLOCKS = 16         # max resident blocks / SM
SM_REGS = 65536            # 64K 32-bit regs / SM
SM_SMEM = 102400           # 100 KB usable smem / SM
REG_ALLOC_UNIT = 256       # regs allocated per-warp, units of 256
SMEM_ALLOC_UNIT = 128      # smem allocation granularity (bytes), sm_8x
SMEM_DRIVER_RSVD = 1024    # ~1KB driver static smem reserve / block


def ceil_to(x, unit):
    return ((x + unit - 1) // unit) * unit


def theoretical_blocks(regs, smem_bytes, block_threads):
    warps_per_block = (block_threads + 31) // 32
    regs_per_warp = ceil_to(regs * 32, REG_ALLOC_UNIT)
    regs_per_block = regs_per_warp * warps_per_block
    blk_reg = SM_REGS // regs_per_block if regs_per_block else SM_MAX_BLOCKS
    smem_per_block = ceil_to(smem_bytes + SMEM_DRIVER_RSVD, SMEM_ALLOC_UNIT)
    blk_smem = SM_SMEM // smem_per_block if smem_per_block else SM_MAX_BLOCKS
    blk_thread = SM_MAX_WARPS // warps_per_block
    blk = min(blk_reg, blk_smem, blk_thread, SM_MAX_BLOCKS)
    return blk, dict(reg=blk_reg, smem=blk_smem, thr=blk_thread)


def build(colored, warps, stream, min_blocks):
    os.environ["FEATHER_PGS_MFGS_COLORED"] = "1" if colored else "0"
    os.environ["FEATHER_PGS_MFGS_MAX_COLORS"] = str(MAX_COLORS)
    os.environ["FEATHER_PGS_MFGS_WARPS"] = str(warps)
    os.environ["FEATHER_PGS_MFGS_STREAM_LAMBDA"] = "1" if stream else "0"
    os.environ["FEATHER_PGS_MFGS_STREAM_DRIVE"] = "1" if stream else "0"
    if min_blocks:
        os.environ["FEATHER_PGS_MFGS_MIN_BLOCKS"] = str(min_blocks)
    else:
        os.environ.pop("FEATHER_PGS_MFGS_MIN_BLOCKS", None)
    _get_pgs_solve_mf_gs_kernel.cache_clear()
    k = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode=FRICTION)
    k.module.load(DEV)  # force NVRTC -> .sm86.ptx emission
    # locate the just-built .sm86.ptx by the module name (= cache dir prefix)
    name = k.key
    modname = k.module.name  # e.g. pgs_solve_mf_gs_..._colored16_w4_slam_sdrv_aaf588f3
    # cache layout: <CACHE>/<warp_version>/wp_<modname>_<shorthash>/<...>.sm86.ptx
    pats = glob.glob(os.path.join(CACHE, "*", f"wp_{modname}*", "*.sm86.ptx"))
    if not pats:
        pats = glob.glob(os.path.join(CACHE, "**", f"*{modname}*.sm86.ptx"),
                         recursive=True)
    if not pats:
        raise RuntimeError(f"no .sm86.ptx for module {modname}")
    ptx = max(pats, key=os.path.getmtime)
    return ptx, name


REG_RE = re.compile(r"Used (\d+) registers")
SMEM_RE = re.compile(r"(\d+) bytes smem")
SPILL_S_RE = re.compile(r"(\d+) bytes spill stores")
SPILL_L_RE = re.compile(r"(\d+) bytes spill loads")


def ptxas_v(ptx):
    r = subprocess.run([PTXAS, "-arch=sm_86", "-v", ptx, "-o", "/dev/null"],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    regs = smem = None
    ss = sl = 0
    m = REG_RE.search(out)
    if m:
        regs = int(m.group(1))
    m = SMEM_RE.search(out)
    if m:
        smem = int(m.group(1))
    m = SPILL_S_RE.search(out)
    if m:
        ss = int(m.group(1))
    m = SPILL_L_RE.search(out)
    if m:
        sl = int(m.group(1))
    return regs, smem, ss, sl, out


CONFIGS = [
    ("serial-resident", False, 1, False, None),
    ("serial-stream",   False, 1, True,  None),
    ("W1-stream",       True,  1, True,  None),
    ("W2-resident",     True,  2, False, None),
    ("W2-stream",       True,  2, True,  None),
    ("W4-stream",       True,  4, True,  None),
    ("W4-stream-mb4",   True,  4, True,  4),
    ("W4-stream-mb6",   True,  4, True,  6),
    ("W4-stream-mb8",   True,  4, True,  8),
    ("W8-stream",       True,  8, True,  None),
    ("W8-stream-mb4",   True,  8, True,  4),
    ("W8-stream-mb6",   True,  8, True,  6),
    ("W8-stream-mb8",   True,  8, True,  8),
]

print("\n[ptxas] === OCCUPANCY TABLE (sm_86, real ptxas -v) ===", flush=True)
hdr = (f"{'config':<17} {'W':>2} {'strm':>4} {'cap':>4} | {'regs':>4} {'smem':>6} "
       f"{'spill s/l':>10} | {'blk/SM':>6} {'lim r/s/t':>10} {'warps/SM':>8}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
rows = []
for label, colored, w, stream, mb in CONFIGS:
    ptx, name = build(colored, w, stream, mb)
    regs, smem, ss, sl, raw = ptxas_v(ptx)
    bd = 32 * w
    if regs is None or smem is None:
        print(f"{label:<17} PARSE-FAIL regs={regs} smem={smem} ptx={os.path.basename(ptx)}", flush=True)
        print("   " + "\n   ".join(raw.strip().splitlines()[-4:]), flush=True)
        continue
    blk, lim = theoretical_blocks(regs, smem, bd)
    warps_sm = blk * (bd // 32)
    limstr = f"{lim['reg']}/{lim['smem']}/{lim['thr']}"
    print(f"{label:<17} {w:>2} {'Y' if stream else 'N':>4} {str(mb or '-'):>4} | "
          f"{regs:>4} {smem:>6} {f'{ss}/{sl}':>10} | {blk:>6} "
          f"{limstr:>10} {warps_sm:>8}",
          flush=True)
    rows.append(dict(label=label, W=w, stream=stream, min_blocks=mb, regs=regs,
                     smem=smem, spill_store=ss, spill_load=sl, blk_per_sm=blk,
                     lim=lim, warps_per_sm=warps_sm))

with open(os.path.join(os.path.dirname(__file__), "regcap_ptxas_result.json"), "w") as f:
    json.dump(rows, f, indent=2)
print("\n[ptxas] PTXAS_DONE", flush=True)
