"""ptxas occupancy (blocks/SM) for the warp-per-world and thread-per-world FPGS
solve kernels, ON THIS GPU's arch. We let warp build each kernel module (which
emits a .cu / .ptx into the warp kernel cache), find the generated source, and
run `ptxas -v --gpu-name sm_<arch>` to get registers/smem per kernel. Then we
compute blocks/SM from the arch's limits (regs/SM, smem/SM, max blocks/SM).

This answers: is the warp kernel actually occupancy-starved on Blackwell, and
does tpw lift blocks/SM the way it does on the 3090 (64->1536 threads/SM)?
"""
import glob
import os
import re
import subprocess
import sys

REPO = os.environ.get("TPW_REPO", "/home/dturpin/repos/il-newton-dev/.claude/worktrees/tpw")
sys.path.insert(0, REPO)

import warp as wp
import newton as _n  # noqa: F401

from newton._src.solvers.feather_pgs.solver_feather_pgs import (  # noqa: E402
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_mf_gs_kernel_tpw,
)

DEV = "cuda:0"
CACHE = os.environ.get("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))
wp.config.kernel_cache_dir = CACHE
# make warp keep readable source + emit verbose
wp.config.verbose = True
wp.config.verify_fp = False
wp.init()

dev = wp.get_device(DEV)
arch = int(dev.arch)
M_D, M_MF, D = 512, 4096, 609

# Per-SM hardware limits by compute capability. (regs/SM, smem/SM bytes, max blocks/SM, max threads/SM)
# Blackwell consumer/workstation (sm_120 / sm_100) and Ampere (sm_86 3090) for reference.
LIMITS = {
    86: dict(regs=65536, smem=102400, max_blocks=16, max_threads=1536, name="Ampere GA10x (RTX 3090)"),
    89: dict(regs=65536, smem=102400, max_blocks=24, max_threads=1536, name="Ada AD10x"),
    90: dict(regs=65536, smem=233472, max_blocks=32, max_threads=2048, name="Hopper sm_90 (H100/B200-as-90?)"),
    100: dict(regs=65536, smem=233472, max_blocks=32, max_threads=2048, name="Blackwell sm_100 (B200/GB200)"),
    120: dict(regs=65536, smem=102400, max_blocks=24, max_threads=1536, name="Blackwell sm_120 (RTX PRO 6000 / 5090)"),
}


def build_and_get_module_dir(kernel):
    """Force-load the module so warp writes the .cu; return its cache dir + module name."""
    mod = kernel.module
    # building requires a launch context; load on device
    try:
        mod.load(dev)
    except Exception as e:
        print(f"  (module.load warning: {e})")
    return mod, getattr(mod, "name", "?")


def find_cu_for_module(mod_name):
    # warp writes wp_<modulehash>.cu under <cache>/<version>/gen/ ; search broadly
    cands = []
    for pat in ("**/*.cu", "**/*.cpp"):
        cands += glob.glob(os.path.join(CACHE, pat), recursive=True)
    return cands


def ptxas_verbose(cu_path, arch, kernel_substr):
    """Compile a .cu to cubin via nvcc and capture ptxas -v for the kernel."""
    out = subprocess.run(
        ["nvcc", f"-arch=sm_{arch}", "-cubin", "-Xptxas", "-v",
         "-o", "/dev/null", cu_path,
         "-I", os.path.join(CACHE)],
        capture_output=True, text=True,
    )
    return out.stdout + out.stderr


def parse_regs_smem(text, kernel_substr):
    """Parse 'Function properties for ... pgs_solve...' / 'Used N registers, M bytes smem'."""
    blocks = re.split(r"ptxas info\s*:\s*Compiling entry function '([^']+)'", text)
    # blocks: [pre, name1, body1, name2, body2, ...]
    results = {}
    for i in range(1, len(blocks), 2):
        name = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        regs = re.search(r"Used (\d+) registers", body)
        smem = re.search(r"(\d+) bytes smem", body)
        cmem = re.search(r"(\d+) bytes cmem\[0\]", body)
        results[name] = dict(
            regs=int(regs.group(1)) if regs else None,
            smem=int(smem.group(1)) if smem else None,
            cmem=int(cmem.group(1)) if cmem else None,
        )
    # return entries whose demangled-ish name contains the substring
    return {k: v for k, v in results.items() if kernel_substr in k}


def occupancy(regs, smem_static, block_dim, lim):
    """blocks/SM limited by regs, smem, hard block cap, and threads/SM."""
    # register-limited blocks: regs/SM // (regs_per_thread * threads_per_block), rounded to warp alloc granularity
    threads = block_dim
    if regs and regs > 0:
        # warp register alloc granularity is 256 on recent archs; round regs/thread up per-warp
        regs_per_warp = ((regs * 32 + 255) // 256) * 256
        warps_per_block = (threads + 31) // 32
        regs_per_block = regs_per_warp * warps_per_block
        reg_blocks = lim["regs"] // regs_per_block if regs_per_block else lim["max_blocks"]
    else:
        reg_blocks = lim["max_blocks"]
    if smem_static and smem_static > 0:
        smem_blocks = lim["smem"] // smem_static
    else:
        smem_blocks = lim["max_blocks"]
    thread_blocks = lim["max_threads"] // threads if threads else lim["max_blocks"]
    occ = min(reg_blocks, smem_blocks, lim["max_blocks"], thread_blocks)
    return occ, dict(reg_blocks=reg_blocks, smem_blocks=smem_blocks,
                     thread_blocks=thread_blocks, hard_cap=lim["max_blocks"])


if __name__ == "__main__":
    lim = LIMITS.get(arch, LIMITS[120])
    warp_bd = 32  # cap892 block_dim (warp-per-world uses one warp = 32 threads/block effectively via launch_tiled bd)
    tpw_bd = int(os.environ.get("TPW_BD", "256"))
    print("==================== PTXAS OCCUPANCY (this GPU) ====================", flush=True)
    print(f"GPU = {dev.name}  arch = sm_{arch}  ({lim['name']})", flush=True)
    print(f"per-SM limits: regs={lim['regs']} smem={lim['smem']}B max_blocks={lim['max_blocks']} "
          f"max_threads={lim['max_threads']}", flush=True)

    warp_k = _get_pgs_solve_mf_gs_kernel(M_D, M_MF, D, arch, friction_mode="current")
    tpw_k = _get_pgs_solve_mf_gs_kernel_tpw(M_D, M_MF, D, 256, arch, friction_mode="current")
    print(f"warp kernel key = {warp_k.key}", flush=True)
    print(f"tpw  kernel key = {tpw_k.key}", flush=True)

    for k in (warp_k, tpw_k):
        build_and_get_module_dir(k)

    cu_files = find_cu_for_module(None)
    print(f"\nfound {len(cu_files)} generated .cu under {CACHE}", flush=True)

    # Run ptxas -v on each .cu and collect the matching entry functions
    found = {}
    for cu in cu_files:
        txt = ptxas_verbose(cu, arch, "pgs_solve_mf_gs")
        for sub in ("pgs_solve_mf_gs",):
            res = parse_regs_smem(txt, sub)
            found.update(res)

    print(f"\nptxas entries matching pgs_solve_mf_gs: {len(found)}", flush=True)
    for name, info in found.items():
        is_tpw = "_tpw_" in name
        bd = tpw_bd if is_tpw else warp_bd
        occ, det = occupancy(info["regs"], info["smem"], bd, lim)
        tag = "TPW " if is_tpw else "WARP"
        print(f"\n[{tag}] {name}", flush=True)
        print(f"   regs/thread={info['regs']}  static_smem={info['smem']}B  cmem0={info['cmem']}B  block_dim={bd}", flush=True)
        print(f"   blocks/SM = {occ}  (reg-limited={det['reg_blocks']} smem-limited={det['smem_blocks']} "
              f"thread-limited={det['thread_blocks']} hard_cap={det['hard_cap']})", flush=True)
        if is_tpw:
            print(f"   threads/SM ~= {occ * bd}", flush=True)
        else:
            print(f"   warps/SM ~= {occ}  (one warp per block) -> threads/SM ~= {occ * 32}", flush=True)
    print("==================== PTXAS DONE ====================", flush=True)
