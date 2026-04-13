#!/usr/bin/env python3
"""Generate FeatherPGS kernel-work and memory-layout artifacts for the explainer lane."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / ".agent" / "data" / "fpgs-matrix-free-dense-explainer"
SCENARIO_ROOT = ARTIFACT_ROOT / "scenarios"
KERNEL_ROOT = ARTIFACT_ROOT / "kernels"
SUMMARY_PATH = ARTIFACT_ROOT / "m4-kernel-memory-summary.md"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def logical_buffer_index(data: dict) -> dict[str, dict]:
    return {item["name"]: item for item in data["logical_buffers"]}


def scenario_metrics() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for scenario in ("g1_flat", "h1_tabletop"):
        out[scenario] = {}
        for preset in ("fpgs_dense_row", "fpgs_matrix_free"):
            data = load_json(SCENARIO_ROOT / scenario / f"{preset}.json")
            buffers = logical_buffer_index(data)
            out[scenario][preset] = {
                "counts": data["world_counts"],
                "buffers": buffers,
                "buffer_bytes": {
                    name: buffers[name]["total_bytes"]
                    for name in (
                        "C",
                        "rhs",
                        "diag",
                        "impulses",
                        "J_world",
                        "Y_world",
                        "mf_J_a",
                        "mf_J_b",
                        "mf_MiJt_a",
                        "mf_MiJt_b",
                        "mf_rhs",
                        "mf_impulses",
                        "mf_eff_mass_inv",
                        "mf_meta_packed",
                    )
                    if name in buffers
                },
            }
    return out


def dense_row_artifact(commit: str, generated_at: str, metrics: dict[str, dict[str, dict]]) -> dict:
    g1 = metrics["g1_flat"]["fpgs_dense_row"]
    h1 = metrics["h1_tabletop"]["fpgs_dense_row"]
    return {
        "schema_version": "1.0.0",
        "provenance": {
            "generator": "scripts/analysis/capture_fpgs_kernel_memory_artifacts.py",
            "git_commit": commit,
            "generated_at_utc": generated_at,
            "source_files": [
                "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
                "scripts/analysis/capture_fpgs_kernel_memory_artifacts.py",
                ".agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_dense_row.json",
                ".agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json",
            ],
        },
        "kernel": {
            "id": "dense-row",
            "name": "Dense tiled-row PGS solve",
            "source_file": "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
            "source_symbol": "TiledKernelFactory._build_pgs_solve_tiled_row_kernel",
            "phase": "stage5_dense_pgs",
            "modes": ["dense"],
        },
        "launch": {
            "granularity": "one warp (32 threads) per world; one dense contact system per launch tile",
            "threads_per_block": 32,
            "worlds_per_block": 1,
            "sizing_symbols": ["M = dense_max_constraints", "TILE_TRI = M * (M + 1) / 2"],
        },
        "memory_layout": {
            "global_reads": [
                {
                    "name": "world_constraint_count",
                    "location": "global",
                    "shape": "[world_count]",
                    "dtype": "int32",
                    "notes": "Per-world active row count m.",
                },
                {
                    "name": "world_C",
                    "location": "global",
                    "shape": "[world_count, M, M]",
                    "dtype": "float32",
                    "notes": (
                        "Full Delassus matrix stored globally, then staged as a packed lower triangle in shared memory. "
                        f"Scenario bytes: g1_flat {g1['buffer_bytes']['C']} B, h1_tabletop {h1['buffer_bytes']['C']} B."
                    ),
                },
                {
                    "name": "world_rhs",
                    "location": "global",
                    "shape": "[world_count, M]",
                    "dtype": "float32",
                    "notes": "Bias plus baked J*v_hat term.",
                },
                {
                    "name": "world_diag",
                    "location": "global",
                    "shape": "[world_count, M]",
                    "dtype": "float32",
                    "notes": "Diagonal used as the PGS denominator.",
                },
                {
                    "name": "world_impulses",
                    "location": "global",
                    "shape": "[world_count, M]",
                    "dtype": "float32",
                    "notes": "Warm-start values loaded before the sweep.",
                },
                {
                    "name": "world_row_type/world_row_parent/world_row_mu",
                    "location": "global",
                    "shape": "[world_count, M]",
                    "dtype": "int32/float32",
                    "notes": "Projection metadata for contact, friction, and joint-limit rows.",
                },
            ],
            "global_writes": [
                {
                    "name": "world_impulses",
                    "location": "global",
                    "shape": "[world_count, M]",
                    "dtype": "float32",
                    "notes": "Final impulses written back once after all iterations.",
                }
            ],
            "shared_allocations": [
                {
                    "name": "s_Ctri",
                    "location": "shared",
                    "shape": "[M * (M + 1) / 2]",
                    "dtype": "float32",
                    "notes": "Packed lower triangle of C; this is the dominant shared-memory footprint.",
                },
                {
                    "name": "s_lam/s_rhs/s_diag",
                    "location": "shared",
                    "shape": "[M] each",
                    "dtype": "float32",
                    "notes": "Impulse state and scalar row data resident across the full GS sweep.",
                },
                {
                    "name": "s_rtype/s_parent/s_mu",
                    "location": "shared",
                    "shape": "[M] each",
                    "dtype": "int32/int32/float32",
                    "notes": "Per-row projection metadata staged once per world.",
                },
            ],
            "register_resident": [
                {
                    "name": "my_sum/dot_sum/w_val/delta/new_impulse",
                    "location": "register",
                    "shape": "scalars per lane",
                    "dtype": "float32",
                    "notes": "Lane-local accumulators for the current row update.",
                },
                {
                    "name": "j_k/jb_k/base_i",
                    "location": "register",
                    "shape": "small integer scalars per lane",
                    "dtype": "int32",
                    "notes": "Precomputed packed-triangle indexing terms.",
                },
            ],
        },
        "operations": {
            "streamed_inputs": [
                "The kernel streams the full dense Delassus matrix C from global memory during the load phase.",
                "world_constraint_count gates how much of the staged shared state is actually used.",
            ],
            "preloaded_inputs": [
                "C is repacked into s_Ctri once, then reused for every PGS iteration.",
                "rhs, diag, impulses, and row metadata are loaded into shared memory before the solve loop.",
            ],
            "recomputed_values": [
                "Each row recomputes the full dot product sum_j C_ij * lambda_j from shared memory every sweep.",
                "Friction sibling projection recomputes tangential magnitude from the updated shared impulses.",
            ],
            "dominant_work": [
                "For each active row i, evaluate w_i = rhs_i + sum_j C_ij * lambda_j.",
                "Project lambda_i for unilateral contact / joint limit rows and the paired friction rows.",
                "Store only the final impulse vector; no velocity vector is updated in this kernel.",
            ],
        },
        "observations": [
            (
                "The dense-row kernel removes repeated global reads of C during the sweep by staging the lower triangle in shared memory, "
                "but it still requires the full C allocation to exist beforehand."
            ),
            (
                f"In the review scenarios the explicit C allocation is modest for g1_flat ({g1['buffer_bytes']['C']} B) but already "
                f"65,536 B for h1_tabletop at M=128, before counting J/Y construction and Delassus assembly."
            ),
            (
                f"h1_tabletop dense-row keeps all {h1['counts']['dense_constraint_rows']} active rows in the dense system; "
                "there is no free-rigid off-ramp in this mode."
            ),
        ],
    }


def matrix_free_artifact(commit: str, generated_at: str, metrics: dict[str, dict[str, dict]]) -> dict:
    g1 = metrics["g1_flat"]["fpgs_matrix_free"]
    h1 = metrics["h1_tabletop"]["fpgs_matrix_free"]
    return {
        "schema_version": "1.0.0",
        "provenance": {
            "generator": "scripts/analysis/capture_fpgs_kernel_memory_artifacts.py",
            "git_commit": commit,
            "generated_at_utc": generated_at,
            "source_files": [
                "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
                "newton/_src/solvers/feather_pgs/kernels.py",
                "scripts/analysis/capture_fpgs_kernel_memory_artifacts.py",
                ".agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json",
                ".agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_matrix_free.json",
            ],
        },
        "kernel": {
            "id": "matrix-free-gs",
            "name": "Two-phase articulated plus free-rigid matrix-free GS solve",
            "source_file": "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
            "source_symbol": "TiledKernelFactory._build_pgs_solve_mf_gs_kernel",
            "phase": "stage6_matrix_free_pgs",
            "modes": ["matrix_free", "split"],
        },
        "launch": {
            "granularity": "one warp (32 threads) per world; dense articulated phase followed by a free-rigid MF phase",
            "threads_per_block": 32,
            "worlds_per_block": 1,
            "sizing_symbols": ["M_D = dense_max_constraints", "M_MF = mf_max_constraints", "D = max_world_dofs"],
        },
        "memory_layout": {
            "global_reads": [
                {
                    "name": "world_constraint_count/world_dof_start",
                    "location": "global",
                    "shape": "[world_count]",
                    "dtype": "int32",
                    "notes": "Sizes the dense articulated phase and locates the world velocity slice.",
                },
                {
                    "name": "J_world/Y_world",
                    "location": "global",
                    "shape": "[world_count, M_D, D]",
                    "dtype": "float32",
                    "notes": (
                        "World-indexed articulated Jacobian rows and H^-1 J^T rows. "
                        f"Scenario bytes: g1_flat {g1['buffer_bytes']['J_world'] + g1['buffer_bytes']['Y_world']} B combined, "
                        f"h1_tabletop {h1['buffer_bytes']['J_world'] + h1['buffer_bytes']['Y_world']} B combined."
                    ),
                },
                {
                    "name": "rhs/diag/world_impulses",
                    "location": "global",
                    "shape": "[world_count, M_D]",
                    "dtype": "float32",
                    "notes": "Articulated row bias, diagonal, and impulse state. rhs omits baked J*v_hat in matrix-free mode.",
                },
                {
                    "name": "mf_meta_packed",
                    "location": "global",
                    "shape": "[world_count, M_MF * 4]",
                    "dtype": "int32",
                    "notes": (
                        "128-bit packed free-rigid metadata with dof offsets, eff_mass_inv, rhs, and row projection metadata. "
                        f"Scenario bytes: {h1['buffer_bytes']['mf_meta_packed']} B."
                    ),
                },
                {
                    "name": "mf_J_a/mf_J_b/mf_MiJt_a/mf_MiJt_b",
                    "location": "global",
                    "shape": "[world_count, M_MF, 6]",
                    "dtype": "float32",
                    "notes": "Free-rigid Jacobians and H^-1 J^T vectors streamed during the second phase.",
                },
                {
                    "name": "mf_impulses",
                    "location": "global",
                    "shape": "[world_count, M_MF]",
                    "dtype": "float32",
                    "notes": "Warm-start state for the free-rigid rows.",
                },
            ],
            "global_writes": [
                {
                    "name": "v_out",
                    "location": "global",
                    "shape": "[total_world_dofs]",
                    "dtype": "float32",
                    "notes": "The live velocity vector is written back after the combined articulated and MF sweep.",
                },
                {
                    "name": "world_impulses/mf_impulses",
                    "location": "global",
                    "shape": "[world_count, M_D] and [world_count, M_MF]",
                    "dtype": "float32",
                    "notes": "Final impulse vectors for both phases.",
                },
            ],
            "shared_allocations": [
                {
                    "name": "s_v",
                    "location": "shared",
                    "shape": "[D]",
                    "dtype": "float32",
                    "notes": "World velocity vector lives in shared memory across both phases of the solve.",
                },
                {
                    "name": "s_lam_dense/s_rhs_dense/s_diag_dense/s_rtype_dense/s_parent_dense/s_mu_dense",
                    "location": "shared",
                    "shape": "[M_D] each",
                    "dtype": "float32/int32",
                    "notes": "Articulated row state and metadata preloaded before the dense phase.",
                },
                {
                    "name": "s_lam_mf",
                    "location": "shared",
                    "shape": "[M_MF]",
                    "dtype": "float32",
                    "notes": "Free-rigid impulse state kept in shared memory for the second phase.",
                },
            ],
            "register_resident": [
                {
                    "name": "prefetched J/Y lane slices",
                    "location": "register",
                    "shape": "ceil(D / 32) float pairs per lane",
                    "dtype": "float32",
                    "notes": "Software-pipeline registers for the current and next articulated row.",
                },
                {
                    "name": "jv/residual/delta/new_impulse",
                    "location": "register",
                    "shape": "scalars per lane",
                    "dtype": "float32",
                    "notes": "Current articulated-row update state.",
                },
                {
                    "name": "packed metadata fields",
                    "location": "register",
                    "shape": "small scalar set per lane",
                    "dtype": "int32/float32",
                    "notes": "Decoded from mf_meta_packed for the free-rigid branch.",
                },
            ],
        },
        "operations": {
            "streamed_inputs": [
                "Articulated rows stream J_world and Y_world row slices instead of a preassembled Delassus matrix.",
                "The free-rigid phase streams mf_J_* and mf_MiJt_* plus packed metadata from global memory.",
            ],
            "preloaded_inputs": [
                "The live world velocity vector v_out is preloaded into shared memory as s_v and reused across all iterations.",
                "Dense articulated rhs, diag, impulses, and row metadata are staged into shared memory once per world.",
                "Free-rigid impulses are staged into shared memory once; metadata is compressed into mf_meta_packed for coalesced loads.",
            ],
            "recomputed_values": [
                "Every articulated row recomputes J_i * v from the live shared velocity vector instead of reading C_i:.",
                "Every free-rigid row recomputes J_i * v from the current shared body/world velocity state.",
                "Only the diagonal C_ii is materialized explicitly for articulated rows; off-diagonal couplings are reconstructed on the fly through J and Y.",
            ],
            "dominant_work": [
                "Articulated phase: evaluate w_i = rhs_i + J_i * v, project lambda_i, then apply v += Y_i * delta_lambda_i.",
                "Free-rigid phase: evaluate w_i = mf_rhs_i + J_i * v, project lambda_i, then apply v += MiJt_i * delta_lambda_i.",
                "One shared velocity vector couples the articulated and free-rigid phases without assembling a dense mixed-world C matrix.",
            ],
        },
        "observations": [
            (
                f"In g1_flat the matrix-free preset still allocates the larger articulated buffers ({g1['buffer_bytes']['J_world']} B each for J_world and Y_world) "
                "but gains no free-rigid rows, so it is mostly a control case for the docs."
            ),
            (
                f"In h1_tabletop the matrix-free path routes {h1['counts']['matrix_free_constraint_rows']} rows through the free-rigid branch and leaves only "
                f"{h1['counts']['dense_constraint_rows']} articulated rows in the dense articulated phase."
            ),
            (
                f"For h1_tabletop the articulated matrix-free storage is dominated by J_world + Y_world ({h1['buffer_bytes']['J_world'] + h1['buffer_bytes']['Y_world']} B), "
                f"while the free-rigid branch adds {sum(h1['buffer_bytes'][name] for name in ('mf_J_a', 'mf_J_b', 'mf_MiJt_a', 'mf_MiJt_b', 'mf_rhs', 'mf_impulses', 'mf_eff_mass_inv', 'mf_meta_packed'))} B."
            ),
            "This kernel is the main explanation target for the surprising result: it does not need a shared-memory copy of a dense Delassus matrix, only the velocity vector and scalar row state.",
        ],
    }


def build_summary(metrics: dict[str, dict[str, dict]]) -> str:
    g1_dense = metrics["g1_flat"]["fpgs_dense_row"]
    g1_mf = metrics["g1_flat"]["fpgs_matrix_free"]
    h1_dense = metrics["h1_tabletop"]["fpgs_dense_row"]
    h1_mf = metrics["h1_tabletop"]["fpgs_matrix_free"]

    return f"""# M4 Kernel Work and Memory Summary

This pass captures the first reviewable M4 slice: the dense tiled-row PGS kernel and the fused articulated-plus-free-rigid matrix-free GS kernel.

## Matrix and vector work

- Dense tiled-row solve operates on the explicit Delassus system `C = J H^{{-1}} J^T` and updates impulses from `w_i = rhs_i + sum_j C_ij lambda_j`.
- Articulated matrix-free solve keeps only `J_world`, `Y_world`, `diag`, `rhs`, and the live world velocity `v`; each row updates from `w_i = rhs_i + J_i v` and applies `v += Y_i delta_lambda_i`.
- Free-rigid matrix-free rows use `mf_J_*`, `mf_MiJt_*`, `mf_eff_mass_inv`, and `mf_rhs`; each row updates from `w_i = mf_rhs_i + J_i v` and applies `v += MiJt_i delta_lambda_i`.

## Scenario-backed sizing

| Scenario | Preset | Active dense rows | Active MF rows | Key dense storage | Key matrix-free storage |
| --- | --- | ---: | ---: | --- | --- |
| `g1_flat` | `fpgs_dense_row` | {g1_dense["counts"]["dense_constraint_rows"]} | 0 | `C`: {g1_dense["buffer_bytes"]["C"]} B | n/a |
| `g1_flat` | `fpgs_matrix_free` | {g1_mf["counts"]["dense_constraint_rows"]} | {g1_mf["counts"]["matrix_free_constraint_rows"]} | `rhs`+`diag`+`impulses`: {g1_mf["buffer_bytes"]["rhs"] + g1_mf["buffer_bytes"]["diag"] + g1_mf["buffer_bytes"]["impulses"]} B | `J_world`+`Y_world`: {g1_mf["buffer_bytes"]["J_world"] + g1_mf["buffer_bytes"]["Y_world"]} B |
| `h1_tabletop` | `fpgs_dense_row` | {h1_dense["counts"]["dense_constraint_rows"]} | 0 | `C`: {h1_dense["buffer_bytes"]["C"]} B | n/a |
| `h1_tabletop` | `fpgs_matrix_free` | {h1_mf["counts"]["dense_constraint_rows"]} | {h1_mf["counts"]["matrix_free_constraint_rows"]} | `rhs`+`diag`+`impulses`: {h1_mf["buffer_bytes"]["rhs"] + h1_mf["buffer_bytes"]["diag"] + h1_mf["buffer_bytes"]["impulses"]} B | `J_world`+`Y_world`: {h1_mf["buffer_bytes"]["J_world"] + h1_mf["buffer_bytes"]["Y_world"]} B; free-rigid `mf_*`: {sum(h1_mf["buffer_bytes"][name] for name in ("mf_J_a", "mf_J_b", "mf_MiJt_a", "mf_MiJt_b", "mf_rhs", "mf_impulses", "mf_eff_mass_inv", "mf_meta_packed"))} B |

## Practical interpretation

- `g1_flat` is a useful control case: matrix-free allocates larger world-indexed buffers, but the scenario has no free-rigid MF rows, so there is no row-routing benefit yet.
- `h1_tabletop` is the important mixed-world case: dense-row keeps all {h1_dense["counts"]["dense_constraint_rows"]} active rows in the explicit Delassus system, while matrix-free moves {h1_mf["counts"]["matrix_free_constraint_rows"]} rows to the free-rigid branch and keeps only {h1_mf["counts"]["dense_constraint_rows"]} articulated rows on the articulated side.
- The dense-row kernel relies on shared memory for the packed lower triangle of `C`. The matrix-free GS kernel instead spends shared memory on the live velocity vector plus scalar row state and reconstructs couplings through `J*v` and `Y*delta_lambda`.
- This is the first concrete evidence bundle for the docs claim that matrix-free can perform well without a shared-memory Delassus tile, while the dense path still pays the up-front `C` materialization cost.
"""


def main() -> None:
    KERNEL_ROOT.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    generated_at = utc_now()
    metrics = scenario_metrics()

    dense = dense_row_artifact(commit, generated_at, metrics)
    matrix_free = matrix_free_artifact(commit, generated_at, metrics)

    (KERNEL_ROOT / "dense-row.json").write_text(json.dumps(dense, indent=2) + "\n")
    (KERNEL_ROOT / "matrix-free-gs.json").write_text(json.dumps(matrix_free, indent=2) + "\n")
    SUMMARY_PATH.write_text(build_summary(metrics))


if __name__ == "__main__":
    main()
