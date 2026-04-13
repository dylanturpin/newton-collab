# M4 Kernel Work and Memory Summary

This pass captures the first reviewable M4 slice: the dense tiled-row PGS kernel and the fused articulated-plus-free-rigid matrix-free GS kernel.

## Matrix and vector work

- Dense tiled-row solve operates on the explicit Delassus system `C = J H^{-1} J^T` and updates impulses from `w_i = rhs_i + sum_j C_ij lambda_j`.
- Articulated matrix-free solve keeps only `J_world`, `Y_world`, `diag`, `rhs`, and the live world velocity `v`; each row updates from `w_i = rhs_i + J_i v` and applies `v += Y_i delta_lambda_i`.
- Free-rigid matrix-free rows use `mf_J_*`, `mf_MiJt_*`, `mf_eff_mass_inv`, and `mf_rhs`; each row updates from `w_i = mf_rhs_i + J_i v` and applies `v += MiJt_i delta_lambda_i`.

## Scenario-backed sizing

| Scenario | Preset | Active dense rows | Active MF rows | Key dense storage | Key matrix-free storage |
| --- | --- | ---: | ---: | --- | --- |
| `g1_flat` | `fpgs_dense_row` | 24 | 0 | `C`: 4096 B | n/a |
| `g1_flat` | `fpgs_matrix_free` | 24 | 0 | `rhs`+`diag`+`impulses`: 1536 B | `J_world`+`Y_world`: 50176 B |
| `h1_tabletop` | `fpgs_dense_row` | 117 | 0 | `C`: 65536 B | n/a |
| `h1_tabletop` | `fpgs_matrix_free` | 30 | 102 | `rhs`+`diag`+`impulses`: 1536 B | `J_world`+`Y_world`: 136192 B; free-rigid `mf_*`: 63488 B |

## Practical interpretation

- `g1_flat` is a useful control case: matrix-free allocates larger world-indexed buffers, but the scenario has no free-rigid MF rows, so there is no row-routing benefit yet.
- `h1_tabletop` is the important mixed-world case: dense-row keeps all 117 active rows in the explicit Delassus system, while matrix-free moves 102 rows to the free-rigid branch and keeps only 30 articulated rows on the articulated side.
- The dense-row kernel relies on shared memory for the packed lower triangle of `C`. The matrix-free GS kernel instead spends shared memory on the live velocity vector plus scalar row state and reconstructs couplings through `J*v` and `Y*delta_lambda`.
- This is the first concrete evidence bundle for the docs claim that matrix-free can perform well without a shared-memory Delassus tile, while the dense path still pays the up-front `C` materialization cost.
