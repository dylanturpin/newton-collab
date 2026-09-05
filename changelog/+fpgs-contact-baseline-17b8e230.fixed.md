Bound the complete FeatherPGS implicit drive reaction by finite positive effort limits, including CUDA tiled row solves; transport warm-started friction into the current tangent frame and clamp it to the current material's friction cone.

Finite positive effort limits now use bounded PGS drive rows even with the default `drive_mode="augmented"`, including the builder default of 1e6. These drives cold-start each step and depend on the PGS iteration budget; zero iterations applies no bounded drive impulse. This corrects actuator saturation but changes default robot formulation and may increase runtime. Unlimited drives retain the augmented fold.

Automatically reserve dense internal rows in addition to the default 32 contact rows when `dense_max_constraints` is omitted. Explicit capacities remain fixed total budgets. Reject `tiled_contact` and `streaming` for drive-capable models instead of silently selecting another kernel.

Clear stale matching indices when a contact buffer changes to a producer without matching, refresh only reset-selected articulation factors, and include global mesh-reducer table and buffer losses in FeatherPGS capacity status.

Handle dense row capacities that are not divisible by 32 without out-of-bounds CUDA accesses in the tiled row solver.
