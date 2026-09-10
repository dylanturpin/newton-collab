Replace FeatherPGS's experimental per-contact friction anchors with persistent
body-pair friction patches. Set `friction_anchor_beta` to enable up to two anchors
per compatible region, sharing the region's normal load while preserving every
normal contact. Collision contact matching is no longer required for anchor
history. The existing `friction_anchor_beta` remains the single anchor control;
correlation and release use geometry-scaled internal tolerances.

Patch history survives reused contact buffers across solver substeps. Masked
resets track world ownership even when contact rows are rejected. Shape geometry
updates retire affected anchors while preserving unrelated bodies' history.
Carried surface witnesses also retire anchors lifted by rocking or partial
release, without discarding history during decompression that remains in contact.

Uniform pad-friction randomization should assign one sampled coefficient to all
of the pad's convex shapes; genuinely different coefficients define separate
regions. Twisting resistance comes only from the anchor lever arms, so a region
with one anchor has no independent torsional stiction constraint.

Shape-property notifications synchronize geometry to the host and must run
outside CUDA graph capture. `SolverFeatherPGS.update_contacts()` exports linear
forces only: torque entries are zero placeholders, and combining the current
contact point with the exported force does not reconstruct a patch's wrench.
