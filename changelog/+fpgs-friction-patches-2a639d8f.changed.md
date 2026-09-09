Replace FeatherPGS's experimental per-contact friction anchors with persistent
body-pair friction patches. Set `friction_anchor_beta` to enable up to two anchors
per compatible region, sharing the region's normal load while preserving every
normal contact. Collision contact matching is no longer required for anchor
history. The existing `friction_anchor_beta` remains the single anchor control;
correlation and release use geometry-scaled internal tolerances.
