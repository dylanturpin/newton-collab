Deprecate FeatherPGS's `contact_friction_anchor_limit` approximation in favor of
`friction_anchor_beta`. When `friction_anchor_beta` is zero and the friction mode and
PGS kernel support patch friction, positive legacy values enable persistent patch
friction with `friction_anchor_beta = max(pgs_beta, 0.2)` and warn; otherwise they
warn and have no effect. The positional argument remains available for compatibility.
