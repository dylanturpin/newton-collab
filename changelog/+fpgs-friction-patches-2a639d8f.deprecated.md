Deprecate FeatherPGS's `contact_friction_anchor_limit` approximation in favor of
`friction_anchor_beta`. Positive legacy values warn and have no effect. Persistent
patch friction is enabled by default; an explicit `friction_anchor_beta=0` opts
out and is never overridden by the deprecated setting. The positional argument
remains available for compatibility.
