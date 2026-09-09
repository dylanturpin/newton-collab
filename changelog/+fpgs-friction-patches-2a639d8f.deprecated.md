Deprecate FeatherPGS's `contact_friction_anchor_limit` approximation in favor of
`friction_anchor_beta`. Positive legacy values enable persistent patch friction
with at most two anchors per region; the positional argument remains available
for compatibility.
