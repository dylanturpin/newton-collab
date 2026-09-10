Add positional friction anchors to FeatherPGS (`friction_anchor_beta`): compatible
contact regions retain body-local material point pairs and the friction rows
correct their tangential separation, bounding accumulated drift of held objects.
Enabled by default at `friction_anchor_beta=0.2`; anchor history is independent
of collision contact matching. Set the gain to zero to explicitly opt out.
