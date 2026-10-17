Add `propagation_constraint_max` to `SolverFeatherPGS` to size the propagation contact-row family independently of `free_body_constraint_max`, which then only has to hold the velocity-limit rows.
