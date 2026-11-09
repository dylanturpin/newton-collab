# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact adhesion: a per-contact pull of ``steel:adhesion`` [N] solved by a change of variables.

The sweeps solve ``lambda' = lambda + A dt >= 0`` in their unchanged cones while the velocity carries
``Y (lambda' - A dt)``: a cold row starts at ``lambda' = A dt`` with no velocity change, and a warm-started
row installs its carried impulse less the pull where warm-start impulses are installed.
"""

import warnings

import numpy as np
import warp as wp

from ...exceptions import NewtonWarning
from .kernels import WARP_DETERMINISM_OFF

# SolverSteel rejects models with ``requires_grad``, so no kernel needs an adjoint.
wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

ATTRIBUTE = "adhesion"


def configure_contact_adhesion(solver, enable_sleeping: bool) -> None:
    """Enable adhesion when any shape has a positive ``steel:adhesion``, and reject unsupported options."""
    values = _adhesion_values(solver)
    _validate(values)
    solver._adhesion_enabled = bool(np.any(values > 0.0))
    solver._contact_adhesion_impulse = None
    if not solver._adhesion_enabled:
        return
    unsupported = [
        name
        for name, requested in (
            ("enable_sleeping=True", enable_sleeping),
            ("contact_compliance=True", solver._contact_compliance),
            ("contact_regularization > 0", solver._regularization_enabled),
            ("contact torsion", solver._contact_torsion_enabled),
        )
        if requested
    ]
    if unsupported:
        raise NotImplementedError(f"steel:adhesion does not support {', '.join(unsupported)}")


def validate_contact_adhesion(solver) -> None:
    """Re-check ``steel:adhesion`` on the host after a shape notification."""
    values = _adhesion_values(solver)
    _validate(values)
    if not solver._adhesion_enabled and np.any(values > 0.0):
        warnings.warn(
            "steel:adhesion is ignored by a solver built without adhesive shapes; reconstruct the solver",
            NewtonWarning,
            stacklevel=3,
        )


def allocate_contact_adhesion(solver) -> None:
    """Allocate the per-contact pull and one net-impulse scratch per row family, so steps can be captured."""
    device = solver.model.device
    solver._contact_adhesion_impulse = wp.zeros(solver._max_contacts_alloc, dtype=wp.float32, device=device)
    families = (solver._impulses, solver._mf_impulses, solver._propagation_impulses)
    solver._adhesion_scratch = {route: wp.empty_like(impulses) for route, impulses in enumerate(families)}


def prepare_contact_adhesion(solver, contacts, dt: float, seed: bool) -> None:
    """Write each contact's pull impulse ``A dt`` for the rows it has this step, zero for the others.

    With ``seed``, also add the pull to the contact's cold-started normal row.
    """
    model = solver.model
    wp.launch(
        compute_contact_adhesion_impulse,
        dim=solver._contact_adhesion_impulse.shape[0],
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            solver._contact_world,
            solver._contact_slot,
            solver._contact_path,
            solver._constraint_count,
            solver._mf_constraint_count,
            solver._propagation_constraint_count,
            model.steel.adhesion,
            dt,
            int(seed),
        ],
        outputs=[
            solver._contact_adhesion_impulse,
            solver._impulses,
            solver._mf_impulses,
            solver._propagation_impulses,
        ],
        device=model.device,
    )


def net_row_impulses(solver, route: int, impulses: wp.array) -> wp.array:
    """Return a copy of a row family's impulses with each adhesive normal row's pull ``A dt`` subtracted."""
    scratch = solver._adhesion_scratch[route]
    wp.copy(scratch, impulses)
    pull = solver._contact_adhesion_impulse
    wp.launch(
        subtract_contact_adhesion_impulse,
        dim=pull.shape[0],
        inputs=[solver._contact_world, solver._contact_slot, solver._contact_path, pull, route],
        outputs=[scratch],
        device=solver.model.device,
    )
    return scratch


@wp.kernel
def compute_contact_adhesion_impulse(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    propagation_constraint_count: wp.array[wp.int32],
    shape_adhesion: wp.array[wp.float32],
    dt: float,
    seed: int,
    # outputs
    contact_adhesion_impulse: wp.array[wp.float32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    propagation_impulses: wp.array2d[wp.float32],
):
    """Write ``(a0 + a1) dt`` for a contact with a row, zero for dropped and absent contacts.

    With ``seed``, also add it to the contact's normal row.
    """
    c = wp.tid()
    impulse = float(0.0)
    if c < wp.min(contact_count[0], contact_shape0.shape[0]):
        slot = contact_slot[c]
        path = contact_path[c]
        if slot >= 0 and path >= 0:
            world = contact_world[c]
            count = mf_constraint_count[world]
            if path == 0:
                count = world_constraint_count[world]
            elif path == 2:
                count = propagation_constraint_count[world]
            if slot < count:
                shape_a = contact_shape0[c]
                shape_b = contact_shape1[c]
                force = float(0.0)
                if shape_a >= 0:
                    force += shape_adhesion[shape_a]
                if shape_b >= 0:
                    force += shape_adhesion[shape_b]
                impulse = force * dt
                if seed != 0 and impulse != 0.0:
                    if path == 0:
                        world_impulses[world, slot] = world_impulses[world, slot] + impulse
                    elif path == 1:
                        mf_impulses[world, slot] = mf_impulses[world, slot] + impulse
                    else:
                        propagation_impulses[world, slot] = propagation_impulses[world, slot] + impulse
    contact_adhesion_impulse[c] = impulse


@wp.kernel
def subtract_contact_adhesion_impulse(
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    contact_adhesion_impulse: wp.array[wp.float32],
    route: int,
    # in/out
    impulses: wp.array2d[wp.float32],
):
    """Subtract each adhesive contact's pull from its normal row in the row family ``route``."""
    c = wp.tid()
    impulse = contact_adhesion_impulse[c]
    if impulse != 0.0:
        if contact_path[c] == route:
            world = contact_world[c]
            slot = contact_slot[c]
            impulses[world, slot] = impulses[world, slot] - impulse


def _adhesion_values(solver) -> np.ndarray:
    model = solver.model
    namespace = getattr(model, "steel", None)
    array = getattr(namespace, ATTRIBUTE, None)
    if array is None:
        return np.zeros(0, dtype=np.float32)
    values = array.numpy()
    if values.shape != (model.shape_count,):
        raise ValueError(f"steel:adhesion has shape {values.shape}, expected ({model.shape_count},)")
    variant = model._variant_shape_custom.get(f"steel:{ATTRIBUTE}")
    if variant is not None:
        _rows, options = model._variant_row_options()
        values = np.concatenate((values, variant[0].numpy()[options]))
    return values


def _validate(values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("steel:adhesion must be finite and non-negative [N]")
