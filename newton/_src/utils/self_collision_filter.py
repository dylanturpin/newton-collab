# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pose-sampled static self-collision filtering.

Robots with tightly clustered links (multi-finger hands, wrist stacks) carry
shape pairs whose bounding boxes overlap in essentially every reachable
configuration while their surfaces never come within contact range. Such
pairs pass the broad phase every step and pay narrow-phase distance queries
that can never produce a contact. This module samples the articulation's
joint space to identify those pairs so they can be excluded up front,
following the same idea as sampling-based allowed-collision-matrix
generation in motion-planning stacks.
"""

from __future__ import annotations

import numpy as np

from ..sim.articulation import eval_fk
from ..sim.builder import ModelBuilder
from ..sim.collide import CollisionPipeline
from ..sim.enums import JointType

__all__ = ["filter_static_self_collision_pairs", "find_static_self_collision_filters"]


def find_static_self_collision_filters(
    builder: ModelBuilder,
    num_samples: int = 256,
    seed: int = 42,
    min_candidate_fraction: float = 0.0,
    unbounded_half_range: float = 0.5 * np.pi,
    device=None,
) -> list[tuple[int, int]]:
    """Find shape pairs that reach the broad phase but never come within contact range.

    Samples ``num_samples`` joint configurations (plus the rest pose) uniformly
    within joint limits, runs the collision pipeline at each, and returns the
    body-body shape pairs that were broad-phase candidates in at least one
    sample (or ``min_candidate_fraction`` of them) while producing a contact
    (within the margin+gap detection shell) in **none** of them. This mirrors
    sampling-based allowed-collision-matrix generation: the safety of the
    "never touches" claim comes from the sample count, so raise
    ``num_samples`` for robots whose close-approach poses occupy a small
    fraction of joint space. Any pair that contacts in even one sample is
    kept.

    Only revolute, prismatic, and D6 joint coordinates are sampled; free and
    ball joints keep their rest configuration. Joints without finite limits
    are sampled in ``rest +- unbounded_half_range``. Links connected only by
    fixed joints (e.g. welded fingers) keep their relative pose in every
    sample, so their mutual never-touching classification is exact rather
    than statistical.

    Args:
        builder: The model builder to analyze. Not modified; a throwaway copy
            is finalized internally. Shape indices in the result refer to this
            builder's shapes.
        num_samples: Number of random joint configurations to sample.
        seed: RNG seed for deterministic results.
        min_candidate_fraction: Minimum fraction of samples in which a pair
            must appear as a broad-phase candidate to be considered for
            filtering. ``0.0`` (default) considers every observed candidate
            pair; raise it to restrict filtering to permanently-overlapping
            pairs only.
        unbounded_half_range: Sampling half-range [rad or m] around the rest
            coordinate for joints without finite limits.
        device: Device to run the sampling on. Defaults to the current device.

    Returns:
        Sorted list of ``(shape_a, shape_b)`` pairs (``shape_a < shape_b``)
        that never produced a contact in any sampled configuration.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    # Finalize a throwaway copy so the caller's builder stays untouched.
    # add_builder appends entities in order, so shape indices match.
    probe = ModelBuilder(up_axis=builder.up_axis)
    probe.rigid_gap = builder.rigid_gap
    probe.add_builder(builder)
    model = probe.finalize(device=device)

    state = model.state()
    pipeline = CollisionPipeline(model)
    contacts = pipeline.contacts()

    joint_type = model.joint_type.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()
    limit_lower = model.joint_limit_lower.numpy()
    limit_upper = model.joint_limit_upper.numpy()
    shape_body = model.shape_body.numpy()
    rest_q = model.joint_q.numpy().copy()

    sampled = []  # (q_index, lo, hi) triples for the coordinates we vary
    for j in range(model.joint_count):
        jt = joint_type[j]
        if jt not in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.D6):
            continue
        axis_count = int(joint_dof_dim[j, 0] + joint_dof_dim[j, 1])
        for a in range(axis_count):
            dof = int(joint_qd_start[j]) + a
            qi = int(joint_q_start[j]) + a
            lo = float(limit_lower[dof])
            hi = float(limit_upper[dof])
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                lo = rest_q[qi] - unbounded_half_range
                hi = rest_q[qi] + unbounded_half_range
            sampled.append((qi, lo, hi))

    rng = np.random.default_rng(seed)
    candidate_hits: dict[tuple[int, int], int] = {}
    contact_seen: set[tuple[int, int]] = set()
    total = 0

    for sample in range(num_samples + 1):
        q = rest_q.copy()
        if sample > 0:  # sample 0 is the rest pose
            for qi, lo, hi in sampled:
                q[qi] = rng.uniform(lo, hi)
        model.joint_q.assign(q)
        eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline.collide(state, contacts)

        pair_count = int(pipeline.broad_phase_pair_count.numpy()[0])
        if pair_count > 0:
            pairs = pipeline.broad_phase_shape_pairs.numpy()[:pair_count]
            for s0, s1 in pairs:
                if shape_body[s0] < 0 or shape_body[s1] < 0:
                    continue  # static (e.g. ground) pairs are not self-collision
                key = (int(min(s0, s1)), int(max(s0, s1)))
                candidate_hits[key] = candidate_hits.get(key, 0) + 1

        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        if contact_count > 0:
            c0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
            c1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
            for s0, s1 in zip(c0, c1, strict=True):
                contact_seen.add((int(min(s0, s1)), int(max(s0, s1))))
        total += 1

    min_hits = max(1.0, min_candidate_fraction * float(total))
    filters = [pair for pair, hits in candidate_hits.items() if hits >= min_hits and pair not in contact_seen]
    return sorted(filters)


def filter_static_self_collision_pairs(
    builder: ModelBuilder,
    num_samples: int = 256,
    seed: int = 42,
    min_candidate_fraction: float = 0.0,
    unbounded_half_range: float = 0.5 * np.pi,
    device=None,
) -> list[tuple[int, int]]:
    """Find and exclude broad-phase candidate pairs that never come within contact range.

    Convenience wrapper around :func:`find_static_self_collision_filters` that
    registers each returned pair via
    :meth:`ModelBuilder.add_shape_collision_filter_pair`. Call after adding the
    articulation to ``builder`` and before replicating or finalizing it.

    Returns:
        The list of filtered ``(shape_a, shape_b)`` pairs.
    """
    pairs = find_static_self_collision_filters(
        builder,
        num_samples=num_samples,
        seed=seed,
        min_candidate_fraction=min_candidate_fraction,
        unbounded_half_range=unbounded_half_range,
        device=device,
    )
    for shape_a, shape_b in pairs:
        builder.add_shape_collision_filter_pair(shape_a, shape_b)
    return pairs
