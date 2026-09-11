# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve the tangential friction block at a fixed normal load."""

import warp as wp


@wp.func
def friction_pair_candidate(
    a: float, c: float, d: float, residual: wp.vec2, old: wp.vec2, radius: float, omega: float
) -> wp.vec2:
    """Minimize the tangent quadratic on a disk, then apply relaxation.

    The caller projects the relaxed result back onto the disk. Sticking uses
    the block inverse; sliding solves (A + alpha I)x = b with |x| = radius.
    Incremental sticking avoids perturbing fixed points. An orthonormal
    eigenbasis avoids cancellation in nearly singular blocks.
    Bisection retains the feasible endpoint. Normal load is fixed in this solve.
    """
    if radius <= 0.0:
        return wp.vec2(0.0)
    scale = wp.max(wp.max(a, d), 1.0e-20)
    a /= scale
    c /= scale
    d /= scale
    largest = 0.5 * (a + d + wp.sqrt((a - d) * (a - d) + 4.0 * c * c))
    smallest = float(0.0)
    if largest > 0.0:
        smallest = wp.max((a * d - c * c) / largest, 0.0)
    axis = wp.vec2(c, largest - a)
    if a >= d:
        axis = wp.vec2(largest - d, c)
    if wp.length(axis) > 0.0:
        axis = wp.normalize(axis)
    else:
        axis = wp.vec2(1.0, 0.0)
    perpendicular = wp.vec2(-axis[1], axis[0])
    residual_rotated = wp.vec2(wp.dot(axis, residual), wp.dot(perpendicular, residual)) / scale
    result = old
    sticking = bool(False)
    if largest > 0.0 and (smallest > 0.0 or residual_rotated[1] == 0.0):
        correction = (residual_rotated[0] / largest) * axis
        if smallest > 0.0:
            correction += (residual_rotated[1] / smallest) * perpendicular
        # Incremental sticking preserves a feasible zero-residual impulse exactly.
        result = old - correction
        sticking = wp.length(result) <= radius
    if not sticking:
        old_rotated = wp.vec2(wp.dot(axis, old), wp.dot(perpendicular, old))
        b = wp.vec2(largest * old_rotated[0], smallest * old_rotated[1]) - residual_rotated
        solution = wp.vec2(0.0)
        lo = float(0.0)
        hi = wp.length(b) / radius
        if hi > 0.0:
            for _ in range(24):
                alpha = 0.5 * (lo + hi)
                trial = wp.vec2(b[0] / (largest + alpha), b[1] / (smallest + alpha))
                if wp.length(trial) > radius:
                    lo = alpha
                else:
                    hi = alpha
            solution = wp.vec2(b[0] / (largest + hi), b[1] / (smallest + hi))
        result = solution[0] * axis + solution[1] * perpendicular
    return old + omega * (result - old)


# Native kernels use the same bounded solve as the Warp function above.
FRICTION_PAIR_CUDA = """
    const auto friction_pair_candidate = [](
        float a, float c, float d, float r0, float r1,
        float old0, float old1, float radius, float omega) {
        if (radius <= 0.0f) return make_float2(0.0f, 0.0f);
        float scale = fmaxf(fmaxf(a, d), 1.0e-20f);
        a /= scale; c /= scale; d /= scale;
        float largest = 0.5f * (a + d + sqrtf((a-d)*(a-d) + 4.0f*c*c));
        float smallest = largest > 0.0f ? fmaxf((a*d-c*c) / largest, 0.0f) : 0.0f;
        float vx = c, vy = largest-a;
        if (a >= d) { vx = largest-d; vy = c; }
        float norm = sqrtf(vx*vx + vy*vy);
        if (norm > 0.0f) { vx /= norm; vy /= norm; }
        else { vx = 1.0f; vy = 0.0f; }
        float r0_rotated = (vx*r0 + vy*r1) / scale;
        float r1_rotated = (-vy*r0 + vx*r1) / scale;
        float result0 = old0, result1 = old1;
        bool sticking = false;
        if (largest > 0.0f && (smallest > 0.0f || r1_rotated == 0.0f)) {
            float dx = r0_rotated / largest;
            float dy = smallest > 0.0f ? r1_rotated / smallest : 0.0f;
            result0 = old0 - (vx*dx - vy*dy);
            result1 = old1 - (vy*dx + vx*dy);
            sticking = sqrtf(result0*result0 + result1*result1) <= radius;
        }
        if (!sticking) {
            float b0 = largest * (vx*old0 + vy*old1) - r0_rotated;
            float b1 = smallest * (-vy*old0 + vx*old1) - r1_rotated;
            float x = 0.0f, y = 0.0f;
            float lo = 0.0f;
            float hi = sqrtf(b0*b0 + b1*b1) / radius;
            if (hi > 0.0f) {
                for (int iteration = 0; iteration < 24; ++iteration) {
                    float alpha = 0.5f * (lo + hi);
                    float tx = b0 / (largest + alpha);
                    float ty = b1 / (smallest + alpha);
                    if (sqrtf(tx*tx + ty*ty) > radius) lo = alpha;
                    else hi = alpha;
                }
                x = b0 / (largest + hi);
                y = b1 / (smallest + hi);
            }
            result0 = vx*x - vy*y;
            result1 = vy*x + vx*y;
        }
        return make_float2(old0 + omega*(result0-old0), old1 + omega*(result1-old1));
    };
"""
