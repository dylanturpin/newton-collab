# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Releases of CUDA resources that wait until no graph capture is active.

Warp frees device memory capture-safely but destroys textures and pinned host memory immediately, which
invalidates an active stream capture. The garbage collector can finalize either inside any capture.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import warp as wp

_pending_releases: list[Callable[[], None]] = []


def release_deferred() -> None:
    """Run the releases a graph capture deferred, unless a capture is still active."""
    if not _pending_releases or _any_cuda_device_capturing():
        return
    pending = _pending_releases[:]
    del _pending_releases[: len(pending)]
    for release in pending:
        release()


class CaptureSafeTexture3D(wp.Texture3D):
    """A :class:`warp.Texture3D` whose destruction waits until no graph capture is active."""

    def __del__(self):
        try:
            if self.device.is_cuda:
                # The bound release keeps the texture alive until it runs; Python finalizes an object once.
                _release_or_defer(functools.partial(wp.Texture3D.__del__, self))
                return
        except (TypeError, AttributeError):
            # Interpreter shutdown, as in Warp's own finalizers.
            return
        super().__del__()


def make_pinned_release_capture_safe(array: wp.array) -> wp.array:
    """Make a pinned host array's ``cudaFreeHost`` wait until no graph capture is active."""
    free = array.deleter
    array.deleter = lambda ptr, capacity: _release_or_defer(functools.partial(free, ptr, capacity))
    return array


def _any_cuda_device_capturing() -> bool:
    return any(device.is_capturing for device in wp.get_cuda_devices())


def _release_or_defer(release: Callable[[], None]) -> None:
    if _any_cuda_device_capturing():
        _pending_releases.append(release)
        return
    release()
    release_deferred()
