# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from .feather_pgs import SolverFeatherPGS
from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags

try:
    from .implicit_mpm import SolverImplicitMPM
except Exception:  # warp.fem may be unavailable (e.g. omni.warp.core in GUI/Kit mode)
    SolverImplicitMPM = None
from .kamino import SolverKamino
from .mujoco import SolverMuJoCo
from .semi_implicit import SolverSemiImplicit
from .solver import SolverBase
from .style3d.solver_style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD

__all__ = [
    "SolverBase",
    "SolverFeatherPGS",
    "SolverFeatherstone",
    "SolverImplicitMPM",
    "SolverKamino",
    "SolverMuJoCo",
    "SolverNotifyFlags",
    "SolverSemiImplicit",
    "SolverStyle3D",
    "SolverVBD",
    "SolverXPBD",
]
