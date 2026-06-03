# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from typing import TYPE_CHECKING

from .feather_pgs import SolverFeatherPGS
from .featherstone import SolverFeatherstone
from .flags import SolverNotifyFlags
from .kamino import SolverKamino
from .mujoco import SolverMuJoCo
from .semi_implicit import SolverSemiImplicit
from .solver import SolverBase
from .style3d.solver_style3d import SolverStyle3D
from .vbd import SolverVBD
from .xpbd import SolverXPBD

if TYPE_CHECKING:
    from .implicit_mpm import SolverImplicitMPM

_LAZY_EXPORTS = {
    "SolverImplicitMPM": ".implicit_mpm",
}

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


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
