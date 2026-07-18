# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inverse-kinematics submodule."""

from .ik_common import IKJacobianType
from .ik_lbfgs_optimizer import IKOptimizerLBFGS
from .ik_lm_optimizer import IKOptimizerLM
from .ik_objectives import (
    IKObjective,
    IKObjectiveAxisAlignment,
    IKObjectiveJointLimit,
    IKObjectivePosition,
    IKObjectivePositionSet,
    IKObjectiveRotation,
    IKObjectiveRotationSet,
)
from .ik_solver import IKOptimizer, IKSampler, IKSolver
from .ik_trajectory_objectives import (
    IKObjectiveApparentGravity,
    IKObjectiveFootContact,
    IKObjectiveFootSkate,
    IKObjectiveGravityTorque,
    IKObjectiveJointReference,
    IKObjectiveSelfCollision,
    IKObjectiveSmoothness,
    IKObjectiveTemporal,
    IKObjectiveVelocityLimit,
    IKObjectiveWorldPlane,
    IKObjectiveWorldPlaneCapsule,
)
from .ik_trajectory_solver import IKLinearSolver, IKSolverTrajectory

__all__ = [
    "IKJacobianType",
    "IKLinearSolver",
    "IKObjective",
    "IKObjectiveApparentGravity",
    "IKObjectiveAxisAlignment",
    "IKObjectiveFootContact",
    "IKObjectiveFootSkate",
    "IKObjectiveGravityTorque",
    "IKObjectiveJointLimit",
    "IKObjectiveJointReference",
    "IKObjectivePosition",
    "IKObjectivePositionSet",
    "IKObjectiveRotation",
    "IKObjectiveRotationSet",
    "IKObjectiveSelfCollision",
    "IKObjectiveSmoothness",
    "IKObjectiveTemporal",
    "IKObjectiveVelocityLimit",
    "IKObjectiveWorldPlane",
    "IKObjectiveWorldPlaneCapsule",
    "IKOptimizer",
    "IKOptimizerLBFGS",
    "IKOptimizerLM",
    "IKSampler",
    "IKSolver",
    "IKSolverTrajectory",
]
