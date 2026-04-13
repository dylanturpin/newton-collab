# TGS Feasibility Study Notes

This report is the main artifact for the `dt/tgs-feather-pgs-study` branch. It records what the current Isaac Lift configuration actually does, what PhysX TGS means in source terms, and what Stage 2 should implement next.

## Scope of this pass

This is the Stage 1 slice only. No selectable TGS experiment path exists yet. The work here establishes the baseline configuration and answers whether “TGS” is likely to be a pure settings change or a deeper solver change.

Repository baseline for this report:

- `newton-collab`: `79f10ef391c3931d135db76c6b9fe572c09895b3`
- PhysX reference checkout: `/tmp/physics` at `ed6e5ca2474c9c80ad4f4826591b88476779c6ef`

## Current Isaac Lift setup

The task of interest is `Isaac-Lift-Cube-Franka-v0`, registered from `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py` and backed by `joint_pos_env_cfg.FrankaCubeLiftEnvCfg`.

The shared lift defaults in `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py` currently set:

- `self.decimation = 2`
- `self.sim.dt = 0.01`
- `self.sim.render_interval = self.decimation`
- `LiftPhysicsCfg.feather_pgs = NewtonCfg(..., num_substeps=2, solver_cfg=FeatherPGSSolverCfg(...))`

The FeatherPGS preset in the same file currently sets:

- `pgs_iterations=8`
- `pgs_beta=0.05`
- `pgs_omega=1.0`
- `pgs_cfm=1.0e-6`
- `dense_max_constraints=64`
- `enable_joint_limits=True`
- `pgs_warmstart=False`

The Franka-specific config in `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py` adds three limit-based early terminations on the arm joints:

- `joint_pos_out_of_limit`
- `joint_vel_out_of_limit`
- `joint_effort_out_of_limit`

This matters because the user-visible workaround under study is the velocity-limit termination. Stage 3 needs runs both with and without that guard.

## What Newton substeps mean today

The important timing relation is split across Isaac Lab config and Newton manager code.

- `decimation` is the number of physics frames per policy action. In Lift it is `2`.
- `sim.dt` is the physics frame size passed to the physics manager. In Lift it is `0.01` seconds.
- `num_substeps` is Newton-only solver subdivision inside one physics frame. In Lift FeatherPGS it is `2`.

`source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py` sets:

- `_num_substeps = cfg.num_substeps`
- `_solver_dt = cls.get_physics_dt() / cls._num_substeps`

and then runs the solver once per substep in:

    for i in range(cls._num_substeps):
        step_fn(cls._state_0, cls._state_1)

So the current Lift FeatherPGS schedule is:

- policy action period: `decimation * sim.dt = 2 * 0.01 = 0.02 s`
- physics frame size: `0.01 s`
- FeatherPGS solver step size inside a frame: `0.01 / 2 = 0.005 s`
- FeatherPGS solver calls per policy action: `decimation * num_substeps = 4`

This already gives FeatherPGS smaller outer timesteps than `sim.dt`, but it is still not PhysX TGS semantics.

## What FeatherPGS does inside one solver call

`newton/_src/solvers/feather_pgs/solver_feather_pgs.py` runs:

1. FK, inverse dynamics, and mass matrix setup.
2. Cholesky and predictor velocity construction.
3. Contact row building.
4. `pgs_iterations` projected Gauss-Seidel iterations in dense, split, or matrix-free form.
5. A single final integration pass in `_stage6_integrate(...)`.

The key point is that FeatherPGS does not update generalized coordinates between PGS iterations. The inner PGS loop operates on velocities and impulses, then the solver integrates once at the end of the step. That means increasing `pgs_iterations` is not equivalent to shrinking the step and re-evaluating contacts/poses each inner iteration.

## What PhysX TGS means in source terms

The PhysX public description in `physx/include/PxSceneDesc.h` is already stronger than “more solver iterations”:

- `PxSolverType::eTGS` is described as a non-linear iterative solver.
- The same header states that TGS applies friction throughout all position and velocity iterations, while PGS has different friction scheduling.
- `PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS` applies gravity and other external forces in each TGS position iteration sub-time-step, explicitly changing dynamics as iteration count changes.

The implementation-facing immediate-mode API in `physx/include/PxImmediateMode.h` exposes distinct TGS entry points such as:

- `PxComputeUnconstrainedVelocitiesTGS(...)`
- `PxCreateContactConstraintsTGS(...)`
- `PxSolveConstraintsTGS(...)`
- `PxIntegrateSolverBodiesTGS(...)`
- `PxUpdateArticulationBodiesTGS(...)`

The clearest operational example is `physx/snippets/snippetimmediatearticulation/SnippetImmediateArticulation.cpp`, which computes:

    const float stepDt = dt / float(gNbIterPos);

and then passes `stepDt` into `PxSolveConstraintsTGS(...)`, followed by body and articulation updates through `PxIntegrateSolverBodiesTGS(...)` and `PxUpdateArticulationBodiesTGS(...)`.

The strongest plain-language comment is in `physx/source/lowleveldynamics/shared/DyCpuGpu1dConstraint.h`, which defines the TGS position-iteration time step explicitly as:

    stepDt = simDt / posIterationCount

and then discusses how geometric error and elapsed time are recomputed as transforms are integrated during position iterations.

## Initial conclusion

PhysX TGS is not just “run more PGS iterations.” It changes the solve semantics in at least three ways that FeatherPGS does not currently reproduce:

- it uses an explicit per-position-iteration timestep (`stepDt`)
- it updates poses and articulation bodies during the TGS solve, not only after the solve
- it applies some force and friction logic with TGS-specific per-iteration behavior

That said, the user’s proposed approximation is still a valid Stage 2 experiment. In this codebase the closest low-risk approximation is:

1. increase `decimation`
2. decrease `sim.dt` by the same factor
3. keep `num_substeps` explicit and simple, ideally `1` for the new smaller frame unless profiling says otherwise
4. leave baseline FeatherPGS untouched and expose the new regime as a selectable preset

That experiment should be described as “smaller-step FeatherPGS at fixed control rate,” not “true TGS FeatherPGS.”

## Stage 2 recommendation

Stage 2 should implement an additive configuration path for `Isaac-Lift-Cube-Franka-v0` that makes the smaller-step regime selectable without hand edits. The first version does not need solver-kernel changes. It should:

- preserve the current FeatherPGS baseline untouched
- add a named preset or variant for the smaller-step schedule
- make it easy in Stage 3 to run with and without the Franka velocity-limit termination

If that smaller-step path materially reduces velocity spikes or unlocks stable learning, it would justify a later deeper milestone to prototype true TGS-like inner-step pose updates in FeatherPGS. If it does not, a deeper TGS port is probably not worth the complexity.

## Commands run in Stage 1

Configuration inspection:

    sed -n '1,240p' /workspace/skild-IL-solver/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py
    sed -n '1,260p' /workspace/skild-IL-solver/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py
    sed -n '430,550p' /workspace/skild-IL-solver/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py
    sed -n '190,330p' /workspace/skild-IL-solver/source/isaaclab_newton/isaaclab_newton/physics/newton_manager_cfg.py

FeatherPGS inspection:

    rg -n "def step\\(|pgs_iterations|integrate|pgs_mode" /workspace/newton-collab/newton/_src/solvers/feather_pgs/solver_feather_pgs.py /workspace/newton-collab/newton/_src/solvers/feather_pgs/kernels.py
    sed -n '1378,1475p' /workspace/newton-collab/newton/_src/solvers/feather_pgs/solver_feather_pgs.py
    sed -n '2844,2898p' /workspace/newton-collab/newton/_src/solvers/feather_pgs/solver_feather_pgs.py

PhysX inspection:

    git clone --depth 1 https://github.com/NVIDIA-Omniverse/PhysX.git /tmp/physics
    sed -n '56,92p' /tmp/physics/physx/include/PxSceneDesc.h
    sed -n '270,292p' /tmp/physics/physx/include/PxSceneDesc.h
    sed -n '568,650p' /tmp/physics/physx/include/PxImmediateMode.h
    sed -n '640,716p' /tmp/physics/physx/source/lowleveldynamics/shared/DyCpuGpu1dConstraint.h
    sed -n '1760,1945p' /tmp/physics/physx/snippets/snippetimmediatearticulation/SnippetImmediateArticulation.cpp

## Validation recorded for this slice

This pass only changed documentation artifacts. Focused validation was:

    git -C /workspace/newton-collab diff --check
    python3 -c "from pathlib import Path; assert Path('/workspace/newton-collab/report.md').read_text(); assert 'Stage 2 recommendation' in Path('/workspace/newton-collab/report.md').read_text(); assert 'Stage 1 slice completed' in Path('/workspace/.agent/execplans/tgs-feather-pgs-investigation.md').read_text()"
