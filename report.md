# TGS Feasibility Study Notes

This report is the main artifact for the `dt/tgs-feather-pgs-study` branch. It records what the current Isaac Lift configuration actually does, what PhysX TGS means in source terms, and the selectable smaller-step FeatherPGS path added in Stage 2.

## Scope of this pass

This report now covers Stage 1, the Stage 2 implementation slice, and a Stage 3A bring-up pass that narrowed the current runtime blockers. The codebase now has a selectable smaller-step FeatherPGS path for Franka lift, but it still does not have completed rollout, training, or throughput comparisons.

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

Stage 2 is now implemented as additive task variants in `skild-IL-solver`. The current baseline stays untouched, and Stage 3 can select the smaller-step path without local edits.

## Selectable Stage 2 path

The Stage 2 implementation lives in:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py`

The important correction from Stage 1 is that the existing Franka FeatherPGS baseline already executes four FeatherPGS solver calls per action:

- `decimation=2`
- `sim.dt=0.01`
- `num_substeps=2`
- solver step size `0.01 / 2 = 0.005 s`
- solver calls per action `2 * 2 = 4`

Because of that, a naive “`decimation=4`, `dt=0.005`, `num_substeps=1`” variant would not actually increase solver update frequency. It would preserve the same `0.005 s` FeatherPGS step size and the same four solver calls per action.

The implemented smaller-step path instead halves the FeatherPGS step size and doubles solver updates per action:

- `Isaac-Lift-Cube-Franka-FeatherPGS-TGS-v0`
- `Isaac-Lift-Cube-Franka-FeatherPGS-TGS-Play-v0`
- `Isaac-Lift-Cube-Franka-FeatherPGS-TGS-NoVelGuard-v0`
- `Isaac-Lift-Cube-Franka-FeatherPGS-TGS-NoVelGuard-Play-v0`

Those variants use:

- `decimation=8`
- `sim.dt=0.0025`
- `sim.render_interval=8`
- `LiftPhysicsCfg.feather_pgs_tgs`
- `num_substeps=1`

So the new schedule is:

- policy action period: `8 * 0.0025 = 0.02 s`
- FeatherPGS solver step size: `0.0025 / 1 = 0.0025 s`
- FeatherPGS solver calls per action: `8 * 1 = 8`

The `NoVelGuard` variants set `joint_vel_out_of_limit = None` and leave the position and effort guards unchanged. This gives Stage 3 a direct diagnosis path for “smaller-step FeatherPGS with the guard” versus “smaller-step FeatherPGS without the guard.”

This is still a smaller-step approximation, not a true PhysX-style TGS port. It changes the outer simulation schedule, not the solver’s inner “update poses during position iterations” semantics.

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

Stage 2 config validation:

    python3 - <<'PY'
    import sys
    sys.path.insert(0, '/workspace/skild-IL-solver/source/isaaclab_tasks')
    sys.path.insert(0, '/workspace/skild-IL-solver/source/isaaclab_newton')
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    base = load_cfg_from_registry('Isaac-Lift-Cube-Franka-v0', 'env_cfg_entry_point')
    tgs = load_cfg_from_registry('Isaac-Lift-Cube-Franka-FeatherPGS-TGS-v0', 'env_cfg_entry_point')
    no_guard = load_cfg_from_registry('Isaac-Lift-Cube-Franka-FeatherPGS-TGS-NoVelGuard-v0', 'env_cfg_entry_point')
    play = load_cfg_from_registry('Isaac-Lift-Cube-Franka-FeatherPGS-TGS-Play-v0', 'env_cfg_entry_point')

    assert base.decimation == 2
    assert abs(base.sim.dt - 0.01) < 1e-12
    assert base.sim.physics.feather_pgs.num_substeps == 2

    assert tgs.decimation == 8
    assert abs(tgs.sim.dt - 0.0025) < 1e-12
    assert tgs.sim.render_interval == 8
    assert tgs.sim.physics.num_substeps == 1
    assert tgs.terminations.joint_vel_out_of_limit is not None

    assert no_guard.terminations.joint_vel_out_of_limit is None
    assert play.scene.num_envs == 50
    assert play.observations.policy.enable_corruption is False
    print('lift_tgs_config_ok')
    PY

## Validation recorded for this slice

Stage 1 validation remains the same. Stage 2 focused validation was:

    python3 - <<'PY'
    import sys
    sys.path.insert(0, '/workspace/skild-IL-solver/source/isaaclab_tasks')
    sys.path.insert(0, '/workspace/skild-IL-solver/source/isaaclab_newton')
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    ...
    print('lift_tgs_config_ok')
    PY
    git -C /workspace/skild-IL-solver diff --check
    git -C /workspace/newton-collab diff --check

## Stage 3A runtime bring-up findings

This pass did not produce the requested MJWarp versus FeatherPGS versus TGS-style comparison plots yet. It did establish the current launch recipe and the concrete blockers that prevented a same-pass comparison run.

The first blocker was non-interactive Kit bootstrap. Isaac Lab in this workspace prompts for the Omniverse EULA unless `OMNI_KIT_ACCEPT_EULA=YES` is exported. Without that variable, even a trivial inline Python launcher fails before simulation startup with:

    Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF when reading a line

The second blocker is the intended PhysX baseline. After setting `OMNI_KIT_ACCEPT_EULA=YES`, the default Franka lift task still cannot launch in this workspace when `sim.physics` remains PhysX-backed because the environment is missing `omni.physics`:

    ValueError: Could not resolve the input string 'isaaclab_physx.physics.physx_manager:PhysxManager' into callable object.
    Received the error:
     No module named 'omni.physics'.

That means the practical Stage 3 rollout harness in this workspace must compare Newton-only configurations: MJWarp via an explicit `NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1, ...)`, baseline FeatherPGS via `LiftPhysicsCfg().feather_pgs`, and the Stage 2 TGS-style task or equivalent `LiftPhysicsCfg().feather_pgs_tgs`.

The Newton-only FeatherPGS bring-up got furthest with this command:

    cd /workspace/skild-IL-solver
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p - <<'PY'
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftPhysicsCfg

    cfg = load_cfg_from_registry('Isaac-Lift-Cube-Franka-v0', 'env_cfg_entry_point')
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    cfg.seed = 7
    cfg.sim.physics = LiftPhysicsCfg().feather_pgs
    env = gym.make('Isaac-Lift-Cube-Franka-v0', cfg=cfg)
    obs, info = env.reset()
    ...
    PY

That run progressed past config parsing and USD stage construction, then emitted rigid-body inertia warnings such as:

    Warning: The rigid body at /World/envs/env_0/Object has a possibly invalid inertia tensor ...

but it did not reach the first printed rollout metrics within the pass after more than 60 seconds of runtime. I therefore do not treat it as completed rollout evidence yet.

The standalone throughput path is also not ready yet. Even timeout-bounded minimal benchmark runs failed to complete and produced no artifact directory:

    cd /workspace/newton-collab
    timeout 120 python3 newton/tools/solver_benchmark.py --benchmark --scenario g1_flat --solver mujoco --num-worlds 64 --warmup-frames 5 --measure-frames 10 --viewer null --out /tmp/tgs_bench_mujoco
    timeout 30 python3 newton/tools/solver_benchmark.py --benchmark --scenario g1_flat --solver mujoco --num-worlds 1 --warmup-frames 1 --measure-frames 1 --viewer null --out /tmp/tgs_bench_mujoco_1

Both commands exited with status `124`, and `/tmp/tgs_bench_mujoco*` was not created afterward.

## Current recommendation after Stage 3A

The next pass should stay narrow:

- finish one completed Newton-only short rollout harness that prints per-step metrics for MJWarp, baseline FeatherPGS, and the TGS-style FeatherPGS path
- only after that succeeds, start the longer Franka lift training runs
- treat PhysX baseline and standalone throughput as environment blockers until `omni.physics` and/or a completing benchmark path is available in this workspace
