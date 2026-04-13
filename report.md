# TGS Feasibility Study Notes

This report is the main artifact for the `dt/tgs-feather-pgs-study` branch. It records what the current Isaac Lift configuration actually does, what PhysX TGS means in source terms, the selectable smaller-step FeatherPGS path added in Stage 2, the completed Newton-only rollout comparison from Stage 3B, and the completed Stage 3C training study up through the monitored 4096-environment runs.

## Scope of this pass

This report now covers Stage 1, the Stage 2 implementation slice, the Stage 3A bring-up pass that narrowed the runtime blockers, the completed Stage 3B rollout comparison slice, the initial Stage 3C boot-only guard comparison, and the monitored 4096-environment Stage 3C training study. The codebase now has selectable Newton Franka lift tasks for MJWarp, baseline FeatherPGS, and the smaller-step TGS-style FeatherPGS path, and the report now includes training-quality and throughput evidence for those runnable Newton baselines in this workspace. Stage 4 final recommendation work is still outstanding.

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

## Stage 3B rollout comparison

Stage 3B is now implemented with a checked-in Newton-only harness at:

- `skild-IL-solver/scripts/benchmarks/compare_franka_lift_newton_rollout.py`

The script runs one short deterministic rollout case at a time and prints a JSON summary containing:

- mean wall-clock time per environment step
- peak absolute joint velocity and peak velocity-limit ratio
- object height and object linear speed bounds
- which termination terms fired and on which steps

The Stage 3B command set was:

    cd /workspace/skild-IL-solver
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/benchmarks/compare_franka_lift_newton_rollout.py --case mjwarp --num_envs 1 --steps 32 --headless --summary-json /tmp/franka_rollout_mjwarp.json
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/benchmarks/compare_franka_lift_newton_rollout.py --case feather_pgs --num_envs 1 --steps 32 --headless --summary-json /tmp/franka_rollout_feather_pgs.json
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/benchmarks/compare_franka_lift_newton_rollout.py --case feather_pgs_tgs --num_envs 1 --steps 32 --headless --summary-json /tmp/franka_rollout_feather_pgs_tgs.json

Each run used:

- seed `7`
- `num_envs=1`
- `steps=32`
- action mode `seeded-random`
- action scale `0.2`

The resulting summaries were:

    {
      "case": "mjwarp",
      "mean_step_sec": 2.9311,
      "peak_joint_vel": 0.0,
      "peak_joint_vel_ratio": 0.0,
      "peak_object_speed": 0.0,
      "terminated_steps": [0, 1, ..., 31],
      "termination_counts": {"joint_vel_out_of_limit": 32, ...}
    }

    {
      "case": "feather_pgs",
      "mean_step_sec": 0.0803,
      "peak_joint_vel": 1.9995,
      "peak_joint_vel_ratio": 2.7522,
      "peak_object_speed": 0.7848,
      "terminated_steps": [],
      "termination_counts": {"joint_vel_out_of_limit": 0, ...}
    }

    {
      "case": "feather_pgs_tgs",
      "mean_step_sec": 0.1480,
      "peak_joint_vel": 1.9658,
      "peak_joint_vel_ratio": 2.7272,
      "peak_object_speed": 0.7848,
      "terminated_steps": [],
      "termination_counts": {"joint_vel_out_of_limit": 0, ...}
    }

The practical Stage 3B conclusions from those runs are:

- The short deterministic seeded-random rollout does not show an obvious stability win for the smaller-step TGS-style path over baseline FeatherPGS. Their peak joint velocity, peak normalized velocity, object-speed envelope, reward total, and no-reset behavior are nearly identical over 32 environment steps.
- The smaller-step path is slower than baseline FeatherPGS in this harness. Baseline FeatherPGS averaged `0.0803 s` per environment step, while the TGS-style path averaged `0.1480 s` per step, roughly `1.84x` slower.
- The current MJWarp comparison path is not healthy in this workspace for Franka lift under the same seeded action trace. It terminated on every single step with `joint_vel_out_of_limit`, and it was also dramatically slower than either FeatherPGS path because solver initialization alone took about `39 s`.

This is enough to answer the Stage 3B question at rollout level:

- baseline FeatherPGS and the TGS-style smaller-step path both complete the same short Newton-only rollout without resets
- the TGS-style path does not obviously reduce short-horizon velocity spikes relative to baseline FeatherPGS
- the available MJWarp baseline path is currently a failure case rather than a stable reference in this workspace

The MJWarp result needs one caution. The harness reads kinematic state after `env.step(...)`, and Isaac Lab auto-resets terminated environments inside `step()`. So the `peak_joint_vel=0.0` result for MJWarp should be interpreted as “the environment reset immediately on `joint_vel_out_of_limit` before a useful post-step state was observable,” not as proof that the offending pre-reset velocity was literally zero.

## Stage 3C training slice: velocity guard comparison

The first completed Stage 3C slice stayed deliberately narrow. Instead of trying to solve throughput and longer-horizon baseline parity in the same pass, it answered the immediate reviewer question: what happens if the TGS-style Franka lift task trains with and without `joint_vel_out_of_limit`?

The command set was:

    cd /workspace/skild-IL-solver
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-FeatherPGS-TGS-v0 --num_envs 32 --seed 7 --max_iterations 2 --headless
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-FeatherPGS-TGS-NoVelGuard-v0 --num_envs 32 --seed 7 --max_iterations 2 --headless

Both runs used:

- `num_envs=32`
- `seed=7`
- `max_iterations=2`
- the same TGS-style smaller-step FeatherPGS schedule from Stage 2 (`dt=0.0025`, `decimation=8`, `num_substeps=1`)

The generated log directories were:

- guarded TGS: `skild-IL-solver/logs/rsl_rl/franka_lift/2026-04-13_17-15-13`
- no-guard TGS: `skild-IL-solver/logs/rsl_rl/franka_lift/2026-04-13_17-16-18`

The dumped `env.yaml` files confirm that these two runs match on solver schedule and differ only in the velocity-limit termination term:

- guarded run: `joint_vel_out_of_limit` is present
- no-guard run: `joint_vel_out_of_limit: null`

The key iteration-2 metrics from the TensorBoard event files were:

    guarded TGS (2026-04-13_17-15-13)
      Train/mean_reward = 0.3553
      Train/mean_episode_length = 1.01
      Perf/total_fps = 214
      Episode_Termination/joint_vel_out_of_limit = 1.0
      Metrics/object_pose/position_error = 0.2093

    no-guard TGS (2026-04-13_17-16-18)
      Train/mean_reward = 0.7557
      Train/mean_episode_length = 27.5
      Perf/total_fps = 225
      Episode_Termination/time_out = 0.0391
      Metrics/object_pose/position_error = 0.2068

The console logs tell the same story from the first iteration onward:

- guarded TGS reached mean episode length `1.00` at iteration 1 and `1.01` at iteration 2, with `Episode_Termination/joint_vel_out_of_limit: 1.0000` throughout
- no-guard TGS reached mean episode length `12.00` at iteration 1 and `27.50` at iteration 2, with no velocity-limit term remaining and only a small `time_out` fraction

The practical conclusion from this slice is direct: on the selectable smaller-step FeatherPGS path, the velocity-limit guard is still the dominant early-learning failure mode in this workspace. Removing only that guard materially changes training behavior immediately, while keeping the same solver settings leaves the agent trapped in one-step episodes.

This slice does not settle the full TGS question yet. It does not compare against baseline FeatherPGS training under a like-for-like selectable Newton training path, and it does not answer the throughput question beyond showing that guarded and no-guard runs under the same TGS-style schedule landed at similar training-loop FPS (`214` versus `225`).

Human review changed how to interpret this slice. The `32`-environment, `2`-iteration runs are useful only as boot checks. They establish that the selectable task IDs launch and that removing `joint_vel_out_of_limit` changes the immediate failure mode, but they are not sufficient for training-quality conclusions.

## Stage 3C monitored 4096-env training study

To close Stage 3C at a meaningful scale, this pass used three monitored 4096-environment runs and compared them against an external reference curve supplied by the investigator. That reference says a healthy Franka lift run should show visible reward signal by about iteration `100`, strong growth by about `200`, and a plateau by about `300-500`. In approximate mean-reward terms, the supplied baseline landmarks were:

- FeatherPGS reference: about `5` at `0-50`, `15` at `100`, `20` at `150`, `35` at `200`, and `75-88` by `300-600`
- PhysX reference: about `5` at `0-50`, `20` at `100`, `80` at `150`, `120` at `200`, and `120-140` by `250-400`
- MJWarp reference: about `5` at `0-50`, `25` at `100`, `75` at `150`, `110` at `200`, and roughly `120-135` by `250-400`

Those reference numbers matter because this workspace still cannot run the stock PhysX-backed `Isaac-Lift-Cube-Franka-v0` task. The task resolves to a PhysX manager that imports `omni.physics`, and that module is still missing here. To keep the comparison reproducible without ad hoc config edits, this pass added explicit Newton task IDs:

- `Isaac-Lift-Cube-Franka-MJWarp-v0`
- `Isaac-Lift-Cube-Franka-FeatherPGS-v0`
- `Isaac-Lift-Cube-Franka-FeatherPGS-NoVelGuard-v0`
- plus their `-Play-v0` variants

The monitored command set was:

    cd /workspace/skild-IL-solver
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-MJWarp-v0 --num_envs 4096 --seed 7 --max_iterations 100 --headless 2>&1 | tee /tmp/mjwarp_4096_train.log
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-FeatherPGS-TGS-NoVelGuard-v0 --num_envs 4096 --seed 7 --max_iterations 300 --headless 2>&1 | tee /tmp/tgs_noguard_4096_train.log
    OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-Franka-FeatherPGS-v0 --num_envs 4096 --seed 7 --max_iterations 300 --headless 2>&1 | tee /tmp/featherpgs_4096_train.log

The generated log directories were:

- MJWarp: `skild-IL-solver/logs/rsl_rl/franka_lift/2026-04-13_17-34-19`
- TGS no-guard: `skild-IL-solver/logs/rsl_rl/franka_lift/2026-04-13_17-40-30`
- baseline FeatherPGS: `skild-IL-solver/logs/rsl_rl/franka_lift/2026-04-13_17-53-44`

The monitoring rule was to inspect progress roughly every `50` iterations and kill a run once it had clearly flatlined after more than `100` iterations. That rule changed two of the runs:

- TGS no-guard was launched for `300` iterations but stopped manually after iteration `207` because reward plateaued in the mid-teens between iterations `100` and `200`
- baseline FeatherPGS was launched for `300` iterations but stopped manually after iteration `167` because reward remained near `1-2`, far below both the reference FeatherPGS curve and the TGS no-guard run

The key checkpoints from the captured trainer logs were:

    MJWarp (17:34:19, completed 100 iterations in 00:04:57)
      iter 0:   reward 0.35, episode length 1.02, steps/s 12639, joint_vel_out_of_limit 0.9996
      iter 50:  reward 0.68, episode length 1.95, steps/s 32887, joint_vel_out_of_limit 1.0000
      iter 99:  reward 0.70, episode length 1.99, steps/s 34833, joint_vel_out_of_limit 1.0000

    TGS no-guard (17:40:30, stopped at iter 207 after 00:12:02)
      iter 0:   reward 0.70, episode length 21.99, steps/s 15623, time_out 0.0493
      iter 50:  reward 8.93, episode length 235.29, steps/s 28748, time_out 0.8864
      iter 100: reward 18.28, episode length 211.03, steps/s 27599, time_out 0.6499
      iter 150: reward 16.88, episode length 206.06, steps/s 28966, time_out 0.6339
      iter 200: reward 16.69, episode length 203.31, steps/s 27627, time_out 0.6454
      iter 207: reward 15.62, episode length 200.15, steps/s 28956, time_out 0.6259

    baseline FeatherPGS with guard (17:53:44, stopped at iter 167 after 00:05:29)
      iter 0:   reward 0.37, episode length 1.04, steps/s 19829, joint_vel_out_of_limit 0.9987
      iter 50:  reward 0.70, episode length 6.55, steps/s 52989, joint_vel_out_of_limit 1.0000
      iter 100: reward 1.37, episode length 203.74, steps/s 50130, joint_vel_out_of_limit 0.2576
      iter 150: reward 1.99, episode length 226.76, steps/s 49587, joint_vel_out_of_limit 0.1351
      iter 167: reward 1.37, episode length 223.63, steps/s 48414, joint_vel_out_of_limit 0.1459

The practical conclusions from this monitored Stage 3C study are:

- The runnable MJWarp Franka baseline in this workspace does not resemble the supplied healthy MJWarp curve. It stayed stuck near `0.7` reward through `100` iterations, with almost every episode terminating on `joint_vel_out_of_limit`.
- The TGS-style no-guard path is the only runnable path in this workspace that shows clear learning signal by iteration `100`. Its reward of `18.28` at `100` is consistent with “this run is actually learning,” but it plateaued early in the mid-teens instead of continuing toward the supplied FeatherPGS reference curve.
- Baseline FeatherPGS with the default velocity guard underperforms badly. It eventually escaped the strict one-step regime, but even by `150` iterations it remained around `2` reward, which is far below both the supplied FeatherPGS reference (`~20` by `150`) and the TGS no-guard run.
- Throughput cost is real but not catastrophic at trainer level. The TGS no-guard path ran at about `27.6k-28.9k` steps/s once warmed up, versus about `48.4k-53.0k` steps/s for baseline FeatherPGS and about `32.9k-34.8k` steps/s for MJWarp. Relative to baseline FeatherPGS, the TGS-style schedule was roughly `1.7x-1.9x` slower in these 4096-env training loops.
- The main comparison is therefore not “TGS wins cleanly.” It is “the only path here that learns meaningfully is the smaller-step TGS-style path with the velocity guard removed, and it buys that signal at a significant throughput cost while still plateauing far below the supplied healthy long-run baselines.”

## Current recommendation after the completed Stage 3C slice

Stage 3C is now complete enough to support Stage 4. The next pass should stop changing training harnesses and instead write the final recommendation from the evidence already gathered:

- keep the Stage 3B rollout harness as the reproducible pre-flight check before any further training runs
- treat the `32`-env, `2`-iteration runs as boot evidence only and use the monitored `4096`-env runs for actual training conclusions
- treat the missing `omni.physics` module as a standing limitation on any PhysX-backed comparison in this workspace
- use the supplied reference curve plus the monitored Stage 3C data to decide whether the smaller-step TGS-style path is worth its throughput cost and whether the baseline FeatherPGS result is fundamentally blocked by the velocity guard, by solver quality, or by both
- write the final Stage 4 viability and cost-benefit recommendation so the study questions are answered in one place without inference
