# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unitree G1 benchmark / example:

- Builds a replicated G1 scene from USD.
- Can use either:
    * newton.solvers.SolverMuJoCo  (implicit, hardcoded parameters)
    * newton.solvers.SolverFeatherPGS (CRBA + PGS contacts)

Usage examples:

Interactive (viewer) with FeatherPGS:
    uv run newton/examples/robot/example_robot_g1_benchmark.py \
        --solver feather_pgs --sim-substeps 8 --num-worlds 1

Headless benchmark with FeatherPGS:
    uv run newton/examples/robot/example_robot_g1_benchmark.py \
        --solver feather_pgs --benchmark --viewer null \
        --num-worlds 8192 --sim-substeps 8 --pgs-iterations 1

Headless benchmark with MuJoCo:
    uv run newton/examples/robot/example_robot_g1_benchmark.py \
        --solver mujoco --benchmark --viewer null --num-worlds 1024
"""

import time
import numpy as np

import warp as wp

import newton
import newton.examples
import newton.utils

try:
    import torch  # optional, used only for GPU memory stats
except ImportError:
    torch = None


class Example:
    def __init__(
        self,
        viewer,
        num_worlds: int = 4,
        sim_substeps: int = 8,
        solver_type: str = "feather_pgs",
        update_mass_matrix_interval: int = 1,
        pgs_iterations: int = 12,
        pgs_max_constraints: int = 32,
        pgs_beta: float = 0.01,
        pgs_cfm: float = 1.0e-6,
        pgs_omega: float = 1.0,
        pgs_warmstart: bool = False,
    ):
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.sim_substeps = sim_substeps
        self.sim_dt = self.frame_dt / float(self.sim_substeps)

        self.num_worlds = num_worlds
        self.viewer = viewer
        self.solver_type = solver_type

        # ------------------------------------------------------------------
        # Build G1 model
        # ------------------------------------------------------------------
        g1 = newton.ModelBuilder()

        # Register attributes used by the MuJoCo-style import, regardless of solver
        newton.solvers.SolverMuJoCo.register_custom_attributes(g1)

        g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=2.5, friction=1e-5)
        g1.default_shape_cfg.ke = 5.0e4
        g1.default_shape_cfg.kd = 5.0e2
        g1.default_shape_cfg.kf = 1.0e3
        g1.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")

        g1.add_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        # Joint drive gains for actuated joints
        for i in range(6, g1.joint_dof_count):
            g1.joint_target_ke[i] = 1000.0
            g1.joint_target_kd[i] = 5.0

        # Approximate meshes for faster collision detection
        g1.approximate_meshes("bounding_box")

        builder = newton.ModelBuilder()
        builder.replicate(g1, self.num_worlds)
        builder.add_ground_plane()

        self.model = builder.finalize()

        # ------------------------------------------------------------------
        # Create solver
        # ------------------------------------------------------------------
        if solver_type == "feather_pgs":
            solver_kwargs = dict(
                update_mass_matrix_interval=update_mass_matrix_interval,
                pgs_iterations=pgs_iterations,
                pgs_beta=pgs_beta,
                pgs_cfm=pgs_cfm,
                pgs_omega=pgs_omega,
                pgs_max_constraints=pgs_max_constraints,
                pgs_warmstart=pgs_warmstart,
            )
            self.solver = newton.solvers.SolverFeatherPGS(self.model, **solver_kwargs)

        elif solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=False,
                solver="newton",
                integrator="implicit",
                njmax=210,
                ncon_per_env=35,
                ls_parallel=True,
                iterations=100,
                ls_iterations=50,
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        print("PGS params:",
            "iter", self.solver.pgs_iterations,
            "beta", self.solver.pgs_beta,
            "cfm", self.solver.pgs_cfm,
            "omega", self.solver.pgs_omega,
            "max_constraints", self.solver.pgs_max_constraints,
            "pgs_warmstart", self.solver.pgs_warmstart)

        # ------------------------------------------------------------------
        # Allocate state/control/contacts
        # ------------------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.capture_cuda_graph()

    # ----------------------------------------------------------------------
    # CUDA graph capture (optional)
    # ----------------------------------------------------------------------
    def capture_cuda_graph(self):
        self.graph = None
        device = wp.get_device()
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ----------------------------------------------------------------------
    # Simulation / stepping
    # ----------------------------------------------------------------------
    def simulate(self):
        for _ in range(self.sim_substeps):
            # Broad/narrow-phase collision for current state
            self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)

            # Clear forces from previous step
            self.state_0.clear_forces()

            # Apply viewer forces only if a viewer exists (headless-safe)
            if self.viewer is not None:
                self.viewer.apply_forces(self.state_0)

            # Advance solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    # ----------------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------------
    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


# --------------------------------------------------------------------------
# Benchmark runner
# --------------------------------------------------------------------------
def run_benchmark(
    example: Example,
    warmup_frames: int,
    measure_frames: int,
    check_stability: bool,
    stability_threshold: float,
):
    # Warmup
    for _ in range(warmup_frames):
        example.step()

    wp.synchronize_device()

    initial_q = None
    if check_stability:
        initial_q = example.state_0.joint_q.numpy()

    # Benchmark loop
    total_env_frames = example.num_worlds * measure_frames

    t_start = time.time()
    for _ in range(measure_frames):
        example.step()
    wp.synchronize_device()
    t_end = time.time()

    elapsed = t_end - t_start
    fps_env = total_env_frames / elapsed if elapsed > 0.0 else 0.0

    # GPU memory (optional)
    gpu_used_gb = None
    gpu_total_gb = None
    if torch is not None and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_used_gb = (total - free) / 1024**3
        gpu_total_gb = total / 1024**3

    # Print performance summary
    print("\n=== Benchmark Summary ===")
    print(f"Solver:              {example.solver_type}")
    print(f"Worlds:              {example.num_worlds}")
    print(f"Sim substeps/frame:  {example.sim_substeps}")
    print(f"Warmup frames:       {warmup_frames}")
    print(f"Measured frames:     {measure_frames}")
    print(f"Total env-frames:    {total_env_frames}")
    print(f"Elapsed time (s):    {elapsed:.6f}")
    print(f"Env-FPS (env/s):     {fps_env:,.2f}")
    if gpu_used_gb is not None:
        print(f"GPU memory used (GB):   {gpu_used_gb:.3f}")
        print(f"GPU memory total (GB):  {gpu_total_gb:.3f}")
    print("=========================\n")

    # Optional stability check
    if check_stability and initial_q is not None:
        final_q = example.state_0.joint_q.numpy()

        try:
            max_drift = np.max(np.abs(initial_q - final_q))
            is_stable = np.allclose(initial_q, final_q, atol=stability_threshold)

            print("=== Stability Check ===")
            print(f"Max |Δjoint_q|:      {max_drift:.6e}")
            print(f"Threshold (atol):    {stability_threshold:.6e}")
            if is_stable:
                print("Result:              PASS")
            else:
                print("Result:              FAIL")
            print("=======================\n")
        except Exception as e:
            print("=== Stability Check Failed ===")
            print(f"Error while comparing joint_q: {e}")
            print("===============================\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = newton.examples.create_parser()

    # Core configuration
    parser.add_argument("--num-worlds", type=int, default=16, help="Total number of simulated worlds.")
    parser.add_argument("--sim-substeps", type=int, default=8, help="Simulation substeps per frame.")

    # Solver selection
    parser.add_argument(
        "--solver",
        type=str,
        choices=["mujoco", "feather_pgs"],
        default="feather_pgs",
        help="Which articulated solver to use.",
    )

    # FeatherPGS-specific parameters
    parser.add_argument("--pgs-iterations", type=int, default=12, help="Number of PGS iterations per frame.")
    parser.add_argument(
        "--pgs-max-constraints",
        type=int,
        default=32,
        help="Maximum number of stored contact constraints per articulation for PGS.",
    )
    parser.add_argument(
        "--pgs-beta",
        type=float,
        default=0.01,
        help="ERP-style position correction factor for PGS.",
    )
    parser.add_argument(
        "--pgs-cfm",
        type=float,
        default=1.0e-6,
        help="Compliance/regularization added to the Delassus diagonal for PGS.",
    )
    parser.add_argument(
        "--pgs-omega",
        type=float,
        default=1.0,
        help="Successive over-relaxation factor for the PGS sweep.",
    )
    parser.add_argument(
        "--pgs-warmstart",
        action="store_true",
        help="Re-use impulses from the previous frame when contacts persist.",
    )
    parser.add_argument(
        "--update-mass-matrix-interval",
        type=int,
        default=1,
        help="How often to update the mass matrix (every n-th step).",
    )

    # Benchmark options
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (no interactive loop).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=16,
        help="Number of warmup frames before benchmarking.",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=64,
        help="Number of frames to measure during benchmarking.",
    )
    parser.add_argument(
        "--check-stability",
        action="store_true",
        help="Also check joint_q drift between start and end of benchmark.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=5e-3,
        help="Max allowed absolute drift in joint_q for the stability check.",
    )

    viewer, args = newton.examples.init(parser)

    example = Example(
        viewer=viewer,
        num_worlds=args.num_worlds,
        sim_substeps=args.sim_substeps,
        solver_type=args.solver,
        update_mass_matrix_interval=args.update_mass_matrix_interval,
        pgs_iterations=args.pgs_iterations,
        pgs_max_constraints=args.pgs_max_constraints,
        pgs_beta=args.pgs_beta,
        pgs_cfm=args.pgs_cfm,
        pgs_omega=args.pgs_omega,
        pgs_warmstart=args.pgs_warmstart,
    )

    if args.benchmark:
        # Headless-style benchmark run (viewer may still exist but is not used)
        run_benchmark(
            example,
            warmup_frames=args.warmup_frames,
            measure_frames=args.measure_frames,
            check_stability=args.check_stability,
            stability_threshold=args.stability_threshold,
        )
        if viewer is not None:
            viewer.close()
    else:
        # Interactive mode (default)
        if viewer is None:
            print("No viewer created (e.g., --viewer null).")
            print("Use --benchmark to run a timed benchmark headlessly.")
        else:
            print(f"Starting interactive run with solver={args.solver}")
            newton.examples.run(example, args)
