# Reconstructed Manda rigid scenes

This harness rebuilds rigid fixtures from [Manda Robotics' comparison](https://mandarobotics.com/blog/comparing-physics-engines/index.html)
and runs the same authored MJCF and frozen command reference with **Newton FPGS**
and **native MuJoCo CPU**. It is a correctness and sensitivity runner; diagnostic
readbacks are included in its wall time. The article's Newton route uses
MuJoCo-Warp, so these FPGS runs add a different solver to the comparison.

## Available scenes

| CLI scene | Reconstructed physical inputs | Remaining differences from the article |
|---|---|---|
| `slide` | 40 mm, 100 g cube; initial speed 1 m/s; friction 0.4; 2 ms steps | FPGS contact law differs from MuJoCo's soft contact |
| `drop` | Same cube; initial COM height 350 mm; zero velocity; 1 ms steps; 1 s | Contact law and force peaks differ |
| `hinge` | 1 kg, 40 × 40 × 300 mm link; Y-axis hinge at 450 mm; COM offset 150 mm; initial angle 0.5 rad; +0.2 Nm pulse on [0.1, 0.2) s | Independent scalar RK4 reference; FPGS and MuJoCo Euler integration |
| `collision` | Rolling 30 mm sphere, 60 mm cube, 90 mm equilateral prism; published masses, poses, velocities and analytic inertias; friction 0.3; 0.5 ms steps | Original prism mesh yaw was not specified; our prism points along +X |
| `panda_effort` | Pinned Panda frames/inertias; fixed open fingers; initial configuration and frozen gravity feedforward; published effort pulses | Primitive noncolliding display geometry replaces source meshes |
| `grasp` | Independently driven fingers; published pad geometry, gains, effort caps, 40 mm cube, friction 0.5, close/lift/release timings | Table position and unloaded IK/feedforward tape generated here |
| `stack` | Two dynamic 40 mm, 100 g cubes; 100 mm transfer; six-second sequence; published release clearance and offset variants | Table/initial placement and IK/feedforward tape generated here |
| `push` | Three 40 mm, 100 g cubes; 84 mm initial front span; 80/90 mm channels; 240 mm push; shared 1 kHz compliant controller | Initial X layout, wall length/height, table and motion tape generated here |

The robot arm and display geometry do not collide. Only finger-pad/object,
object/object and object/table/wall pairs collide. Pads do not collide with the
table, walls or each other. All native drives, armature, damping, friction loss,
tendons and finger equality constraints are removed. Both solvers receive
external joint efforts; simulated objects are never welded or teleported.
Joint limits are monitored rather than enforced, matching the published audit.

The complete 15-object sweep and precision insertion are deferred: their exact
layout, motion tapes and bore/peg construction are not available in the linked
public sources. Deformable-ball experiments are outside this rigid suite.

## Run

Use a **new output directory** for every command; the runner rejects existing
directories to preserve prior trials. No asset downloads are needed.

```bash
# Eight full protocols, both solvers, CPU correctness comparison
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene all --solver both --output /tmp/manda-rigid-baseline

# View an individual FPGS scene
uv run --extra examples python -m scripts.benchmarks.manda_rigid \
    --scene grasp --viewer gl --output /tmp/manda-grasp-view

# CUDA correctness run (requires CUDA Warp); matrix-free FPGS selected on CUDA
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene all --device cuda:0 --iterations 64 --output /tmp/manda-rigid-cuda

# Half-step refinement, preserving the same 1 kHz robot controller
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene all --solver both --dt 0.0005 --output /tmp/manda-rigid-half-step

# Crowded contact variants
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene push --solver both --width 0.08 --repeats 3 --output /tmp/manda-push-80
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene stack --solver both --stack-offset 0.022 --output /tmp/manda-stack-22
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene grasp --solver both --grasp-offset 0.008 --output /tmp/manda-grasp-offset

# Short instrumentation smoke test; this does not exercise the whole task
uv run --extra dev python -m scripts.benchmarks.manda_rigid \
    --scene all --solver both --duration 0.01 --output /tmp/manda-rigid-smoke

uv run --extra dev python -m unittest newton.tests.test_manda_rigid_benchmarks
```

Render a completed run into individual MP4 clips and a combined reel:

```bash
uv run --extra dev --with pillow --with imageio-ffmpeg python -m \
    scripts.benchmarks.render_manda_rigid --input /tmp/manda-rigid-baseline
```

Videos go into a new `videos` directory under the run. Select one scene with
`--scene grasp`, or native MuJoCo recordings with `--solver mujoco` and a
different `--output` directory. Playback speed is labelled. The renderer reads
the saved poses and joints, checks replayed body positions and orientations,
and adds cameras, lighting and simplified noncolliding gripper display shapes.
It calls forward kinematics for display and never advances the dynamics.

Use `--viewer viser` for a browser viewer. Robot physics timesteps must divide
1 ms; efforts are held between controller ticks. `--iterations` changes FPGS
iterations, and is saved in the result. Native MuJoCo uses Euler, Newton solver,
100 iterations, elliptic friction, `solref=(0.01,1)` and
`solimp=(0.99,0.999,0.001,0.5,2)`. FPGS uses zero angular damping, 64 iterations
by default, zero torsion, no friction anchors, default ERP/regularization, and
deterministic collision ordering. These settings are **not equivalent contact
laws**. CPU uses split FPGS; CUDA uses matrix-free FPGS.

## Outputs and checks

Each scene directory contains `scene.xml` and a frozen `reference.npz` with
joint targets, target velocities and unloaded inverse-dynamics feedforward.
Each solver/repeat directory contains `trace.npz` and `result.json`. The root
`summary.json` retains every completed or failed trial. Scene and tape hashes
identify the exact shared inputs.

Traces store every physics-step pose, COM linear velocity, controlled joint
position, preceding-step effort and net contact force for tracked bodies.
Quaternions use XYZW; distances are metres, force is newtons, and arm/finger
efforts are Nm/N. Post-step poses and preceding-step forces are explicitly
labelled. Contact forces are native readbacks, not estimated from acceleration.
FPGS exports linear forces only; no contact torque or wrist-wrench accuracy
claim is made. Finger forces are retained separately because opposing squeeze
forces cancel in the cube's net force.

Before FPGS runs, canonical MuJoCo and Newton are compared for every imported
body's mass, COM, full inertia tensor, initial frame and rotation, and every
realized collision pair. Failures stop the run. Simulation nonfiniteness,
constraint capacity and arm joint limits are checked. Physical task failure
(a missed grasp, toppled stack or blocked push) is saved as an outcome, rather
than hidden by tuning or causing the evidence to be discarded.

Metrics include sliding travel, drop height, hinge error against RK4, cube
lift/hold, finger closing forces, final-half-second geometric stack survival,
and number of blocks past the channel goal. Stack settling speed is reported
separately; geometric survival does not certify continuous support contact.
Linear impulse residuals audit force/velocity consistency for free objects.
These are reconstruction outcomes, not a reproduction of the article's
numerical rankings or evidence of real-world accuracy.

## Sources and asset attribution

- [Sliding specification](https://mandarobotics.com/blog/comparing-physics-engines/assets/sliding-block/index.html)
- [Drop specification](https://mandarobotics.com/blog/comparing-physics-engines/assets/drop-settle/index.html)
- [Hinge audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/hinge/audit.md)
- [Collision-chain audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/collision-chain/audit.md)
- [Panda effort audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/panda/audit.md)
- [Grasp audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/grasp/audit.md)
- [Stack audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/stack/audit.md)
- [Push audit](https://mandarobotics.com/blog/comparing-physics-engines/assets/push/audit.md)

`assets/manda_panda.xml` is a modified subset of
[MuJoCo Menagerie's Panda XML](https://github.com/google-deepmind/mujoco_menagerie/blob/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/franka_emika_panda/panda.xml),
at revision `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7`, distributed under
[Apache-2.0](https://github.com/google-deepmind/mujoco_menagerie/blob/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/LICENSE),
the same license included in this repository's `LICENSE.md`. Body transforms,
COMs, masses and full inertias are retained. Joint defaults are expanded;
source meshes, native actuators, equality/tendon constraints and passive
damping/armature are removed. Our display primitives are generated separately.
