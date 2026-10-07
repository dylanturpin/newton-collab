# Kitchen pour and spill on CUDA FPGS

This reconstructs the supplied September 28 `dense60-balls-source` recording:
60 kitchen objects drop into a transparent, open bin, which tips and moves away
to leave a pile on a table. The bin follows a prescribed motion. Every released
object is simulated by FPGS; no recorded source trajectories are replayed.

The meshes use the original kitchen authoring profiles, including hollow cups,
plates, bowls, jars, vases, and mug handles. The object mix and seed follow the
original balls-family exporter. The exact imported scene JSON and mouse-drag
trajectory were unavailable, so the release sequence and bin motion are a
reconstruction. Material density and available mass, COM, and inertia data are
preserved. Native spheres and capsules handle the small balls and rods;
the other objects use nonconvex mesh/SDF collisions at 0.75 mm resolution.

## Prepare portable assets

Run in this checkout, substituting the locations of the source authoring module
and the supplied CMG kitchen pack:

```bash
uv run --extra dev --with trimesh python -m scripts.benchmarks.prepare_kitchen_pour \
    --kitchen-source /path/to/solid_runtime/scenes/kitchen.py \
    --cmg-pack /path/to/kitchen-pour-cmg \
    --output /path/to/kitchen-assets
```

This creates `scene.json`, visual OBJ meshes, and collision mesh arrays. Copy the
directory and this Newton checkout to a CUDA node. The rollout does not need the
source authoring package or CMG files afterward.

## Simulate and render

```bash
uv run --extra dev python -m scripts.benchmarks.fpgs_kitchen_pour \
    --asset /path/to/kitchen-assets --cache /path/to/kitchen-cache \
    --output /path/to/kitchen-run

uv run --extra dev --with pillow --with imageio-ffmpeg python -m \
    scripts.benchmarks.render_kitchen_pour \
    --asset /path/to/kitchen-assets --run /path/to/kitchen-run \
    --output /path/to/kitchen-video.mp4
```

The defaults are 23 seconds, 60 output frames per simulated second, four
substeps (240 Hz), 10 FPGS iterations, and colored propagation on CUDA. Output
paths must be new. SDF and Warp compilation are cached separately from runs.
The run writes `trace.npz` and `result.json`; the renderer writes a 30 fps MP4 at
normal playback speed and several PNG stills. MuJoCo is used only to render
the FPGS poses, with collisions disabled and no dynamics steps.

Timing covers the complete evolving collision/solver loop, bin and release
control, CUDA graph launches, and GPU trace recording. It excludes initial SDF
construction, compilation, graph setup, CPU output copies, and video encoding.
The result includes per-interval progress and checks for nonfinite states,
invalid rotations, contact/constraint capacity, and objects falling through the
table interior. Bin occupancy before and after tipping is reported as a
coarse behavior check.

Use `--iterations`, `--substeps`, `--voxel`, and `--contacts` to explore quality
and speed. Keep substeps even so each captured frame returns to the same pair
of state buffers. The current shared-memory solver path cannot compile an
arbitrarily large legacy matrix-free allocation. This all-free-body scene
reserves 8,192 contacts and 24,640 propagation rows through the combined row
budgets, while keeping the unused legacy matrix-free budget small. Overflow
checks remain enabled.

## Recorded run, October 5, 2026

Slurm job `677374` on `skildai-gpu02`, NVIDIA RTX A6000, Warp 1.17.0,
Newton commit `96f5cf095d989c0947eddc6408a11e237bd0c373` plus these benchmark
scripts:

| Measurement | Result |
| --- | --- |
| Simulated duration | 23.0 s |
| Timed simulation loop | 24.31 s |
| Overall speed | 0.946x real time, 56.8 simulated frames/wall-second |
| Physics step | 1/240 s, 10 iterations |
| Peak contacts across all substeps | 2,055 of 8,192 |
| Peak propagation rows | 6,165 of 24,640 |
| Dropped contacts/constraint rows | 0 |
| Nonfinite states / invalid rotations | 0 |
| Bodies below table interior | 0 |
| Object COMs within bin before tip / after spill | 57 / 0 |

The slowest measured two-second interval ran at approximately 0.64x real time;
the overall rate includes the faster initial fill and final settling phases.
Rendering is offline and is excluded from this physics measurement.

The portable assets, exact trace, result JSON, GPU log, and final video are under
`newton/tests/outputs/kitchen-pour-20261005/` in this workspace. The accepted run
is `run-c8192-10x4`; the final video directory is `validated-video`. The remote
checkout and cache are at `/data/home/liinbor/src/fpgs-kitchen-pour-20261005`.

Earlier trials exposed an oversized legacy matrix-free shared-memory allocation
and an undersized contact buffer; neither trial was accepted as a result. One
trial on `skildai-gpu01` stopped making progress after two simulated seconds and
was canceled; its cause was not established. The completed result above was
produced on `skildai-gpu02`.

## Two-substep comparison

An initial misinterpretation of the request led to job `677638` on `skildai-gpu02`
with only `--substeps 2` changed: a 1/120 s physics step instead of 1/240 s.
The user subsequently clarified that the intended count was eight substeps.
The solver script and asset hashes match the four-substep run above. Geometry,
materials, 10 solver iterations, collision settings, release rules, bin motion,
duration, and rendering settings were unchanged.

The rollout simulated 23 seconds in approximately 17.023 seconds: **1.351x
real time**, versus 0.946x for four substeps. This is a 1.43x throughput increase.
It **failed the tabletop check**: chopstick body 45 had its COM below the table
interior at 9.5667 seconds. Bowl body 5 also exited the bin side near its bottom
at approximately 3.85 seconds. Reducing the substep count did not resolve the
penetration issue.

Reinspection also confirmed that bowl body 1 had exited through a side wall
around 3.5667 seconds in the original four-substep run. The original finite-state,
capacity, floor, and occupancy checks did not test object/object or bin-wall
penetration; passing them did not establish collision correctness.

The two-substep trace and log are in `run-c8192-10x2`. Because the runner saved
the trace before raising its validation error, the full failed rollout is
available. Its local `result.json` was reconstructed for rendering from that
trace and the millisecond-rounded progress log, and explicitly records the
failure. The video is `video-2-substeps/kitchen-pour-2-substeps.mp4`, at normal
playback speed. No physics code was changed for this comparison.

## Eight-substep comparison

The corrected request was run as job `677812` on the same `skildai-gpu02`
RTX A6000, with only `--substeps 8` changed from the original four-substep
configuration. This halves the physics timestep to 1/480 s (2.0833 ms), while
retaining 10 iterations per substep. Solver script and asset hashes match.

| Measurement | 4 substeps | 8 substeps |
| --- | --- | --- |
| Simulated duration | 23 s | 23 s |
| Timed simulation loop | 24.310 s | 27.536 s |
| Real-time factor | 0.946x | 0.835x |
| Peak contacts | 2,055 | 1,213 |
| Dropped contacts/constraint rows | 0 | 0 |
| Object COMs in bin before tip / after spill | 57 / 0 | 59 / 0 |

The eight-substep run passed the existing finite-state, rotation, contact
capacity, and tabletop checks. A separate inspection of the saved 60 Hz poses
found no early side-wall escape by any of the five bowls. Their sampled mesh
vertices below the rim remained inside the bin's outer wall planes during
filling. The first bowl's closest approach to an outer wall plane was 10.891 mm.
This does not imply zero overlap with the 12 mm thick wall.

An offline rod check sampled 41 centerline points per chopstick or rolling pin
against the closed object meshes at 5, 12, 13.5, 17, 20, and 23 seconds. It found
no centerline points more than 1 mm inside another object's solid material.
This spot check does not exhaustively test capsule surfaces or every substep.
The physics and rendering settings were unchanged for these inspections.

The trace, original result JSON, GPU log, and supplemental checks are in
`run-c8192-10x8`. The normal-speed video is
`video-8-substeps/kitchen-pour-8-substeps.mp4`.

## Eight substeps, four iterations

Job `678484` on the same RTX A6000 changed only the iteration count from 10 to
4 relative to the eight-substep run. The timestep remained 1/480 s, and the
solver script, assets, collision settings, release rules, bin trajectory, and
rendering were unchanged. The original source and asset hashes match.

| Measurement | 8 substeps, 10 iterations | 8 substeps, 4 iterations |
| --- | --- | --- |
| Simulated duration | 23 s | 23 s |
| Timed simulation loop | 27.536 s | 18.279 s |
| Real-time factor | 0.835x | 1.258x |
| Peak contacts | 1,213 | 1,806 |
| Peak propagation rows | 3,639 | 5,418 |
| Dropped contacts/constraint rows | 0 | 0 |
| Object COMs in bin before tip / after spill | 59 / 0 | 59 / 25 |

The run passed the existing finite-state, rotation, capacity, and tabletop
checks. No bowl side-wall escapes were observed during filling. The same
six-time, 41-point rod centerline spot check found no points more than 1 mm
inside another object's solid mesh. These remain limited checks, as described
above. The spill differed materially: 25 object COMs remained within the bin
bounds at the end, compared with zero in the ten-iteration run.

Artifacts are in `run-c8192-4x8`, and the normal-speed video is
`video-8ss-4iter/kitchen-pour-8ss-4iter.mp4`. Only command-line settings changed;
the runner defaults and physics code were left intact.

## Local Blender replay, eight substeps and ten iterations

The local `run-c8192-10x8/trace.npz` and `result.json` match the remote originals
byte for byte. The trace SHA-256 is
`5016520a1f33a104f08b89367ed5c045901317826aea14f148abda99f0b0ca2e`.
It contains 1,381 samples at 60 Hz for the bin and 60 kitchen objects.

The Blender renderer imports the same visual mesh vertices and triangles,
keys every recorded pose, and runs no Blender physics. All 60 Hz samples are
retained as whole-frame and half-frame keys on a 30 fps timeline. Curved
surfaces use smooth normals; edges with adjacent-face angles above 30 degrees
remain sharp. This preserves rims and flat bases without modifying geometry.

On the local Mac, build a portable project and render its 691 PNG frames with:

```bash
uv run --no-project /Applications/Blender.app/Contents/MacOS/Blender \
    --background --factory-startup --python-exit-code 1 \
    --python scripts/benchmarks/render_kitchen_pour_blender.py -- \
    --asset newton/tests/outputs/kitchen-pour-20261005/assets \
    --run newton/tests/outputs/kitchen-pour-20261005/run-c8192-10x8 \
    --output /path/to/new-blender-output --render-animation
```

The default is Cycles on Metal, 16 samples with denoising, at 1280 by 720.
The project embeds replay metadata and checks positions and quaternion
rotations against the trace, including half-frame keys. The saved scene
contains its meshes, materials, lights, camera, and animation, so it does not
need external assets for playback. Rendering time is separate from physics
performance. The local output directory is `blender-sharp-8ss-10iter`.

The initial semi-transparent-box animation was stopped at the user's request
before completion. The revised renderer uses a Glass BSDF with IOR 1.5 and
roughness 0.015, without a transparent-shader blend. A continuous, watertight
shell has the same exterior, interior, and 12 mm thickness as the five collision
panels; internal panel joins are removed from the optical surface. This changes
only rendering, not the recorded simulation. A 12-second still in
`blender-glass-preview-8ss-10iter` uses 128 samples at 1920 by 1080. The glass
preview precedes any new full-video render.

The approved glass animation uses eight samples per pixel at 1280 by 720,
with OpenImageDenoise on the Metal GPU, the fast prefilter, and balanced
denoising quality. Glass bounce limits and all scene materials remain as in
the approved still. Sampled frames took approximately 0.42–0.45 seconds after
startup. The full output directory is `blender-glass-video-8ss-10iter`.

## AVBD Metal HQ replay and clear glass

`kitchen_hq_replay/` consumes the same 8-substep, 10-iteration recording through
the local `avbd-metal` renderer, without stepping an AVBD solver. See its
`README.md` for the exporter, camera format, quality controls, and replay commands.
The export preserves every recorded position and quaternion and the Blender
meshes' sharp corner normals. Only the camera and presentation floor change.

The October 5 comparison isolated a MetalFX reconstruction issue: increasing
the ray budget to High retained frosted-looking transmission, while bypassing
MetalFX restored sharp detail. The fix in the local renderer's
`MetalFXReconstruction.swift` supplies native denoise-strength and reactive
masks for active dielectric camera transport. Opaque surfaces keep denoising;
transmission remains sharp as contents move behind stationary glass. The new
GPU regression failed before the fix and passes at native resolution and 2x
upscaling; all 17 targeted renderer tests pass.

Local outputs under `newton/tests/outputs/kitchen-pour-20261005/avbd-hq/` include
matched `high-denoised`, `high-unfiltered`, and `high-fixed` stills, three camera
previews, regression logs, and `glass-fix-validation.json`. The full video is
`kitchen-pour-avbd-hq-glass-8ss-10iter.mp4`: 691 frames at 1280x720 and 30 fps,
using Balanced HQ, native resolution, MetalFX enabled, and a fixed side view.
GPU rendering averaged 30.1 ms per frame on Apple M5; writing the PNG sequence
took 64.1 seconds. `video-fixed/render-report.json` records the settings and
trace hash, and `video-validation.json` records full video decode verification.
Its portable project retains all 1,381 recorded poses; an independent check
of every body at every sample found zero position or quaternion differences.

The full glass render completed in 351.21 seconds (5 minutes 51 seconds).
Encoding and decoding verification took another 5.64 seconds. The finished
`kitchen-pour-glass-8ss-10iter.mp4` contains all 691 frames at 30 fps,
1280 by 720, for 23.03 seconds of normal-speed playback. Every frame decoded
successfully. The same directory contains the self-contained `.blend`, the
PNG frame sequence, and the replay/render/video validation JSON files.

The 6 mm glass preview revealed black edge segments where the 24-interface
camera path budget exhausted during total internal reflection. A GPU light-pipe
regression fails with the previous 32-interface override cap and passes with
an extended cap of 256. Quality preset defaults remain unchanged; this replay
explicitly opts into 256. A matched 16-second diagnostic separates this defect
from single-sample camera-ray aliasing. The revised raw HQ replay renders at
3840x2160 and spatially downsamples to 2560x1440, without MetalFX or upscaling.
Its output is `avbd-hq/kitchen-pour-glass-6mm-antialiased-8ss-10iter.mp4`;
`video-glass-6mm-aa/render-report.json` records its timing and exact settings.
Sharp corner reflections remain visible; this is bounded preview transport.

## Checkerboard floor variant

The requested floor variant uses a new remote FPGS rollout, with the finite
support table replaced by an infinite collision plane at Z = 0.005 m. This
matches AVBD Metal's built-in checkerboard surface. The box starts at Z =
0.006 m, 1 mm above the floor; release poses and the bin trajectory shift by
the same 5 mm. Object geometry, release order, solver settings, and material
friction remain as in the 8-substep / 10-iteration run.

Remote job 681045 ran on `skildai-gpu02` / NVIDIA RTX A6000. The 23-second
rollout took 26.690 seconds (0.862x real time), with 1,210 peak contacts, no
capacity overflow, finite state, and unit quaternion checks passing. Local
trace and result hashes match the remote originals. The exported AVBD replay
preserves every recorded pose and contains only the box and 60 objects; it
omits both the table and the authored studio floor.

Use `fpgs_kitchen_pour.py --support floor --substeps 8 --iterations 10` for
simulation, then `export_kitchen_hq.py --builtin-floor --glass-wall-thickness
0.006`. The replay enables the renderer's default checkerboard from its scene
metadata. The final camera is `floor-high`; High raw HQ renders at 3840x2160
and downsamples to 2560x1440, with a 256-interface glass path budget. Results
are in `run-floor-8ss-10iter/` and `avbd-hq/checker-floor/`. The full video is
`kitchen-pour-checker-floor-8ss-10iter.mp4`.

The floor check bounds body COMs, not every visual vertex. A separate mesh
inspection records a transient chopstick penetration of 22.2 mm at 18.833 s;
minimum final visual vertex height is 4.28 mm below the floor. Those contact
residuals are retained in the replay and recorded in `local-validation.json`.

The finished video contains 691 frames at 30 fps and 1440p, with every frame
decoded successfully. Rendering averaged 133.95 ms of GPU work per frame on
Apple M5; writing the PNG sequence took 357.82 seconds. Encoding overlapped
rendering. `video-validation.json` includes the complete render settings.

The cinematic revision keeps the same floor recording and starts close to the
box, centered on its contents. From 11.5 to 14 seconds, the camera smoothly
pulls back to frame the box tipping and the spilled objects. Its camera JSON
is `avbd-hq/checker-floor/camera-cinematic.json`; the replacement video is
`kitchen-pour-cinematic-8ss-10iter.mp4`.

Matched final-pose diagnostics isolated the dark surface streaks to primary
reflection and shadow self-intersections. Reducing the replay's effective
camera clipping range from 0.01–100 m to 0.1–30 m removes the visible blotches
without changing AVBD's ray offsets or disabling reflections, shadows, or
diffuse lighting. The comparison is saved in
`avbd-hq/checker-floor/surface-tight-clipping/comparison.png`. All temporary
renderer diagnostics were restored; this revision changes only the replay
camera and clipping settings.

The cinematic video contains 691 verified frames at 2560x1440 and 30 fps.
GPU rendering averaged 184.69 ms per frame; rendering and PNG writes took
432.45 seconds, with encoding overlapping production. Exact camera settings,
source hashes, and full decode verification are saved in
`cinematic-provenance.json` and `cinematic-video-validation.json`.

The subsequent `pile-follow` camera starts roughly 18% farther from the box.
It retains the 11.5–14 s pullback, then eases its target toward the pile center
at X = 0.75 m through 18.5 s as the objects spill. The camera position follows
that target shift, holding its viewing direction over the settled pile.
`camera-pile-follow.json` stores the four camera keyframes; its video is
`kitchen-pour-pile-follow-8ss-10iter.mp4`. The simulation and render quality
settings remain the same as the cinematic revision.
