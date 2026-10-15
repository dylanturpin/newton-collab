# FPGS hero montage

**Set-dressing rule:** use deliberate task- or class-specific decorations only.
No automatic clutter, random stationery scatter, or generic edge-filling fallback.
An absent or empty decoration plan leaves the table undecorated except for its
explicit class-specific props. See [AGENTS.md](AGENTS.md).

The curated roster contains 20 stations across 15 task families. All worlds
step together on an NVIDIA RTX A6000 with `SolverFeatherPGS`, matrix-free
solving, immediate articulated contact response, eight substeps and 32
iterations. `assets/hero_roster.json` defines the lineup and display layout.
The solver reserves 8192 matrix-free and 2048 articulated constraint rows per
world. SAP broad phase and body-pair patch reduction limit redundant convex-cell
contacts. Acceptance checks actual row watermarks, reduction fallbacks, finite
states and task outcomes.

The recorded 50 Hz body poses are rendered locally with the AVBD Metal HQ
renderer at native 2560×1440, without MetalFX upscaling. Each task receives six
seconds at its original simulation speed; drawer opening and placement share
that allocation. Only the ending shot pulls back to reveal the complete batch.
There are no titles, HUD overlays or animated object pose assignments.

| Task | Actual interaction |
| --- | --- |
| Allegro hand | Public HORA policy rotates a freely simulated cube with the fingers while the wrist stays still |
| Balance scale | Franka and UR10 place a missing stock counterweight onto a passive spring-stabilized balance board |
| Drawer | The arm grasps the handle and opens an unpowered prismatic drawer, then places stock proc-gen cutlery inside |
| Hardware pile | A scraper pushes 96 randomly blue, teal and gold nuts, bolts and washers off a raised deck into a lower collection bin |
| Wrecking crane | A gripper slews an unpowered stock crane, swinging a five-link steel chain and heavy ball into a furnished three-floor miniature building |
| Knife insertion | The arm lifts a stock chef’s knife and seats its blade in a 5 mm slotted knife block |
| Shadow hand | A floating five-finger hand supports a free lacquer-and-brass lighter in its palm; the hinge is on the left and the thumb approaches from the right (one contact-only take accepted for the teaser) |
| Serving | Different arms grasp a loose dinner plate and seat it in a stock wire rack with two free plates already resting inside |
| Toy vehicles | An arm pushes a train or articulated dump truck on independently rolling wheels |
| Air hockey | A physical striker sends a puck through a goal |
| Jenga | Different arms add a loose block to a dynamic alternating tower |
| Brick chutes | A stock uncapped glass jar pours six shapes of studded toy bricks into three inclined runs and terminal pockets |
| Toy assembly | Different arms seat removable viaduct tiers on keyed tube connectors |
| G1 and Go2 | Public pretrained locomotion policies run through Warp-NN CUDA inference |

### Beer-pong addition

`beer_pong_task.py` adds a separate seven-second Franka shot without changing
the accepted montage roster. The gripper picks up a 40 mm, 2.7 g ball from a
small tee, accelerates it, and opens its jaws. The free ball bounces once on
the table and lands in an open tapered cup. Only the arm and finger joints
are actuated: no ball pose assignment, launch impulse, attachment, or flight
controller is used. Cup collision walls follow the rendered hollow profile.
Ten cups form four triangular rows (1, 2, 3, 4), with the blue landing cup
at the front and nine teal cups behind it.

```bash
uv run --with imageio-ffmpeg --with pyyaml python scripts/benchmarks/hero_montage/run.py \
  --assets /path/to/hero-assets.json \
  --roster scripts/benchmarks/hero_montage/assets/beer_pong_roster.json \
  --output /tmp/beer-pong --duration 7 --substeps 8 --iterations 16 --no-render
```

The task audit requires a successful lift, exactly one table bounce outside
the cup, measured free-flight acceleration within 0.1 m/s² of gravity, and
containment for the final second. It uses the tapered inner wall rather than
a cylindrical approximation when checking the settled ball. The standalone
roster also works with `export_metal.py`; the packed geometry and recorded
poses use the existing AVBD/Cycles replay pipeline and studio lighting.

### Thread assembly and spinning top

`screw_task.py` uses two physical grippers: a Franka holds the bolt head while
a UR10 turns the nut through one full revolution. The threaded meshes come
from the same Isaac Gym factory assets as Newton's nut/bolt SDF example,
at twice their original scale. Both objects are free bodies. The second arm
holds its axial target fixed; thread contact produces the nut's axial travel.
The audit checks rotation, descent, thread alignment, and bolt stability.

`spinning_top_task.py` creates a painted toy with a light shell, a dense low
rim, and a rounded tip. A UR10-mounted rotary chuck grips its stem, spins it
up, opens, and withdraws. Only robot/chuck joints receive motor targets;
the released top has no drive, attachment, or imposed angular velocity.
The audit requires multiple free revolutions with an upright axis and
bounded travel after withdrawal.

Run either standalone roster with the same runner and replay exporters:
`assets/screw_roster.json` for 10 seconds or
`assets/spinning_top_roster.json` for 9 seconds, with 8 substeps and
16 solver iterations. These additions leave the accepted montage roster
unchanged.

### Bird automaton

`bird_automaton_task.py` imports the user-supplied CAD and contact-gear MJCF
from `assets/bird_automaton` at 2.5× scale. The original crank actuator is
removed. The UR10 grasps the existing cylindrical crank handle and traces a
57.5 mm radius circle around the input axle. Its wrist follows the handle,
keeping the jaws tangent to the crank path. There is no rotary chuck, hub
extension, added visible grip, or driven joint in the mechanism.

The handle collider matches the original CAD cylinder exactly: radius 4 mm,
length 17 mm, center (-21, 0, -23) mm before scaling. The original CAD inertia
is preserved. Gear-tooth contacts drive the camshaft; wings, jaw, and followers
remain passive. Slide limits are scaled explicitly to match geometry scaling.
CAD visuals retain flat face normals.

Use `assets/bird_automaton_roster.json`, a 12-second duration, 8 substeps, and
16 iterations. The accepted run turns the input 1.498 revolutions and the
output 0.997 revolutions, sweeping the wings 70° and the beak 30°. The audit
checks the hand's actual circular orbit, proximity to the original handle,
input rotation, the 24:36 gear ratio, and wing/beak travel. Disabling only
robot–handle contacts leaves the crank stationary despite identical arm
commands. This replaces the rejected hub-driven preview. Sources, physical
recordings, controls, geometry verification, and the Cycles video are saved in
`newton/tests/outputs/bird-crank-20261014/`. The supplied asset README documents
the original design; its ideal-gear validation is separate from these Newton runs.

The lighter has a passive bistable spring-detent potential and a freely
simulated case; its hinge has no position target or motor. The previous assisted opening has been rejected. Direct hinge and contact-gated
opening-force paths have been removed. The user accepted the specific `contact-tilted/shadow-3` CUDA recording for
the teaser: about 65 degrees of opening, with 9.6 mm case slip. It does not
pass the earlier full-open or repeatability gates; acceptance of that take
does not relax those gates for other recordings.
Finger joint actuators remain the only active drives. The lid's spring and detent
depend only on hinge angle and velocity, never the animation clock.
The jar is a physically fixed tool on the arm flange, while every studded
brick is an independent free body. Rack plates are free bodies, including
the two plates initially resting in its slots.

The hands’ mounting arms, wrist sleeves and forearms are omitted only from the
render export. Their simulated joints and collisions are preserved. The hand,
fingers and manipulated objects remain visible. This is a presentation choice,
not a change to the recorded physics.

## Fresh checkout

From the repository root on a CUDA machine:

```bash
uv sync --extra examples --extra onnx --extra dev
uv run --with imageio-ffmpeg --with pyyaml python scripts/benchmarks/hero_montage/prepare_assets.py --menagerie /path/to/mujoco_menagerie --output /tmp/hero-assets.json
uv run --with imageio-ffmpeg --with pyyaml python scripts/benchmarks/hero_montage/run.py --assets /tmp/hero-assets.json --roster scripts/benchmarks/hero_montage/assets/hero_roster.json --output /tmp/hero-run --duration 15 --substeps 8 --iterations 32 --no-render --use-graph
```

Clone MuJoCo Menagerie and check out the revision listed below before running
the asset setup. Setup downloads the pinned Newton robot/policy assets and
writes machine-local paths; compiled procedural assets and HORA inputs are
already bundled. For Metal rendering, set `AVBD_METAL_REPO` to your AVBD Metal
checkout and follow `../hero_hq_replay/README.rst`. The renderer is a separate
repository dependency. Recorded traces and videos are not bundled.

## Procedural assets

Task objects and set dressing come from
[proc-gen-3d](https://github.com/LuckyIYI/proc-gen-3d), pinned to
`45739ebfef8a8e933ba08b0c087618cd14dc54c9`. The exporters preserve original
visual surfaces, authored mass properties, convex collision cells and joint
topology. A knife block retains its open narrow slots, a mug its hollow interior,
and vehicles their free wheels and other articulated parts. Macro fastener
collision preserves shanks, heads and open nut/socket bores; thread-scale
microcontacts are omitted for bulk handling. Full thread geometry is rendered.

`prepare_task_assets.py`, `prepare_revision_assets.py` and
`prepare_catalog_revision.py` compile the manipulated designs. `task_assets.py`
imports them into Newton. `catalog_tasks.py` implements balance and knife tasks;
`crane_task.py` preserves the crane’s passive mechanism and adds a jointed chain,
3.84 kg steel ball, dry-stacked columns, separable floor panels and loose furniture.
The arm drives only its own joints; ball motion, impact and collapse come from
the simulated contacts and passive mechanism. The balance pivot has a
4 N·m/rad passive torsion spring, zero rest angle and a visible bearing housing.
Its missing weight changes its equilibrium; no motor commands its beam angle.
The knife block's longitudinal slots are fitted to 38 mm while retaining the
original catalog knives, 5 mm kerf and wood outline. Its lip supports the knife
after the gripper releases. Acceptance measures clipped blade clearance and the
native handle's supporting surface, rather than treating a tilted knife's
centerline as its physical envelope.

`prepare_furnishings.py` compiles selected tables and curated decorative assets.
`furnishings.py` places explicit task layouts and class-specific decorations;
it does not generate clutter or fill empty table edges. These fixed decorations
are visual-only and are not counted as simulated manipulation tasks. Wood albedo fields come
from the source repository’s CC0 scan-fitted directional spectra. Manufactured
edges retain crease-aware normals. Robot materials preserve their native look;
manipulated handles and weights use the kitchen scene’s blue, teal and gold.

## Validation and rendering

`audit.py` independently checks the recorded trace: finger-driven cube rotation
and retention, opposed hand grasp and pouring, drawer travel and mesh containment,
clipped knife-blade clearance, seating and stable release, balance equilibrium,
pre-impact building stability, ball contact, floor collapse, chain attachment, wheel
rotation against travel, hardware arrivals and supported placements. Ordinary
outcome checks and zero-overflow constraint checks must also pass. A diagnostic
export can inspect a failing probe, but `assemble_metal_teaser.py` refuses to
assemble a deliverable from it.

Keep each trace’s exact `source/` snapshot beside it. Export geometry with that
snapshot, preserving body counts and labels. Recorded object poses are never
replaced with controller targets. Rendering interpolates translations and
sign-corrected normalized quaternions to 30 fps. Temporal history resets for
every replay frame. `plan_metal_teaser.py` plans cameras; the Swift
`scripts/benchmarks/hero_hq_replay` tool renders the exported buffers;
`assemble_metal_teaser.py` verifies trace hashes, resolution, timing, frame counts
and decode integrity before delivering the video.

`assemble_retake_teaser.py` can replace selected cuts using separately validated
20-world batch recordings. It checks each recording and delivery hash, preserves
each task's time allocation, retains the unchanged encoded clips, and fully
decodes the completed edit. Its manifest records the source batch for every cut;
an edit containing retakes is explicitly identified as multiple recordings.
The quadruped lead-in and final pullback must come from the same recording so
neighboring stations and lighting remain continuous during the grid reveal.

```bash
# In the isolated remote workspace, with Newton on PYTHONPATH:
uv run --no-project --python .venv/bin/python python run.py \
  --output accepted-run --roster assets/hero_roster.json \
  --duration 15 --substeps 8 --iterations 32 --no-render --use-graph

# Geometry export may run on CPU locally; simulation remains on CUDA:
uv run python source/export_metal.py --run accepted-run \
  --assets local-assets.json --roster source/assets/hero_roster.json \
  --device cpu --output metal-data
uv run python source/plan_metal_teaser.py --data metal-data \
  --trace accepted-run/trace.npz --output shots.json
```

## Sources

- [Newton robot assets and policies](https://github.com/newton-physics/newton-assets),
  pinned to `f8fb7abcbeba2318814a74f3eeb02780ad7925d6`.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), pinned to
  `f054586a8e90465d49ee5be15335c4a0c7f57caf`: Kuka, Allegro, Shadow, Kinova,
  xArm and UR robot models. Preserve each model’s license when distributing assets.
- [HORA](https://github.com/HaozhiQi/hora): public in-hand rotation policy,
  pretrained weights and grasp configurations. Policy inference and object
  motion are evaluated in this FPGS scene, not copied from a source animation.
- [Mandara comparison](https://mandarobotics.com/blog/comparing-physics-engines/index.html)
  and [4DCodeBench](https://4dcodebench.com/): scene and presentation references.
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym): public
  hardware-demo reference. G1 and Go2 shots here are simulations; public hardware
  footage does not validate this FPGS run or establish identical policies.

## Separate editorial clips (October 7)

The wrecking crane releases its handle at 5.85 s and lifts clear by 7.1 s.
`validate_crane_release.py RUN` checks the recorded opening, retreat, and
settled gripper speed. The check fails on probe-109 and passes on
`wrecking-release` (A6000, 8 substeps, 32 iterations, 15 seconds). The
demolition gate still passes with all six upper floor panels collapsed.

`crane_visuals.py` adds cab glazing, panels, rollers, vents, pivot caps,
and hoses only during Metal export. These are zero-density, noncolliding
meshes on existing bodies; body masses and counts are asserted unchanged.

`compose_replay.py --spec SPEC --output DIR` assembles a presentation-only
overview from recorded worlds. It preserves every sampled pose (apart from
display translations), rejects out-of-range times, and records each source.
This overview is a composite, not evidence of a new simultaneous batch run.

`package_edit_clips.py --spec SPEC --output DIR` copies the individually
named close-ups and final zoom-out, fully decodes each MP4, verifies native
2560x1440 at 30 fps and frame count, and writes a provenance manifest.
The accepted lighter is one 4.5-second shot; the new wrecking shot is
10 seconds to include the turn, impact, release, and retreat. Existing
unchanged close-ups are reused without re-encoding.

## 96-tile paper teaser

`expand_replay_still.py --source METAL_EXPORT --output DIR` creates a
single-frame, seeded 12x8 arrangement of 96 replicas from the accepted
recordings. It balances template counts, varies recorded action phases,
whole-tile placement and table finishes. Tables remain aligned by default;
`--yaw-range` optionally adds rotation. Every pose is transformed
rigidly as a whole scene; inverse-transform assertions preserve the source
configuration. This is a presentation of recorded replicas, not validation
of a new 96-world simultaneous simulation.

The optional display-mesh reduction uses dependencies available with
`uv run --with fast-simplification --with scipy`.

`layout_paper_teaser.py FOLDER` assembles the 3600x2400 overview and three
1600x1000 manipulation renders into a 4920x2448 PNG and SVG with a clean
right-hand column. It uses Matplotlib and adds no labels or overlays.

Overview reduction welds duplicated hard-normal seams before simplifying.
If a reduced material region retains less than 90% of source surface area,
or exceeds 108%, its original mesh is preserved. This prevents disconnected
export triangles and thin regions from becoming incomplete robot skins.
`validate_overview_meshes.py --source SOURCE --overview OVERVIEW` checks
robot and gripper surface coverage against the original detailed export;
it fails on the damaged reduction and passes after correction.

For a full-resolution instanced paper still, use Blender's
`render_instanced_paper.py` importer. Each replica links the same original
mesh datablocks; only body transforms and table material slots vary.
The final overview and close-ups use Cycles with Metal acceleration,
adaptive sampling, and denoising. AgX at gamma 1 receives scene-linear
material colors converted from the exported sRGB palette.
No overview simplification is used. Triangle corner normals preserve the
source hard edges and avoid Blender's custom-normal seam failure. Textures
are resolved absolutely and packed into the saved `.blend` file.

The revised paper layout uses a tightly framed overview whose tile grid
continues beyond the image edges, with lighter, bolts sweep, and plate
placement stacked in the right-hand column. The sweep is captured earlier
while hardware remains ahead of the pusher; the lighter uses teal enamel
and warm yellow metallic trim. `render_cycles_teaser.py` renders each
close-up directly from the recorded poses selected by
`cycles-miniatures.json`, using the same Cycles configuration.

`arrange_paper_still.py SOURCE REPLICAS` rearranges the existing recorded
tiles while preserving template counts. Matching task types are separated
in all eight neighboring grid directions, including across variants.
Floating hand tasks are centered by their visible hand bounds; the hand
and manipulated objects move together while the table stays fixed.
Relative recorded body configurations and centered hand bounds are checked.

## Metal decor and lighting study

Keep an approved copy of the paper folder before a composition study.
`dress_paper_still.py SOURCE OUTPUT` adds fixed proc-gen storage boxes,
spare parts, paper trays, plate/block stacks, wood table variants, and
painted guides around hand stations. It checks table footprints and
existing decoration bounds, and preserves the recorded pose files byte
for byte. Assets retain their source commit in the furnishings manifest.

`prepare_metal_paper_view.py ORIGINAL DRESSED CAMERA OUTPUT` restores the
original detailed mesh surfaces and compacts visible tiles with a two-metre
margin for edges and shadow casters. The 96-tile layout remains in the
metadata; offscreen geometry is omitted only from this camera-specific
preview export. Rebuild it after changing the camera.

The native HQ replay tool accepts `--lighting soft-studio`, `warm-window`,
`cool-lab`, or `warm-gallery`. For still comparisons, `--frame-count 1`
renders a single accumulated frame. Use native resolution and the same
quality settings for every lighting option.

The approved pre-decor paper folder is kept as
`paper-teaser-checkpoint-approved` beside the active output. New studies use
`paper-teaser/decor-study`, so returning to the accepted composition requires
no inverse edits. Keep sage and light gray plastic worktops dominant; use
wood for accents. Wrecking-ball side clusters use proc-gen excavator and
dump-truck toys, while assembly/hand stations get gear kits and precision
tools, and kitchen stations get plate stacks. When moving this revision to
Blender, import the dressed metadata, including its appended static meshes,
rather than rebuilding only the original template geometry.

The furnishings baker preserves per-part polymer finishes before unioning
toy geometry. Their main colors follow the teaser's teal, blue, and yellow
palette, while dark chassis and metallic joints retain separate materials.

The next curation pass replaces the tiny generic office accessories with
larger activity-specific groups. It checks the saved task geometry near the
worktop as well as the table outline before placing each group. Tables keep
their orientation; red wood tints become neutral ash, and sage/gray plastic
remains dominant. Two or three groups fit each bench without crowding the
main action. This is a still-composition clearance check, not validation of
the entire recorded motion against new decoration.

| Activity | Proc-gen categories and selected objects |
| --- | --- |
| Plates and kitchen | Countertop microwaves, retro toasters, gooseneck kettles, ceramic canisters, mixing bowls, crocks, cutting boards |
| Demolition | Excavator and dump-truck toys, stacked timber |
| Workshop and weighing | Ventilated carrying crates, lidded totes, cordless drills |
| Assembly and play | Gear kits, shape puzzles, toy trains, ring stacks, brick crates |
| Floating hand stations | Desk fans, mantel clocks, articulated desk lamps, bound journals |
| Air hockey | Twin-bell clock and parts storage |

Rejected details include the filled slot caddy, whose thin rods read poorly
from the overview, heavily distorted vessels, and the three-part stereo
station. The final selection removes decorative vases/audio and favors
appliances and clocks. Plastic worktops are mostly light asphalt gray,
with sage on every fifth eligible tile. Recorded body poses and neighboring
task separation remain unchanged. `decor-study/appliance-scene-96` and
`appliance-metal` hold this revision; earlier studies are retained separately.

The final color revision restores the approved per-table material mix from
`paper-teaser/scene-96`: cream, neutral gray, some blue-gray and sage, plus
light and darker wood. Only red-biased random wood tints are neutralized;
the original wood textures and darker values remain. The two explicit
close-up overrides are sage for the lighter and blue-gray for hardware;
the plate scene keeps its wood surface. Miniature bookcases are replaced
by proc-gen anglepoise desk lamps with colored shades. This version is
stored as `decor-study/balanced-scene-96`; the overly uniform blue-gray
study remains separately available for comparison.

## Final decorated teaser exports

`prepare_paper_closeups.py SOURCE DRESSED CAMERAS OUTPUT` extracts the
accepted lighter (3.4 s), early hardware sweep (1.6 s), and plate placement
(4.7 s) from the original recording. It keeps the chosen decorated table
variants, restores full-resolution geometry, and translates the camera with
the centered hand. The resulting single-frame exports are shared by native
HQ Metal and Blender, so their recorded configurations match exactly.

The Cycles importer follows the dressed mesh inventory. It links surviving
original meshes, imports appended decor, respects deleted accessories, and
uses each tile's material overrides. Texture lookup includes the decorated
export, and texture multipliers use scene-linear colors. An import regression
check on the plate close-up found zero of the 13 expected decoration regions
before this change; afterward, all 13 and exactly the approved total mesh
count were instantiated.

Final outputs are kept separately in `paper-teaser/final-avbd-balanced` and
`paper-teaser/final-cycles-balanced`, preserving the approved checkpoint. Native HQ
uses 64 accumulated samples and eight diffuse samples with no upscaling.
Cycles uses 128 adaptive samples, Metal acceleration, and denoising. The
overview is 3600x2400, each inset 1600x1000, and the assembled figure
4920x2448. Pass `--renderer` to `layout_paper_teaser.py` to record the actual
renderer in the figure manifest. The saved Cycles files pack their textures.

The subsequent `cleaned` exports retain all 194 table material regions and
the recorded pose buffers unchanged. They remove the sweep pegboard and
Jenga ring-stack decoration. The desk lamp's upper pivot is raised 14 cm in
the source asset and connects vertically through the shade's top opening.
Boolean intersection of the support and shade is zero; the original support
intersected 1.37e-6 cubic metres of the shade.
The `refined` revision also omits the gray decorative spring rods/collars on
the lower lamp arm and shifts only the plate inset's toaster left of the wrist.

The native replay chooses a constant near plane per shot at 2% of its closest
focus distance, bounded to 5 cm–2 m. The previous fixed 5 cm near plane caused
white depth artifacts on distant thin containers. A distant tote regression
found 6,981 overly bright pixels in a 94,352-pixel body region before this
change and zero afterward, against a higher-precision clipping reference.
This leaves the tote geometry and normals unchanged. `--near-clip` overrides
the automatic setting in renderer units (scaled by `sceneLengthScale=0.1`);
render reports record the actual world-space clipping distances.

## Decorated editorial rerenders

`prepare_edit_replays.py ROOT DRESSED OUTPUT` applies the approved decorated
table variants to the 16 accepted individual takes and the final 96-tile
zoom-out. It preserves each source recording's time window and rotations,
and verifies body translations against that recording. Removed decorative
drills have empty reserved slots, so other props retain their positions.
The Blender teaser without drills is under `final-cycles-no-drills`.

The exported `render-plan.json` gives each clip's data, camera, frame count,
and recording provenance. Render with native 2560x1440 AVBD HQ, four samples
per frame, eight diffuse samples, `--reset-history-per-frame`, and
`--direct-video`; package with `package_edit_clips.py` for a full decode check.
The continuous final replay replicates accepted recordings across 96 tiles;
it is not a new simultaneous simulation. Camera culling covers the complete
zoom path, and its fixed focal length avoids overshooting the final layout.
Independent-frame rendering adapts the near plane throughout the pullback
to retain precision on distant thin surfaces.

The `visible-tasks` layout swaps drawer tile 13 with humanoid tile 58 to bring
the open drawer into the central crop. `swap_paper_tiles.py SOURCE OUTPUT
CAMERA 13 58` preserves all tile-relative transforms and material assignments,
checks eight-direction neighbor separation, and writes a visibility audit.
The original crop contained 14 of 15 task types; the revised crop contains all
15, with the humanoid still visible on the left. Use this arrangement for the
overview and final zoom-out. The inset cameras and individual clips are
unchanged. The revised Cycles teaser is under `final-cycles-visible-tasks`.

`assemble_video_teaser.py ROOT RENDERER --wait-for-clips` waits for the 17
decode-validated editorial exports, then assembles a 24-second video teaser.
The left panel is a dedicated 1800x1200 native HQ replay ending at the paper
camera; `playbackSpeed: 0.25` in its camera JSON slows the recorded motion
independently of the camera path. The default replay speed remains 1.
Three 624x390 rows show every task at normal speed with staggered 4–5-second
cuts. Drawer opening and placement are joined into one shot. The final video
is 2460x1224 at 30 fps, with white gutters and no labels or overlays.
`timeline.json` and `teaser-report.json` retain edit timings and provenance.

Knife-holder wood is baked from proc-gen's original `render_detailed.material`
oak shader. Run Blender with `bake_knife_oak.py -- PROC_GEN_CHECKOUT` to rebuild
the two UV atlases in `assets/knife_oak`. `texture_knife_holders.py EXPORT...`
applies them to replay visuals, preserving pose buffers and the existing blue
and green handle materials. Editorial replay preparation applies this finish
automatically. The bake contains base color; AVBD uses a satin roughness of
0.38 rather than the source Blender shader's procedural bump and clearcoat.
The approved-style refinement uses finer longitudinal mapping and a restrained
neutral color ramp, removing broad cloudy bands. Texture refreshes preserve
the original UV atlas, geometry, and blue/green handle materials.

Replay material decoding must forward optional `transmission` and `ior`
fields to AVBD's `previewOptics`. Omitting them turns the pouring jar opaque
despite the export retaining its glass settings. Missing fields default to
opaque materials at IOR 1.5; the pouring jar uses transmission 0.98, IOR 1.47.

For composition iterations, use the realtime ray-tracing preset, four passes,
one diffuse sample, and retained temporal history. The 24-second pullback now
reveals the 96-table layout with a frame-filling crop, retaining the accepted
viewing direction and avoiding a large empty floor border. Its fixed near plane is 1 m; the replay renderer
extends the far plane to cover the full camera path and scene bounds.
`prepare_realtime_overview.py FULL DRESSED OUTPUT` retains full geometry for
the opening wrecking-ball tile and validated overview meshes for distant tiles. It
also merges identical material references so repeated geometry can be shared
by the ray tracer. This avoids the full 96-tile export's memory bottleneck.
The video teaser reuses the completed 24-second wide clip via a central crop
and downsampling instead of rendering the pullback a second time.

## CUDA Cycles replay clips

`export_cycles_replays.py` runs inside Blender and packs the accepted replay
geometry, materials, UVs, and original 50 Hz pose arrays. Shared meshes attach
to per-body controllers. The overview instances 96 dressed worlds four times
on a 24 by 16 grid, with the outer border outside the final camera frustum.
The pullback starts on a central wrecking-ball scene.

`render_cycles_replay.py` uses Cycles OptiX at 1920 by 1080: 16 samples for
close-ups and 8 for the overview, GPU denoising, persistent data, and six
maximum bounces. It interpolates recorded translations and quaternion SLERP;
no physics steps run during rendering. Each output carries its trace hash,
actual frame timings, camera settings, and a full video decode check.
`assemble_cycles_montage.py` composes the separate clips into a 1920 by 956
14-second teaser without upscaling the source images. The opening three
close-ups share a six-second window, followed by four two-second groups of
other tasks. The lighter's 4.5-second demonstration plays over six seconds;
the remaining close-ups retain real-time playback. The overview stretches
six seconds of composite recordings over the full 14-second pullback.

The baked knife-holder atlases restore their original Blender UV orientation
on import; the Metal replay stores their vertical texture coordinates flipped.

## Open toy well (October 14)

`well_task.py` adds a tall open frame, passive spool and crank, freely rotating
handle, 44-link rope, and hanging bucket. The UR10 follows a circular crank path
and turns its wrist with the crank so that the jaws transmit tangential force
through contact. Only the robot joints receive targets. The 30 mm radius drum
allows the rope to wind over itself for more than one full revolution.
The rope has fixed-length capsules and passive planar hinges; it bends in the
spool plane, while the bucket's bail connection is a passive spherical joint.
Drum contact physically winds the rope; no rope shortening or bucket drive is used.

```bash
uv run python scripts/benchmarks/hero_montage/run.py \
  --assets ASSETS.json --cases well:0 --duration 12 \
  --iterations 16 --substeps 8 --no-render --output OUTPUT
```

The grip closes over one second with 1200 N/m stiffness and a 12 N effort limit.
The original instant closure with 6000 N/m stiffness caused visible contact
chatter. On `fpgs-main` commit `14cf39b775dc4786f63903d4068c42aaad03f3ce`,
changing only the grip reduced resting finger travel from approximately 0.8 mm
to 0.002 mm at the same solver budget. No solver modification was needed.

The thinner-drum recording turns the spool 1.654 revolutions and raises the
bucket 0.377 m, with 34 wrapped rope segments and 1.714 turns of rope winding.
The audit checks winding, bucket rise, finite state, joint anchors, and resting
contact/rope vibration. Use `assets/well_roster.json` to export the recorded replay.
Source snapshots, an adapter for the current FPGS constructor names, comparison
recordings, validation reports, Blender input, and the Cycles preview are under
`newton/tests/outputs/well-stability-20261014/`. The original preview is preserved
under `newton/tests/outputs/well-20261014/`.

The negative control disables only robot–mechanism contacts, preserving all
robot commands and mechanism joints. Opening the jaws alone is not an isolating
control: the moving fingers can still push the crank.

## Buried mug unpacking (October 14)

`unpacking_task.py` places a blue proc-gen ceramic mug inside an open carton
beneath 275 lightweight white packing pieces. Each piece is a free rigid body
with a capsule collider, density 18 kg/m³, and a softly irregular visible shell.
The carton is fixed; only robot joints are actuated. The robot pushes through
the packing, closes its jaws around the mug, lifts it clear, and places it on a
coaster outside the carton. Packing and mug poses are never prescribed.

```bash
uv run python scripts/benchmarks/hero_montage/run.py \
  --assets ASSETS.json --cases unpacking:0 --duration 15 \
  --iterations 16 --substeps 8 --no-render --output OUTPUT
```

The presentation starts after three seconds of physical settling. The accepted
run displaces 36 packing pieces by more than 4 cm, lifts the mug 0.323 m, and
places it upright within 12 mm of the coaster target. The opening-frame
visibility check casts 4,026 surface rays toward the mug and finds no visible
mug surfaces. The first, less-filled prototype exposed the blue rim and was
rejected despite passing the placement check.

Use `assets/unpacking_roster.json` for replay export. The local package at
`newton/tests/outputs/unpacking-20261014/` retains the accepted `probe04` trace,
source snapshot, visibility check, Cycles scene, and video. Fine foam and paper
surface finishes are added by its saved `pack_unpacking.py`; they do not change
collision geometry or recorded motion.

### Pelican bicycle with a physical remote fader

`--cases pelican:0 --duration 14` retains the complete supplied pelican bicycle,
including its handlebars. A UR10 grips a passive 240 mm slider on a separate
remote console and moves it from zero to full scale. Each physics substep reads
the actual slider coordinate and maps it to a 0–60 rpm wheel velocity target,
with a finite 0.20 Nm motor torque limit. Only the robot and this explicitly
requested electric drive are actuated; gears, cams, wings, beak, and pedal
linkages remain passive. One fixed imported root keeps the two physical ankle
connect constraints inside the same articulation.

The source CAD and contact MJCF live in `assets/pelican_bicycle/`, and
`assets/pelican_roster.json` exports the scene through the existing replay
rendering pipeline. The task audit checks slider travel, wheel rotation,
gear ratio, wing/beak motion, initial standstill, and ankle closure.

### Contact-driven bottle flip

`--cases bottle_flip:0 --duration 7 --iterations 16 --substeps 8` adds a UR10
that grips, lifts, and flips a passive bottle. The robot uses the joint targets
in `assets/bottle_flip_throw.json`; the bottle receives no pose commands,
release impulses, attachments, or flight correction. The swing stays in a
vertical plane to reduce sideways rotation. Visible tapered geometry follows
the convex collision envelope.

The bottle uses a rigid partial-fill approximation: 185 g total mass, a 30 mm
center of mass, and fixed inertia. This is not a liquid/slosh simulation.
The audit requires one complete turn, a ballistic airborne interval, and an
upright, stationary finish on the tabletop. The accepted standalone recording
uses the unchanged FPGS solver at `14cf39b775dc4786f63903d4068c42aaad03f3ce`.
The motion is a calibrated demonstration, not a robust bottle-flipping policy;
small solver or contact differences can change whether the landing succeeds.

`assets/bottle_flip_roster.json` exports this scene. The local package under
`newton/tests/outputs/bottle-flip-20261014/` retains the accepted `timing-01`
recording, its audit, a robot-contact-disabled negative control, source snapshot,
and the Cycles replay/video. Replays preserve the recorded motion at real time.

### Two-arm box quarter-turns

`box_turn_task.py` adds a free cardboard box and two UR10 arms with the
original Franka gripper collision geometry. Both grippers keep a fixed
12 mm per-finger aperture. Opposing fingertip contacts lift the box, roll it
approximately 90 degrees, yaw it 90 degrees, and lower it onto the table.
Only robot joint targets are commanded; the box has no drive, attachment,
pose override, or applied task force. Box markings are visual-only.

```bash
uv run --extra examples python scripts/benchmarks/hero_montage/run.py \
  --assets assets.json --cases box_turn:0 --duration 13 \
  --substeps 8 --iterations 16 --no-render --output box-turn
```

The validated local package is `newton/tests/outputs/box-turn-20261015/`.
The accepted `probe12` take has 5.29 degrees of intermediate flip error,
0.72 degrees of final orientation error, and no measured motion during the
last second. A separate contact-instrumented rerun also passes. Its contact
log contains only the two original finger collision shapes per robot and
the table touching the box. The negative control disables robot-box
contacts while keeping table contacts: the box stays at its starting pose
and correctly fails the task gate. The grippers remain at fixed aperture
throughout both runs. This is a calibrated robot trajectory, not a learned
or disturbance-robust manipulation policy.

Simulation used solver snapshot `14cf39b775dc4786f63903d4068c42aaad03f3ce`
at 50 Hz. The Cycles preview replays the accepted physical trace at 30 fps
and real-time speed; rendering does not resimulate or alter the box motion.

### Task-specific dressing and bird cameras

`assets/task_decor.json` defines 27 larger proc-gen props across the nine
new tasks. `furnishings.dress()` uses these explicit layouts. The former
generic office-scatter fallback has been removed. All additions are visual-only and remain inside the
table footprint. Existing older-task dressing is preserved.

`style_task_replays.py` applies the same layouts to saved Blender replays,
preserves their lighting and recorded motion, and saves separate editable
scenes plus Cycles/Metal previews. The bird automaton uses a closer oblique
view of the crank hand and bird profile; the pelican view retains both the
fader interaction and complete bird silhouette. Bird flat normals are
preserved. The original replay packages are not overwritten.

The nine updated replay packages, preview sheet, and unchanged-motion hash
checks are in `newton/tests/outputs/styled-scenes-20261015/`. Each package
contains `scene.blend`, `motion.npz`, `camera.json`, `metadata.json`,
`dressing.json`, and `preview.png`; it can be passed directly to the existing
Cycles video renderer. The saved configuration reproduces the previews:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b --python \
  scripts/benchmarks/hero_montage/style_task_replays.py -- \
  newton/tests/outputs/styled-scenes-20261015/config.json
```

### Orbit close-ups and expanded demo edit

The saved `orbit-montage-20261015` output package preserves the approved
close-up recordings and adds a five-degree eased camera arc to seven of the
nine newer tasks, beginning at 2.5 seconds. Beer pong and bottle flip keep
fixed cameras for their fast flight trajectories. The package includes
reproducible preparation and assembly scripts, camera previews, and motion
hash validation. The bottle cap clearance fix and tangram without its backing
board are retained.

Its overview keeps 384 linked collection instances and replaces 156 tiles
with the nine newer decorated tasks. All nine are represented near the final
view center, with no matching eight-neighbor task pairs. The 10-second camera
path starts at the wrecking ball and retains the approved boundary-free end
framing. This overview combines accepted recorded replays; it is not a new
384-world simulation. The older overview recordings retain their previous
playback timing, while the new task recordings advance at native speed.

The full edit follows the pullback with a staggered 2x2 grid of all 25 robot
demos. Intermediate close-ups use 3.6-second action excerpts, with changes spaced
0.9 seconds apart in top-left, top-right, bottom-left, bottom-right order.
Only the final clip in each cell plays through its ending. Quick excerpts
use the earlier teaser speed of 2/1.8 (1.111x); finals use 5/4.6 (1.087x). The nine-task
3x3 montage and latest-video collection are refreshed from the same clips.

### Toaster loading

`toaster_task.py` adds a two-slice breakfast task. One free bread body rests
in the raised carriage; the UR10 inserts the other, releases it, approaches
from the front, depresses the paddle, and withdraws. The original proc-gen
`toaster_studio` retro housing retains its open wells and articulated lift.
A passive return spring, spring-loaded rear ratchet, and tapered centering
wires complete the mechanism. The ratchet catches through collision with a
carriage tab. Only robot joint targets vary over time; neither bread nor the
carriage receives a scripted trajectory or assisted force. This is a rigid
mechanism demonstration, without thermal cooking or deformable bread.

Generate the source asset with `prepare_toaster_asset.py --source PROC_GEN`,
then record with:

```bash
uv run scripts/benchmarks/hero_montage/run_toaster.py --assets ASSETS.json --output OUTPUT
```

The runner requires the current FPGS `matrix_free` API and CUDA. It verifies
insertion, carriage travel, pre-press support, final bread support, resting
motion, and solver capacity. Export with `export_metal.py` and
`assets/toaster_roster.json`; the existing Cycles replay pipeline supplies
studio lighting and a subtle camera orbit. The bread outline and crumb
texture are generated locally in the task source.

### Video style reference (2026-10-15 correction)

Use `video_style_reference.py` for all new close-up cameras and edit timing.
The user rejected the negative-azimuth camera path: the required orbit is
**positive world-Z azimuth**, usually 5 degrees (6 for the kitchen/Unitree
shots), starting around 2.5 seconds with the existing smooth interpolation.
Do not negate it based on apparent subject rotation. The helper checks the
positive cross product explicitly. Keep accepted static shots static.

Preserve `export_cycles_replays.studio`: AgX / Medium High Contrast, exposure
0, gamma 1, world strength 0.5, existing warm key/cool fill and soft sun.
Use the saved scene lighting for rerenders. Cycles OptiX, 1080p30, 16 samples
and denoising are the close-up baseline. Preserve the soft green, light
bluish gray, cream and neutral wood table mixture; no blanket recoloring.
Props should be sizable, coherent with the task, and clear of manipulation.
Task objects must follow recorded physics; editing changes camera/playback
only. Keep a hash of the replay and preserve the original videos.

For a new packed camera:

```bash
uv run scripts/benchmarks/hero_montage/video_style_reference.py input-camera.json camera.json
```

Keep the 10-second wrecking-ball zoom-out and boundary-free framing. In the
2x2 montage, stagger the first cells by 0/0.9/1.8/2.7 seconds, then use
3.6-second core-action excerpts. Only each cell's final clip may run long;
end the montage when the last final action finishes. Do not substitute
full-length intermediate sources. Select source trim windows visually.

### HQ wrecking crane: interactive branch brake

The October 15 HQ revision ports the side grasp handle and 3 N·m dry slew
brake from `codex/fpgs-interactive-scenes` (commit `51f909129`). The arm
moves the handle in a circle through contact, then releases and retreats.
The passive brake resists the ball back-driving the core; it is not a
pose target. Crane masses match the branch's USD asset.

Use `run_wrecking.py` with current FPGS (joint-friction support required),
4 substeps at 60 Hz and 32 iterations. It enables joint friction explicitly
and checks bilateral handle contact, impact, collapse, chain attachment and
less than 0.05 rad core drift after release. The legacy batch runner does
not enable this brake; use this dedicated runner for new HQ crane traces.
Export with `assets/wrecking_roster.json`, retaining the existing visual
trim and `video_style_reference.py` camera direction. The overview is not
rerendered as part of this close-up revision.

When refreshing an existing HQ scene's physics, restore its approved dressing
from the previous exported `scene.json` and mesh buffers with
`restore_scene_dressing.py SOURCE REFERENCE OUTPUT` before packing Cycles.
The builder defaults are not the approved editorial decorations. For the
wrecking scene, the approved props are the excavator and dump truck, using
their original transforms and materials; no parts tray or office clutter.

### Miniature G1 lava fall

`lava_fall_task.py` builds a rigid 30%-scale G1 toy with spread arms on a
cantilevered stand, a deep blue pot, and 200 lightweight flame-shaped pieces.
The toy's mass and inertia derive from the imported G1, with per-link convex
collision shapes added because its walking USD omits limb collisions. Gravity
starts the backward fall; neither the figure nor the pieces receive animated
poses, release impulses, or sinking forces.

Run `run_lava_fall.py --assets ASSETS.json --output OUTPUT` with the current FPGS
CUDA environment. `ASSETS.json` needs a `unitree_g1` asset-directory entry.
The checked-in `assets/lava_settled_pile.npz` is a physical initial packing.
Regenerate it with `--prepare-pile --duration 3`; that preparatory trace is not
a demonstration. The normal run checks contact capacity and that the complete
visible figure ends inside the pot and below the pile surface. Use
`assets/lava_fall_roster.json` for replay export, then the shared Cycles lighting
and `video_style_reference.counterclockwise_camera` for rendering.

### Wrecking crane: stock fingers and rear-body grasp

The current `crane_task.py` grips the original rear counterweight directly,
across its 52 mm depth. The yellow side handle and long jaw extensions are
removed; the stock finger geometry and normal 103.4 mm TCP offset are used.
The passive 3 N·m slew brake, articulated chain, building, approved excavator,
dump truck, and original table remain. `run_wrecking.py` checks bilateral
contact against the rotating crane body and rejects the obsolete added shapes.
The validated replacement recording is `wrecking-rear-grasp-20261015/probe1`;
older handle-based recordings are historical references only.
