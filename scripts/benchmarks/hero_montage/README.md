# FPGS hero montage

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

`prepare_furnishings.py` compiles varied tables, tools, plants and 16 office-clutter
designs. `furnishings.py` places small, irregular clusters of pens, stationery,
clips and other props at table edges, checking their footprints against the
actual tabletop and existing decor. These fixed decorations are visual-only
and are not counted as simulated manipulation tasks. Wood albedo fields come
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
