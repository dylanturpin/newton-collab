# Kitchen replay with AVBD Metal HQ

This small renderer consumes the recorded FPGS kitchen poses. It does not create
or advance an AVBD solver. The local `avbd-metal` checkout supplies
`GPUSimRenderer`, with HQ ray tracing and MetalFX reconstruction required.

The exporter preserves the approved Blender meshes and their per-corner normals,
including sharp edges, and supplies the original 60 Hz position/quaternion
samples. Two additional static bodies represent the table and studio floor.
Binary vertices use the renderer's documented 64-byte ABI; positions and xyzw
quaternions each use 16 bytes per body per sample. Glass uses the renderer's
experimental dielectric transport at transmission 1 and IOR 1.5. It has the
renderer’s documented limits, including no caustics or nested media.

The exporter accepts `--glass-wall-thickness METRES` (default `0.012`). For
example, `--glass-wall-thickness 0.006` halves the visible walls and base to 6 mm,
preserving the exterior dimensions and recorded poses. This changes render
geometry only; it does not rerun the FPGS simulation.

From the Newton checkout, export the existing Blender replay:

```bash
uv run --no-project /Applications/Blender.app/Contents/MacOS/Blender \
    --background /path/to/kitchen-pour-8ss-10iter.blend --python-exit-code 1 \
    --python scripts/benchmarks/export_kitchen_hq.py -- \
    --run /path/to/run-c8192-10x8 --output /path/to/hq/data
```

The package defaults to the sibling `avbd-metal` checkout. Set
`AVBD_METAL_REPO=/absolute/path/to/avbd-metal` to choose another checkout.
The renderer repo is consumed as a local dependency, including local edits.

```bash
uv run --no-project swift build -c release \
    --package-path scripts/benchmarks/kitchen_hq_replay

uv run --no-project scripts/benchmarks/kitchen_hq_replay/.build/release/KitchenHQReplay \
    --data /path/to/hq/data --output /path/to/hq/previews \
    --cameras /path/to/hq/cameras.json --times 12,20 --accumulation 32
```

The camera JSON is an array of objects with `name`, three-component `position`
and `target` vectors in metres, and vertical `fov` in degrees. Previews settle
the HQ temporal history independently at each pose. The renderer fails if HQ
is unavailable or silently falls back to Fast.

Optional `endPosition`, `endTarget`, `endFov`, `moveStart`, and `moveEnd` fields
animate the view between recording times in seconds. Position, target, and FOV
interpolate with a quintic easing curve, keeping both endpoints stationary.
Omitting the endpoint fields preserves a fixed camera.
For multiple moves, an optional `keyframes` array takes precedence. Each entry
contains `time` in seconds, `position`, `target`, and `fov`; times must increase.
The view eases between adjacent entries and holds outside their time range.

Add `--video-camera CAMERA_NAME` to render a full 30 fps PNG sequence from the
60 Hz recording, keeping temporal history as the bodies move. The report records
the camera, renderer, device, source trace hash, timing, and physics settings.

The replay defaults to a 0.1–30 m clipping range, adjustable with `--near-clip`
and `--far-clip` in metres. The previous 0.01–100 m range lost enough depth
precision to produce self-intersection blotches in primary reflections and
shadows. Tightening the range clears the matched settled-pile comparison with
the renderer's original ray offsets and all lighting effects enabled.

Use `--quality high` for the higher ray budget, `--scale 0.5` for MetalFX 2x
upscaling, and `--no-denoise` to inspect native-resolution HDR lighting before
MetalFX. Keep the camera, pose and ray budget identical for comparisons.
The renderer checkout's clear-glass reconstruction fix masks transmitted pixels
out of neural denoising and temporal history while preserving opaque denoising.

For raw exports, `--supersample 1.5 --width 2560 --height 1440` renders at
3840x2160 and spatially downsamples each frame to 1440p. This reduces geometric
and refracted-edge aliasing without upscaling or MetalFX history. The adapter
accepts factors from 1 through 2; reports record render and output dimensions.
Repeated raw draws via `--accumulation` do not average camera samples.

`--interfaces 256` lets rays follow repeated total internal reflections inside
thin glass. The default remains 24; budgets above 32 require the renderer's
extended interface-budget fix. This changes rendering only. The 6 mm replay
uses `--no-denoise --quality high --interfaces 256 --supersample 1.5`.

To render the resimulated floor variant, run the solver with `--support floor`
and export with `--builtin-floor`. This omits both authored support meshes and
sets `builtin_ground` in the replay metadata, enabling AVBD's checkerboard.
The solver aligns its infinite collision plane to the renderer's Z = 5 mm
floor and starts the box 1 mm above it. Pass the new run's directory to the
exporter so the rendering uses the new poses.
