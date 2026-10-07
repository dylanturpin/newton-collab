# CUDA FPGS house of cards

Standalone prototype: 15 dynamic cards, three tiers, and a dynamic ball launched
at 3 seconds. Cards are supported by contact and friction throughout; there are
no joints, glue, or scripted card transforms. Overlapping roof cards start at
their geometric resting angles rather than intersecting.

Run with the Newton environment on a CUDA node:

```bash
uv run python simulate.py --output run --substeps 8 --iterations 32
MUJOCO_GL=egl uv run --with mujoco --with pillow --with imageio-ffmpeg python render.py --run run
```

The renderer replays `trace.npz` using MuJoCo for drawing only. It never advances
MuJoCo physics. The output is a 1280x720, 30 fps, eight-second MP4.

Measured on RTX A6000, 2026-10-05:

| Substeps | Iterations | Thickness | Simulation ms/frame | Before impact |
| --- | --- | --- | --- | --- |
| 8 | 32 | 1 mm | 13.07 | Pass: max drift 3.3 mm, max rotation 2.4 degrees |
| 4 | 8 | 1 mm | 3.69 | Collapses early |
| 4 | 16 | 1 mm | 4.31 | Collapses early |
| 8 | 8 | 1 mm | 7.11 | Collapses early |
| 4 | 8 | 2 mm | 3.35 | Still standing, but drifts 16.4 mm; fails the 15 mm stability limit |

These timings include collision, solve, launch control, GPU pose recording, and
graph launches. They exclude compilation and video rendering. All listed runs
use propagation-colored FPGS, friction coefficient 0.8, friction-anchor beta
0.2, and a 0.5 mm speculative contact gap. Both standing houses collapse after
the ball hits. `simulate.py` exits unsuccessfully for failed validation while
preserving the trace and diagnostics for inspection.

Remote files are isolated in `/data/home/liinbor/src/fpgs-card-house-20261005`.
The batch script imports the existing validated Newton checkout without editing
it. The successful thin-card run is `run-09-gap-8ss-32iter`; the thicker 4/8
preview is `run-10-2mm-4ss-8iter`.
