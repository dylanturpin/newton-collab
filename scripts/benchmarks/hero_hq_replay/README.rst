Direct native video export
==========================

Build the replay tool through ``uv``. Use a separate scratch directory while
an earlier exporter binary is still running::

    uv run --no-project swift build --package-path scripts/benchmarks/hero_hq_replay \
      --scratch-path /tmp/hero-video-build -c release

Export the recorded poses straight to hardware-encoded MP4 clips::

    uv run --no-project /tmp/hero-video-build/release/HeroHQReplay \
      --data accepted-run/metal-data --cameras accepted-run/shots.json \
      --output direct-clips --direct-video --width 2560 --height 1440

``--video`` and ``--direct-video`` use hardware streaming by default, with
one realtime-quality HQ pass per output frame,
with native resolution, denoising, and motion-aware temporal history. Scene
changes reset history. ``--quality high`` retains the higher ray budgets;
``--quality balanced`` selects intermediate ray budgets. The realtime video
preset uses emitted-power-weighted area-light sampling to share its ray budget
between lights, and caps diffuse sampling at one ray per pixel per pass. This
trades single-frame noise for speed while retaining temporal denoising. Explicit
``--samples-per-frame``, ``--accumulation`` and ``--reset-history-per-frame``
remain available. These settings are saved in each render report. Rendering
quality and rendering throughput must be measured together.

Video frames go from Metal textures to a bounded six-buffer IOSurface pool.
VideoToolbox hardware H.264 encoding is required, and compression overlaps
subsequent rendering. There are no CPU pixel readbacks, channel-swapping loops,
PNG files or later video re-encoding. The renderer and repeated world geometry
are reused across shots; pose recordings are mapped read-only.
Still export uses PNGs. Legacy video-frame PNG output remains available with
``--video --png-frames``; intermediate accumulation passes no longer read
pixels back.

Join clips and check their packet counts and timestamps::

    uv run --with imageio-ffmpeg --with pillow python \
      scripts/benchmarks/hero_montage/assemble_metal_teaser.py \
      --data accepted-run/metal-data --frames direct-clips \
      --shots accepted-run/shots.json --output teaser

The assembler validates the accepted simulation, trace hashes, cut timing,
frame counts and native resolution. Direct clips are remuxed without another
encoding pass. ``--full-decode`` adds a full decoded-frame check; this is
reported separately from the encoding/assembly timings. Legacy PNG export
continues to use its existing software encoding and full decoding checks.

Measure capture and encoding separately from scene rendering::

    uv run --no-project /tmp/hero-video-build/release/HeroHQReplay \
      --benchmark-encoding --benchmark-seconds 60 --output encoder-benchmark
    uv run --no-project --with imageio-ffmpeg --with numpy python \
      scripts/benchmarks/hero_hq_replay/validate_encoding_benchmark.py encoder-benchmark

The test generates a moving native-resolution pattern on the GPU, requires
hardware encoding, and checks the decoded frame count, duration, color-channel
order, orientation and embedded frame ordinals. Its timing excludes scene
rendering. Reports distinguish all GPU passes per output frame, setup time,
GPU capture-copy time, encoder submission, backpressure and final drain time.
