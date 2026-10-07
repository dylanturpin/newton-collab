# Physics scene archive

This branch preserves the scene prototypes and their supporting tools. Start
with the individual guides:

- [Manda rigid fixtures](manda_rigid.md): eight reconstructed solver benchmarks.
- [Kitchen pour](kitchen_pour.md): heterogeneous dishes, glass container,
  recording, Blender rendering, and AVBD Metal replay.
- [Card house](card_house_demo/README.md): card-stack simulation and comparison.
- [Hero montage](hero_montage/README.md): 20 heterogeneous stations, including
  wrecking-ball demolition, thumb-operated lighter, plate rack, drawer,
  knife insertion, hardware collection, toy-brick pouring, and locomotion.
- [Hero Metal replay](hero_hq_replay/README.rst): native-resolution local rendering.

Procedural meshes, convex collision cells, textures, task configurations and
policy inputs are included. Robot downloads use the pinned upstream revisions
in the hero guide. Machine-specific recordings, videos, virtual environments
and Swift build products are excluded. Historical Slurm scripts contain the
original cluster paths and need adapting to another installation.

The final 96-second 1440p teaser used separately validated 20-world CUDA batches
(probes 63, 109 and 142), with six seconds per task family and a final pullback.
Its SHA-256 is
`fdf278231adb7c0292d8fa2a4e3ba0ddbcbb465313308d082b49dde43da6500f`.
The latest batch used 8 substeps and 32 iterations at 50 Hz. This archive
preserves the latest scene implementation; reproducing an earlier cut exactly
requires that recording's frozen source snapshot and trace.

Third-party provenance is recorded beside assets and in the scene guides.
HORA's MIT license is included under `hero_montage/assets/hora/LICENSE`.
Proc-gen-3d inputs derive from revision
`45739ebfef8a8e933ba08b0c087618cd14dc54c9`; that checkout contains no top-level
license file, so the Newton license should not be interpreted as a license
grant for those upstream designs. Preserve upstream robot licenses when
redistributing downloaded models.
