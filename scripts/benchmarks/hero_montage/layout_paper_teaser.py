# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Assemble a publication teaser from native simulation render panels."""

import argparse
import json
from pathlib import Path


def layout(folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    width, height = 4920, 2448
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    panels = [(folder / "overview-render/overview-96/frame_0000.png", (24, 24, 3600, 2400))]
    panels.extend(
        (folder / "miniatures" / name / "frame_0000.png", (3648, 24 + i * 810, 1248, 780))
        for i, name in enumerate(("07-lighter-contact", "04-hardware-bin", "08-plate-rack"))
    )
    for path, (x, y, w, h) in panels:
        image = mpimg.imread(path)
        assert abs(image.shape[1] / image.shape[0] - w / h) < 1e-6
        axes = figure.add_axes((x / width, 1 - (y + h) / height, w / width, h / height))
        axes.imshow(image, interpolation="lanczos", aspect="auto")
        axes.set_axis_off()
    figure.savefig(folder / "paper-teaser-96.png", dpi=100, facecolor="white")
    figure.savefig(folder / "paper-teaser-96.svg", dpi=100, facecolor="white")
    plt.close(figure)
    (folder / "figure-manifest.json").write_text(
        json.dumps(
            {
                "width": width,
                "height": height,
                "scene_tiles": 96,
                "panels": [str(path.resolve()) for path, _ in panels],
                "right_column": ["Contact-only lighter", "Bolts sweep", "Plate placement"],
                "source": "Replicated accepted CUDA recordings with render-only layout and material variation",
                "simultaneous_heterogeneous_batch": False,
                "overview_renderer": "Blender Cycles, linked full-resolution source meshes",
                "overview_mesh_simplification": False,
                "labels": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    layout(parser.parse_args().folder)
