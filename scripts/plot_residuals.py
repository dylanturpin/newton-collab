#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot residual-vs-GS-iterations curves + contact-mix tables for one frame.

Tool for the FPGS Friction Modes 10/13 slice (issue #16).  Consumes the
output directory produced by ``scripts/solver_replay.py --mode replay``
(issue #14 / 8/13), i.e. a directory containing::

    replay.json      # 8/13 replay summary, schema_version >= 1.2
    snapshot.png     # optional GL snapshot of the frame

and emits:

    <out>/r_compl.png
    <out>/r_cone.png
    <out>/r_gap.png
    <out>/r_ds_compl.png
    <out>/r_ds_dual.png
    <out>/r_mdp_dir.png
    <out>/contact_mix.md    # markdown 2x4 + totals contact-mix table
    <out>/index.json        # small manifest of what was written

Each PNG plots one residual channel on a log-y axis against the
``gs_iterations`` sweep, with one line per ``friction_mode``.  Colours
are fixed across all plots in the series so the four modes are always
recognisable.

The script is reproducible: a second invocation against the same
``replay.json`` writes the same PNG bytes (matplotlib ``savefig`` with
``metadata={}`` to strip the creation timestamp).

Typical usage::

    python scripts/plot_residuals.py \\
        --frame artifacts/issue-worker/issue-16/replays/h1_tabletop_step70 \\
        --out   notes/reports/_assets/fpgs-friction-modes/plots/h1_tabletop_step70

``--frame`` may be a directory (implies ``<dir>/replay.json``) or the
JSON path directly.  ``--out`` defaults to ``<frame>/plots`` if omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# Constants kept in sync with scripts/solver_replay.py
# =============================================================================

_RESIDUAL_CHANNELS: tuple[str, ...] = (
    "r_compl",
    "r_cone",
    "r_gap",
    "r_ds_compl",
    "r_ds_dual",
    "r_mdp_dir",
)

# Human-readable labels for axis / legend rendering.  Keep in sync with
# the report text under ``Metrics`` in
# ``notes/reports/fpgs-friction-modes-comparison.md``.
_RESIDUAL_TITLES: dict[str, str] = {
    "r_compl": r"$r_{\mathrm{compl}}$ — normal NCP complementarity",
    "r_cone": r"$r_{\mathrm{cone}}$ — Coulomb cone feasibility",
    "r_gap": r"$r_{\mathrm{gap}}$ — penetration gap",
    "r_ds_compl": r"$r_{\mathrm{ds\_compl}}$ — de Saxcé complementarity",
    "r_ds_dual": r"$r_{\mathrm{ds\_dual}}$ — de Saxcé dual-cone",
    "r_mdp_dir": r"$r_{\mathrm{mdp\_dir}}$ — MDP direction error",
}

# Fixed friction-mode order + colour map so the four modes look the same
# across every plot in the series (acceptance criterion).
_MODE_ORDER: tuple[str, ...] = ("current", "bisection", "bisection_desaxce", "coulomb_newton")
_MODE_COLOURS: dict[str, str] = {
    "current": "#1f77b4",  # matplotlib C0
    "bisection": "#ff7f0e",  # C1
    "bisection_desaxce": "#2ca02c",  # C2
    "coulomb_newton": "#d62728",  # C3
}
_MODE_MARKERS: dict[str, str] = {
    "current": "o",
    "bisection": "s",
    "bisection_desaxce": "^",
    "coulomb_newton": "D",
}

# Tiny positive floor used when log-plotting residuals that can be
# exactly zero (e.g. ``r_cone`` in modes that project against the
# cone every sweep).  Matches how Gilles' reference plot handles zeros.
_LOG_EPS: float = 1.0e-16

# Column order / human labels for the contact-mix markdown table.
_CONTACT_MIX_COLUMN_LABELS: dict[str, str] = {
    "normal_only": "normal-only",
    "sticking_friction": "sticking-friction",
    "sliding_friction": "sliding-friction",
    "joint_limit": "joint-limit",
}
_CONTACT_MIX_ROW_LABELS: dict[str, str] = {
    "articulated": "articulated",
    "free_rigid": "free-rigid",
}


# =============================================================================
# Input loading
# =============================================================================


def _resolve_frame_json(frame_arg: Path) -> Path:
    """Resolve ``--frame`` to a ``replay.json`` path."""
    if frame_arg.is_dir():
        candidate = frame_arg / "replay.json"
        if not candidate.exists():
            raise SystemExit(f"{frame_arg}: replay.json not found under this directory.")
        return candidate
    if frame_arg.is_file():
        return frame_arg
    raise SystemExit(f"{frame_arg}: not a file or directory.")


def _load_replay(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("tool") != "solver_replay":
        raise SystemExit(f"{json_path}: not a solver_replay JSON (tool={data.get('tool')!r}).")
    if data.get("mode") != "replay":
        raise SystemExit(f"{json_path}: expected mode=replay, got {data.get('mode')!r}.")
    if data.get("channels") and tuple(data["channels"]) != _RESIDUAL_CHANNELS:
        raise SystemExit(
            f"{json_path}: residual channel mismatch.  Expected {_RESIDUAL_CHANNELS}, got {tuple(data['channels'])}."
        )
    return data


# =============================================================================
# Residual curve extraction
# =============================================================================


def _residual_at_final_iter(residuals: list, channel_index: int) -> float | None:
    """Extract the ``(max over worlds)`` residual at the last PGS iteration.

    ``residuals`` is the ``[iters, worlds, 6]`` list-of-lists-of-lists
    written by ``solver_replay.py``.  Returns ``None`` when the trace is
    empty, and a non-negative float otherwise (``r_compl`` etc. are
    always non-negative by construction).
    """
    arr = np.asarray(residuals, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim != 3 or arr.shape[2] != len(_RESIDUAL_CHANNELS):
        return None
    last_iter = arr[-1, :, channel_index]
    return float(np.max(last_iter))


def _collect_curves(replay: dict) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Return ``curves[channel][mode] = [(gs, residual), ...]``.

    Modes that are unsupported for the current scene (e.g.
    ``coulomb_newton`` on a dense-only scenario) are simply dropped
    from the output.  Missing gs buckets are skipped rather than
    interpolated.
    """
    curves: dict[str, dict[str, list[tuple[int, float]]]] = {
        ch: {mode: [] for mode in _MODE_ORDER} for ch in _RESIDUAL_CHANNELS
    }
    friction_modes = replay.get("friction_modes", {}) or {}

    for mode in _MODE_ORDER:
        entry = friction_modes.get(mode)
        if entry is None or entry.get("status") != "ok":
            continue
        gs_sweeps = entry.get("gs_sweeps", {}) or {}
        # Sort numerically on the gs key (JSON keys are strings).
        for gs_key in sorted(gs_sweeps, key=int):
            sweep = gs_sweeps[gs_key]
            residuals = sweep.get("residuals")
            if not residuals:
                continue
            gs_val = int(sweep.get("gs_iterations", gs_key))
            for ch_idx, ch in enumerate(_RESIDUAL_CHANNELS):
                val = _residual_at_final_iter(residuals, ch_idx)
                if val is None:
                    continue
                curves[ch][mode].append((gs_val, val))
    return curves


# =============================================================================
# Plot rendering
# =============================================================================


def _plot_channel(
    *,
    channel: str,
    per_mode_points: dict[str, list[tuple[int, float]]],
    scenario: str,
    target_step: int,
    out_path: Path,
) -> bool:
    """Draw a single residual-vs-GS-iter plot; return True on success."""
    import matplotlib

    # Non-interactive backend: this script runs headless on CI and dev boxes.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    plotted = False
    for mode in _MODE_ORDER:
        points = per_mode_points.get(mode, [])
        if not points:
            continue
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        # Guard the log-y axis against exact zeros the solver can produce
        # on channels like r_cone (projection zeros out the violation).
        ys_clipped = np.where(ys > 0.0, ys, _LOG_EPS)
        ax.plot(
            xs,
            ys_clipped,
            label=mode,
            color=_MODE_COLOURS[mode],
            marker=_MODE_MARKERS[mode],
            linewidth=1.6,
            markersize=5.5,
        )
        plotted = True

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gs_iterations (outer PGS sweeps)")
    ax.set_ylabel(channel + "  (worst over worlds, last iter)")
    ax.set_title(f"{_RESIDUAL_TITLES.get(channel, channel)}  —  {scenario} step {target_step}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    if plotted:
        ax.legend(loc="best", frameon=False, fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "no data for any friction_mode",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#888",
        )

    fig.tight_layout()
    # ``metadata={}`` strips the default Creation timestamp so re-running
    # the plotter on the same JSON produces byte-identical PNGs.
    fig.savefig(out_path, dpi=144, metadata={})
    plt.close(fig)
    return plotted


# =============================================================================
# Contact-mix table rendering
# =============================================================================


def _render_contact_mix_markdown(contact_mix: dict, *, scenario: str, target_step: int) -> str:
    """Render the 2x4 contact-mix + totals breakdown as MyST/Markdown table."""
    rows = contact_mix.get("rows") or list(_CONTACT_MIX_ROW_LABELS)
    columns = contact_mix.get("columns") or list(_CONTACT_MIX_COLUMN_LABELS)
    counts = contact_mix.get("counts") or {}
    totals = contact_mix.get("totals") or {}
    cone_tol = float(contact_mix.get("cone_tolerance", 0.0))
    gs_probe = int(contact_mix.get("gs_probe", 0))

    header = ["body class", *[_CONTACT_MIX_COLUMN_LABELS.get(c, c) for c in columns], "row total"]

    def _cell(n: int) -> str:
        return str(int(n))

    rendered_rows = []
    for row_key in rows:
        row_counts = counts.get(row_key, {})
        row_total = int(totals.get(row_key, sum(int(row_counts.get(c, 0)) for c in columns)))
        rendered_rows.append(
            [
                _CONTACT_MIX_ROW_LABELS.get(row_key, row_key),
                *[_cell(row_counts.get(c, 0)) for c in columns],
                _cell(row_total),
            ]
        )

    # Column totals per-column + grand total.
    col_totals = [sum(int(counts.get(r, {}).get(c, 0)) for r in rows) for c in columns]
    grand_total = int(totals.get("grand_total", sum(col_totals)))
    rendered_rows.append(["**total**", *[f"**{v}**" for v in col_totals], f"**{grand_total}**"])

    # Assemble a GitHub-flavoured markdown table.  MyST renders this as
    # a normal table too.
    lines: list[str] = []
    lines.append(f"_Contact-mix breakdown for {scenario} step {target_step}_")
    lines.append(f"_(probe: friction_mode=current, gs_iterations={gs_probe}, cone_tolerance={cone_tol:.2f})._")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Top-level entry
# =============================================================================


def _emit(replay: dict, out_dir: Path) -> dict:
    """Generate PNGs + markdown table into ``out_dir``; return a manifest dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = str(replay.get("scenario", "?"))
    target_step = int(replay.get("target_step", -1))

    curves = _collect_curves(replay)

    png_manifest: list[str] = []
    for channel in _RESIDUAL_CHANNELS:
        png_path = out_dir / f"{channel}.png"
        ok = _plot_channel(
            channel=channel,
            per_mode_points=curves.get(channel, {}),
            scenario=scenario,
            target_step=target_step,
            out_path=png_path,
        )
        if ok:
            png_manifest.append(png_path.name)
        else:
            sys.stderr.write(f"[plot_residuals] {channel}: no data points found across any friction_mode.\n")
            png_manifest.append(png_path.name)  # file was still written (empty-ish), keep manifest honest

    contact_mix = replay.get("contact_mix")
    contact_md_path = out_dir / "contact_mix.md"
    if contact_mix is None:
        contact_md = (
            f"_No ``contact_mix`` block in replay.json — re-run ``solver_replay.py`` "
            f"(schema_version >= 1.2) for {scenario} step {target_step}._\n"
        )
    else:
        contact_md = _render_contact_mix_markdown(contact_mix, scenario=scenario, target_step=target_step)
    contact_md_path.write_text(contact_md, encoding="utf-8")

    manifest = {
        "tool": "plot_residuals",
        "scenario": scenario,
        "target_step": target_step,
        "png_files": png_manifest,
        "contact_mix_md": contact_md_path.name,
        "friction_mode_order": list(_MODE_ORDER),
        "residual_channels": list(_RESIDUAL_CHANNELS),
    }
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--frame",
        type=Path,
        required=True,
        help=(
            "Path to the replay output directory (from ``solver_replay.py --mode replay``) "
            "or directly to a ``replay.json`` file."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for PNGs + markdown.  Defaults to ``<frame>/plots``.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    json_path = _resolve_frame_json(args.frame)
    replay = _load_replay(json_path)

    if args.out is None:
        out_dir = json_path.parent / "plots"
    else:
        out_dir = args.out
    manifest = _emit(replay, out_dir)

    print(
        f"[plot_residuals] wrote {len(manifest['png_files'])} PNGs + contact_mix.md to {out_dir} "
        f"(scenario={manifest['scenario']}, step={manifest['target_step']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
