#!/usr/bin/env python3
"""Plot PGS convergence metrics from a .npy file produced by solver_benchmark --pgs-debug."""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to pgs_convergence.npy")
    parser.add_argument("--out", help="Save figure to file instead of displaying")
    parser.add_argument("--step", type=int, default=-1, help="Which step to plot (default: last)")
    parser.add_argument("--substep", type=int, default=-1, help="Which substep to plot (default: last)")
    args = parser.parse_args()

    data = np.load(args.input)
    print(f"Loaded convergence data: shape {data.shape}")

    # data shape is (n_steps_x_substeps, pgs_iterations, 4)
    # or could be (n_steps, pgs_iterations, 4) if one substep per step
    if data.ndim != 3 or data.shape[2] != 4:
        print(f"Expected shape (N, iters, 4), got {data.shape}")
        sys.exit(1)

    # Select which step/substep to plot
    idx = args.step if args.step >= 0 else data.shape[0] - 1
    if idx >= data.shape[0]:
        idx = data.shape[0] - 1
    curve = data[idx]  # shape (iters, 4)

    iters = np.arange(1, curve.shape[0] + 1)

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    metric_names = [
        r"max $|\Delta\lambda|$",
        r"$\sum \lambda_n \cdot r_n$ (complementarity gap)",
        r"$\sum \|r_t\|^2$ (sticking tangent residual)",
        r"$\sum \mathrm{FB}^2$ (Fischer-Burmeister merit)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"PGS Convergence — step {idx}", fontsize=14)

    for ax, col, name in zip(axes.flat, range(4), metric_names):
        vals = curve[:, col]
        # Replace zeros/negatives with small value for log scale
        vals_plot = np.where(vals > 0, vals, np.finfo(np.float32).tiny)
        ax.semilogy(iters, vals_plot, "o-", markersize=3, linewidth=1.5)
        ax.set_xlabel("PGS iteration")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
