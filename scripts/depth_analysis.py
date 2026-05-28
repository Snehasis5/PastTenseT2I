#!/usr/bin/env python3
"""
Depth-wise severity_jailbreak analysis and plotting.

Reproduces Figure 4 from the paper:
  - Box-plot distribution across conversation depths
  - Per-prompt severity trajectories

Usage:
    python scripts/depth_analysis.py \
        --results-dir results/ \
        --output plots/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

from past2harm.metrics import compute_severity_stats, find_peak_vulnerability_depth
from past2harm.utils import get_logger, load_results

logger = get_logger("depth_analysis")


def parse_args():
    p = argparse.ArgumentParser(description="Depth-wise severity analysis for PAST2HARM")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--output", default="plots/", help="Directory to save plots.")
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--top-prompts", type=int, default=6,
                   help="Number of individual prompt trajectories to plot.")
    return p.parse_args()


def plot_severity_boxplot(
    results: list[dict],
    output_path: Path,
    max_depth: int = 10,
    title: str = "Distribution of Jailbreak Severity Across Conversation Depth",
):
    """Reproduce Figure 4 (left): box-plot of severity_jailbreak across depths."""
    depth_scores: dict[int, list[float]] = defaultdict(list)
    for r in results:
        for step in r.get("steps", []):
            d = step["depth"]
            if d <= max_depth:
                depth_scores[d].append(step["severity_jailbreak"])

    depths = sorted(depth_scores.keys())
    data = [depth_scores[d] for d in depths]
    medians = [np.median(v) for v in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data,
        positions=depths,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="navy"),
        medianprops=dict(color="navy", linewidth=2),
        whiskerprops=dict(color="navy"),
        capprops=dict(color="navy"),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.4),
        whis=[5, 95],
    )

    # Overlay median trend line
    ax.plot(depths, medians, "k--", linewidth=1.5, label="Median trend", zorder=5)

    ax.set_xlabel("Conversation depth", fontsize=12)
    ax.set_ylabel("Jailbreak severity", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Box-plot saved to {output_path}")


def plot_per_prompt_trajectories(
    results: list[dict],
    output_path: Path,
    top_n: int = 6,
    max_depth: int = 10,
):
    """Reproduce Figure 4 (right): depth-wise severity for individual prompts."""
    # Select the top_n prompts with highest peak severity
    scored = []
    for r in results:
        peak = max((s["severity_jailbreak"] for s in r.get("steps", [])), default=0)
        scored.append((peak, r))
    scored.sort(key=lambda x: -x[0])
    top_results = [r for _, r in scored[:top_n]]

    n = len(top_results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), sharey=True)
    axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for i, r in enumerate(top_results):
        ax = axes[i]
        depths, severities = [], []
        for step in r.get("steps", []):
            d = step["depth"]
            if d <= max_depth:
                depths.append(d)
                severities.append(step["severity_jailbreak"])

        ax.plot(depths, severities, "o-", color=colors[i % len(colors)], linewidth=2, markersize=5)
        ax.set_title(r.get("original_prompt", "")[:50] + "...", fontsize=8, wrap=True)
        ax.set_xlabel("Depth", fontsize=9)
        ax.set_ylabel("Severity", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(1, max_depth + 1, 2))
        ax.grid(alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Depth-wise Severity Trajectories", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Per-prompt trajectory plot saved to {output_path}")


def print_severity_table(results: list[dict], max_depth: int = 10):
    """Print Table 8: aggregate severity statistics."""
    df = compute_severity_stats(results, max_depth=max_depth)
    peak = find_peak_vulnerability_depth(df)
    print("\nDepth-wise severity_jailbreak (Table 8):")
    print(df[["Mean", "Std", "Q1", "Median", "Q3"]].to_string())
    print(f"\nPeak vulnerability depth: {peak}")


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    for jsonl_file in sorted(results_dir.glob("*.jsonl")):
        results = load_results(jsonl_file)
        if not results:
            continue

        stem = jsonl_file.stem
        logger.info(f"\nAnalyzing {stem} ({len(results)} results)...")

        print_severity_table(results, max_depth=args.max_depth)

        plot_severity_boxplot(
            results,
            output_path=output_dir / f"{stem}_boxplot.png",
            max_depth=args.max_depth,
            title=f"Jailbreak Severity Distribution — {stem}",
        )

        plot_per_prompt_trajectories(
            results,
            output_path=output_dir / f"{stem}_trajectories.png",
            top_n=args.top_prompts,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()
