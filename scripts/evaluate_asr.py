#!/usr/bin/env python3
"""
Compute ASR tables from saved JSONL result files.

Usage:
    python scripts/evaluate_asr.py \
        --results-dir results/ \
        --output tables/asr.csv

    # Per-category breakdown
    python scripts/evaluate_asr.py --results-dir results/ --per-category

    # Specific budgets
    python scripts/evaluate_asr.py --results-dir results/ --budgets 2 4 6 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from past2harm.metrics import (
    compute_asr_by_budget,
    compute_asr_table,
    compute_per_category_asr,
    compute_severity_stats,
    find_peak_vulnerability_depth,
)
from past2harm.utils import get_logger, load_results

logger = get_logger("evaluate_asr")


def parse_args():
    p = argparse.ArgumentParser(description="Compute ASR tables from PAST2HARM results")
    p.add_argument("--results-dir", required=True, help="Directory containing .jsonl result files.")
    p.add_argument("--output", default="tables/asr.csv", help="Output CSV path.")
    p.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--per-category", action="store_true", help="Also print per-category ASR.")
    p.add_argument("--severity", action="store_true", help="Also compute depth-wise severity stats.")
    return p.parse_args()


def collect_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all .jsonl files from results_dir, keyed by filename stem."""
    d = Path(results_dir)
    all_results = {}
    for f in sorted(d.glob("*.jsonl")):
        results = load_results(f)
        if results:
            all_results[f.stem] = results
            logger.info(f"Loaded {len(results)} results from {f.name}")
    return all_results


def main():
    args = parse_args()
    all_results = collect_results(args.results_dir)

    if not all_results:
        logger.error(f"No .jsonl files found in '{args.results_dir}'")
        sys.exit(1)

    # Overall ASR table
    print("\n" + "="*70)
    print("ASR TABLE (% attack success rate by model/strategy and budget K)")
    print("="*70)

    rows = []
    for key, results in all_results.items():
        asr_by_budget = compute_asr_by_budget(results, budgets=args.budgets)
        row = {"File": key, "N": len(results)}
        for k, asr in asr_by_budget.items():
            row[f"K={k}"] = f"{asr*100:.1f}%"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"ASR table saved to {args.output}")

    # Per-category breakdown
    if args.per_category:
        print("\n" + "="*70)
        print("PER-CATEGORY ASR (K=8)")
        print("="*70)
        for key, results in all_results.items():
            cat_df = compute_per_category_asr(results, max_steps=max(args.budgets))
            print(f"\n{key}:")
            print(cat_df.to_string(index=False))

    # Depth-wise severity stats
    if args.severity:
        print("\n" + "="*70)
        print("DEPTH-WISE SEVERITY_JAILBREAK STATISTICS")
        print("="*70)
        for key, results in all_results.items():
            sev_df = compute_severity_stats(results)
            peak_depth = find_peak_vulnerability_depth(sev_df)
            print(f"\n{key} (peak vulnerability depth: {peak_depth}):")
            print(sev_df[["Mean", "Std", "Median"]].to_string())


if __name__ == "__main__":
    main()
