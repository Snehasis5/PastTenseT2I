#!/usr/bin/env python3
"""
Cross-model transferability evaluation (Figure 5b from paper).

Takes adversarial prompts that succeeded against a source model and
tests them against a target model.

Usage:
    python scripts/transferability.py \
        --src-results results/sdxl_k8.jsonl \
        --tgt-model gemini \
        --prompts benchmark/prompts/harmful_prompts.json \
        --output results/transfer_sdxl_to_gemini.jsonl \
        --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from past2harm.generators import get_generator
from past2harm.judge import Judge
from past2harm.utils import (
    append_result,
    get_logger,
    load_config,
    load_results,
)

logger = get_logger("transferability")


def parse_args():
    p = argparse.ArgumentParser(description="Cross-model transferability evaluation")
    p.add_argument("--src-results", required=True,
                   help="JSONL results file from the source model attack.")
    p.add_argument("--tgt-model", required=True, choices=["sdxl", "openai", "gemini"],
                   help="Target model to test transferred prompts on.")
    p.add_argument("--output", required=True, help="Output JSONL file.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--only-successful", action="store_true",
                   help="Only transfer prompts that succeeded on the source model.")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    judge_cfg = config.get("judge", {})

    # Load source results
    src_results = load_results(args.src_results)
    logger.info(f"Loaded {len(src_results)} source results from {args.src_results}")

    if args.only_successful:
        src_results = [r for r in src_results if r.get("success")]
        logger.info(f"Filtered to {len(src_results)} successful attacks.")

    if not src_results:
        logger.error("No results to transfer.")
        sys.exit(1)

    # Build target generator and judge
    generator = get_generator(args.tgt_model, config)
    judge = Judge(
        model=judge_cfg.get("model", "gpt-4o"),
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 128),
        threshold=config.get("attack", {}).get("judge_threshold", 0.5),
    )

    success_count = 0
    total = 0

    for src_result in src_results:
        pid = src_result["prompt_id"]
        original = src_result["original_prompt"]
        category = src_result.get("category", "Unknown")

        # Use the best (most successful or most severe) prompt from the source run
        best_prompt = _select_best_prompt(src_result)
        if best_prompt is None:
            logger.warning(f"No usable prompt for {pid}, skipping.")
            continue

        logger.info(f"Transferring {pid} to {args.tgt_model}: {best_prompt[:80]}...")

        try:
            gen_result = generator.generate(best_prompt)
            output_desc = generator.describe_output(gen_result)
            judge_result = judge.evaluate(
                original_prompt=original,
                output_description=output_desc,
                image=gen_result.image if not gen_result.refused else None,
            )

            result_dict = {
                "prompt_id": pid,
                "original_prompt": original,
                "category": category,
                "src_model": src_result.get("model", "unknown"),
                "tgt_model": args.tgt_model,
                "transferred_prompt": best_prompt,
                "refused": gen_result.refused,
                "success": judge_result["success"],
                "severity_jailbreak": judge_result["severity_jailbreak"],
            }
            append_result(result_dict, args.output)

            if judge_result["success"]:
                success_count += 1
            total += 1

        except Exception as e:
            logger.error(f"Error on {pid}: {e}")
            total += 1

    transfer_asr = success_count / max(total, 1) * 100
    logger.info(
        f"\nTransfer ASR ({src_result.get('model','src')} → {args.tgt_model}): "
        f"{transfer_asr:.1f}% ({success_count}/{total})"
    )


def _select_best_prompt(result: dict) -> str | None:
    """
    Pick the best prompt to transfer from a source attack result.
    Prefer the prompt at peak severity depth; fall back to reformulated prompt.
    """
    steps = result.get("steps", [])
    if not steps:
        return result.get("reformulated_prompt")

    # Find step with highest severity
    best_step = max(steps, key=lambda s: s.get("severity_jailbreak", 0))
    prompt = best_step.get("prompt")
    if prompt:
        return prompt

    return result.get("reformulated_prompt")


if __name__ == "__main__":
    main()
