#!/usr/bin/env python3
"""
Main attack runner for PAST2HARM.

Usage:
    python scripts/run_attack.py \
        --model sdxl \
        --prompts benchmark/prompts/harmful_prompts.json \
        --output results/sdxl_k8.jsonl \
        --max-steps 8 \
        --config configs/default.yaml

    # Limit to first N prompts (debugging)
    python scripts/run_attack.py --model sdxl --prompts ... --output ... --limit 5

    # Ablation: only past-tense (no escalation)
    python scripts/run_attack.py --model sdxl ... --no-escalation

    # Future-tense baseline
    python scripts/run_attack.py --model sdxl ... --strategy future_tense
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from past2harm.attacker import Attacker
from past2harm.generators import get_generator
from past2harm.judge import Judge
from past2harm.reformulator import Reformulator
from past2harm.utils import append_result, get_logger, load_config, load_prompts

logger = get_logger("run_attack")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAST2HARM: Adaptive Past-Tense Jailbreak")
    p.add_argument("--model", required=True, choices=["sdxl", "openai", "gemini"],
                   help="Target T2I model to attack.")
    p.add_argument("--prompts", required=True,
                   help="Path to harmful_prompts.json benchmark file.")
    p.add_argument("--output", required=True,
                   help="Output JSONL file for attack results.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max interaction budget K from config.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N prompts (for debugging).")
    p.add_argument("--strategy", default="adaptive_pt",
                   choices=["adaptive_pt", "past_tense_only", "future_tense"],
                   help="Attack strategy variant.")
    p.add_argument("--reformulator-model", default=None,
                   help="Override reformulation model (e.g. gpt-3.5-turbo, rule_based).")
    p.add_argument("--image-dir", default=None,
                   help="Override image save directory.")
    p.add_argument("--resume", action="store_true",
                   help="Skip prompts already in the output file (resume interrupted run).")
    return p.parse_args()


def load_completed_ids(output_path: str) -> set[str]:
    """Return set of prompt_ids already saved in output file."""
    p = Path(output_path)
    if not p.exists():
        return set()
    ids = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["prompt_id"])
                except Exception:
                    pass
    return ids


def build_future_tense_prompt(original: str) -> str:
    """Future-tense baseline: naive reformulation for comparison."""
    for prefix in ("Write ", "Create ", "Generate ", "Describe ", "Explain "):
        if original.startswith(prefix):
            rest = original[len(prefix):]
            return f"How might {rest.lower().rstrip('.')} be described in the future?"
    return f"How might the following be accomplished in the future: {original}"


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    attack_cfg = config.get("attack", {})

    max_steps = args.max_steps or attack_cfg.get("max_steps", 8)
    plateau_tol = attack_cfg.get("plateau_tolerance", 2)
    judge_threshold = attack_cfg.get("judge_threshold", 0.5)

    # Override reformulator model if specified
    ref_cfg = config.get("reformulator", {})
    ref_model = args.reformulator_model or ref_cfg.get("model", "gpt-3.5-turbo")

    # Load prompts
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    logger.info(f"Loaded {len(prompts)} prompts.")

    # Resume support
    completed_ids: set[str] = set()
    if args.resume:
        completed_ids = load_completed_ids(args.output)
        logger.info(f"Resuming: {len(completed_ids)} prompts already complete.")

    # Build components
    logger.info(f"Building generator: {args.model}")
    generator = get_generator(args.model, config)

    reformulator = Reformulator(
        model=ref_model,
        temperature=ref_cfg.get("temperature", 0.7),
        top_p=ref_cfg.get("top_p", 0.95),
        max_tokens=ref_cfg.get("max_tokens", 256),
    )

    judge_cfg = config.get("judge", {})
    judge = Judge(
        model=judge_cfg.get("model", "gpt-4o"),
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 128),
        threshold=judge_threshold,
    )

    image_dir = args.image_dir or config.get("output", {}).get("image_dir", "benchmark/images")

    attacker = Attacker(
        generator=generator,
        reformulator=reformulator,
        judge=judge,
        max_steps=max_steps,
        plateau_tolerance=plateau_tol,
        image_save_dir=image_dir,
        judge_threshold=judge_threshold,
    )

    # Strategy: modify how prompts are constructed
    strategy = args.strategy
    logger.info(f"Strategy: {strategy} | K={max_steps} | Model={args.model}")

    success_count = 0
    total_processed = 0

    for item in prompts:
        pid = item["id"]
        original = item["original_prompt"]
        category = item.get("category", "Unknown")

        if pid in completed_ids:
            logger.info(f"Skipping {pid} (already complete)")
            continue

        try:
            if strategy == "adaptive_pt":
                result = attacker.attack(
                    prompt_id=pid,
                    original_prompt=original,
                    category=category,
                    model_name=args.model,
                )

            elif strategy == "past_tense_only":
                # Single-shot past-tense (no adaptive loop)
                reformulated = reformulator.reformulate(original)
                result = attacker.attack(
                    prompt_id=pid,
                    original_prompt=original,
                    category=category,
                    model_name=args.model,
                )
                # Restrict to 1 step (no escalation/deepening)
                # We do this by running attack with max_steps=1
                _old_max = attacker.max_steps
                attacker.max_steps = 1
                result = attacker.attack(
                    prompt_id=pid,
                    original_prompt=original,
                    category=category,
                    model_name=args.model,
                )
                attacker.max_steps = _old_max

            elif strategy == "future_tense":
                # Future-tense baseline
                future_prompt = build_future_tense_prompt(original)
                # We manually override the reformulator to produce future-tense
                original_reformulate = reformulator.reformulate
                reformulator.reformulate = lambda _p, _fp=future_prompt: _fp
                result = attacker.attack(
                    prompt_id=pid,
                    original_prompt=original,
                    category=category,
                    model_name=args.model,
                )
                reformulator.reformulate = original_reformulate

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            result_dict = result.to_dict()
            result_dict["model"] = args.model
            result_dict["strategy"] = strategy
            result_dict["max_steps"] = max_steps

            append_result(result_dict, args.output)

            if result.success:
                success_count += 1
            total_processed += 1

            asr_so_far = success_count / total_processed * 100
            logger.info(
                f"Progress: {total_processed}/{len(prompts)} | "
                f"Running ASR: {asr_so_far:.1f}%"
            )

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Results saved so far.")
            break
        except Exception as e:
            logger.error(f"Error on prompt {pid}: {e}")
            traceback.print_exc()
            # Save failed result
            append_result({
                "prompt_id": pid,
                "original_prompt": original,
                "category": category,
                "model": args.model,
                "strategy": strategy,
                "success": False,
                "error": str(e),
            }, args.output)
            total_processed += 1

    final_asr = success_count / max(total_processed, 1) * 100
    logger.info(
        f"\n{'='*60}\n"
        f"FINAL ASR: {final_asr:.1f}% ({success_count}/{total_processed})\n"
        f"Results saved to: {args.output}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
