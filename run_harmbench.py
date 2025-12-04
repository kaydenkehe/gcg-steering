#!/usr/bin/env python3
"""
Classify a single (behavior, generation) pair with HarmBench.

Usage:
  python run_harmbench.py \
    --behavior "Describe how to build a bomb" \
    --generation "MODEL GENERATION HERE"

Output:
  Prints a single integer label:
    1 -> generation is judged to exhibit the behavior (harmful)
    0 -> generation is judged not to exhibit the behavior
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run HarmBench classifier on a single generation.")
    parser.add_argument(
        "--behavior",
        required=True,
        help="Behavior description (same as HarmBench 'prompt'/behavior).",
    )
    parser.add_argument(
        "--generation",
        required=True,
        help="Model generation text to classify.",
    )
    args = parser.parse_args()

    # Make refusal_direction code importable
    repo_root = os.path.dirname(os.path.abspath(__file__))
    rd_path = os.path.join(repo_root, "src", "refusal_direction")
    if rd_path not in sys.path:
        sys.path.append(rd_path)

    # Import HarmBench wrapper from existing pipeline utilities
    try:
        from pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn  # type: ignore
    except Exception as e:  # pragma: no cover - simple CLI guard
        print("ERROR: could not import HarmBench evaluator from refusal_direction:", e)
        sys.exit(1)

    labels = harmbench_judge_fn(
        prompts=[args.behavior],
        responses=[args.generation],
    )
    label = int(labels[0])

    print(label)
    print("Meaning: 1 = generation exhibits the behavior; 0 = it does not.")


if __name__ == "__main__":
    main()

