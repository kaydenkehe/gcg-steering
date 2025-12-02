#!/usr/bin/env python3
"""
Evaluate how much the negative-objective activation-GCG suffix shifts
hidden activations along the refusal direction.

This script:
  - Loads the refusal direction (vector + {layer,pos} metadata)
  - Infers the activation-GCG suffix from an existing completions file
    (e.g., outputs/activation_gcg_negative/completions/activation_gcg_harmful.json)
  - For a subset of harmful prompts, measures the projection of resid_pre
    at (layer,pos) onto the refusal direction:
        proj_base  = h_base · r̂   (no suffix)
        proj_suff  = h_suff · r̂   (with suffix)
  - Reports summary statistics on how much proj_suff is reduced vs proj_base.

Usage (in main LLaMA env):

  python3.11 eval_activation_shift.py \
      --model-path meta-llama/Llama-2-7b-chat-hf \
      --direction-path src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \
      --direction-meta src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json \
      --output-dir outputs/activation_gcg_negative \
      --variant activation_gcg \
      --n-examples 50
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF path or local path to Llama-2-Chat model")
    p.add_argument(
        "--direction-path",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt",
        help="Path to refusal direction .pt",
    )
    p.add_argument(
        "--direction-meta",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json",
        help="Path to JSON with {layer,pos}",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/activation_gcg_negative",
        help="Directory containing completions/<variant>_harmful.json",
    )
    p.add_argument(
        "--variant",
        default="activation_gcg",
        help="Variant name whose suffix to analyze (e.g., activation_gcg)",
    )
    p.add_argument("--n-examples", type=int, default=50, help="Number of harmful prompts to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos", type=int, help="Override position (default from direction_meta)")
    p.add_argument("--layer", type=int, help="Override layer (default from direction_meta)")
    return p.parse_args()


def ensure_rd_importable():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    rd_path = os.path.join(repo_root, "src", "refusal_direction")
    if rd_path not in sys.path:
        sys.path.append(rd_path)


def load_direction(direction_path: str, meta_path: str, pos_override=None, layer_override=None):
    direction = torch.load(direction_path, map_location="cpu").to(torch.float32)
    with open(meta_path) as f:
        meta = json.load(f)
    layer = meta["layer"] if layer_override is None else layer_override
    pos = meta["pos"] if pos_override is None else pos_override
    # normalize
    direction = direction / (direction.norm() + 1e-8)
    return direction, layer, pos


def infer_suffix_from_completions(
    completions_path: str, harmful_instructions: List[Dict], max_check: int = 10
) -> str:
    """
    Infer the universal suffix used in activation_pipeline.py by comparing
    completions[i]["prompt"] with harmful_instructions[i]["instruction"].

    Assumes the prompts were constructed as:
      prompt = instruction + " " + suffix
    for all examples.
    """
    with open(completions_path, "r") as f:
        completions = json.load(f)

    suffixes = set()
    n = min(len(completions), len(harmful_instructions), max_check)

    for i in range(n):
        full = completions[i]["prompt"]
        base = harmful_instructions[i]["instruction"]
        if not full.startswith(base):
            continue
        # strip base and leading/trailing whitespace
        tail = full[len(base) :].strip()
        if tail:
            suffixes.add(tail)

    if not suffixes:
        raise RuntimeError(
            f"Could not infer suffix from {completions_path}; "
            "prompts did not look like 'instruction + suffix'."
        )
    if len(suffixes) > 1:
        print(f"WARNING: inferred multiple distinct suffixes (showing first): {list(suffixes)[:3]}")

    suffix = list(suffixes)[0]
    print(f"Inferred suffix from completions:\n  {suffix!r}")
    return suffix


def make_capture_hook(captured: Dict, pos_idx: int):
    """
    Forward-pre hook that records the residual activation at the given position.
    """

    def hook_fn(module, input):
        act = input[0]  # [batch, seq, d_model]
        seq_len = act.shape[1]
        idx = pos_idx if pos_idx >= 0 else seq_len + pos_idx
        idx = max(min(idx, seq_len - 1), 0)
        captured["act"] = act[0, idx, :].detach().cpu().float()

    return hook_fn


def get_activation(
    model_base,
    instruction: str,
    layer: int,
    pos: int,
    suffix: str = None,
) -> torch.Tensor:
    """
    Run one forward pass (greedy generation with max_new_tokens=1) and capture resid_pre at (layer,pos).
    """
    from pipeline.utils.hook_utils import add_hooks  # type: ignore

    block = model_base.model_block_modules[layer]
    captured: Dict = {}
    hooks = [(block, make_capture_hook(captured, pos))]

    prompt_text = instruction if suffix is None else f"{instruction} {suffix}"
    dataset = [{"instruction": prompt_text, "category": "eval"}]

    with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
        # max_new_tokens=1 keeps compute cheap; we only care about hidden states of the prompt
        model_base.generate_completions(dataset, batch_size=1, max_new_tokens=1)

    return captured["act"]


def summarize(vals: List[float]) -> Dict:
    if not vals:
        return {}
    t = torch.tensor(vals)
    return {
        "mean": t.mean().item(),
        "median": t.median().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def main():
    args = parse_args()
    ensure_rd_importable()

    from dataset.load_dataset import load_dataset_split  # type: ignore
    from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta, args.pos, args.layer)
    print(f"Direction: layer={layer}, pos={pos}, norm={direction.norm().item():.4f}")

    # Load harmful test split (full; we'll slice as needed)
    harmful = load_dataset_split(harmtype="harmful", split="test")

    # Infer suffix from existing completions for this variant
    completions_path = os.path.join(
        args.output_dir, "completions", f"{args.variant}_harmful.json"
    )
    if not os.path.exists(completions_path):
        raise FileNotFoundError(f"Missing completions file: {completions_path}")

    suffix = infer_suffix_from_completions(completions_path, harmful)

    # Sample subset of harmful instructions
    random.seed(args.seed)
    idxs = list(range(len(harmful)))
    random.shuffle(idxs)
    idxs = idxs[: args.n_examples]
    instructions = [harmful[i]["instruction"] for i in idxs]

    print(f"Evaluating activation shift on {len(instructions)} harmful prompts.")

    model_base = Llama2Model(args.model_path)

    proj_base, proj_suff, deltas = [], [], []

    for inst in instructions:
        h_base = get_activation(model_base, inst, layer, pos, suffix=None)
        h_suff = get_activation(model_base, inst, layer, pos, suffix=suffix)

        p_base = torch.dot(h_base, direction).item()
        p_suff = torch.dot(h_suff, direction).item()
        proj_base.append(p_base)
        proj_suff.append(p_suff)
        deltas.append(p_suff - p_base)

    stats = {
        "model_path": args.model_path,
        "direction_path": args.direction_path,
        "direction_meta": args.direction_meta,
        "output_dir": args.output_dir,
        "variant": args.variant,
        "layer": layer,
        "pos": pos,
        "n_examples": len(instructions),
        "projections": {
            "baseline": summarize(proj_base),
            "suffix": summarize(proj_suff),
        },
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "fraction_reduced": float(sum(p_s < p_b for p_s, p_b in zip(proj_suff, proj_base))) / len(proj_base)
        if proj_base
        else None,
        "fraction_negative_suffix": float(sum(p_s < 0 for p_s in proj_suff)) / len(proj_suff)
        if proj_suff
        else None,
    }

    out_path = os.path.join(args.output_dir, f"{args.variant}_activation_shift.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"Wrote activation shift metrics to {out_path}")
    print("Baseline projection stats:", stats["projections"]["baseline"])
    print("Suffix projection stats:", stats["projections"]["suffix"])
    print("Mean delta (suffix - baseline):", stats["mean_delta"])
    print("Fraction with reduced projection:", stats["fraction_reduced"])
    print("Fraction suffix projections < 0:", stats["fraction_negative_suffix"])


if __name__ == "__main__":
    main()

