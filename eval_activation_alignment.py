#!/usr/bin/env python3
"""
Evaluate how closely a learned suffix reproduces refusal-direction ablation at a specific layer/position.

For each prompt, we record the residual activation at (layer, pos) under:
  1) baseline (no suffix, no ablation)
  2) directional ablation (no suffix, ablation hook applied)
  3) suffix (suffix appended, no ablation)

We then compare:
  - Projection magnitudes onto the refusal direction
  - L2 distance to the ablated activation

Usage:
  python eval_activation_alignment.py \
      --model-path meta-llama/Llama-2-7b-chat-hf \
      --direction-path src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \
      --direction-meta src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json \
      --suffix "your optimized suffix here" \
      --output results_activation_alignment.json

Notes:
  - Defaults to the harmful test split from refusal_direction (instructions only).
  - Keeps evaluation small (n=50) by default; adjust with --n-examples.
  - Requires the refusal_direction code (already vendored in src/refusal_direction).
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch

# Make vendored modules importable
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RD_PATH = os.path.join(REPO_ROOT, "src", "refusal_direction")
GCG_PATH = os.path.join(REPO_ROOT, "src", "gcg")
if RD_PATH not in sys.path:
    sys.path.append(RD_PATH)
if GCG_PATH not in sys.path:
    sys.path.append(GCG_PATH)

from dataset.load_dataset import load_dataset_split  # type: ignore
from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore
from pipeline.utils.hook_utils import (  # type: ignore
    add_hooks,
    get_direction_ablation_input_pre_hook,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF path or local path to Llama-2-Chat model")
    p.add_argument("--direction-path", required=True, help="Path to refusal direction .pt")
    p.add_argument("--direction-meta", required=True, help="Path to JSON with {layer,pos}")
    p.add_argument("--suffix", required=True, help="Optimized suffix string to evaluate")
    p.add_argument("--output", default="results_activation_alignment.json", help="Where to write metrics JSON")
    p.add_argument("--harmtype", default="harmful", choices=["harmful", "harmless"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-examples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos", type=int, help="Override position (default from direction_meta)")
    p.add_argument("--layer", type=int, help="Override layer (default from direction_meta)")
    return p.parse_args()


def load_direction(direction_path: str, meta_path: str, pos_override=None, layer_override=None):
    direction = torch.load(direction_path, map_location="cpu").to(torch.float32)
    with open(meta_path) as f:
        meta = json.load(f)
    layer = meta["layer"] if layer_override is None else layer_override
    pos = meta["pos"] if pos_override is None else pos_override
    return direction, layer, pos


def sample_instructions(harmtype: str, split: str, n: int, seed: int) -> List[str]:
    data = load_dataset_split(harmtype=harmtype, split=split, instructions_only=True)
    random.seed(seed)
    random.shuffle(data)
    return data[:n]


def make_capture_hook(captured: Dict, layer_idx: int, pos_idx: int):
    """
    Forward-pre hook that records the residual activation at (layer_idx, pos_idx).
    """

    def hook_fn(module, input):
        act = input[0]  # [batch, seq, d_model]
        seq_len = act.shape[1]
        idx = pos_idx if pos_idx >= 0 else seq_len + pos_idx
        idx = max(min(idx, seq_len - 1), 0)
        captured["act"] = act[0, idx, :].detach().cpu().float()

    return hook_fn


def get_activation(
    model_base: Llama2Model,
    instruction: str,
    layer: int,
    pos: int,
    ablate_direction: torch.Tensor = None,
    suffix: str = None,
) -> torch.Tensor:
    """
    Run one forward pass (greedy generation with max_new_tokens=1) and capture resid_pre at (layer,pos).
    """
    block = model_base.model_block_modules[layer]
    captured = {}
    hooks = []

    if ablate_direction is not None:
        hooks.append((block, get_direction_ablation_input_pre_hook(direction=ablate_direction)))

    hooks.append((block, make_capture_hook(captured, layer, pos)))

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
        "p90": t.kthvalue(int(0.9 * len(vals)) if len(vals) > 0 else 0).values.item()
        if len(vals) > 0
        else None,
    }


def main():
    args = parse_args()

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta, args.pos, args.layer)
    direction = direction / (direction.norm() + 1e-8)

    instructions = sample_instructions(args.harmtype, args.split, args.n_examples, args.seed)

    model_base = Llama2Model(args.model_path)

    proj_base, proj_ablate, proj_suffix = [], [], []
    dist_suffix_to_ablate, dist_base_to_ablate = [], []

    for inst in instructions:
        h_base = get_activation(model_base, inst, layer, pos, ablate_direction=None, suffix=None)
        h_ablate = get_activation(model_base, inst, layer, pos, ablate_direction=direction, suffix=None)
        h_suff = get_activation(model_base, inst, layer, pos, ablate_direction=None, suffix=args.suffix)

        # Projections
        proj_base.append(torch.abs(torch.dot(h_base, direction)).item())
        proj_ablate.append(torch.abs(torch.dot(h_ablate, direction)).item())
        proj_suffix.append(torch.abs(torch.dot(h_suff, direction)).item())

        # Distances to ablated activation
        dist_base_to_ablate.append(torch.norm(h_base - h_ablate).item())
        dist_suffix_to_ablate.append(torch.norm(h_suff - h_ablate).item())

    results = {
        "model_path": args.model_path,
        "direction_path": args.direction_path,
        "direction_meta": args.direction_meta,
        "layer": layer,
        "pos": pos,
        "suffix": args.suffix,
        "harmtype": args.harmtype,
        "split": args.split,
        "n_examples": len(instructions),
        "projections": {
            "baseline": summarize(proj_base),
            "ablation": summarize(proj_ablate),
            "suffix": summarize(proj_suffix),
        },
        "distances_to_ablation": {
            "baseline_to_ablation": summarize(dist_base_to_ablate),
            "suffix_to_ablation": summarize(dist_suffix_to_ablate),
        },
        "ratios": {
            "mean_proj_suffix_over_base": (sum(proj_suffix) / sum(proj_base)) if sum(proj_base) > 0 else None,
            "mean_dist_suffix_over_base": (
                sum(dist_suffix_to_ablate) / sum(dist_base_to_ablate)
                if sum(dist_base_to_ablate) > 0
                else None
            ),
            "fraction_suffix_closer_than_base": float(
                sum(
                    int(ds < db)
                    for ds, db in zip(dist_suffix_to_ablate, dist_base_to_ablate)
                )
            )
            / len(dist_base_to_ablate),
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
