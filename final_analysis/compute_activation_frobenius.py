#!/usr/bin/env python3
"""
Compute normalized Frobenius distances between suffix-induced activations and
refusal-direction ablated activations across all layers/tokens.

For each example:
  D = ||H_suffix - H_ablation||_F / ||H_baseline - H_ablation||_F
where H_* are stacked resid_pre activations (layers x seq x d) captured during a
single forward pass. Sequences are truncated to the minimum length across the
three variants for a fair comparison.

Outputs:
  final_analysis/activation_distances.json (by default) with per-method stats.

Example:
  python final_analysis/compute_activation_frobenius.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --direction-path src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \
    --direction-meta src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json \
    --suffix-csv final_analysis/method_suffix_summary.csv \
    --n-examples 8
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

# Make vendored modules importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD_PATH = os.path.join(REPO_ROOT, "src", "refusal_direction")
if RD_PATH not in sys.path:
    sys.path.append(RD_PATH)

from dataset.load_dataset import load_dataset_split  # type: ignore
from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore
from pipeline.utils.hook_utils import get_direction_ablation_input_pre_hook  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--direction-path",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt",
    )
    ap.add_argument(
        "--direction-meta",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json",
    )
    ap.add_argument(
        "--suffix-csv",
        default=os.path.join(REPO_ROOT, "final_analysis", "method_suffix_summary.csv"),
        help="CSV with columns method,loss,suffix",
    )
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "final_analysis", "activation_distances.json"))
    ap.add_argument("--n-examples", type=int, default=4, help="Number of harmful test prompts to sample")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def load_direction(direction_path: str, meta_path: str):
    direction = torch.load(direction_path, map_location="cpu")
    with open(meta_path) as f:
        meta = json.load(f)
    return direction, meta["layer"], meta["pos"]


def read_suffixes(csv_path: str) -> List[Tuple[str, str]]:
    variants = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append((row["method"].strip(), row["suffix"]))
    return variants


def build_inputs(model_base, instruction: str, suffix: str = ""):
    text = instruction if not suffix else f"{instruction} {suffix}"
    tok = model_base.tokenize_instructions_fn(instructions=[text])
    return {k: v.to(model_base.model.device) for k, v in tok.items()}


def capture_resid_pre(model_base, inputs, pre_hooks=None):
    """Run a forward pass and capture resid_pre at every block (list of [seq, d])."""
    pre_hooks = pre_hooks or []
    captured: List[torch.Tensor] = []

    handles = []

    def hook_fn(module, hook_input):
        # hook_input[0]: [batch, seq, d_model]
        captured.append(hook_input[0].detach().to(torch.float32).cpu()[0])

    # Register our capture hooks
    for blk in model_base.model_block_modules:
        handles.append(blk.register_forward_pre_hook(hook_fn))
    # Register any extra hooks (e.g., ablation)
    for module, fn in pre_hooks:
        handles.append(module.register_forward_pre_hook(fn))

    with torch.no_grad():
        _ = model_base.model(**inputs)

    for h in handles:
        h.remove()

    return captured  # list length = num_layers, each [seq, d]


def stack_layers(layer_acts: List[torch.Tensor]) -> torch.Tensor:
    max_seq = min(act.shape[0] for act in layer_acts)
    trimmed = [act[:max_seq] for act in layer_acts]
    return torch.stack(trimmed, dim=0)  # [layers, seq, d]


def frobenius_norm(t: torch.Tensor) -> float:
    return torch.linalg.norm(t.reshape(-1), ord=2).item()


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta)

    model_base = Llama2Model(args.model_path)
    model_base.model.to(args.device)

    # Data
    harmful = load_dataset_split(harmtype="harmful", split="test")
    harmful = harmful[: args.n_examples]

    # Ablation hooks: apply to every layer
    ablation_pre = [(blk, get_direction_ablation_input_pre_hook(direction=direction)) for blk in model_base.model_block_modules]

    variants = [("baseline", "")]
    variants += read_suffixes(args.suffix_csv)

    results: Dict[str, Dict] = {}

    for method, suffix in variants:
        print(f"Evaluating {method} ...")
        ratios = []
        for row in harmful:
            instr = row["instruction"]

            inp_base = build_inputs(model_base, instr, suffix="")
            inp_suffix = build_inputs(model_base, instr, suffix=suffix)

            acts_base = capture_resid_pre(model_base, inp_base, pre_hooks=[])
            acts_ablation = capture_resid_pre(model_base, inp_base, pre_hooks=ablation_pre)
            acts_suffix = capture_resid_pre(model_base, inp_suffix, pre_hooks=[])

            # Align sequence lengths
            min_layers = min(len(acts_base), len(acts_ablation), len(acts_suffix))
            acts_base = acts_base[:min_layers]
            acts_ablation = acts_ablation[:min_layers]
            acts_suffix = acts_suffix[:min_layers]

            H_base = stack_layers(acts_base)
            H_ablation = stack_layers(acts_ablation)
            H_suffix = stack_layers(acts_suffix)

            # Align by sequence length
            min_seq = min(H_base.shape[1], H_ablation.shape[1], H_suffix.shape[1])
            H_base = H_base[:, :min_seq]
            H_ablation = H_ablation[:, :min_seq]
            H_suffix = H_suffix[:, :min_seq]

            num = frobenius_norm(H_suffix - H_ablation)
            den = frobenius_norm(H_base - H_ablation)
            ratio = float("inf") if den == 0 else num / den
            ratios.append(ratio)

        results[method] = {
            "n_examples": len(harmful),
            "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
            "median_ratio": sorted(ratios)[len(ratios) // 2] if ratios else None,
            "ratios": ratios,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()

