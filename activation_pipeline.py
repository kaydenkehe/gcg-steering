"""
Unified CLI to run activation-based GCG (refusal-direction objective),
standard GCG, and refusal-direction ablation baselines on Llama-2-7B-Chat.

Outputs:
  - Optimized suffixes for activation-GCG and standard GCG (optional)
  - Completions for harmful/harmless datasets
  - Jailbreak ASR metrics (substring + optional LlamaGuard2)
  - CE/perplexity metrics on harmless data
Artifacts are written under --output-dir.
"""
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import torch

# Local imports
sys.path.append("src/gcg")
sys.path.append("src/refusal_direction")

from llm_attacks import get_workers
from llm_attacks.activation_gcg import (
    AttackPrompt as ActAttackPrompt,
    PromptManager as ActPromptManager,
    MultiPromptAttack as ActMultiPromptAttack,
)
from llm_attacks.gcg import AttackPrompt as GcgAttackPrompt
from llm_attacks.gcg import PromptManager as GcgPromptManager
from llm_attacks.gcg import GCGMultiPromptAttack

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.llama2_model import Llama2Model
from pipeline.utils.hook_utils import (
    get_all_direction_ablation_hooks,
    get_direction_ablation_input_pre_hook,
)
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


def parse_args():
    p = argparse.ArgumentParser(description="Activation-based GCG pipeline (Llama-2-7B-Chat)")
    p.add_argument("--model-path", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument(
        "--direction-path",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt",
        help="Precomputed refusal direction .pt file",
    )
    p.add_argument(
        "--direction-meta",
        default="src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json",
        help="JSON with {layer,pos}",
    )
    p.add_argument("--output-dir", default="outputs/activation_gcg")
    p.add_argument("--n-train", type=int, default=25, help="Number of harmful train prompts")
    p.add_argument("--n-steps", type=int, default=200, help="Optimization steps for activation GCG")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--allow-non-ascii", action="store_true", default=False)
    p.add_argument("--run-gcg-baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-ablation-baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disable-llamaguard", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_direction(direction_path, direction_meta):
    direction = torch.load(direction_path, map_location="cpu")
    if direction_meta and os.path.exists(direction_meta):
        with open(direction_meta) as f:
            meta = json.load(f)
        layer = meta["layer"]
        pos = meta["pos"]
    else:
        layer = 0
        pos = -1
    return direction, layer, pos


def build_params(model_path, device):
    # Minimal params object for get_workers
    params = SimpleNamespace()
    params.tokenizer_paths = [model_path]
    params.tokenizer_kwargs = [{"use_fast": False}]
    params.model_paths = [model_path]
    params.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
    params.conversation_templates = ["llama-2"]
    params.devices = [device]
    params.num_train_models = 1
    return params


def sample_harmful_prompts(n_train):
    harmful = load_dataset_split(harmtype="harmful", split="train", instructions_only=True)
    random.seed(42)
    random.shuffle(harmful)
    return harmful[:n_train]


def run_activation_gcg(args, direction, layer, pos):
    print("=== Running activation-based GCG ===")
    params = build_params(args.model_path, args.device)
    workers, test_workers = get_workers(params)

    goals = sample_harmful_prompts(args.n_train)
    targets = ["Sure."] * len(goals)  # placeholder; not used in activation objective

    managers = {"AP": ActAttackPrompt, "PM": ActPromptManager, "MPA": ActMultiPromptAttack}

    attack = ActMultiPromptAttack(
        goals,
        targets,
        workers,
        direction=direction,
        layer=layer,
        pos=pos,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        managers=managers,
    )
    suffix, loss = attack.run(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        temp=args.temp,
        allow_non_ascii=args.allow_non_ascii,
        target_weight=1.0,
        control_weight=0.0,
        test_steps=50,
        stop_on_success=False,
        verbose=False,
        filter_cand=True,
        anneal=False,
    )[:2]

    for w in workers + test_workers:
        w.stop()
    return suffix


def run_standard_gcg(args):
    print("=== Running standard GCG baseline ===")
    params = build_params(args.model_path, args.device)
    workers, test_workers = get_workers(params)
    goals = sample_harmful_prompts(args.n_train)
    targets = ["Sure, here is"] * len(goals)
    managers = {"AP": GcgAttackPrompt, "PM": GcgPromptManager, "MPA": GCGMultiPromptAttack}
    attack = GCGMultiPromptAttack(
        goals,
        targets,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        managers=managers,
    )
    suffix, loss = attack.run(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        temp=args.temp,
        allow_non_ascii=args.allow_non_ascii,
        target_weight=1.0,
        control_weight=0.1,
        test_steps=50,
        stop_on_success=False,
        verbose=False,
        filter_cand=True,
        anneal=False,
    )[:2]
    for w in workers + test_workers:
        w.stop()
    return suffix


def generate_completions(model_base, dataset, suffix=None, fwd_pre_hooks=None, fwd_hooks=None, max_new_tokens=128):
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []
    # Append suffix to instruction if provided
    augmented = []
    for row in dataset:
        inst = row["instruction"] + (f" {suffix}" if suffix else "")
        augmented.append({"instruction": inst, "category": row.get("category", "unknown")})
    return model_base.generate_completions(
        augmented,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=8,
        max_new_tokens=max_new_tokens,
    )


def load_eval_sets():
    harmful_eval = load_dataset_split(harmtype="harmful", split="test")
    harmless_eval = load_dataset_split(harmtype="harmless", split="test")
    return harmful_eval, harmless_eval


def main():
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "completions"))

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta)
    print(f"Loaded direction: layer={layer}, pos={pos}, norm={direction.norm().item():.4f}")

    # Attack runs
    activation_suffix = run_activation_gcg(args, direction, layer, pos)
    print("Activation-GCG suffix:", activation_suffix)

    gcg_suffix = None
    if args.run_gcg_baseline:
        gcg_suffix = run_standard_gcg(args)
        print("Standard GCG suffix:", gcg_suffix)

    # Prepare model for evaluation (reuse refusal-direction utilities)
    model_base = Llama2Model(args.model_path)
    harmful_eval, harmless_eval = load_eval_sets()

    variants = []
    variants.append(("activation_gcg", activation_suffix, [], []))
    if gcg_suffix:
        variants.append(("gcg", gcg_suffix, [], []))

    if args.run_ablation_baseline:
        ablation_pre, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
        variants.append(("ablation", None, ablation_pre, ablation_hooks))

    # Generate completions and evaluate
    for name, suffix, pre_hooks, hooks in variants:
        print(f"=== Generating completions for {name} ===")
        harm_comp = generate_completions(
            model_base, harmful_eval, suffix=suffix, fwd_pre_hooks=pre_hooks, fwd_hooks=hooks, max_new_tokens=128
        )
        harmless_comp = generate_completions(
            model_base, harmless_eval, suffix=suffix, fwd_pre_hooks=pre_hooks, fwd_hooks=hooks, max_new_tokens=128
        )

        harm_path = os.path.join(args.output_dir, "completions", f"{name}_harmful.json")
        harmless_path = os.path.join(args.output_dir, "completions", f"{name}_harmless.json")
        json.dump(harm_comp, open(harm_path, "w"), indent=4)
        json.dump(harmless_comp, open(harmless_path, "w"), indent=4)

        # Jailbreak evals (substring + optional LlamaGuard2)
        meths = ["substring_matching"]
        if not args.disable_llamaguard and os.getenv("TOGETHER_API_KEY"):
            meths.append("llamaguard2")

        print(f"=== Evaluating jailbreak metrics for {name} ===")
        eval_harm = evaluate_jailbreak(completions=harm_comp, methodologies=meths)
        eval_harmless = evaluate_jailbreak(completions=harmless_comp, methodologies=meths)

        json.dump(
            eval_harm,
            open(os.path.join(args.output_dir, f"{name}_harmful_eval.json"), "w"),
            indent=4,
        )
        json.dump(
            eval_harmless,
            open(os.path.join(args.output_dir, f"{name}_harmless_eval.json"), "w"),
            indent=4,
        )

        print(f"{name} substring ASR (harmful): {eval_harm.get('substring_matching_success_rate', 'n/a')}")

        # CE / perplexity on harmless completions (acts as on-distribution)
        print(f"=== Evaluating CE/perplexity for {name} on harmless completions ===")
        loss_metrics = evaluate_loss(
            model_base,
            fwd_pre_hooks=pre_hooks,
            fwd_hooks=hooks,
            batch_size=2,
            n_batches=256,
            dataset_labels=["alpaca_custom_completions"],
            completions_file_path=harmless_path,
        )
        json.dump(
            loss_metrics,
            open(os.path.join(args.output_dir, f"{name}_loss_eval.json"), "w"),
            indent=4,
        )

    print("All done. Artifacts in", args.output_dir)


if __name__ == "__main__":
    main()
