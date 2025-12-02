#!/usr/bin/env python3
"""
Minimal GCG-style suffix optimization for Gemma chat models (Gemma-2B-IT).

Notes:
- Runs a universal attack over AdvBench harmful behaviors (goal/target pairs) like the original GCG.
- Uses Gemma's chat template directly (no FastChat dependency).
- Optimizes a single suffix over n_train prompts, with greedy coordinate gradient (top-k sampling).
- Default objective is the standard CE on the target string (same as original GCG).

Example:
  python gemma_gcg_baseline.py \
    --model-path google/gemma-2b-it \
    --n-train 25 \
    --n-steps 200 \
    --batch-size 512 \
    --topk 256 \
    --output outputs/gemma_gcg_suffix.txt
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

# Ensure refusal_direction's internal `pipeline` package is importable when
# running from the repo root.
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
REFUSAL_ROOT = os.path.join(REPO_DIR, "src", "refusal_direction")
if REFUSAL_ROOT not in sys.path:
    sys.path.append(REFUSAL_ROOT)

from pipeline.model_utils.gemma_model import GemmaModel


GEMMA_USER_PREFIX = "<start_of_turn>user\n"
GEMMA_MODEL_PREFIX = "<end_of_turn>\n<start_of_turn>model\n"


def load_advbench_pairs(n: int) -> Tuple[List[str], List[str]]:
    """Load and shuffle AdvBench goal/target pairs."""
    path = os.path.join("src", "gcg", "data", "advbench", "harmful_behaviors.csv")
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    goals = df["goal"].tolist()[:n]
    targets = df["target"].tolist()[:n]
    return goals, targets


def build_input_ids(tokenizer, goal: str, control_ids: List[int], target: str, device):
    """Construct input_ids and slices for goal + control + target using Gemma chat template."""
    # Tokenize parts
    user_prefix = tokenizer(GEMMA_USER_PREFIX, add_special_tokens=False).input_ids
    goal_ids = tokenizer(goal, add_special_tokens=False).input_ids
    space_ids = tokenizer(" ", add_special_tokens=False).input_ids  # to separate goal/control
    mid_ids = tokenizer(GEMMA_MODEL_PREFIX, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids

    # Assemble
    control_start = len(user_prefix) + len(goal_ids) + len(space_ids)
    control_stop = control_start + len(control_ids)
    target_start = control_stop + len(mid_ids)
    target_stop = target_start + len(target_ids)

    input_ids = (
        user_prefix
        + goal_ids
        + space_ids
        + list(control_ids)
        + mid_ids
        + target_ids
    )
    input_ids = torch.tensor(input_ids, device=device)

    control_slice = slice(control_start, control_stop)
    target_slice = slice(target_start, target_stop)
    loss_slice = slice(target_start - 1, target_stop - 1)
    return input_ids, control_slice, target_slice, loss_slice


def _model_device(model) -> torch.device:
    """Helper to get the device a model is on."""
    return next(model.parameters()).device


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Compute gradients of CE loss w.r.t. control token embeddings (GCG objective).
    """
    device = _model_device(model)
    embed = model.get_input_embeddings()
    embed_weights = embed.weight

    # One-hot for control tokens
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].to(device).unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # Stitch together embeddings
    embeds = embed(input_ids.unsqueeze(0).to(device)).detach()
    full_embeds = torch.cat(
        [embeds[:, : input_slice.start, :], input_embeds, embeds[:, input_slice.stop :, :]],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice].to(device)
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    return grad


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    """Sample new control tokens using coordinate-wise gradient information."""
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device,
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos],
        1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    return new_control_toks


def _forward_batches(*, model, input_ids, attention_mask, batch_size=512):
    device = _model_device(model)
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size].to(device)
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i : i + batch_size].to(device)
        else:
            batch_attention_mask = None
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)
        del batch_input_ids, batch_attention_mask
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return torch.cat(logits, dim=0)


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):
    """
    Replace control_slice with each candidate control and get logits.
    """
    device = _model_device(model)

    if isinstance(test_controls, torch.Tensor):
        # Direct tensor of token IDs: shape [batch, control_len]
        test_ids = test_controls.to(device)
        pad_tok = -1
    elif isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(
                tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                device=device,
            )
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if test_ids[0].shape[0] != control_slice.stop - control_slice.start:
        raise ValueError(
            f"test_controls must have shape (n, {control_slice.stop - control_slice.start}), got {test_ids.shape}"
        )

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(device),
        1,
        locs,
        test_ids,
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        logits = _forward_batches(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del locs, test_ids
        return logits, ids
    else:
        logits = _forward_batches(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del locs, test_ids, ids
        return logits


def target_loss(logits, ids, target_slice):
    """Per-example CE loss over target_slice tokens."""
    crit = nn.CrossEntropyLoss(reduction="none")
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-train", type=int, default=25)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--allow-non-ascii", action="store_true", default=False)
    ap.add_argument("--output", default="outputs/gemma_gcg_suffix.txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model/tokenizer (no FastChat)
    model_base = GemmaModel(args.model_path)
    model = model_base.model.to(device)
    tokenizer = model_base.tokenizer
    tokenizer.padding_side = "left"

    # Load AdvBench goal/target pairs
    goals, targets = load_advbench_pairs(args.n_train)

    # Initialize control string/tokens
    control_str = "! " * 20  # 20 tokens approx
    control_ids = tokenizer(control_str, add_special_tokens=False).input_ids

    best_control_ids = control_ids
    best_loss = float("inf")

    for step in range(args.n_steps):
        # Aggregate gradients across all prompts
        agg_grad = None
        for goal, target in zip(goals, targets):
            input_ids, control_slice, target_slice, loss_slice = build_input_ids(
                tokenizer, goal, control_ids, target, device
            )
            grad = token_gradients(
                model, input_ids, control_slice, target_slice, loss_slice
            )
            grad = grad / grad.norm(dim=-1, keepdim=True)
            if agg_grad is None:
                agg_grad = torch.zeros_like(grad)
            agg_grad += grad

        # Sample candidate controls
        control_cands = sample_control(
            torch.tensor(control_ids, device=device),
            agg_grad,
            batch_size=args.batch_size,
            topk=args.topk,
            temp=args.temp,
            not_allowed_tokens=None if args.allow_non_ascii else None,
        )

        # Evaluate candidates
        losses = torch.zeros(control_cands.shape[0], device=device)
        with torch.no_grad():
            for goal, target in zip(goals, targets):
                # Build prompt for each candidate batch
                goal_tokens = tokenizer(goal, add_special_tokens=False).input_ids
                target_tokens = tokenizer(target, add_special_tokens=False).input_ids

                # Prepare batched input_ids
                batch_inputs = []
                control_slice = None
                target_slice = None
                loss_slice = None
                for cand in control_cands:
                    input_ids, c_slice, t_slice, l_slice = build_input_ids(
                        tokenizer,
                        goal,
                        cand.tolist(),
                        target,
                        device,
                    )
                    batch_inputs.append(input_ids)
                    control_slice = c_slice
                    target_slice = t_slice
                    loss_slice = l_slice

                batch_inputs = torch.stack(batch_inputs, dim=0)
                logits, ids = get_logits(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=batch_inputs[0],  # shape: sequence
                    control_slice=control_slice,
                    test_controls=control_cands,
                    return_ids=True,
                    batch_size=128,
                )
                # logits shape: [batch, seq, vocab]
                tloss = target_loss(logits, ids, target_slice)
                losses += tloss.to(device)

        min_idx = losses.argmin().item()
        best_cand_ids = control_cands[min_idx].tolist()
        best_step_loss = losses[min_idx].item() / len(goals)

        # Update control
        control_ids = best_cand_ids
        if best_step_loss < best_loss:
            best_loss = best_step_loss
            best_control_ids = best_cand_ids

        # Print progress
        if step % 10 == 0 or step == args.n_steps - 1:
            print(f"Step {step+1}/{args.n_steps} | loss {best_step_loss:.4f}")

    # Save final suffix
    final_suffix = tokenizer.decode(best_control_ids, skip_special_tokens=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(final_suffix)
    print(f"Saved Gemma GCG suffix to {args.output}")
    print(final_suffix)


if __name__ == "__main__":
    main()
