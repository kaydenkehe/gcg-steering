# Activation-based GCG Integration (Llama-2)

What changed
- Added `llm_attacks/activation_gcg` implementing a Greedy Coordinate Gradient attack whose objective is the projection of a chosen residual stream activation onto the refusal direction (objective A).
- Extended `attack_manager` worker to support activation-based gradient and score calls.
- Exposed activation attack classes via `llm_attacks/__init__.py` for compatibility.
- Added `activation_pipeline.py` — a unified CLI to:
  - run activation-GCG (and optional standard GCG baseline) on Llama‑2‑7B‑Chat,
  - optionally evaluate a refusal-direction ablation baseline,
  - generate harmful/harmless completions,
  - compute jailbreak ASR (substring, optional LlamaGuard2) and CE/perplexity metrics.

How to run
1) Activation-GCG (uses precomputed refusal direction by default):
```
python activation_pipeline.py \
  --model-path meta-llama/Llama-2-7b-chat-hf \
  --output-dir outputs/activation_gcg
```
   - If you have TogetherAI for LlamaGuard2, set `TOGETHER_API_KEY` and omit `--disable-llamaguard`.
   - To skip baselines: add `--no-run-gcg-baseline --no-run-ablation-baseline`.

2) Outputs (under `--output-dir`):
   - `completions/{variant}_harmful.json`, `{variant}_harmless.json`
   - `{variant}_harmful_eval.json`, `{variant}_harmless_eval.json` (ASR metrics)
   - `{variant}_loss_eval.json` (CE/perplexity on harmless completions)
   - Printed suffix strings for activation-GCG (and standard GCG if enabled)

Defaults and assumptions
- Model: Llama-2-7B-Chat.
- Refusal direction loaded from `src/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt` (layer/pos from accompanying metadata). Override with `--direction-path/--direction-meta`.
- Train set: 25 harmful instructions sampled from `refusal_direction/dataset` split `harmful_train`.
- Objective: minimize mean activation projection onto refusal direction at the selected layer/position.

Notes
- CE evaluation uses the harmless completions for each variant as “custom completions” (same pattern as the refusal-direction code).
- The new attack currently targets Llama-style models; extending to other families would require minor hook adjustments.
