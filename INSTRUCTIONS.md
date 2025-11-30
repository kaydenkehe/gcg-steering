# INSTRUCTIONS.md

## Project Overview

This project investigates whether the Greedy Coordinate Gradient (GCG) attack strategy can be modified to optimize suffixes that suppress internal “refusal” activations in a language model, rather than maximizing output token probability. This is a white-box attack relying on full model access, targeting LLaMA-2-7B-Chat.

You will adapt the optimization method from the GCG paper using concepts and directional data from the Refusal Direction paper. The goal is to determine if this activation-based optimization yields higher success at eliciting harmful behavior.

## Objectives

You are responsible for developing a new adversarial attack method with the following properties:

- Leverages the discrete greedy update pattern of GCG
- Replaces GCG’s log-probability objective with an activation-based objective derived from the Refusal Direction paper
- Works in both single-prompt (individual) and multi-prompt (universal) attack settings
- Optimizes a suffix to minimize the model's projection onto the learned refusal direction
- Evaluates whether the resulting completions comply with the harmful prompt (not whether they match a target string)
- Compares this method against existing GCG and Refusal Direction baselines

## Conceptual Background

The Refusal Direction paper shows that refusal behavior in aligned language models is governed by a single learned direction in the model’s residual stream. Projecting the hidden state onto this direction yields a scalar “refusal activation score.” Higher scores correlate with the model refusing a prompt.

The GCG method performs greedy discrete optimization over suffix tokens to maximize the likelihood of compliant outputs. In this project, you will maintain GCG’s greedy update structure but define a new optimization objective: **minimize the refusal activation score** at a designated location in the model’s hidden states.

## What You Will Implement

You are to:

1. Implement a discrete suffix optimization method that:
   - Iteratively selects token substitutions at each suffix position
   - Uses model gradient or forward evaluations to decide the best substitution
   - Uses the refusal activation score (not output token probabilities) as its optimization target

2. Support two modes of operation:
   - **Individual attack**: one suffix is optimized per harmful prompt
   - **Universal attack**: one suffix is optimized jointly across multiple prompts

3. Integrate this method with the existing model infrastructure (using LLaMA-2-7B-Chat)
   - Load a model with internal access to activations
   - Use or compute the refusal direction
   - Compute the refusal activation score during optimization

4. Evaluate attack success
   - Generate completions using the optimized suffixes
   - Determine whether the model complies with the harmful behavior
   - Compute the Attack Success Rate (ASR)

## Evaluation

You should implement evaluation code that:

- Measures the fraction of completions that exhibit the harmful behavior (ASR)
- Runs on the same model and prompt set used during optimization
- Compares this activation-based method against:
  - Standard GCG (log-probability-based)
  - Refusal direction baseline from the original paper

## Implementation Constraints

You should:

- Use the LLaMA-2-7B-Chat model for all attacks and evaluations
- Preserve a modular architecture separating:
  - Suffix optimization
  - Model evaluation
  - Prompt loading
  - Refusal direction scoring
- Reuse or adapt components from both the `llm-attacks` and `refusal_direction` codebases where appropriate
- Log and report all experiment metadata, including:
  - Prompts used
  - Final suffixes
  - Refusal activation scores
  - Evaluation results

## Design Flexibility

You are free to determine:

- How to extract the relevant hidden state for computing the refusal activation score
- Which layer and position in the model to use for projection
- How to structure the optimization loop and token selection strategy
- How to balance efficiency and accuracy
- Whether to apply batching, caching, or sampling strategies

## Deliverables

You must produce:

- A working implementation of the activation-based GCG suffix optimizer
- Scripts for evaluating the attack in both individual and universal settings
- A comparison summary of ASR across all attack methods
- A structured output log of all results for further analysis

## References

- Refusal Direction Paper: activation target concept, refusal vector, scoring methodology
- GCG Paper: optimization strategy, suffix structure, evaluation framing

Your implementation should reflect the goals and assumptions of both papers while introducing a novel hybrid method.
