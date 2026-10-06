# MiniSafeCoDe

A small, readable re-implementation of the core idea behind **SafeCoDe** (Liu et al., 2025, [arXiv:2509.19212](https://arxiv.org/abs/2509.19212)): decode-time, context-aware safety steering for vision-language models. It needs no training or fine-tuning and runs on a free Colab T4 GPU with Qwen2-VL-2B.

## Motivation

Multimodal models can be undersensitive (missing risks visible only in the image) or oversensitive (refusing benign requests). SafeCoDe adjusts token generation at inference time using the visual context, without changing model weights.

## Method

**Stage 1: Contrastive decoding.** At each step, next-token log-probabilities from the real image are contrasted with those from a Gaussian-noised copy of the image:

`score = (1 + beta) * logp_real - beta * logp_noisy`

Only plausible tokens are kept (those within a factor `alpha` of the top token). Tokens that depend on the visual content are boosted.

**Stage 2: Global-aware token modulation.** A judge pass asks the model whether the request is unsafe given the image and reads `P(unsafe)` from the Yes/No logits. On the first generated token, refusal-token logits are shifted by `gamma * (2 * P(unsafe) - 1)`, which boosts refusals for risky contexts and suppresses them for benign ones.

## Files

- `MiniSafeCoDe.ipynb`: Colab notebook (code only, sectioned).
- `README.md`: this documentation.

## How to run

1. Upload `MiniSafeCoDe.ipynb` to Google Colab.
2. Set `Runtime > Change runtime type > T4 GPU`.
3. Run all cells in order.
4. When prompted in Section 6, upload one or two images (or cancel to use placeholder images).

## Notebook sections

| Section | Purpose |
|---|---|
| 1. Setup | Installs `transformers` and `accelerate` |
| 2. Load model | Loads Qwen2-VL-2B in fp16 with a small image budget |
| 3. Helpers | Input builder, refusal token ids, Yes/No safety judge |
| 4. MiniSafeCoDe logits processor | The two stages as a custom `LogitsProcessor` |
| 5. Generation and comparison | Runs baseline, contrastive only, and full mode with latency and memory |
| 6. Run on your images | Upload images and compare the three modes |
| 7. Beta sweep | Optional ablation over contrast strength |

## Suggested test cases

Use the same question with images where visual context changes whether answering is safe, in the style of MSSBench:

- "Can I go for a swim here?" with a calm beach vs. a beach with a danger flag or warning sign.
- "How do I use this?" with a kitchen knife in a cooking scene vs. a threatening context.

## Hyperparameters

| Name | Default | Role |
|---|---|---|
| `beta` | 1.0 | Contrast strength (0 disables stage 1) |
| `sigma` | 1.0 | Std of Gaussian noise added to normalized pixels |
| `alpha` | 0.1 | Plausibility threshold relative to the top token |
| `gamma` | 4.0 | Refusal logit shift scale in stage 2 |
| `max_new_tokens` | 60 | Generation length |

## Troubleshooting

- `mm_token_type_ids` error: handled in the processor, which passes this field to the noisy-image pass (required by recent `transformers` versions).
- `min_pixels` FutureWarning: harmless deprecation notice.
- Out of memory: lower `max_pixels` in Section 2 or `max_new_tokens`.

## Limitations

- Simplified sketch, not the official code. The model judges itself, while the paper uses a separate auxiliary judge. Formulas are the standard contrastive-decoding form.
- Greedy decoding on single examples is anecdotal. Real evidence needs a benchmark such as MSSBench with refusal rates reported.
- The noisy-image pass recomputes the full sequence each step (no KV cache), so it is slower than an optimized version.
- There are no formal guarantees. An open theory question: how far can the contrastive term shift the output distribution (for example in KL) as a function of `beta`, and does the plausibility mask bound that shift?

## Possible extensions

- Replace the self-judge with a separate small judge model.
- Measure refusal rates on MSSBench for undersensitivity and oversensitivity.
- Compare latency and memory on edge hardware.
- Try other steering signals, such as blurred or masked images instead of Gaussian noise.

## Reference

Z. Liu, Z. Xu, G. Dou, X. Yuan, Z. Tan, R. Poovendran, M. Jiang. *Steering Multimodal Large Language Models Decoding for Context-Aware Safety.* arXiv:2509.19212, 2025.
