<!-- File: docs/references/test_specs/phase-31-gpt.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPEC FILE TEMPLATES -->

# Phase 31: GPT Architecture - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_31_gpt_forward_shape` | `input_ids {2, 16}` (batch=2, seq=16) | Output logits shape `{2, 16, 50257}` |
| `test_phase_31_gpt_causal` | Seq `[A,B,C,D]`, compute logits | `logits[2]` (for C) identical whether D present or not |
| `test_phase_31_gpt_parameter_count` | GPT-2 Small config | Total params in `[118M, 130M]` (~124M ± 5%) |
| `test_phase_31_gpt_gradient` | Forward + backward on random input | All named parameters have non-zero `.grad` |
| `test_phase_31_gpt_embedding_tied` | Check `wte.weight` and `lm_head.weight` | Same underlying data pointer |
| `test_phase_31_gpt_layer_order` | Hook each layer, forward pass | Layers execute in order 0,1,2,...,11 |

## Implementation Notes

- GPT-2 Small config: 12 layers, 768 hidden, 12 heads, 50257 vocab, 1024 ctx
- Parameter count breakdown: wte(38.6M) + wpe(0.8M) + 12*layer(7.1M) + ln_f(1.5K) ≈ 124M
- Embedding tying: `lm_head.weight = wte.weight` (shared, not copied)
