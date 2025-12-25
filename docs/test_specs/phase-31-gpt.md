# Phase 31: GPT Architecture - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_gpt_forward_shape` | `[B, S]` input | Output `[B, S, V]` |
| `test_gpt_causal` | Sequence | Position i output depends only on 0..i |
| `test_gpt_parameter_count` | GPT-2 Small config | ~124M parameters (±5%) |
| `test_gpt_gradient` | Full backward | All parameters have gradients |
| `test_gpt_embedding_tied` | wte and lm_head | Share same weight matrix |
| `test_gpt_layer_order` | 12 layers | Layers execute in correct order |

## Implementation Notes

- GPT-2 Small: 12 layers, 768 hidden, 12 heads, 50257 vocab, 1024 context
- Parameter count: ~124M (verify within 5% tolerance)
- Weight tying: lm_head.weight is wte.weight (same memory)
- Layer order matters for weight loading from HuggingFace
