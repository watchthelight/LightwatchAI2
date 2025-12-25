<!-- File: docs/references/test_specs/phase-16-mha.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 16: Multi-Head Attention - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_16_mha_output_shape` | `input {2, 16, 768}, heads=12, d_model=768` | Output shape `{2, 16, 768}` |
| `test_phase_16_mha_head_split` | `768-dim input, 12 heads` | Each head receives 64-dim input |
| `test_phase_16_mha_gradient` | Forward + backward on random input | All projection weights have non-zero gradients |
| `test_phase_16_mha_attention_pattern` | First head, causal mask | Attention weights lower-triangular |
| `test_phase_16_mha_projection` | `input {1, 4, 768}` | `W_q, W_k, W_v` all have shape `{768, 768}` |
| `test_phase_16_mha_concatenation` | 12 heads, 64-dim each | After concat: `{batch, seq, 768}` |

## Implementation Notes

- GPT-2 uses pre-norm architecture (LayerNorm before attention)
- Output projection is `{768, 768}` (concat heads → hidden dim)
- Each head: Q, K, V projections from 768 → 64
