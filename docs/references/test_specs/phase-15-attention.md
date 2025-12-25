<!-- File: docs/references/test_specs/phase-15-attention.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPEC FILE TEMPLATES -->

# Phase 15: Single-Head Attention - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 8

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_15_attention_shape` | `Q,K,V all {2,12,64,64}` | Output shape `{2,12,64,64}` |
| `test_phase_15_attention_causal` | `S=4, query at pos 2` | Attention weights: `w[3]==0.0` (future masked) |
| `test_phase_15_attention_softmax` | `S=8, any Q,K,V` | Each row of attention weights sums to 1.0 (tol 1e-5) |
| `test_phase_15_attention_gradient` | `Q,K,V {1,1,4,8} randn` | Numerical gradient check passes (tol 1e-3) |
| `test_phase_15_attention_scale` | `d_k=64` | Pre-softmax scores scaled by `1/8.0` (sqrt(64)) |
| `test_phase_15_attention_mask_inf` | Masked position `K[0,0,3,:]=999.0` | Output unaffected by masked position values |
| `test_phase_15_attention_single_token` | `S=1, Q,K,V {1,1,1,64}` | Output shape `{1,1,1,64}`, no NaN |
| `test_phase_15_attention_long_sequence` | `S=1024, B=1, H=1, D=64` | Completes without OOM, output shape correct |

## Implementation Notes

- Attention formula: `softmax(Q @ K.T / sqrt(d_k) + mask) @ V`
- Causal mask: `-inf` for positions where `j > i`
- Use `-1e9` instead of `-inf` to avoid NaN in softmax gradients
