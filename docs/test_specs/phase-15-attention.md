# Phase 15: Single-Head Attention - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 8

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_attention_shape` | `Q,K,V [B,H,S,D]` | Output `[B,H,S,D]` |
| `test_attention_causal` | Position i query | Weights 0 for j > i |
| `test_attention_softmax` | Any input | Weights sum to 1.0 per query |
| `test_attention_gradient` | Backward pass | Matches numerical gradient (tol 1e-4) |
| `test_attention_scale` | `d_k=64` | Scaled by `1/sqrt(64)` |
| `test_attention_mask_inf` | Masked positions | Output unaffected by masked values |
| `test_attention_single_token` | `S=1` | No crash, correct output |
| `test_attention_long_sequence` | `S=1024` | Memory efficient, no OOM |

## Implementation Notes

- Causal mask: position i can only attend to positions 0..i
- Scaling factor: 1/sqrt(d_k) where d_k is the key dimension
- Use -inf (or very large negative) for masked positions before softmax
- For numerical stability, subtract max before exp in softmax
