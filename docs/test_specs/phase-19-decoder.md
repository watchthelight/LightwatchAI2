# Phase 19: Transformer Decoder - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 5

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_decoder_block_shape` | `x [B,S,D]` | Output `[B,S,D]` |
| `test_decoder_block_causal` | Full sequence | Position i uses only 0..i |
| `test_decoder_block_gradient` | Backward pass | All parameters have gradients |
| `test_decoder_residual` | Skip connections | Output = input + sublayer(input) |
| `test_decoder_prenorm` | LayerNorm position | Applied before attention/FFN |

## Implementation Notes

- GPT-2 uses pre-norm architecture: LN before attention and FFN
- Residual connections: x + Attention(LN(x)) and x + FFN(LN(x))
- Each block has: LN1, Attention, LN2, FFN
- Final LN applied after all blocks (ln_f in GPT-2)
