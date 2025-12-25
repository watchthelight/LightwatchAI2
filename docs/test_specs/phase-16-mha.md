# Phase 16: Multi-Head Attention - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_mha_shape` | `x [B,S,D]`, 12 heads | Output `[B,S,D]` |
| `test_mha_head_split` | 12 heads, D=768 | Each head gets 64 dims |
| `test_mha_projection` | Input/output | Wq, Wk, Wv, Wo correctly sized |
| `test_mha_gradient` | Full backward | All projections have gradients |
| `test_mha_causal` | Decoder attention | Causal mask applied |
| `test_mha_cross_attention` | Encoder-decoder | Cross-attention works |

## Implementation Notes

- GPT-2 Small: 12 heads, 768 hidden, 64 per head
- Combined QKV projection is common optimization: W_qkv [768, 2304]
- Output projection: W_o [768, 768]
- Cross-attention uses different K,V source than Q
