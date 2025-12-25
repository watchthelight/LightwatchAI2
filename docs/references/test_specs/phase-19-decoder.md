<!-- File: docs/references/test_specs/phase-19-decoder.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 19: Transformer Decoder - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 5

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_19_decoder_forward_shape` | `input {2, 16, 768}` | Output shape `{2, 16, 768}` |
| `test_phase_19_decoder_residual` | `input {1, 4, 768}` | `output.mean()` close to `input.mean()` at init |
| `test_phase_19_decoder_causal` | Seq `[A,B,C,D]`, check output at pos 2 | Same whether D present or not (causal isolation) |
| `test_phase_19_decoder_gradient` | Forward + backward | All layer parameters have non-zero grad |
| `test_phase_19_decoder_layer_order` | Hook MHA and FFN | MHA called before FFN in each block |

## Implementation Notes

- Pre-norm: LayerNorm → MHA → residual → LayerNorm → FFN → residual
- GPT-2 uses GELU activation in FFN
- Dropout applied after attention and FFN (during training)
