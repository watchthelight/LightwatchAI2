# Phase 19: Transformer Decoder

## Objective
Implement a single transformer decoder block with causal self-attention (GPT-style).

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 16 | MultiHeadAttention |
| 17 | FFN |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 13 | include/lightwatch/nn/normalization.hpp | LayerNorm |
| 15 | include/lightwatch/nn/attention.hpp | causal_mask |
| 16 | include/lightwatch/nn/attention.hpp | MultiHeadAttention |
| 17 | include/lightwatch/nn/ffn.hpp | FFN |
| 14 | include/lightwatch/nn/dropout.hpp | Dropout |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/transformer.hpp | TransformerDecoderBlock (added) | Phase 20, 31 |
| src/nn/transformer.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class TransformerDecoderBlock : public Module {
public:
    TransformerDecoderBlock(
        size_t embed_dim,
        size_t num_heads,
        size_t ffn_dim,
        float dropout_p = 0.1);

    // Causal self-attention (automatically applies causal mask)
    autograd::Variable forward(const autograd::Variable& input) override;

    // With custom mask (for KV-cache)
    autograd::Variable forward(
        const autograd::Variable& input,
        const Tensor<bool>* mask);

    LayerNorm ln1;
    MultiHeadAttention attn;
    Dropout dropout1;
    LayerNorm ln2;
    FFN ffn;
    Dropout dropout2;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Causal self-attention**: Apply causal mask to prevent future token attention
2. **Pre-norm architecture**: LayerNorm before attention and FFN
3. **Residual connections**: Add input to sublayer outputs
4. **Forward**:
   - x = x + dropout(attn(ln1(x), causal_mask))
   - x = x + dropout(ffn(ln2(x)))

### Performance Constraints
- O(seq² * embed_dim) for self-attention
- Causal mask generation: O(seq²)

## Required Tests
See `docs/test_specs/phase-19-decoder.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_19_decoder_shape` | {2, 16, 768} | Output {2, 16, 768} |
| `test_phase_19_decoder_causal` | Check attention | No future leakage |
| `test_phase_19_decoder_backward` | Backward pass | All components have grads |
| `test_phase_19_decoder_autoregressive` | Generate token-by-token | Consistent outputs |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_19" --output-on-failure` exits 0
- [ ] Decoder block applies causal masking
- [ ] No information leakage from future tokens

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-450 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 is decoder-only transformer
- Causal mask is lower triangular (True where i >= j)
- Pre-norm: norm before attention, not after
- This block is the core building block for Phase 31 (GPT)
