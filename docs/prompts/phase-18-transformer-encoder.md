# Phase 18: Transformer Encoder

## Objective
Implement a single transformer encoder block with pre-norm architecture.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 16 | MultiHeadAttention |
| 17 | FFN |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 13 | include/lightwatch/nn/normalization.hpp | LayerNorm |
| 16 | include/lightwatch/nn/attention.hpp | MultiHeadAttention |
| 17 | include/lightwatch/nn/ffn.hpp | FFN |
| 14 | include/lightwatch/nn/dropout.hpp | Dropout |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/transformer.hpp | TransformerEncoderBlock | Phase 20 |
| src/nn/transformer.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class TransformerEncoderBlock : public Module {
public:
    TransformerEncoderBlock(
        size_t embed_dim,
        size_t num_heads,
        size_t ffn_dim,
        float dropout_p = 0.1,
        bool pre_norm = true);

    autograd::Variable forward(const autograd::Variable& input) override;

    // With attention mask
    autograd::Variable forward(
        const autograd::Variable& input,
        const Tensor<bool>* mask);

    LayerNorm ln1;
    MultiHeadAttention attn;
    Dropout dropout1;
    LayerNorm ln2;
    FFN ffn;
    Dropout dropout2;

private:
    bool pre_norm_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Pre-norm (GPT-2 style)**:
   - x = x + dropout(attn(ln1(x)))
   - x = x + dropout(ffn(ln2(x)))
2. **Post-norm (original Transformer)**:
   - x = ln1(x + dropout(attn(x)))
   - x = ln2(x + dropout(ffn(x)))
3. **Residual connections**: Add input to output of each sublayer

### Performance Constraints
- O(seq² * embed_dim) for attention
- O(seq * embed_dim * ffn_dim) for FFN

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_18_encoder_shape` | {2, 16, 768} | Output {2, 16, 768} |
| `test_phase_18_encoder_residual` | Check residual | Input contributes to output |
| `test_phase_18_encoder_backward` | Backward pass | All components have grads |
| `test_phase_18_encoder_prenorm` | pre_norm=true | Norm before attention |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_18" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/transformer.hpp`
- [ ] Encoder block produces correct output shape

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 250-400 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses pre-norm architecture
- Encoder blocks don't have causal masking (bidirectional)
- This is for completeness; GPT-2 is decoder-only
- Residual connections are critical for gradient flow
