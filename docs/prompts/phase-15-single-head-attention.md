# Phase 15: Single-Head Attention

## Objective
Implement scaled dot-product attention with causal masking support.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11 | Linear layer |
| 12 | Softmax activation |
| 13 | LayerNorm |
| 14 | Dropout |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable, ops |
| 11 | include/lightwatch/nn/linear.hpp | Linear |
| 12 | include/lightwatch/nn/activations.hpp | Softmax |
| 14 | include/lightwatch/nn/dropout.hpp | Dropout |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/attention.hpp | ScaledDotProductAttention | Phase 16 |
| src/nn/attention.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class ScaledDotProductAttention : public Module {
public:
    explicit ScaledDotProductAttention(float dropout_p = 0.0);

    // q, k, v: {batch, seq_len, head_dim}
    // Returns: {batch, seq_len, head_dim}
    autograd::Variable forward(
        const autograd::Variable& query,
        const autograd::Variable& key,
        const autograd::Variable& value,
        const Tensor<bool>* mask = nullptr);  // Optional attention mask

    autograd::Variable forward(const autograd::Variable& input) override;

private:
    Dropout dropout_;
    float scale_;
};

// Utility: create causal (autoregressive) mask
Tensor<bool> causal_mask(size_t seq_len);

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Attention scores**: scores = (Q @ K^T) / sqrt(d_k)
2. **Masking**: scores = scores + mask (where mask is -inf for blocked positions)
3. **Softmax**: weights = softmax(scores, dim=-1)
4. **Dropout**: weights = dropout(weights)
5. **Output**: output = weights @ V

### Performance Constraints
- O(seq² * head_dim) time complexity
- Memory: O(seq²) for attention weights

## Required Tests
See `docs/test_specs/phase-15-attention.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_15_attention_shape` | Q,K,V {2, 8, 64} | Output {2, 8, 64} |
| `test_phase_15_attention_scale` | Check scaling | Divided by sqrt(64) |
| `test_phase_15_causal_mask` | seq_len=4 | Lower triangular True |
| `test_phase_15_masked_attention` | With causal mask | No future leakage |
| `test_phase_15_attention_backward` | Backward pass | Gradients flow |
| `test_phase_15_attention_weights_sum` | Softmax output | Rows sum to 1 |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_15" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/attention.hpp`
- [ ] Causal masking prevents attending to future tokens

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 1 |
| Complexity | HIGH |

## Notes
- Scale factor: 1/sqrt(head_dim) = 1/sqrt(64) = 0.125 for GPT-2
- Causal mask: lower triangular matrix (True where i >= j)
- Mask value: use -1e9 instead of -inf for numerical stability
- This is single-head only; multi-head in Phase 16
