# Phase 16: Multi-Head Attention

## Objective
Implement multi-head attention with head splitting, parallel attention, and head merging.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 15 | ScaledDotProductAttention |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 15 | include/lightwatch/nn/attention.hpp | ScaledDotProductAttention |
| 11 | include/lightwatch/nn/linear.hpp | Linear |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/attention.hpp | MultiHeadAttention (added) | Phase 18, 19, 31 |
| src/nn/attention.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads,
                       float dropout_p = 0.0, bool bias = true);

    // Self-attention: input is used for Q, K, V
    autograd::Variable forward(const autograd::Variable& input) override;

    // Cross-attention or with explicit K, V
    autograd::Variable forward(
        const autograd::Variable& query,
        const autograd::Variable& key,
        const autograd::Variable& value,
        const Tensor<bool>* mask = nullptr);

    Linear q_proj;  // {embed_dim, embed_dim}
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t head_dim_;
    ScaledDotProductAttention attention_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Project**: Q = Wq @ input, K = Wk @ input, V = Wv @ input
2. **Split heads**: {batch, seq, embed} -> {batch, heads, seq, head_dim}
3. **Attention**: Apply ScaledDotProductAttention per head
4. **Merge heads**: {batch, heads, seq, head_dim} -> {batch, seq, embed}
5. **Output projection**: output = Wo @ merged

### Performance Constraints
- O(seq² * embed_dim) for attention
- Parallel computation across heads (reshape, not loop)

## Required Tests
See `docs/test_specs/phase-16-mha.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_16_mha_shape` | {2, 16, 768} | Output {2, 16, 768} |
| `test_phase_16_mha_heads` | 12 heads, 768 dim | head_dim = 64 |
| `test_phase_16_mha_params` | Count parameters | 4 * 768 * 768 |
| `test_phase_16_mha_causal` | With causal mask | No future leakage |
| `test_phase_16_mha_backward` | Backward pass | All projections have grads |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_16" --output-on-failure` exits 0
- [ ] Multi-head attention produces correct output shape
- [ ] Parameter count matches expected

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 Small: 12 heads, 768 embed_dim, 64 head_dim
- Head splitting is a reshape, not copy
- All heads computed in parallel via batched matmul
- Output projection combines all head outputs
