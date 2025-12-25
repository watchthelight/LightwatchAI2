# Phase 09: Positional Encoding

## Objective
Implement various positional encoding schemes: sinusoidal, learned, RoPE, and ALiBi for flexibility.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 03 | include/lightwatch/tensor.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 03 | include/lightwatch/tensor.hpp | Tensor<float> |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/positional.hpp | SinusoidalPE, RoPE, ALiBi | Phase 10, 15, 31 |
| src/nn/positional.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

// Sinusoidal positional encoding (original Transformer)
class SinusoidalPE : public Module {
public:
    SinusoidalPE(size_t max_seq_len, size_t embed_dim);

    // Returns positional encoding for given sequence length
    Tensor<float> get_encoding(size_t seq_len) const;

    // Add to input embeddings
    autograd::Variable forward(const autograd::Variable& input) override;

private:
    Tensor<float> encodings_;  // Precomputed {max_seq_len, embed_dim}
};

// Rotary Position Embedding (RoPE)
class RoPE {
public:
    RoPE(size_t head_dim, size_t max_seq_len = 2048, float base = 10000.0f);

    // Apply rotation to query and key tensors
    std::pair<Tensor<float>, Tensor<float>> apply(
        const Tensor<float>& q,  // {batch, heads, seq, head_dim}
        const Tensor<float>& k,
        size_t offset = 0) const;

private:
    Tensor<float> cos_cached_;
    Tensor<float> sin_cached_;
};

// Attention with Linear Biases (ALiBi)
class ALiBi {
public:
    explicit ALiBi(size_t num_heads);

    // Returns bias to add to attention scores
    Tensor<float> get_bias(size_t seq_len) const;

private:
    Tensor<float> slopes_;  // Per-head slopes
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Sinusoidal**: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(...)
2. **RoPE**: Rotate pairs of dimensions by position-dependent angle
3. **ALiBi**: Linear bias m * (i - j) where m is head-specific slope
4. **Caching**: Precompute encodings for max_seq_len at construction

### Performance Constraints
- Sinusoidal: O(seq_len * embed_dim) to add
- RoPE: O(seq_len * head_dim) per head
- ALiBi: O(seq_len²) to compute bias matrix

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_09_sinusoidal_shape` | seq=10, dim=64 | Shape {10, 64} |
| `test_phase_09_sinusoidal_values` | Position 0 | Correct sin/cos pattern |
| `test_phase_09_rope_shape` | Apply to q, k | Same shapes as input |
| `test_phase_09_rope_rotation` | Known input | Matches reference |
| `test_phase_09_alibi_shape` | 8 heads, seq=16 | Bias shape {8, 16, 16} |
| `test_phase_09_alibi_slopes` | 8 heads | Correct geometric sequence |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_09" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/positional.hpp`
- [ ] All encoding schemes produce correct shapes

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 4 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses learned positional embeddings (Phase 08), not sinusoidal
- RoPE and ALiBi included for future model support
- Sinusoidal included for completeness and testing
- RoPE requires rotating complex number pairs
- ALiBi slopes: 2^(-8/n) for head i, where n = num_heads
