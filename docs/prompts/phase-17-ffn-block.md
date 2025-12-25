# Phase 17: FFN Block

## Objective
Implement the feed-forward network block used in transformer layers.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11 | Linear layer |
| 12 | GELU activation |
| 13 | LayerNorm |
| 14 | Dropout |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 11 | include/lightwatch/nn/linear.hpp | Linear |
| 12 | include/lightwatch/nn/activations.hpp | GELU |
| 14 | include/lightwatch/nn/dropout.hpp | Dropout |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/ffn.hpp | FFN, SwiGLU | Phase 18, 19, 31 |
| src/nn/ffn.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

// Standard GPT-2 FFN: Linear -> GELU -> Linear
class FFN : public Module {
public:
    FFN(size_t embed_dim, size_t hidden_dim, float dropout_p = 0.0);

    autograd::Variable forward(const autograd::Variable& input) override;

    Linear fc1;   // {embed_dim, hidden_dim}
    Linear fc2;   // {hidden_dim, embed_dim}
    GELU gelu;
    Dropout dropout;

private:
    size_t embed_dim_;
    size_t hidden_dim_;
};

// SwiGLU variant (optional, for modern models)
class SwiGLU : public Module {
public:
    SwiGLU(size_t embed_dim, size_t hidden_dim, float dropout_p = 0.0);

    autograd::Variable forward(const autograd::Variable& input) override;

    Linear gate_proj;  // {embed_dim, hidden_dim}
    Linear up_proj;    // {embed_dim, hidden_dim}
    Linear down_proj;  // {hidden_dim, embed_dim}
    Dropout dropout;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **FFN forward**: dropout(fc2(gelu(fc1(x))))
2. **SwiGLU forward**: dropout(down_proj(silu(gate_proj(x)) * up_proj(x)))
3. **GPT-2 config**: hidden_dim = 4 * embed_dim = 3072

### Performance Constraints
- Two matrix multiplications per FFN
- GELU is the bottleneck (non-linear)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_17_ffn_shape` | {2, 16, 768} | Output {2, 16, 768} |
| `test_phase_17_ffn_hidden` | Check fc1.weight | Shape {3072, 768} |
| `test_phase_17_ffn_backward` | Backward pass | All weights have grads |
| `test_phase_17_swiglu_shape` | {2, 16, 768} | Output {2, 16, 768} |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_17" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/ffn.hpp`
- [ ] FFN produces correct output shape

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 200-350 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 uses FFN (not SwiGLU)
- hidden_dim = 4 * embed_dim is standard
- Dropout applied after fc2
- SwiGLU included for future model support
