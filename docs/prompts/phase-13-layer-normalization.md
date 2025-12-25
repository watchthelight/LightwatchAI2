# Phase 13: Layer Normalization

## Objective
Implement LayerNorm and RMSNorm for transformer normalization with learnable parameters.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11 | Module base class |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable, ops |
| 11 | include/lightwatch/nn/linear.hpp | Module base |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/normalization.hpp | LayerNorm, RMSNorm | Phase 15, 17, 21, 31 |
| src/nn/normalization.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/module.hpp
namespace lightwatch::nn {

class LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5);

    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;  // gamma, shape {normalized_shape}
    autograd::Variable bias;    // beta, shape {normalized_shape}

private:
    size_t normalized_shape_;
    float eps_;
};

class RMSNorm : public Module {
public:
    RMSNorm(size_t normalized_shape, float eps = 1e-6);

    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;  // Scale only, shape {normalized_shape}

private:
    size_t normalized_shape_;
    float eps_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **LayerNorm**:
   - mean = mean(x, dim=-1)
   - var = var(x, dim=-1)
   - y = (x - mean) / sqrt(var + eps) * gamma + beta
2. **RMSNorm**:
   - rms = sqrt(mean(x², dim=-1) + eps)
   - y = x / rms * weight
3. **Backward**: Chain rule through normalization

### Performance Constraints
- Single pass for mean+variance using Welford's algorithm
- Numerically stable variance computation

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_13_layernorm_mean` | LayerNorm output | mean ≈ beta |
| `test_phase_13_layernorm_std` | LayerNorm output | std ≈ gamma |
| `test_phase_13_layernorm_backward` | Backward pass | Correct gradients |
| `test_phase_13_rmsnorm` | RMSNorm output | Scaled by weight |
| `test_phase_13_eps_stability` | Very small variance | No NaN |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_13" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/normalization.hpp`
- [ ] LayerNorm output has zero mean, unit variance

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 250-400 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses LayerNorm (not RMSNorm)
- Weight initialized to 1, bias to 0
- Normalize over last dimension (hidden_size)
- eps prevents division by zero
