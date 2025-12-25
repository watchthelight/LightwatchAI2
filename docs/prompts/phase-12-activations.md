# Phase 12: Activations

## Objective
Implement activation functions: ReLU, GELU, SiLU (Swish), Sigmoid, Tanh, and Softmax with autograd support.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11 | include/lightwatch/nn/linear.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable, Function |
| 04 | include/lightwatch/simd/dispatch.hpp | SIMD activation ops |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/activations.hpp | ReLU, GELU, SiLU, Softmax | Phase 15, 17, 21 |
| src/nn/activations.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class ReLU : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override;
};

class GELU : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override;
};

class SiLU : public Module {  // Swish
public:
    autograd::Variable forward(const autograd::Variable& input) override;
};

class Sigmoid : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override;
};

class Tanh : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override;
};

class Softmax : public Module {
public:
    explicit Softmax(int dim = -1);
    autograd::Variable forward(const autograd::Variable& input) override;
private:
    int dim_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
All inherit from Module and override forward().

### Algorithmic Requirements
1. **ReLU**: max(0, x), grad = 1 if x > 0 else 0
2. **GELU**: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
3. **SiLU**: x * sigmoid(x)
4. **Sigmoid**: 1 / (1 + exp(-x))
5. **Tanh**: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
6. **Softmax**: exp(x - max) / sum(exp(x - max)) for numerical stability

### Performance Constraints
- Element-wise ops vectorized via SIMD
- Softmax: O(n) for n elements along dim

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_12_relu` | [-1, 0, 1] | [0, 0, 1] |
| `test_phase_12_relu_backward` | Backward through ReLU | Correct gradients |
| `test_phase_12_gelu` | [0, 1, 2] | GELU values |
| `test_phase_12_softmax` | [1, 2, 3] | Sums to 1.0 |
| `test_phase_12_softmax_stable` | [1000, 1001, 1002] | No overflow |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_12" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/activations.hpp`
- [ ] All activations match reference implementations

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 uses GELU activation in FFN
- Softmax numerical stability: subtract max before exp
- GELU approximation is standard (not exact erf version)
