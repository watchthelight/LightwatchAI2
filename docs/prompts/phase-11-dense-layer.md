# Phase 11: Dense Layer

## Objective
Implement the Linear (fully connected) layer with proper weight initialization and bias support.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 05 | include/lightwatch/autograd.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable, ops::matmul, ops::add |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/linear.hpp | Linear | Phase 12-19, 31 |
| src/nn/linear.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/module.hpp
namespace lightwatch::nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);

    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;  // Shape: {out_features, in_features}
    autograd::Variable bias;    // Shape: {out_features} or empty

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
```cpp
// Forward: output = input @ weight.T + bias
// input: {*, in_features}
// output: {*, out_features}
autograd::Variable forward(const autograd::Variable& input);
```

### Algorithmic Requirements
1. **Forward pass**: y = xW^T + b
2. **Weight initialization**: Xavier/Glorot uniform by default
3. **Bias initialization**: Zeros
4. **Shape handling**: Support batch dimensions (any number of leading dims)

### Performance Constraints
- Uses SIMD matmul from Phase 04
- Memory: weight + bias only (no extra buffers)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_11_linear_shape` | {2, 10} -> Linear(10, 5) | Output {2, 5} |
| `test_phase_11_linear_3d` | {4, 8, 10} -> Linear(10, 5) | Output {4, 8, 5} |
| `test_phase_11_linear_no_bias` | Linear(10, 5, false) | No bias parameter |
| `test_phase_11_linear_backward` | Forward + backward | Weight/bias grads |
| `test_phase_11_linear_params` | parameters() | Returns weight, bias |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_11" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/linear.hpp`
- [ ] Linear layer produces correct output shapes

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Weight layout: {out_features, in_features} for efficient output computation
- Xavier init: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
- Module base class provides parameter registration
