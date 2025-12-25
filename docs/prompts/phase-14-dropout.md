# Phase 14: Dropout

## Objective
Implement Dropout and DropPath regularization layers with training/eval mode awareness.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11 | Module base class with train()/eval() |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable |
| 11 | include/lightwatch/nn/linear.hpp | Module |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/dropout.hpp | Dropout, DropPath | Phase 15, 17, 31 |
| src/nn/dropout.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/module.hpp
namespace lightwatch::nn {

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1);

    autograd::Variable forward(const autograd::Variable& input) override;

private:
    float p_;  // Drop probability
};

class DropPath : public Module {  // Stochastic Depth
public:
    explicit DropPath(float p = 0.1);

    autograd::Variable forward(const autograd::Variable& input) override;

private:
    float p_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Dropout**:
   - Training: mask = random < (1-p), output = input * mask / (1-p)
   - Eval: output = input (no scaling needed due to inverted dropout)
2. **DropPath**:
   - Training: Drop entire sample with probability p
   - Eval: output = input
3. **RNG**: Use thread-local std::mt19937

### Performance Constraints
- O(n) for n elements
- Inverted dropout (scale during training, not inference)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_14_dropout_train` | p=0.5, training=true | ~50% zeros |
| `test_phase_14_dropout_eval` | p=0.5, training=false | No zeros |
| `test_phase_14_dropout_scale` | Inverted scaling | Mean preserved |
| `test_phase_14_droppath` | p=0.2, batch of 100 | ~20% dropped |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_14" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/dropout.hpp`
- [ ] Dropout respects training/eval mode

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 150-250 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 uses dropout p=0.1 in attention and FFN
- Inverted dropout scales by 1/(1-p) during training
- DropPath is for residual connections (drop entire residual)
- Seed should be configurable for reproducibility
