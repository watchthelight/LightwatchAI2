# Phase 23: Adam Optimizer

## Objective
Implement Adam and AdamW optimizers with first/second moment estimation.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 22 | Optimizer base class |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 22 | include/lightwatch/optim/optimizer.hpp | Optimizer |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/optim/adam.hpp | Adam, AdamW, AdamOptions | Phase 29 |
| src/optim/adam.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/optimizer.hpp
namespace lightwatch::optim {

struct AdamOptions : OptimizerOptions {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    bool amsgrad = false;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;

private:
    AdamOptions options_;
    int step_count_ = 0;
};

class AdamW : public Adam {
public:
    AdamW(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;
};

}  // namespace lightwatch::optim
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Adam update**:
   - m = β1 * m + (1 - β1) * g
   - v = β2 * v + (1 - β2) * g²
   - m_hat = m / (1 - β1^t)
   - v_hat = v / (1 - β2^t)
   - param -= lr * m_hat / (sqrt(v_hat) + eps)
2. **AdamW difference**: Decoupled weight decay
   - param -= lr * weight_decay * param (applied separately)
3. **AMSGrad**: v_max = max(v_max, v), use v_max instead of v

### Performance Constraints
- O(n) for n parameters
- Two state tensors (m, v) per parameter

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_23_adam_basic` | Default options | Converges on simple problem |
| `test_phase_23_adam_bias_correction` | Early steps | Corrected m_hat, v_hat |
| `test_phase_23_adamw_decay` | weight_decay=0.01 | Decoupled decay |
| `test_phase_23_adam_state` | Save/load state | Resumes correctly |
| `test_phase_23_amsgrad` | amsgrad=true | v_max tracked |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_23" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/optim/adam.hpp`
- [ ] Adam matches reference implementation

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 250-400 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 training typically uses AdamW
- Default betas (0.9, 0.999) work well for transformers
- Bias correction is critical for early training steps
- State dict must save m, v, step_count for checkpointing
