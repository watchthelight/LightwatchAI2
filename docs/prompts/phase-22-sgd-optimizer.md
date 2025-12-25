# Phase 22: SGD Optimizer

## Objective
Implement SGD optimizer with momentum and weight decay support.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 05 | Autograd with Variable gradients |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/optim/optimizer.hpp | Optimizer, OptimizerOptions | Phase 23, 24 |
| include/lightwatch/optim/sgd.hpp | SGD, SGDOptions | Phase 29 |
| src/optim/sgd.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/optimizer.hpp
namespace lightwatch::optim {

struct OptimizerOptions {
    float lr = 1e-3;
    float weight_decay = 0.0;
};

class Optimizer {
public:
    explicit Optimizer(std::vector<autograd::Variable*> params,
                       OptimizerOptions options = {});
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();

    float get_lr() const;
    void set_lr(float lr);

protected:
    std::vector<ParamGroup> param_groups_;
    std::unordered_map<autograd::Variable*,
                       std::unordered_map<std::string, Tensor<float>>> state_;
};

struct SGDOptions : OptimizerOptions {
    float momentum = 0.0;
    bool nesterov = false;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<autograd::Variable*> params, SGDOptions options = {});
    void step() override;

private:
    SGDOptions options_;
};

}  // namespace lightwatch::optim
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Basic SGD**: param -= lr * grad
2. **Weight decay**: param -= lr * weight_decay * param
3. **Momentum**:
   - v = momentum * v + grad
   - param -= lr * v
4. **Nesterov**:
   - v = momentum * v + grad
   - param -= lr * (grad + momentum * v)

### Performance Constraints
- O(n) for n parameters
- In-place updates (no extra memory per step)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_22_sgd_basic` | lr=0.1, one step | param -= 0.1 * grad |
| `test_phase_22_sgd_momentum` | momentum=0.9 | Velocity accumulates |
| `test_phase_22_sgd_weight_decay` | wd=0.01 | Params shrink |
| `test_phase_22_sgd_nesterov` | nesterov=true | Lookahead update |
| `test_phase_22_zero_grad` | zero_grad() | All grads zero |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_22" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/optim/sgd.hpp`
- [ ] SGD with momentum matches reference

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 200-350 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Momentum buffer stored in optimizer state
- Weight decay applied to param, not grad (L2 regularization)
- zero_grad clears all parameter gradients
- Base Optimizer class used by all optimizers
