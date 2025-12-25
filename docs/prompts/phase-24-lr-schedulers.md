# Phase 24: LR Schedulers

## Objective
Implement learning rate schedulers: cosine annealing, warmup, and step decay.

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
| include/lightwatch/optim/scheduler.hpp | LRScheduler, CosineAnnealingLR, WarmupLR | Phase 29 |
| src/optim/scheduler.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Defined in docs/contracts/optimizer.hpp
namespace lightwatch::optim {

class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;

    virtual void step() = 0;
    float get_last_lr() const;

protected:
    Optimizer& optimizer_;
    int step_count_ = 0;
    float last_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0);
    void step() override;

private:
    int T_max_;
    float eta_min_;
    float base_lr_;
};

class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer, int warmup_steps, float start_factor = 0.0);
    void step() override;

private:
    int warmup_steps_;
    float start_factor_;
    float base_lr_;
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int step_size, float gamma = 0.1);
    void step() override;

private:
    int step_size_;
    float gamma_;
};

}  // namespace lightwatch::optim
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **CosineAnnealing**: lr = eta_min + (base_lr - eta_min) * (1 + cos(π * t / T_max)) / 2
2. **Warmup**: lr = start_factor * base_lr + (1 - start_factor) * base_lr * t / warmup_steps
3. **StepLR**: lr = base_lr * gamma^(t // step_size)
4. **Chained schedulers**: Support sequential warmup + cosine

### Performance Constraints
- O(1) per step
- No memory allocation during step

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_24_cosine` | T_max=100, 50 steps | LR at half cycle |
| `test_phase_24_cosine_end` | T_max=100, 100 steps | LR = eta_min |
| `test_phase_24_warmup` | 10 warmup steps | Linear increase |
| `test_phase_24_step` | step_size=10, gamma=0.1 | LR drops at 10, 20, ... |
| `test_phase_24_chained` | Warmup then cosine | Smooth transition |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_24" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/optim/scheduler.hpp`
- [ ] Cosine schedule matches reference

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 200-350 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- Warmup typically 5-10% of training steps
- Cosine annealing helps with convergence
- Scheduler.step() called once per batch or epoch
- get_last_lr() returns current learning rate
