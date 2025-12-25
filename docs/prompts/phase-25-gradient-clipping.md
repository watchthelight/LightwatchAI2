# Phase 25: Gradient Clipping

## Objective
Implement gradient clipping by norm and by value for training stability.

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
| include/lightwatch/optim/clip.hpp | clip_grad_norm_, clip_grad_value_ | Phase 29 |
| src/optim/clip.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::optim {

// Clip gradients by global norm (modifies in-place)
// Returns the original total norm before clipping
float clip_grad_norm_(
    std::vector<autograd::Variable*>& params,
    float max_norm,
    float norm_type = 2.0);

// Clip gradients by value (modifies in-place)
void clip_grad_value_(
    std::vector<autograd::Variable*>& params,
    float clip_value);

// Compute gradient norm without clipping
float grad_norm(
    const std::vector<autograd::Variable*>& params,
    float norm_type = 2.0);

}  // namespace lightwatch::optim
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Clip by norm**:
   - total_norm = sqrt(sum(grad.pow(2)))
   - if total_norm > max_norm: grad *= max_norm / total_norm
2. **Clip by value**:
   - grad = clamp(grad, -clip_value, clip_value)
3. **Norm type**: Support L1 (norm_type=1), L2 (norm_type=2), Linf (norm_type=inf)

### Performance Constraints
- O(n) for n parameters
- In-place modification

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_25_clip_norm` | max_norm=1.0, actual=2.0 | Grads scaled by 0.5 |
| `test_phase_25_clip_norm_no_op` | max_norm=1.0, actual=0.5 | Grads unchanged |
| `test_phase_25_clip_value` | clip_value=0.1 | All grads in [-0.1, 0.1] |
| `test_phase_25_grad_norm` | Compute norm | Correct L2 norm |
| `test_phase_25_l1_norm` | norm_type=1 | Correct L1 norm |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_25" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/optim/clip.hpp`
- [ ] Gradient clipping prevents explosion

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 150-250 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 training typically uses max_norm=1.0
- Clip by norm is preferred (preserves gradient direction)
- Return original norm for logging/monitoring
- Apply after backward(), before optimizer.step()
