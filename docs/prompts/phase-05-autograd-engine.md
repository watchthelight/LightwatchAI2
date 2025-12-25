# Phase 05: Autograd Engine

## Objective
Implement automatic differentiation with Variable wrapper, Function base class, and gradient computation through reverse-mode autodiff.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 03 | include/lightwatch/tensor.hpp |
| 04 | include/lightwatch/simd/dispatch.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 03 | include/lightwatch/tensor.hpp | Tensor<float>, matmul |
| 04 | include/lightwatch/simd/dispatch.hpp | simd_matmul, simd operations |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/autograd.hpp | Variable, Function, ops::* | 08, 11-19, 21-25, 31 |
| src/autograd/*.cpp | Implementation | N/A |

## Specification

### Data Structures
See `docs/contracts/autograd.hpp` for the complete API contract.

Key implementation details:
- Computation graph built dynamically during forward pass
- Gradients accumulated in Variable::grad_
- Topological sort for backward traversal
- NoGradGuard disables gradient tracking

### Function Signatures
All signatures in `docs/contracts/autograd.hpp`.

### Algorithmic Requirements
1. **Forward pass**: Operations create Function nodes, store in Variable::grad_fn_
2. **Backward pass**:
   - Start from output variable
   - Topological sort of computation graph
   - Propagate gradients through each Function::backward()
   - Accumulate gradients at leaf variables
3. **Gradient accumulation**: grad += incoming_grad (not replace)
4. **No-grad mode**: Thread-local flag, guard increments/decrements counter

### Performance Constraints
- Memory: Graph nodes must be freed after backward()
- Time: Backward should be ~2x forward time (typical for neural nets)

## Required Tests
See `docs/test_specs/phase-05-autograd.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_05_variable_create` | Variable(Tensor, true) | requires_grad() == true |
| `test_phase_05_add_backward` | y = a + b; y.backward() | a.grad, b.grad both 1.0 |
| `test_phase_05_mul_backward` | y = a * b; y.backward() | a.grad==b, b.grad==a |
| `test_phase_05_matmul_backward` | y = matmul(a, b) | Correct gradients |
| `test_phase_05_chain_rule` | y = a * b + c | All gradients correct |
| `test_phase_05_relu_backward` | y = relu(x) | Grad 1 where x>0, 0 else |
| `test_phase_05_softmax_backward` | y = softmax(x, -1) | Jacobian-vector product |
| `test_phase_05_no_grad` | NoGradGuard scope | No graph construction |
| `test_phase_05_detach` | y = x.detach() | y.grad_fn() == nullptr |
| `test_phase_05_memory_baseline` | 512x768 matmul | RSS < 50MB |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_05" --output-on-failure` exits 0
- [ ] `grep -q "class Variable" include/lightwatch/autograd.hpp`
- [ ] `grep -q "class Function" include/lightwatch/autograd.hpp`
- [ ] All 10 required tests pass

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 1000-1500 |
| New source files | 5 |
| New test files | 2 |
| Complexity | HIGH |

## Notes
- Each Function subclass: AddBackward, MulBackward, MatmulBackward, etc.
- save_for_backward stores tensors/variables needed for gradient computation
- retain_grad() allows non-leaf variables to keep gradients
- Memory baseline test ensures no accidental dependency on heavy libraries
