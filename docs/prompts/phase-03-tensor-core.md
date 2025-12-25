# Phase 03: Tensor Core

## Objective
Implement the core Tensor<T> class with shape management, element access, basic operations, and memory-efficient views.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 02 | include/lightwatch/memory/aligned.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 02 | include/lightwatch/memory/aligned.hpp | aligned_alloc, aligned_free |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/tensor.hpp | Shape, Tensor<T>, matmul, concat, stack | 04, 05, 08, 09, 11-19, 21-25, 31-36 |
| src/tensor.cpp | Implementation | N/A |

## Specification

### Data Structures
See `docs/contracts/tensor.hpp` for the complete API contract.

Key implementation details:
- Row-major memory layout
- Shared data via std::shared_ptr<T[]> for views
- Strides array for non-contiguous tensors
- Offset for sliced views

### Function Signatures
All signatures in `docs/contracts/tensor.hpp`.

### Algorithmic Requirements
1. **Shape broadcasting**: Support NumPy-style broadcasting for element-wise ops
2. **Strided access**: Support non-contiguous memory via strides
3. **View semantics**: reshape/view share data when possible, copy otherwise
4. **Matmul**: Implement naive O(n³) for correctness (optimized in Phase 04)
5. **Reduction ops**: Iterate along specified dimension
6. **Random generation**: Use std::mt19937 with configurable seed

### Performance Constraints
- Element access: O(1) with index computation
- Shape operations: O(1) for metadata, O(n) for copies
- Matmul: O(n³) (will be optimized in Phase 04)

## Required Tests
See `docs/test_specs/phase-03-tensor.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_03_tensor_create` | Shape {2,3} | numel()==6, ndim()==2 |
| `test_phase_03_tensor_zeros` | zeros({3,4}) | All elements 0.0 |
| `test_phase_03_tensor_ones` | ones({2,2}) | All elements 1.0 |
| `test_phase_03_tensor_element_access` | t({0,1}) = 5.0 | t.at({0,1}) == 5.0 |
| `test_phase_03_tensor_reshape` | {2,3} -> {3,2} | Same data, new shape |
| `test_phase_03_tensor_transpose` | {2,3}.transpose(0,1) | Shape {3,2} |
| `test_phase_03_tensor_slice` | {4,3}.slice(0,1,3) | Shape {2,3} |
| `test_phase_03_tensor_add` | ones + ones | All elements 2.0 |
| `test_phase_03_tensor_mul` | 2*ones | All elements 2.0 |
| `test_phase_03_tensor_matmul_2d` | {2,3} @ {3,4} | Shape {2,4} |
| `test_phase_03_tensor_sum` | ones({2,3}).sum(1) | Shape {2}, values 3.0 |
| `test_phase_03_tensor_broadcast` | {2,1} + {1,3} | Shape {2,3} |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_03" --output-on-failure` exits 0
- [ ] `grep -q "class Tensor" include/lightwatch/tensor.hpp`
- [ ] All 12 required tests pass

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 1500-2500 |
| New source files | 6 |
| New test files | 2 |
| Complexity | HIGH |

## Notes
- Template instantiation for float primarily; bool for comparisons
- Negative dimension indices count from end (Python-style)
- Contiguous check: stride[i] == stride[i+1] * shape[i+1] for all i
- Broadcasting rules: dimensions aligned from right, 1s broadcast
