# Phase 03: Tensor Core - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 12

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_tensor_construction` | `Shape{2,3,4}` | `numel()==24`, `ndim()==3` |
| `test_tensor_zeros` | `Shape{3,3}` | All elements == 0.0 |
| `test_tensor_ones` | `Shape{2,2}` | All elements == 1.0 |
| `test_tensor_randn` | `Shape{100}` | Mean ≈ 0.0 (±0.3), Std ≈ 1.0 (±0.3) |
| `test_tensor_matmul_2d` | `A[2,3] @ B[3,4]` | Result shape `[2,4]`, values correct |
| `test_tensor_matmul_batch` | `A[5,2,3] @ B[5,3,4]` | Result shape `[5,2,4]` |
| `test_tensor_broadcast_add` | `A[2,3] + B[3]` | Result shape `[2,3]`, values correct |
| `test_tensor_broadcast_mul` | `A[2,3,4] * B[1,4]` | Result shape `[2,3,4]` |
| `test_tensor_slice` | `T[10,20].slice(0,2,5)` | Result shape `[3,20]` |
| `test_tensor_transpose` | `T[2,3].transpose(0,1)` | Result shape `[3,2]` |
| `test_tensor_contiguous` | Non-contiguous slice | `is_contiguous()==true` after `.contiguous()` |
| `test_tensor_reduction_sum` | `T[2,3].sum(1)` | Result shape `[2]`, values correct |

## Implementation Notes

- All tests should use tolerance of 1e-6 for floating point comparisons
- Random tests should use fixed seed for reproducibility
- Memory layout is row-major (C-style)
