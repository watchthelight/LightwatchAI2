<!-- File: docs/references/test_specs/phase-03-tensor.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPEC FILE TEMPLATES -->

# Phase 03: Tensor Core - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 12

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_03_tensor_construction` | `Tensor<float> t({2,3,4})` | `t.numel()==24`, `t.ndim()==3`, `t.shape()=={2,3,4}` |
| `test_phase_03_tensor_zeros` | `Tensor<float>::zeros({3,3})` | All 9 elements == 0.0f |
| `test_phase_03_tensor_ones` | `Tensor<float>::ones({2,2})` | All 4 elements == 1.0f |
| `test_phase_03_tensor_randn` | `Tensor<float>::randn({1000})` | Mean ∈ [-0.1, 0.1], Std ∈ [0.9, 1.1] |
| `test_phase_03_tensor_matmul_2d` | `A{2,3}=[1..6], B{3,4}=[1..12]` | Result shape `{2,4}`, `C[0,0]==38.0f` |
| `test_phase_03_tensor_matmul_batch` | `A{2,2,3}, B{2,3,4}` randn | Result shape `{2,2,4}`, matches loop impl |
| `test_phase_03_tensor_broadcast_add` | `A{2,3}=[1..6] + B{3}=[10,20,30]` | `C[0,:]={11,22,33}`, `C[1,:]={14,25,36}` |
| `test_phase_03_tensor_broadcast_mul` | `A{2,1,4}=1.0 * B{3,1}=[2,3,4]` | Result shape `{2,3,4}`, all rows scaled |
| `test_phase_03_tensor_slice` | `T{10,20}.slice(0, 2, 5)` | Result shape `{3,20}`, `R[0,:]==T[2,:]` |
| `test_phase_03_tensor_transpose` | `T{2,3}=[1..6].transpose(0,1)` | Result shape `{3,2}`, `R[0,1]==T[1,0]==4` |
| `test_phase_03_tensor_contiguous` | `T{4,4}.slice(0,1,3)` (non-contig) | After `.contiguous()`: `is_contiguous()==true` |
| `test_phase_03_tensor_reduction_sum` | `T{2,3}=[1,2,3,4,5,6].sum(1)` | Result `{2}`, values `[6.0, 15.0]` |

## Implementation Notes

- All operations must handle edge cases: empty tensors, single-element tensors
- Broadcasting follows NumPy rules (right-align shapes, expand dims of size 1)
- Tolerance for floating-point comparisons: 1e-5
