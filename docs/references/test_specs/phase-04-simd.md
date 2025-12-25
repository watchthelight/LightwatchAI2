<!-- File: docs/references/test_specs/phase-04-simd.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 04: SIMD Operations - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_04_simd_dispatch` | Runtime CPU detection | Returns AVX2, SSE4.2, or SCALAR based on CPU features |
| `test_phase_04_simd_dot_product` | Two 1024-element aligned float vectors | Result matches scalar implementation (tol 1e-5) |
| `test_phase_04_simd_unaligned` | 1023-element vectors (not 32-byte aligned) | Correct result, no crash |
| `test_phase_04_simd_matmul_small` | 4x4 @ 4x4 matrices | Result matches naive impl (tol 1e-5) |
| `test_phase_04_simd_matmul_large` | 512x512 @ 512x512 matrices | Result matches naive impl (tol 1e-4), completes in <1s |
| `test_phase_04_simd_element_wise` | 1000-element vectors, add/mul/sub | Results match scalar (tol 1e-6) |

## Implementation Notes

- AVX2 requires 32-byte alignment for optimal performance
- Include scalar fallback for non-SIMD platforms
- Test on both aligned and unaligned memory
- Benchmark should show 4-8x speedup over scalar for large operations
