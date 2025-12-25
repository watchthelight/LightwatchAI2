# Phase 04: SIMD Operations - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_simd_add` | `A[1024], B[1024]` | Element-wise sum matches scalar |
| `test_simd_mul` | `A[1024], B[1024]` | Element-wise product matches scalar |
| `test_simd_dot` | `A[1024], B[1024]` | Dot product matches scalar (tol 1e-5) |
| `test_simd_matmul` | `A[64,64] @ B[64,64]` | Matches scalar implementation |
| `test_simd_exp` | `A[1024]` in range [-5,5] | Matches std::exp (tol 1e-5) |
| `test_simd_alignment` | Unaligned data | No crash, correct results |

## Implementation Notes

- Compare SIMD results against scalar reference implementation
- Test both aligned and unaligned memory access
- Tolerance for transcendental functions (exp, log) may be looser
- Test edge cases: empty arrays, single element, non-power-of-2 sizes
