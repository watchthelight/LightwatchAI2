# Phase 04: SIMD Operations

## Objective
Implement SIMD-accelerated tensor operations using AVX2/SSE intrinsics with runtime dispatch for optimal performance.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 03 | include/lightwatch/tensor.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 03 | include/lightwatch/tensor.hpp | Tensor<T>, matmul |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/simd/dispatch.hpp | simd_matmul, simd_add, simd_mul | Phase 05 |
| include/lightwatch/simd/avx2.hpp | AVX2 implementations | Internal |
| include/lightwatch/simd/sse.hpp | SSE implementations | Internal |
| include/lightwatch/simd/scalar.hpp | Scalar fallback | Internal |

## Specification

### Data Structures
```cpp
// include/lightwatch/simd/dispatch.hpp
namespace lightwatch::simd {

enum class Backend {
    SCALAR,
    SSE,
    AVX2,
    AVX512
};

Backend detect_backend();
const char* backend_name(Backend b);

// Dispatch to fastest available implementation
void matmul(const float* a, const float* b, float* c,
            size_t M, size_t N, size_t K);

void add(const float* a, const float* b, float* c, size_t n);
void mul(const float* a, const float* b, float* c, size_t n);
void scale(const float* a, float scalar, float* c, size_t n);

// Element-wise operations
void exp(const float* a, float* c, size_t n);
void relu(const float* a, float* c, size_t n);
void gelu(const float* a, float* c, size_t n);

// Reductions
float sum(const float* a, size_t n);
float max(const float* a, size_t n);

}  // namespace lightwatch::simd
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Runtime dispatch**: Check CPUID at startup, select best backend
2. **AVX2 matmul**: Process 8 floats per instruction
3. **SSE matmul**: Process 4 floats per instruction
4. **Cache blocking**: Use tiled matmul for L1/L2 cache efficiency
5. **Memory alignment**: Require 32-byte alignment for AVX2, 16-byte for SSE

### Performance Constraints
- AVX2 matmul: >2x speedup over naive
- AVX2 element-wise: >4x speedup over scalar
- Must not crash on non-AVX2 CPUs (graceful fallback)

## Required Tests
See `docs/test_specs/phase-04-simd.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_04_detect_backend` | Call detect_backend() | Returns valid enum |
| `test_phase_04_simd_add` | SIMD add vs scalar | Results match within 1e-5 |
| `test_phase_04_simd_mul` | SIMD mul vs scalar | Results match within 1e-5 |
| `test_phase_04_simd_matmul` | SIMD matmul vs naive | Results match within 1e-4 |
| `test_phase_04_simd_exp` | SIMD exp vs std::exp | Results match within 1e-4 |
| `test_phase_04_simd_gelu` | SIMD GELU vs reference | Results match within 1e-4 |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_04" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/simd/dispatch.hpp`
- [ ] Backend detection works on current platform

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 800-1200 |
| New source files | 4 |
| New test files | 1 |
| Complexity | HIGH |

## Notes
- macOS on Apple Silicon: AVX not available, use scalar or Accelerate
- Cache blocking tile sizes: 32x32 for L1, 64x64 for L2 typical
- GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
- Use compiler intrinsics, not inline assembly for portability
