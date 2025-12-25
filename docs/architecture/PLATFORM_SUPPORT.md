# Platform Support

## Overview

This document specifies the supported platforms, architectures, and SIMD strategies for LightwatchAI2.

## Supported Architectures

| Architecture | Status | SIMD | Notes |
|--------------|--------|------|-------|
| x86-64 (AMD64) | **Primary** | AVX2, SSE4.2 | Full support, primary development target |
| ARM64 (AArch64) | Secondary | NEON | macOS Apple Silicon, Linux ARM servers |
| x86 (32-bit) | Unsupported | - | Memory constraints make GPT-2 impractical |

## Supported Operating Systems

| OS | Status | Tested Versions | Notes |
|----|--------|-----------------|-------|
| Linux | **Primary** | Ubuntu 22.04+, Debian 12+ | Primary CI target |
| macOS | Supported | 13 (Ventura)+ | Intel and Apple Silicon |
| Windows | Supported | Windows 10+ | MSVC 2022, MinGW-w64 |
| FreeBSD | Untested | - | Should work, not CI tested |

## SIMD Strategy

### Detection Hierarchy

Runtime detection determines the best available SIMD implementation:

```cpp
enum class SimdLevel {
    SCALAR,     // Fallback, no SIMD
    SSE42,      // x86-64 baseline
    AVX2,       // Modern x86-64 (Intel Haswell+, AMD Zen+)
    AVX512,     // High-end x86-64 (reserved for future)
    NEON,       // ARM64
};

SimdLevel detect_simd_level();
```

### Compile-Time vs Runtime

| Approach | Usage |
|----------|-------|
| Compile-time | `-march=native` for maximum performance on build machine |
| Runtime | Portable binaries with runtime dispatch |

Default: **Runtime dispatch** for distributed binaries.

### x86-64 SIMD

#### AVX2 (Primary)

Required CPU features:
- AVX2
- FMA3
- BMI1, BMI2

Supported CPUs:
- Intel: Haswell (2013) and later
- AMD: Excavator (2015), Zen (2017) and later

```cpp
// Example: Vectorized dot product
float simd_dot_avx2(const float* a, const float* b, size_t n);
```

#### SSE4.2 (Fallback)

Fallback for older x86-64 CPUs without AVX2.

Supported CPUs:
- Intel: Nehalem (2008) and later
- AMD: Bulldozer (2011) and later

```cpp
float simd_dot_sse42(const float* a, const float* b, size_t n);
```

### ARM64 SIMD (NEON)

NEON is mandatory on all ARM64 processors, no runtime detection needed.

```cpp
float simd_dot_neon(const float* a, const float* b, size_t n);
```

### Scalar Fallback

Always available. Used when:
- SIMD detection fails
- `LIGHTWATCH_NO_SIMD=1` environment variable set
- Testing correctness

```cpp
float simd_dot_scalar(const float* a, const float* b, size_t n);
```

## Compiler Requirements

### Minimum Versions

| Compiler | Minimum | Recommended | C++ Standard |
|----------|---------|-------------|--------------|
| GCC | 10.0 | 13+ | C++17 |
| Clang | 12.0 | 17+ | C++17 |
| MSVC | 19.29 (VS 2019 16.10) | 19.38+ (VS 2022 17.8) | C++17 |
| Apple Clang | 14.0 | 15+ | C++17 |

### Compiler-Specific Notes

#### GCC

```bash
# Recommended flags
-std=c++17 -O3 -march=native -Wall -Wextra

# For portable binary
-std=c++17 -O3 -mavx2 -mfma -Wall -Wextra
```

#### Clang

```bash
# Recommended flags
-std=c++17 -O3 -march=native -Wall -Wextra

# macOS with Apple Silicon
-std=c++17 -O3 -mcpu=apple-m1 -Wall -Wextra
```

#### MSVC

```bash
# Recommended flags
/std:c++17 /O2 /arch:AVX2 /W4

# For portable binary
/std:c++17 /O2 /W4  # Runtime dispatch for SIMD
```

## Memory Alignment

| Requirement | Value | Reason |
|-------------|-------|--------|
| Tensor data | 64 bytes | AVX-512 future-proofing, cache line alignment |
| SIMD vectors | 32 bytes | AVX2 requirement |
| General heap | 16 bytes | Standard malloc guarantee |

```cpp
// Aligned allocation
void* aligned_alloc(size_t alignment, size_t size);

// Portable aligned allocation wrapper
template<typename T>
T* alloc_aligned(size_t count, size_t alignment = 64);
```

## Platform-Specific Considerations

### Linux

- **Memory mapping:** Use `mmap()` for large weight files
- **Huge pages:** Optional support via `madvise(MADV_HUGEPAGE)`
- **Thread affinity:** Optional CPU pinning for benchmarks

### macOS

- **No valgrind:** Use AddressSanitizer instead for memory checks
- **Apple Silicon:** Native ARM64 binary, no Rosetta 2
- **Memory:** Unified memory architecture, no explicit NUMA

```bash
# macOS memory check alternative
cmake -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON ..
```

### Windows

- **Large pages:** Requires `SeLockMemoryPrivilege`
- **Path separators:** Use `std::filesystem::path` for portability
- **Unicode:** UTF-8 manifest for proper path handling

```cpp
// Windows-specific
#ifdef _WIN32
    #include <windows.h>
    // Use VirtualAlloc for large aligned allocations
#endif
```

## Build Configuration

### CMake Presets

```json
{
    "configurePresets": [
        {
            "name": "linux-release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CXX_FLAGS": "-march=native"
            }
        },
        {
            "name": "macos-arm64",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_OSX_ARCHITECTURES": "arm64"
            }
        },
        {
            "name": "windows-release",
            "generator": "Visual Studio 17 2022",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        }
    ]
}
```

### Feature Detection CMake

```cmake
# Detect SIMD support
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    set(LIGHTWATCH_SIMD_AVX2 ON)
endif()

# ARM NEON (always available on ARM64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LIGHTWATCH_SIMD_NEON ON)
endif()
```

## Testing Matrix

CI should test these configurations:

| OS | Arch | Compiler | SIMD | Priority |
|----|------|----------|------|----------|
| Ubuntu 22.04 | x86-64 | GCC 12 | AVX2 | P0 |
| Ubuntu 22.04 | x86-64 | Clang 15 | AVX2 | P1 |
| macOS 14 | ARM64 | Apple Clang | NEON | P0 |
| macOS 14 | x86-64 | Apple Clang | AVX2 | P2 |
| Windows 11 | x86-64 | MSVC 2022 | AVX2 | P1 |

## Performance Expectations

| Platform | Expected tok/s | Notes |
|----------|---------------|-------|
| x86-64 AVX2 | 50-100 | Intel 8th gen+, AMD Zen 2+ |
| x86-64 SSE4.2 | 20-40 | Older CPUs |
| ARM64 NEON | 40-80 | Apple M1/M2, Ampere |
| Scalar | 5-15 | Fallback only |

## Known Limitations

1. **No GPU support:** CPU-only for simplicity
2. **No distributed:** Single-machine only
3. **No quantization:** fp32 only (no int8/fp16)
4. **32-bit unsupported:** Memory requirements too high
