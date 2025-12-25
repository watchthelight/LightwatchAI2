# Phase 01: Build System

## Objective
Establish the complete CMake build infrastructure with proper compiler detection, optimization flags, and test framework integration.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| (none) | Phase 01 has no dependencies |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| (none) | N/A | N/A |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| CMakeLists.txt | Build targets, compiler flags | All phases |
| cmake/LightwatchCompilerFlags.cmake | Compiler configuration | All phases |
| include/lightwatch/config.hpp | Version macros, platform detection | All phases |

## Specification

### Data Structures
```cpp
// include/lightwatch/config.hpp
#pragma once

#define LIGHTWATCH_VERSION_MAJOR 0
#define LIGHTWATCH_VERSION_MINOR 1
#define LIGHTWATCH_VERSION_PATCH 0

#if defined(__AVX2__)
    #define LIGHTWATCH_HAS_AVX2 1
#else
    #define LIGHTWATCH_HAS_AVX2 0
#endif

#if defined(__AVX512F__)
    #define LIGHTWATCH_HAS_AVX512 1
#else
    #define LIGHTWATCH_HAS_AVX512 0
#endif
```

### Function Signatures
N/A (build system phase)

### Algorithmic Requirements
1. Configure CMake with C++17 standard
2. Set up compiler warning flags for GCC/Clang/MSVC
3. Enable position-independent code for library targets
4. Configure output directories (bin/, lib/)
5. Set up FetchContent for nlohmann/json
6. Configure CTest integration
7. Add platform detection macros

### Performance Constraints
- CMake configure time < 30 seconds
- Incremental build time should not regress

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_01_cmake_configure` | `cmake -B build` | Exit code 0 |
| `test_phase_01_cmake_build` | `cmake --build build` | Exit code 0 |
| `test_phase_01_config_header` | Include config.hpp | Compiles successfully |

## Acceptance Criteria
- [ ] `cmake -B build -S .` exits 0
- [ ] `cmake --build build` exits 0
- [ ] `test -f include/lightwatch/config.hpp`
- [ ] `grep -q "LIGHTWATCH_VERSION" include/lightwatch/config.hpp`

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 100-200 |
| New source files | 3 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- Build system must work on Linux, macOS, and Windows
- FetchContent provides nlohmann/json without manual downloads
- Compiler flags should enable all warnings but not treat them as errors yet
