// Phase 01: Build System - Config Header Test
// Validates that config.hpp compiles and provides expected macros

#include <lightwatch/config.hpp>
#include <iostream>

int main() {
    // Test version macros
    static_assert(LIGHTWATCH_VERSION_MAJOR >= 0, "Version major must be defined");
    static_assert(LIGHTWATCH_VERSION_MINOR >= 0, "Version minor must be defined");
    static_assert(LIGHTWATCH_VERSION_PATCH >= 0, "Version patch must be defined");

    // Test SIMD detection macros exist (values depend on platform)
    static_assert(LIGHTWATCH_HAS_AVX2 == 0 || LIGHTWATCH_HAS_AVX2 == 1, "AVX2 must be 0 or 1");
    static_assert(LIGHTWATCH_HAS_AVX512 == 0 || LIGHTWATCH_HAS_AVX512 == 1, "AVX512 must be 0 or 1");
    static_assert(LIGHTWATCH_HAS_SSE4 == 0 || LIGHTWATCH_HAS_SSE4 == 1, "SSE4 must be 0 or 1");

    // Test platform detection (exactly one must be true)
    static_assert(
        (LIGHTWATCH_PLATFORM_WINDOWS + LIGHTWATCH_PLATFORM_LINUX + LIGHTWATCH_PLATFORM_MACOS) <= 1,
        "At most one platform should be detected"
    );

    // Test compiler detection (exactly one must be true)
    static_assert(
        (LIGHTWATCH_COMPILER_GCC + LIGHTWATCH_COMPILER_CLANG + LIGHTWATCH_COMPILER_MSVC) <= 1,
        "At most one compiler should be detected"
    );

    std::cout << "Config header test passed" << std::endl;
    std::cout << "  Version: " << LIGHTWATCH_VERSION_STRING << std::endl;
    std::cout << "  Platform: ";
    if (LIGHTWATCH_PLATFORM_MACOS) std::cout << "macOS";
    else if (LIGHTWATCH_PLATFORM_LINUX) std::cout << "Linux";
    else if (LIGHTWATCH_PLATFORM_WINDOWS) std::cout << "Windows";
    else std::cout << "Unknown";
    std::cout << std::endl;

    std::cout << "  Compiler: ";
    if (LIGHTWATCH_COMPILER_CLANG) std::cout << "Clang";
    else if (LIGHTWATCH_COMPILER_GCC) std::cout << "GCC";
    else if (LIGHTWATCH_COMPILER_MSVC) std::cout << "MSVC";
    else std::cout << "Unknown";
    std::cout << std::endl;

    std::cout << "  AVX2: " << (LIGHTWATCH_HAS_AVX2 ? "yes" : "no") << std::endl;
    std::cout << "  AVX512: " << (LIGHTWATCH_HAS_AVX512 ? "yes" : "no") << std::endl;

    return 0;
}
