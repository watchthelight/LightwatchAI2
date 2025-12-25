#pragma once

// LightwatchAI2 Configuration Header
// Auto-generated platform and version detection

// Version information
#define LIGHTWATCH_VERSION_MAJOR 0
#define LIGHTWATCH_VERSION_MINOR 1
#define LIGHTWATCH_VERSION_PATCH 0
#define LIGHTWATCH_VERSION_STRING "0.1.0"

// SIMD feature detection
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

#if defined(__SSE4_1__)
    #define LIGHTWATCH_HAS_SSE4 1
#else
    #define LIGHTWATCH_HAS_SSE4 0
#endif

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define LIGHTWATCH_PLATFORM_WINDOWS 1
    #define LIGHTWATCH_PLATFORM_LINUX 0
    #define LIGHTWATCH_PLATFORM_MACOS 0
#elif defined(__linux__)
    #define LIGHTWATCH_PLATFORM_WINDOWS 0
    #define LIGHTWATCH_PLATFORM_LINUX 1
    #define LIGHTWATCH_PLATFORM_MACOS 0
#elif defined(__APPLE__)
    #define LIGHTWATCH_PLATFORM_WINDOWS 0
    #define LIGHTWATCH_PLATFORM_LINUX 0
    #define LIGHTWATCH_PLATFORM_MACOS 1
#else
    #define LIGHTWATCH_PLATFORM_WINDOWS 0
    #define LIGHTWATCH_PLATFORM_LINUX 0
    #define LIGHTWATCH_PLATFORM_MACOS 0
#endif

// Compiler detection
#if defined(__GNUC__) && !defined(__clang__)
    #define LIGHTWATCH_COMPILER_GCC 1
    #define LIGHTWATCH_COMPILER_CLANG 0
    #define LIGHTWATCH_COMPILER_MSVC 0
#elif defined(__clang__)
    #define LIGHTWATCH_COMPILER_GCC 0
    #define LIGHTWATCH_COMPILER_CLANG 1
    #define LIGHTWATCH_COMPILER_MSVC 0
#elif defined(_MSC_VER)
    #define LIGHTWATCH_COMPILER_GCC 0
    #define LIGHTWATCH_COMPILER_CLANG 0
    #define LIGHTWATCH_COMPILER_MSVC 1
#else
    #define LIGHTWATCH_COMPILER_GCC 0
    #define LIGHTWATCH_COMPILER_CLANG 0
    #define LIGHTWATCH_COMPILER_MSVC 0
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
    #define LIGHTWATCH_ARCH_X64 1
    #define LIGHTWATCH_ARCH_ARM64 0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define LIGHTWATCH_ARCH_X64 0
    #define LIGHTWATCH_ARCH_ARM64 1
#else
    #define LIGHTWATCH_ARCH_X64 0
    #define LIGHTWATCH_ARCH_ARM64 0
#endif
