#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>
#include <lightwatch/config.hpp>

namespace lightwatch::memory {

inline void* alloc_aligned(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }

    // Ensure alignment is a power of 2
    if ((alignment & (alignment - 1)) != 0) {
        return nullptr;
    }

    // Ensure alignment is at least sizeof(void*)
    if (alignment < sizeof(void*)) {
        alignment = sizeof(void*);
    }

#if LIGHTWATCH_PLATFORM_WINDOWS
    return _aligned_malloc(size, alignment);
#else
    // POSIX systems
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

inline void free_aligned(void* ptr) {
    if (!ptr) {
        return;
    }

#if LIGHTWATCH_PLATFORM_WINDOWS
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

template<typename T>
T* aligned_new(size_t count, size_t alignment = alignof(T)) {
    if (count == 0) {
        return nullptr;
    }

    size_t size = count * sizeof(T);
    void* ptr = alloc_aligned(size, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }

    // Default-construct all elements
    T* typed_ptr = static_cast<T*>(ptr);
    for (size_t i = 0; i < count; ++i) {
        new (&typed_ptr[i]) T();
    }

    return typed_ptr;
}

template<typename T>
void aligned_delete(T* ptr, size_t count) {
    if (!ptr) {
        return;
    }

    // Destruct all elements
    for (size_t i = 0; i < count; ++i) {
        ptr[i].~T();
    }

    free_aligned(ptr);
}

}  // namespace lightwatch::memory
