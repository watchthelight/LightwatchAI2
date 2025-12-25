# Phase 02: Memory Management

## Objective
Implement custom memory allocators (arena, pool) and aligned allocation utilities for efficient tensor memory management.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 01 | CMakeLists.txt, include/lightwatch/config.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 01 | include/lightwatch/config.hpp | Platform macros |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/memory/arena.hpp | Arena | Phase 03, 06 |
| include/lightwatch/memory/pool.hpp | Pool<T> | Phase 03 |
| include/lightwatch/memory/aligned.hpp | aligned_alloc, aligned_free | Phase 03, 04 |

## Specification

### Data Structures
```cpp
// include/lightwatch/memory/arena.hpp
namespace lightwatch::memory {

class Arena {
public:
    explicit Arena(size_t capacity);
    ~Arena();

    void* allocate(size_t size, size_t alignment = 16);
    void reset();  // Resets to beginning, doesn't free

    size_t used() const;
    size_t capacity() const;

private:
    char* data_;
    size_t capacity_;
    size_t offset_;
};

}  // namespace lightwatch::memory

// include/lightwatch/memory/pool.hpp
namespace lightwatch::memory {

template<typename T>
class Pool {
public:
    explicit Pool(size_t initial_capacity = 1024);
    ~Pool();

    T* allocate();
    void deallocate(T* ptr);
    void clear();

    size_t size() const;      // Active allocations
    size_t capacity() const;  // Total slots

private:
    std::vector<T*> free_list_;
    std::vector<std::unique_ptr<T[]>> chunks_;
    size_t chunk_size_;
};

}  // namespace lightwatch::memory

// include/lightwatch/memory/aligned.hpp
namespace lightwatch::memory {

void* aligned_alloc(size_t size, size_t alignment);
void aligned_free(void* ptr);

template<typename T>
T* aligned_new(size_t count, size_t alignment = alignof(T));

template<typename T>
void aligned_delete(T* ptr, size_t count);

}  // namespace lightwatch::memory
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. Arena allocator uses bump pointer allocation
2. Pool allocator maintains free list for O(1) alloc/dealloc
3. Aligned allocation uses platform-specific APIs:
   - POSIX: posix_memalign or aligned_alloc (C11)
   - Windows: _aligned_malloc / _aligned_free
4. All allocators are NOT thread-safe (single-threaded design)

### Performance Constraints
- Arena allocation: O(1) time
- Pool allocation: O(1) average time
- Aligned allocation: O(1) time
- No memory fragmentation in arena allocator

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_02_arena_basic` | Allocate 1KB, 2KB, 4KB | All succeed, offsets correct |
| `test_phase_02_arena_reset` | Allocate, reset, allocate | Reuses same memory |
| `test_phase_02_pool_alloc_dealloc` | Alloc 100, dealloc 50, alloc 50 | All succeed |
| `test_phase_02_aligned_64` | aligned_alloc(1024, 64) | Address % 64 == 0 |
| `test_phase_02_aligned_avx` | aligned_alloc(256, 32) | Address % 32 == 0 |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_02" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/memory/arena.hpp`
- [ ] `test -f include/lightwatch/memory/pool.hpp`
- [ ] `test -f include/lightwatch/memory/aligned.hpp`

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 4 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Thread safety is NOT required (single-threaded inference)
- Arena is useful for temporary allocations during forward pass
- Pool is useful for graph nodes in autograd
- AVX2 requires 32-byte alignment, AVX-512 requires 64-byte
