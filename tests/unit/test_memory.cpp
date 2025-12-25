// Phase 02: Memory Management Tests

#include <lightwatch/memory/arena.hpp>
#include <lightwatch/memory/pool.hpp>
#include <lightwatch/memory/aligned.hpp>
#include <iostream>
#include <cstdint>
#include <cassert>

using namespace lightwatch::memory;

// Test arena basic allocation
bool test_phase_02_arena_basic() {
    Arena arena(16 * 1024);  // 16KB arena

    // Allocate 1KB, 2KB, 4KB
    void* p1 = arena.allocate(1024);
    void* p2 = arena.allocate(2048);
    void* p3 = arena.allocate(4096);

    if (!p1 || !p2 || !p3) {
        std::cerr << "arena_basic: allocation failed" << std::endl;
        return false;
    }

    // Check pointers are within arena and non-overlapping
    if (p2 <= p1 || p3 <= p2) {
        std::cerr << "arena_basic: pointers not sequential" << std::endl;
        return false;
    }

    // Check used space is at least 1K + 2K + 4K = 7K
    if (arena.used() < 7168) {
        std::cerr << "arena_basic: used space too small: " << arena.used() << std::endl;
        return false;
    }

    std::cout << "test_phase_02_arena_basic: PASSED" << std::endl;
    return true;
}

// Test arena reset
bool test_phase_02_arena_reset() {
    Arena arena(4096);

    void* p1 = arena.allocate(1024);
    (void)arena.used();  // Just verify it works

    arena.reset();

    if (arena.used() != 0) {
        std::cerr << "arena_reset: used not 0 after reset" << std::endl;
        return false;
    }

    void* p2 = arena.allocate(1024);

    // After reset, new allocation should start from beginning
    if (p2 != p1) {
        std::cerr << "arena_reset: memory not reused" << std::endl;
        return false;
    }

    std::cout << "test_phase_02_arena_reset: PASSED" << std::endl;
    return true;
}

// Test pool alloc/dealloc
bool test_phase_02_pool_alloc_dealloc() {
    struct TestNode {
        int value;
        TestNode* next;
    };

    Pool<TestNode> pool(128);

    // Allocate 100 nodes
    std::vector<TestNode*> nodes;
    for (int i = 0; i < 100; ++i) {
        TestNode* node = pool.allocate();
        if (!node) {
            std::cerr << "pool_alloc_dealloc: allocation " << i << " failed" << std::endl;
            return false;
        }
        node->value = i;
        nodes.push_back(node);
    }

    if (pool.size() != 100) {
        std::cerr << "pool_alloc_dealloc: size not 100: " << pool.size() << std::endl;
        return false;
    }

    // Deallocate 50 nodes
    for (size_t i = 0; i < 50; ++i) {
        pool.deallocate(nodes[i]);
    }

    if (pool.size() != 50) {
        std::cerr << "pool_alloc_dealloc: size not 50 after dealloc: " << pool.size() << std::endl;
        return false;
    }

    // Allocate 50 more nodes
    for (int i = 0; i < 50; ++i) {
        TestNode* node = pool.allocate();
        if (!node) {
            std::cerr << "pool_alloc_dealloc: reallocation " << i << " failed" << std::endl;
            return false;
        }
    }

    if (pool.size() != 100) {
        std::cerr << "pool_alloc_dealloc: size not 100 after realloc: " << pool.size() << std::endl;
        return false;
    }

    std::cout << "test_phase_02_pool_alloc_dealloc: PASSED" << std::endl;
    return true;
}

// Test 64-byte aligned allocation
bool test_phase_02_aligned_64() {
    void* ptr = alloc_aligned(1024, 64);
    if (!ptr) {
        std::cerr << "aligned_64: allocation failed" << std::endl;
        return false;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % 64 != 0) {
        std::cerr << "aligned_64: address " << addr << " not 64-byte aligned" << std::endl;
        free_aligned(ptr);
        return false;
    }

    free_aligned(ptr);
    std::cout << "test_phase_02_aligned_64: PASSED" << std::endl;
    return true;
}

// Test 32-byte aligned allocation (AVX requirement)
bool test_phase_02_aligned_avx() {
    void* ptr = alloc_aligned(256, 32);
    if (!ptr) {
        std::cerr << "aligned_avx: allocation failed" << std::endl;
        return false;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr % 32 != 0) {
        std::cerr << "aligned_avx: address " << addr << " not 32-byte aligned" << std::endl;
        free_aligned(ptr);
        return false;
    }

    free_aligned(ptr);
    std::cout << "test_phase_02_aligned_avx: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_02_arena_basic()) ++failures;
    if (!test_phase_02_arena_reset()) ++failures;
    if (!test_phase_02_pool_alloc_dealloc()) ++failures;
    if (!test_phase_02_aligned_64()) ++failures;
    if (!test_phase_02_aligned_avx()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 02 tests passed" << std::endl;
    return 0;
}
