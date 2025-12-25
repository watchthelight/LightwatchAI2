#pragma once

#include <cstddef>
#include <vector>
#include <memory>

namespace lightwatch::memory {

template<typename T>
class Pool {
public:
    explicit Pool(size_t initial_capacity = 1024)
        : chunk_size_(initial_capacity), active_count_(0) {
        allocate_chunk();
    }

    ~Pool() = default;

    // Non-copyable
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    // Movable
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;

    T* allocate() {
        if (free_list_.empty()) {
            allocate_chunk();
        }

        T* ptr = free_list_.back();
        free_list_.pop_back();
        ++active_count_;
        return ptr;
    }

    void deallocate(T* ptr) {
        if (ptr) {
            free_list_.push_back(ptr);
            --active_count_;
        }
    }

    void clear() {
        // Return all allocations to free list
        free_list_.clear();
        for (auto& chunk : chunks_) {
            for (size_t i = 0; i < chunk_size_; ++i) {
                free_list_.push_back(&chunk[i]);
            }
        }
        active_count_ = 0;
    }

    size_t size() const {
        return active_count_;
    }

    size_t capacity() const {
        return chunks_.size() * chunk_size_;
    }

private:
    void allocate_chunk() {
        auto chunk = std::make_unique<T[]>(chunk_size_);
        T* base = chunk.get();

        // Add all slots to free list
        for (size_t i = 0; i < chunk_size_; ++i) {
            free_list_.push_back(base + i);
        }

        chunks_.push_back(std::move(chunk));
    }

    std::vector<T*> free_list_;
    std::vector<std::unique_ptr<T[]>> chunks_;
    size_t chunk_size_;
    size_t active_count_;
};

}  // namespace lightwatch::memory
