#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

namespace lightwatch::memory {

class Arena {
public:
    explicit Arena(size_t capacity)
        : capacity_(capacity), offset_(0) {
        if (capacity == 0) {
            throw std::invalid_argument("Arena capacity must be > 0");
        }
        data_ = static_cast<char*>(std::malloc(capacity));
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    ~Arena() {
        std::free(data_);
    }

    // Non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Movable
    Arena(Arena&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_), offset_(other.offset_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
        other.offset_ = 0;
    }

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            std::free(data_);
            data_ = other.data_;
            capacity_ = other.capacity_;
            offset_ = other.offset_;
            other.data_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    void* allocate(size_t size, size_t alignment = 16) {
        if (size == 0) {
            return nullptr;
        }

        // Align the current offset
        size_t aligned_offset = align_up(offset_, alignment);

        // Check if we have enough space
        if (aligned_offset + size > capacity_) {
            return nullptr;  // Out of memory
        }

        void* ptr = data_ + aligned_offset;
        offset_ = aligned_offset + size;
        return ptr;
    }

    void reset() {
        offset_ = 0;
    }

    size_t used() const {
        return offset_;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    static size_t align_up(size_t offset, size_t alignment) {
        return (offset + alignment - 1) & ~(alignment - 1);
    }

    char* data_;
    size_t capacity_;
    size_t offset_;
};

}  // namespace lightwatch::memory
