// Phase 36: KV-Cache
// Key-value caching for efficient incremental generation

#pragma once

#include <lightwatch/tensor.hpp>
#include <vector>
#include <utility>
#include <stdexcept>

namespace lightwatch {

// Key-Value cache for transformer layers
class KVCache {
public:
    KVCache(size_t num_layers, size_t num_heads, size_t head_dim,
            size_t max_seq_len, size_t batch_size = 1)
        : num_layers_(num_layers)
        , num_heads_(num_heads)
        , head_dim_(head_dim)
        , max_seq_len_(max_seq_len)
        , batch_size_(batch_size)
        , current_len_(0)
    {
        // Pre-allocate cache tensors for each layer
        // Shape: {batch, num_heads, max_seq_len, head_dim}
        for (size_t i = 0; i < num_layers; ++i) {
            k_cache_.push_back(Tensor<float>::zeros({batch_size, num_heads, max_seq_len, head_dim}));
            v_cache_.push_back(Tensor<float>::zeros({batch_size, num_heads, max_seq_len, head_dim}));
        }
    }

    // Get cached K, V for a layer (returns view up to current_len)
    std::pair<Tensor<float>, Tensor<float>> get(size_t layer) const {
        if (layer >= num_layers_) {
            throw std::out_of_range("Layer index out of range");
        }

        if (current_len_ == 0) {
            // Return empty tensors
            return {
                Tensor<float>({batch_size_, num_heads_, 0, head_dim_}),
                Tensor<float>({batch_size_, num_heads_, 0, head_dim_})
            };
        }

        // Create views of the cached data up to current_len
        Tensor<float> k_view({batch_size_, num_heads_, current_len_, head_dim_});
        Tensor<float> v_view({batch_size_, num_heads_, current_len_, head_dim_});

        const float* k_src = k_cache_[layer].data();
        const float* v_src = v_cache_[layer].data();
        float* k_dst = k_view.data();
        float* v_dst = v_view.data();

        // Copy only the valid portion
        size_t stride = max_seq_len_ * head_dim_;
        size_t copy_size = current_len_ * head_dim_;

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
                size_t src_offset = (b * num_heads_ + h) * stride;
                size_t dst_offset = (b * num_heads_ + h) * copy_size;

                for (size_t i = 0; i < copy_size; ++i) {
                    k_dst[dst_offset + i] = k_src[src_offset + i];
                    v_dst[dst_offset + i] = v_src[src_offset + i];
                }
            }
        }

        return {k_view, v_view};
    }

    // Update cache with new K, V (appends to existing cache)
    void update(size_t layer, const Tensor<float>& new_k, const Tensor<float>& new_v) {
        if (layer >= num_layers_) {
            throw std::out_of_range("Layer index out of range");
        }

        const auto& k_shape = new_k.shape();
        const auto& v_shape = new_v.shape();

        // Expected shape: {batch, num_heads, new_seq_len, head_dim}
        if (k_shape.size() != 4 || v_shape.size() != 4) {
            throw std::invalid_argument("K/V must be 4D tensors");
        }

        size_t new_seq_len = k_shape[2];
        if (current_len_ + new_seq_len > max_seq_len_) {
            throw std::runtime_error("Cache overflow: exceeds max_seq_len");
        }

        // Copy new K, V into cache at current position
        float* k_dst = k_cache_[layer].data();
        float* v_dst = v_cache_[layer].data();
        const float* k_src = new_k.data();
        const float* v_src = new_v.data();

        size_t stride = max_seq_len_ * head_dim_;

        for (size_t b = 0; b < batch_size_; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
                size_t cache_offset = (b * num_heads_ + h) * stride + current_len_ * head_dim_;
                size_t src_offset = (b * num_heads_ + h) * new_seq_len * head_dim_;

                for (size_t s = 0; s < new_seq_len; ++s) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                        k_dst[cache_offset + s * head_dim_ + d] =
                            k_src[src_offset + s * head_dim_ + d];
                        v_dst[cache_offset + s * head_dim_ + d] =
                            v_src[src_offset + s * head_dim_ + d];
                    }
                }
            }
        }

        // Only update current_len once (all layers share the same sequence position)
        // This is handled by the caller after updating all layers
    }

    // Advance the sequence position (call after updating all layers)
    void advance(size_t new_tokens) {
        current_len_ += new_tokens;
    }

    // Current sequence length in cache
    size_t seq_len() const { return current_len_; }

    // Maximum sequence length
    size_t max_seq_len() const { return max_seq_len_; }

    // Number of layers
    size_t num_layers() const { return num_layers_; }

    // Clear cache for new generation
    void reset() {
        current_len_ = 0;
        // Optionally zero out the tensors (not strictly necessary)
        for (auto& k : k_cache_) {
            std::fill(k.data(), k.data() + k.numel(), 0.0f);
        }
        for (auto& v : v_cache_) {
            std::fill(v.data(), v.data() + v.numel(), 0.0f);
        }
    }

    // Get cache size in bytes
    size_t memory_bytes() const {
        return 2 * num_layers_ * batch_size_ * num_heads_ * max_seq_len_ * head_dim_ * sizeof(float);
    }

    // Accessors
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t batch_size() const { return batch_size_; }

private:
    size_t num_layers_;
    size_t num_heads_;
    size_t head_dim_;
    size_t max_seq_len_;
    size_t batch_size_;
    size_t current_len_;

    std::vector<Tensor<float>> k_cache_;
    std::vector<Tensor<float>> v_cache_;
};

}  // namespace lightwatch
