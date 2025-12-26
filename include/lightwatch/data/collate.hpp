// Phase 28: Batch Processing and Collation

#pragma once

#include <lightwatch/data/dataset.hpp>
#include <lightwatch/tensor.hpp>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace lightwatch {
namespace data {

// Default pad token ID (GPT-2 uses eos_id = 50256 as pad)
constexpr int32_t DEFAULT_PAD_ID = 50256;
constexpr int32_t IGNORE_INDEX = -100;

// Extended Batch structure with metadata
struct BatchEx {
    Tensor<int32_t> input_ids;       // {batch_size, max_seq_len}
    Tensor<int32_t> labels;          // {batch_size, max_seq_len}
    Tensor<int32_t> attention_mask;  // {batch_size, max_seq_len} - 1D mask
    size_t batch_size;
    size_t max_seq_len;
};

// Pad a single sequence to target length
inline Tensor<int32_t> pad_sequence(
    const Tensor<int32_t>& seq,
    size_t target_length,
    int32_t pad_value = DEFAULT_PAD_ID) {

    size_t current_len = seq.numel();

    if (current_len >= target_length) {
        // Truncate if needed
        Tensor<int32_t> result({target_length});
        for (size_t i = 0; i < target_length; ++i) {
            result.data()[i] = seq.data()[i];
        }
        return result;
    }

    // Pad to target length (right padding)
    Tensor<int32_t> result({target_length});
    for (size_t i = 0; i < current_len; ++i) {
        result.data()[i] = seq.data()[i];
    }
    for (size_t i = current_len; i < target_length; ++i) {
        result.data()[i] = pad_value;
    }

    return result;
}

// Collate samples into a batch with configurable pad ID
inline BatchEx collate_fn(
    const std::vector<Sample>& samples,
    int32_t pad_id = DEFAULT_PAD_ID) {

    if (samples.empty()) {
        return {Tensor<int32_t>(), Tensor<int32_t>(), Tensor<int32_t>(), 0, 0};
    }

    // Find max sequence length
    size_t max_len = 0;
    for (const auto& s : samples) {
        max_len = std::max(max_len, s.input_ids.numel());
    }

    size_t batch_size = samples.size();

    BatchEx batch;
    batch.batch_size = batch_size;
    batch.max_seq_len = max_len;
    batch.input_ids = Tensor<int32_t>({batch_size, max_len});
    batch.labels = Tensor<int32_t>({batch_size, max_len});
    batch.attention_mask = Tensor<int32_t>({batch_size, max_len});

    // Initialize with padding values
    batch.input_ids.fill_(pad_id);
    batch.labels.fill_(IGNORE_INDEX);
    batch.attention_mask.fill_(0);

    // Copy each sample with right-padding
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& s = samples[b];
        size_t seq_len = s.input_ids.numel();

        for (size_t i = 0; i < seq_len; ++i) {
            batch.input_ids.data()[b * max_len + i] = s.input_ids.data()[i];
            batch.labels.data()[b * max_len + i] = s.labels.data()[i];
            batch.attention_mask.data()[b * max_len + i] = 1;  // Real token
        }
    }

    return batch;
}

// Create 2D causal attention mask from 1D padding mask
// Output shape: {batch_size, seq_len, seq_len}
// mask[b, i, j] = 1 if position j can attend to position i (i.e., j >= i and both are real tokens)
inline Tensor<float> create_causal_mask(
    const Tensor<int32_t>& attention_mask_1d,
    size_t seq_len) {

    size_t batch_size = attention_mask_1d.shape()[0];
    size_t mask_seq_len = attention_mask_1d.shape()[1];

    // Use provided seq_len or mask's seq_len
    if (seq_len == 0) {
        seq_len = mask_seq_len;
    }

    // Output: {batch_size, seq_len, seq_len}
    Tensor<float> mask({batch_size, seq_len, seq_len});
    mask.fill_(0.0f);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < seq_len; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                // Causal: j <= i (can only attend to current and previous positions)
                // Also check padding: both positions must be real tokens
                bool i_real = (i < mask_seq_len) &&
                              (attention_mask_1d.data()[b * mask_seq_len + i] == 1);
                bool j_real = (j < mask_seq_len) &&
                              (attention_mask_1d.data()[b * mask_seq_len + j] == 1);

                if (i_real && j_real) {
                    mask.data()[b * seq_len * seq_len + i * seq_len + j] = 1.0f;
                }
            }
        }
    }

    return mask;
}

// Create simple causal mask (no padding consideration)
// Returns lower triangular matrix: mask[i,j] = 1 if j <= i
inline Tensor<float> create_causal_mask(size_t seq_len) {
    Tensor<float> mask({seq_len, seq_len});
    mask.fill_(0.0f);

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            mask.data()[i * seq_len + j] = 1.0f;
        }
    }

    return mask;
}

// Convert 1D attention mask to additive mask for attention scores
// 0 -> 0.0 (attend), 1 -> -inf (don't attend) -- inverted for additive
// Actually: 1 -> 0.0 (real token, attend), 0 -> -inf (padding, mask out)
inline Tensor<float> attention_mask_to_additive(
    const Tensor<int32_t>& attention_mask,
    float mask_value = -1e9f) {

    Tensor<float> additive(attention_mask.shape());

    for (size_t i = 0; i < attention_mask.numel(); ++i) {
        additive.data()[i] = (attention_mask.data()[i] == 1) ? 0.0f : mask_value;
    }

    return additive;
}

// Expand 1D attention mask {batch, seq} to 2D {batch, 1, 1, seq} for broadcasting
inline Tensor<float> expand_attention_mask(
    const Tensor<int32_t>& attention_mask,
    float mask_value = -1e9f) {

    size_t batch = attention_mask.shape()[0];
    size_t seq = attention_mask.shape()[1];

    // Shape: {batch, 1, 1, seq} - can broadcast with {batch, heads, seq, seq}
    Tensor<float> expanded({batch, 1, 1, seq});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq; ++s) {
            float val = (attention_mask.data()[b * seq + s] == 1) ? 0.0f : mask_value;
            expanded.data()[b * seq + s] = val;
        }
    }

    return expanded;
}

// Combine causal mask with padding mask
// causal_mask: {seq, seq} or {batch, seq, seq}
// padding_mask: {batch, seq}
// output: {batch, seq, seq} with both masks applied
inline Tensor<float> combine_masks(
    const Tensor<float>& causal_mask,
    const Tensor<int32_t>& padding_mask,
    float mask_value = -1e9f) {

    size_t batch = padding_mask.shape()[0];
    size_t seq = padding_mask.shape()[1];

    bool causal_is_3d = (causal_mask.ndim() == 3);
    size_t causal_seq = causal_mask.shape()[causal_mask.ndim() - 1];

    Tensor<float> combined({batch, seq, seq});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq; ++i) {
            for (size_t j = 0; j < seq; ++j) {
                // Get causal mask value
                float causal_val;
                if (causal_is_3d) {
                    causal_val = causal_mask.data()[b * seq * seq + i * seq + j];
                } else {
                    causal_val = causal_mask.data()[i * causal_seq + j];
                }

                // Check padding for position j (the key position)
                bool j_is_padding = (padding_mask.data()[b * seq + j] == 0);

                // Combined: must pass both causal and padding checks
                if (causal_val == 0.0f || j_is_padding) {
                    combined.data()[b * seq * seq + i * seq + j] = mask_value;
                } else {
                    combined.data()[b * seq * seq + i * seq + j] = 0.0f;
                }
            }
        }
    }

    return combined;
}

}  // namespace data
}  // namespace lightwatch
