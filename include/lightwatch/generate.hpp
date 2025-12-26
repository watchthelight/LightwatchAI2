// Phase 34: Greedy Decode
// Text generation for GPT models

#pragma once

#include <lightwatch/models/gpt.hpp>
#include <lightwatch/autograd.hpp>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>

namespace lightwatch {

using TokenId = int32_t;

// Generation configuration
struct GenerateConfig {
    size_t max_new_tokens = 100;
    TokenId eos_token_id = 50256;  // GPT-2 end of text token
    bool early_stop = true;         // Stop at EOS
};

// Find argmax of a float array
inline size_t argmax(const float* data, size_t size) {
    if (size == 0) return 0;

    size_t max_idx = 0;
    float max_val = data[0];

    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// Greedy generation (argmax at each step)
inline std::vector<TokenId> generate_greedy(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    GenerateConfig config = {}) {

    if (prompt.empty()) {
        return {};
    }

    // Set model to eval mode (disables dropout)
    model.eval();

    // Copy prompt to output
    std::vector<TokenId> output = prompt;

    // Disable gradient computation for inference
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        // Check sequence length limit
        if (output.size() >= max_seq_len) {
            break;
        }

        // Create input tensor from current sequence
        Tensor<int32_t> input_ids({1, output.size()});
        for (size_t i = 0; i < output.size(); ++i) {
            input_ids.data()[i] = output[i];
        }

        // Forward pass
        auto logits = model.forward(input_ids);

        // Get logits for last position: shape is {1, seq_len, vocab_size}
        const auto& shape = logits.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        // Pointer to last position logits
        const float* last_logits = logits.data().data() + (seq_len - 1) * vocab_size;

        // Argmax to get next token
        size_t next_token_idx = argmax(last_logits, vocab_size);
        TokenId next_token = static_cast<TokenId>(next_token_idx);

        // Append token
        output.push_back(next_token);

        // Check for EOS
        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }

    return output;
}

// Streaming callback type
using TokenCallback = std::function<void(TokenId)>;

// Greedy generation with streaming callback
inline void generate_greedy_streaming(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    TokenCallback callback,
    GenerateConfig config = {}) {

    if (prompt.empty()) {
        return;
    }

    // Set model to eval mode
    model.eval();

    // Copy prompt to working sequence
    std::vector<TokenId> sequence = prompt;

    // Disable gradient computation
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        // Check sequence length limit
        if (sequence.size() >= max_seq_len) {
            break;
        }

        // Create input tensor
        Tensor<int32_t> input_ids({1, sequence.size()});
        for (size_t i = 0; i < sequence.size(); ++i) {
            input_ids.data()[i] = sequence[i];
        }

        // Forward pass
        auto logits = model.forward(input_ids);

        // Get logits for last position
        const auto& shape = logits.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];
        const float* last_logits = logits.data().data() + (seq_len - 1) * vocab_size;

        // Argmax
        size_t next_token_idx = argmax(last_logits, vocab_size);
        TokenId next_token = static_cast<TokenId>(next_token_idx);

        // Append token
        sequence.push_back(next_token);

        // Call callback
        callback(next_token);

        // Check for EOS
        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }
}

// Get just the generated tokens (excluding prompt)
inline std::vector<TokenId> get_generated_tokens(
    const std::vector<TokenId>& full_sequence,
    size_t prompt_length) {

    if (full_sequence.size() <= prompt_length) {
        return {};
    }

    return std::vector<TokenId>(
        full_sequence.begin() + static_cast<std::ptrdiff_t>(prompt_length),
        full_sequence.end()
    );
}

}  // namespace lightwatch
