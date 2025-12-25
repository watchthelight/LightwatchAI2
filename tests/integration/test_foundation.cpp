// Phase 10: Foundation Integration Tests
// Comprehensive testing of Phases 1-9 working together

#include <lightwatch/tensor.hpp>
#include <lightwatch/autograd.hpp>
#include <lightwatch/tokenizer/bpe.hpp>
#include <lightwatch/tokenizer/vocabulary.hpp>
#include <lightwatch/nn/embedding.hpp>
#include <lightwatch/nn/positional.hpp>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace lightwatch;
using namespace lightwatch::autograd;
using namespace lightwatch::tokenizer;
using namespace lightwatch::nn;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test tensor-autograd integration
bool test_phase_10_tensor_autograd_integration() {
    // Create variables and perform operations
    Variable a(Tensor<float>::full({2, 3}, 2.0f), true);
    Variable b(Tensor<float>::full({2, 3}, 3.0f), true);

    // Chain of operations: y = (a * b) + a
    auto c = ops::mul(a, b);  // c = 6.0
    auto y = ops::add(c, a);  // y = 8.0

    // Check forward values
    for (size_t i = 0; i < y.numel(); ++i) {
        if (!float_eq(y.data().data()[i], 8.0f)) {
            std::cerr << "tensor_autograd: forward value wrong" << std::endl;
            return false;
        }
    }

    // Backward
    y.backward();

    // dy/da = b + 1 = 4.0
    // dy/db = a = 2.0
    for (size_t i = 0; i < a.numel(); ++i) {
        if (!float_eq(a.grad().data()[i], 4.0f)) {
            std::cerr << "tensor_autograd: a.grad should be 4.0" << std::endl;
            return false;
        }
        if (!float_eq(b.grad().data()[i], 2.0f)) {
            std::cerr << "tensor_autograd: b.grad should be 2.0" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_10_tensor_autograd_integration: PASSED" << std::endl;
    return true;
}

// Test tokenizer-embedding integration
bool test_phase_10_tokenizer_embedding() {
    // Load tokenizer
    BPETokenizer tokenizer = BPETokenizer::gpt2();

    // Create embedding
    Embedding embed(tokenizer.vocab_size(), 64);

    // Encode text
    std::string text = "Hello";
    auto tokens = tokenizer.encode(text);

    if (tokens.empty()) {
        std::cerr << "tokenizer_embedding: no tokens produced" << std::endl;
        return false;
    }

    // Create indices tensor
    Tensor<int32_t> indices({tokens.size()});
    for (size_t i = 0; i < tokens.size(); ++i) {
        indices.data()[i] = tokens[i];
    }

    // Get embeddings
    auto embeddings = embed.forward(indices);

    // Check shape
    if (embeddings.data().shape()[0] != tokens.size() ||
        embeddings.data().shape()[1] != 64) {
        std::cerr << "tokenizer_embedding: wrong embedding shape" << std::endl;
        return false;
    }

    // Check values are finite
    for (size_t i = 0; i < embeddings.numel(); ++i) {
        if (std::isnan(embeddings.data().data()[i]) ||
            std::isinf(embeddings.data().data()[i])) {
            std::cerr << "tokenizer_embedding: NaN/Inf in embeddings" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_10_tokenizer_embedding: PASSED" << std::endl;
    return true;
}

// Test GPT embedding forward pass
bool test_phase_10_gpt_embedding_forward() {
    // Create GPT-style embedding (smaller dimensions for testing)
    GPTEmbedding gpt_embed(1000, 128, 64);  // vocab=1000, seq=128, dim=64

    // Create batch of token IDs
    Tensor<int32_t> token_ids({2, 10});  // batch=2, seq=10
    for (size_t i = 0; i < 20; ++i) {
        token_ids.data()[i] = static_cast<int32_t>(i % 100);
    }

    auto output = gpt_embed.forward(token_ids);

    // Check shape: {batch, seq, embed_dim}
    auto shape = output.data().shape();
    if (shape.size() != 3) {
        std::cerr << "gpt_embedding_forward: expected 3D output" << std::endl;
        return false;
    }

    if (shape[0] != 2 || shape[1] != 10 || shape[2] != 64) {
        std::cerr << "gpt_embedding_forward: wrong shape {"
                  << shape[0] << ", " << shape[1] << ", " << shape[2] << "}" << std::endl;
        return false;
    }

    std::cout << "test_phase_10_gpt_embedding_forward: PASSED" << std::endl;
    return true;
}

// Test memory baseline
bool test_phase_10_memory_baseline() {
    // Create and destroy large tensors
    for (int i = 0; i < 10; ++i) {
        Tensor<float> large = Tensor<float>::randn({1000, 1000});
        // Force some computation
        auto sum = large.sum();
        (void)sum;
    }

    // If we get here without crash, basic memory management works
    std::cout << "test_phase_10_memory_baseline: PASSED" << std::endl;
    return true;
}

// Test numerical stability with small values
bool test_phase_10_numerical_stability() {
    // Test with very small values
    Tensor<float> small({10});
    for (size_t i = 0; i < 10; ++i) {
        small.data()[i] = 1e-30f * static_cast<float>(i + 1);
    }

    Variable v(small, true);

    // Test softmax with small values
    auto sm = ops::softmax(v, 0);

    // Check no NaN/Inf
    for (size_t i = 0; i < sm.numel(); ++i) {
        if (std::isnan(sm.data().data()[i]) || std::isinf(sm.data().data()[i])) {
            std::cerr << "numerical_stability: NaN/Inf in softmax" << std::endl;
            return false;
        }
    }

    // Test with larger values
    Tensor<float> large_vals({10});
    for (size_t i = 0; i < 10; ++i) {
        large_vals.data()[i] = static_cast<float>(i);
    }

    Variable v2(large_vals, true);
    auto sm2 = ops::softmax(v2, 0);

    // Check softmax sums to 1
    float sum = 0;
    for (size_t i = 0; i < sm2.numel(); ++i) {
        sum += sm2.data().data()[i];
    }

    if (!float_eq(sum, 1.0f, 1e-3f)) {
        std::cerr << "numerical_stability: softmax doesn't sum to 1" << std::endl;
        return false;
    }

    std::cout << "test_phase_10_numerical_stability: PASSED" << std::endl;
    return true;
}

// Test large tensor operations
bool test_phase_10_large_tensor() {
    // Create 1M element tensor
    Tensor<float> large = Tensor<float>::randn({1000, 1000});

    // Test sum
    auto sum = large.sum();
    (void)sum;

    // Test reshape
    auto reshaped = large.reshape({500, 2000});
    if (reshaped.numel() != 1000000) {
        std::cerr << "large_tensor: reshape changed element count" << std::endl;
        return false;
    }

    // Test element-wise operations
    auto doubled = large + large;
    (void)doubled;

    std::cout << "test_phase_10_large_tensor: PASSED" << std::endl;
    return true;
}

// Test batch tokenize and embed
bool test_phase_10_batch_tokenize_embed() {
    BPETokenizer tokenizer = BPETokenizer::gpt2();
    Embedding embed(tokenizer.vocab_size(), 32);

    std::vector<std::string> texts = {"a", "bb", "ccc"};
    auto batch_tokens = tokenizer.encode_batch(texts);

    // Find max length
    size_t max_len = 0;
    for (const auto& tokens : batch_tokens) {
        max_len = std::max(max_len, tokens.size());
    }

    // Embed each sequence
    for (size_t b = 0; b < batch_tokens.size(); ++b) {
        const auto& tokens = batch_tokens[b];

        Tensor<int32_t> indices({tokens.size()});
        for (size_t i = 0; i < tokens.size(); ++i) {
            indices.data()[i] = tokens[i];
        }

        auto embeddings = embed.forward(indices);

        if (embeddings.data().shape()[0] != tokens.size()) {
            std::cerr << "batch_tokenize_embed: wrong embedding count" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_10_batch_tokenize_embed: PASSED" << std::endl;
    return true;
}

// Test matmul + autograd
bool test_phase_10_matmul_autograd() {
    Variable a(Tensor<float>::randn({4, 8}), true);
    Variable b(Tensor<float>::randn({8, 6}), true);

    auto c = ops::matmul(a, b);

    // Check shape
    if (c.data().shape()[0] != 4 || c.data().shape()[1] != 6) {
        std::cerr << "matmul_autograd: wrong output shape" << std::endl;
        return false;
    }

    // Backward
    c.backward();

    // Check gradients exist
    if (!a.has_grad() || !b.has_grad()) {
        std::cerr << "matmul_autograd: missing gradients" << std::endl;
        return false;
    }

    // Check gradient shapes
    if (a.grad().shape() != a.data().shape()) {
        std::cerr << "matmul_autograd: a.grad shape mismatch" << std::endl;
        return false;
    }

    std::cout << "test_phase_10_matmul_autograd: PASSED" << std::endl;
    return true;
}

// Test positional encoding integration
bool test_phase_10_positional_encoding_integration() {
    SinusoidalPE pe(100, 64);
    Embedding embed(1000, 64);

    // Create embeddings
    Tensor<int32_t> indices({10});
    for (int i = 0; i < 10; ++i) {
        indices.data()[i] = i * 10;
    }

    auto embeddings = embed.forward(indices);

    // Add positional encoding
    auto with_pos = pe.forward(embeddings);

    // Check shape preserved
    if (with_pos.data().shape() != embeddings.data().shape()) {
        std::cerr << "positional_encoding: shape changed" << std::endl;
        return false;
    }

    // Values should be different (encoding added)
    bool different = false;
    for (size_t i = 0; i < with_pos.numel(); ++i) {
        if (!float_eq(with_pos.data().data()[i], embeddings.data().data()[i])) {
            different = true;
            break;
        }
    }

    if (!different) {
        std::cerr << "positional_encoding: values not changed" << std::endl;
        return false;
    }

    std::cout << "test_phase_10_positional_encoding_integration: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 10: Foundation Integration Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_10_tensor_autograd_integration()) ++failures;
    if (!test_phase_10_tokenizer_embedding()) ++failures;
    if (!test_phase_10_gpt_embedding_forward()) ++failures;
    if (!test_phase_10_memory_baseline()) ++failures;
    if (!test_phase_10_numerical_stability()) ++failures;
    if (!test_phase_10_large_tensor()) ++failures;
    if (!test_phase_10_batch_tokenize_embed()) ++failures;
    if (!test_phase_10_matmul_autograd()) ++failures;
    if (!test_phase_10_positional_encoding_integration()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 10 tests passed (9/9) ===" << std::endl;
    std::cout << "CHECKPOINT 1 PASSED - Foundation components verified" << std::endl;
    return 0;
}
