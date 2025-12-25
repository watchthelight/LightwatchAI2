// Phase 08: Embedding Layer Tests

#include <lightwatch/nn/embedding.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test embedding output shape with 1D indices
bool test_phase_08_embed_shape() {
    Embedding embed(100, 64);  // 100 embeddings, dim 64

    Tensor<int32_t> indices({4});
    indices.data()[0] = 1;
    indices.data()[1] = 5;
    indices.data()[2] = 10;
    indices.data()[3] = 50;

    auto output = embed.forward(indices);

    // Output shape should be {4, 64}
    if (output.data().shape().size() != 2) {
        std::cerr << "embed_shape: expected 2D output" << std::endl;
        return false;
    }
    if (output.data().shape()[0] != 4 || output.data().shape()[1] != 64) {
        std::cerr << "embed_shape: expected shape {4, 64}" << std::endl;
        return false;
    }

    std::cout << "test_phase_08_embed_shape: PASSED" << std::endl;
    return true;
}

// Test embedding values match weight matrix
bool test_phase_08_embed_values() {
    Embedding embed(10, 8);

    // Set specific weight values
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            embed.weight.data()({i, j}) = static_cast<float>(i * 10 + j);
        }
    }

    // Lookup index 5
    Tensor<int32_t> indices({1});
    indices.data()[0] = 5;

    auto output = embed.forward(indices);

    // Check values match row 5 of weight
    for (size_t j = 0; j < 8; ++j) {
        float expected = static_cast<float>(5 * 10 + j);
        if (!float_eq(output.data()({0, j}), expected)) {
            std::cerr << "embed_values: mismatch at " << j << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_08_embed_values: PASSED" << std::endl;
    return true;
}

// Test embedding with batch indices (2D)
bool test_phase_08_embed_batch() {
    Embedding embed(100, 32);

    // Batch of 2 sequences, each with 4 tokens
    Tensor<int32_t> indices({2, 4});
    indices.data()[0] = 1; indices.data()[1] = 2; indices.data()[2] = 3; indices.data()[3] = 4;
    indices.data()[4] = 10; indices.data()[5] = 20; indices.data()[6] = 30; indices.data()[7] = 40;

    auto output = embed.forward(indices);

    // Output shape should be {2, 4, 32}
    auto shape = output.data().shape();
    if (shape.size() != 3) {
        std::cerr << "embed_batch: expected 3D output, got " << shape.size() << "D" << std::endl;
        return false;
    }
    if (shape[0] != 2 || shape[1] != 4 || shape[2] != 32) {
        std::cerr << "embed_batch: expected shape {2, 4, 32}" << std::endl;
        return false;
    }

    std::cout << "test_phase_08_embed_batch: PASSED" << std::endl;
    return true;
}

// Test GPT embedding
bool test_phase_08_gpt_embed() {
    GPTEmbedding gpt_embed(50257, 1024, 768);

    // Single batch, 10 tokens
    Tensor<int32_t> token_ids({1, 10});
    for (int i = 0; i < 10; ++i) {
        token_ids.data()[i] = i * 100;  // Token IDs: 0, 100, 200, ...
    }

    auto output = gpt_embed.forward(token_ids);

    // Output shape should be {1, 10, 768}
    auto shape = output.data().shape();
    if (shape.size() != 3) {
        std::cerr << "gpt_embed: expected 3D output" << std::endl;
        return false;
    }
    if (shape[0] != 1 || shape[1] != 10 || shape[2] != 768) {
        std::cerr << "gpt_embed: expected shape {1, 10, 768}, got {"
                  << shape[0] << ", " << shape[1] << ", " << shape[2] << "}" << std::endl;
        return false;
    }

    std::cout << "test_phase_08_gpt_embed: PASSED" << std::endl;
    return true;
}

// Test embedding backward pass
bool test_phase_08_embed_backward() {
    Embedding embed(10, 4);

    // Initialize weight to zeros for easier testing
    for (size_t i = 0; i < 10 * 4; ++i) {
        embed.weight.data().data()[i] = 0.0f;
    }

    // Lookup indices [2, 5, 2] - note duplicate index 2
    Tensor<int32_t> indices({3});
    indices.data()[0] = 2;
    indices.data()[1] = 5;
    indices.data()[2] = 2;  // Duplicate

    auto output = embed.forward(indices);

    // Create gradient for output (all ones)
    Tensor<float> grad = Tensor<float>::ones({3, 4});
    output.backward(grad);

    // Check gradients accumulated correctly
    // Index 2 was looked up twice, so grad should be 2.0
    // Index 5 was looked up once, so grad should be 1.0
    for (size_t j = 0; j < 4; ++j) {
        if (!float_eq(embed.weight.grad()({2, j}), 2.0f)) {
            std::cerr << "embed_backward: index 2 grad should be 2.0 at "
                      << j << ", got " << embed.weight.grad()({2, j}) << std::endl;
            return false;
        }
        if (!float_eq(embed.weight.grad()({5, j}), 1.0f)) {
            std::cerr << "embed_backward: index 5 grad should be 1.0" << std::endl;
            return false;
        }
    }

    // Other indices should have zero gradient
    for (size_t j = 0; j < 4; ++j) {
        if (!float_eq(embed.weight.grad()({0, j}), 0.0f)) {
            std::cerr << "embed_backward: index 0 grad should be 0.0" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_08_embed_backward: PASSED" << std::endl;
    return true;
}

// Test embedding with Variable input
bool test_phase_08_embed_variable_input() {
    Embedding embed(100, 16);

    // Create Variable with float indices
    Tensor<float> indices_float({3});
    indices_float.data()[0] = 5.0f;
    indices_float.data()[1] = 10.0f;
    indices_float.data()[2] = 15.0f;

    Variable input(indices_float, false);
    auto output = embed.forward(input);

    // Output shape should be {3, 16}
    if (output.data().shape()[0] != 3 || output.data().shape()[1] != 16) {
        std::cerr << "embed_variable_input: wrong output shape" << std::endl;
        return false;
    }

    std::cout << "test_phase_08_embed_variable_input: PASSED" << std::endl;
    return true;
}

// Test out of bounds index
bool test_phase_08_embed_out_of_bounds() {
    Embedding embed(10, 8);

    Tensor<int32_t> indices({1});
    indices.data()[0] = 100;  // Out of bounds

    try {
        embed.forward(indices);
        std::cerr << "embed_out_of_bounds: should throw exception" << std::endl;
        return false;
    } catch (const std::out_of_range&) {
        // Expected
    }

    std::cout << "test_phase_08_embed_out_of_bounds: PASSED" << std::endl;
    return true;
}

// Test parameter count
bool test_phase_08_param_count() {
    GPTEmbedding gpt_embed(50257, 1024, 768);

    // Expected params: wte (50257 * 768) + wpe (1024 * 768)
    size_t expected = 50257 * 768 + 1024 * 768;
    size_t actual = gpt_embed.num_parameters();

    if (actual != expected) {
        std::cerr << "param_count: expected " << expected << ", got " << actual << std::endl;
        return false;
    }

    std::cout << "test_phase_08_param_count: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_08_embed_shape()) ++failures;
    if (!test_phase_08_embed_values()) ++failures;
    if (!test_phase_08_embed_batch()) ++failures;
    if (!test_phase_08_gpt_embed()) ++failures;
    if (!test_phase_08_embed_backward()) ++failures;
    if (!test_phase_08_embed_variable_input()) ++failures;
    if (!test_phase_08_embed_out_of_bounds()) ++failures;
    if (!test_phase_08_param_count()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 08 tests passed (8/8)" << std::endl;
    return 0;
}
