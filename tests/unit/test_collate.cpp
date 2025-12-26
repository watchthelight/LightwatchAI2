// Phase 28: Batch Processing Tests

#include <lightwatch/data/collate.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::data;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Pad sequence
bool test_phase_28_pad_sequence() {
    Tensor<int32_t> seq({3});
    seq.data()[0] = 1;
    seq.data()[1] = 2;
    seq.data()[2] = 3;

    auto padded = pad_sequence(seq, 5, 999);

    // Should be [1, 2, 3, 999, 999]
    if (padded.numel() != 5) {
        std::cerr << "pad_sequence: should have length 5" << std::endl;
        return false;
    }

    if (padded.data()[0] != 1 || padded.data()[1] != 2 || padded.data()[2] != 3) {
        std::cerr << "pad_sequence: original values should be preserved" << std::endl;
        return false;
    }

    if (padded.data()[3] != 999 || padded.data()[4] != 999) {
        std::cerr << "pad_sequence: padding should be 999" << std::endl;
        return false;
    }

    std::cout << "test_phase_28_pad_sequence: PASSED" << std::endl;
    return true;
}

// Test 2: Collate shapes
bool test_phase_28_collate_shapes() {
    std::vector<Sample> samples;

    // Create 4 samples with varying lengths
    for (int len = 3; len <= 6; ++len) {
        Sample s;
        s.input_ids = Tensor<int32_t>({static_cast<size_t>(len)});
        s.labels = Tensor<int32_t>({static_cast<size_t>(len)});
        s.attention_mask = Tensor<int32_t>({static_cast<size_t>(len)});

        for (int i = 0; i < len; ++i) {
            s.input_ids.data()[i] = i + 1;
            s.labels.data()[i] = i + 2;
            s.attention_mask.data()[i] = 1;
        }
        samples.push_back(std::move(s));
    }

    BatchEx batch = collate_fn(samples, 50256);

    // Should have 4 samples, max length 6
    if (batch.batch_size != 4) {
        std::cerr << "collate_shapes: batch_size should be 4, got " << batch.batch_size << std::endl;
        return false;
    }

    if (batch.max_seq_len != 6) {
        std::cerr << "collate_shapes: max_seq_len should be 6, got " << batch.max_seq_len << std::endl;
        return false;
    }

    // Check tensor shapes
    if (batch.input_ids.shape()[0] != 4 || batch.input_ids.shape()[1] != 6) {
        std::cerr << "collate_shapes: input_ids shape should be {4, 6}" << std::endl;
        return false;
    }

    std::cout << "test_phase_28_collate_shapes: PASSED" << std::endl;
    return true;
}

// Test 3: Attention mask
bool test_phase_28_attention_mask() {
    std::vector<Sample> samples;

    // Sample with 3 tokens
    Sample s1;
    s1.input_ids = Tensor<int32_t>({3});
    s1.labels = Tensor<int32_t>({3});
    s1.attention_mask = Tensor<int32_t>({3});
    s1.input_ids.fill_(1);
    s1.labels.fill_(2);
    s1.attention_mask.fill_(1);
    samples.push_back(std::move(s1));

    // Sample with 5 tokens
    Sample s2;
    s2.input_ids = Tensor<int32_t>({5});
    s2.labels = Tensor<int32_t>({5});
    s2.attention_mask = Tensor<int32_t>({5});
    s2.input_ids.fill_(1);
    s2.labels.fill_(2);
    s2.attention_mask.fill_(1);
    samples.push_back(std::move(s2));

    BatchEx batch = collate_fn(samples);

    // First sample: [1, 1, 1, 0, 0] (3 real tokens, 2 padding)
    // Second sample: [1, 1, 1, 1, 1] (5 real tokens)
    size_t max_len = batch.max_seq_len;

    // Check first sample attention mask
    for (size_t i = 0; i < 3; ++i) {
        if (batch.attention_mask.data()[0 * max_len + i] != 1) {
            std::cerr << "attention_mask: first sample pos " << i << " should be 1" << std::endl;
            return false;
        }
    }
    for (size_t i = 3; i < 5; ++i) {
        if (batch.attention_mask.data()[0 * max_len + i] != 0) {
            std::cerr << "attention_mask: first sample pos " << i << " should be 0 (padding)" << std::endl;
            return false;
        }
    }

    // Check second sample attention mask (all 1s)
    for (size_t i = 0; i < 5; ++i) {
        if (batch.attention_mask.data()[1 * max_len + i] != 1) {
            std::cerr << "attention_mask: second sample pos " << i << " should be 1" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_28_attention_mask: PASSED" << std::endl;
    return true;
}

// Test 4: Labels with padding
bool test_phase_28_labels() {
    std::vector<Sample> samples;

    // Sample with 3 tokens
    Sample s;
    s.input_ids = Tensor<int32_t>({3});
    s.labels = Tensor<int32_t>({3});
    s.attention_mask = Tensor<int32_t>({3});
    s.input_ids.data()[0] = 10; s.input_ids.data()[1] = 20; s.input_ids.data()[2] = 30;
    s.labels.data()[0] = 20; s.labels.data()[1] = 30; s.labels.data()[2] = -100;
    s.attention_mask.fill_(1);
    samples.push_back(std::move(s));

    // Pad to 5
    Sample s2;
    s2.input_ids = Tensor<int32_t>({5});
    s2.labels = Tensor<int32_t>({5});
    s2.attention_mask = Tensor<int32_t>({5});
    s2.input_ids.fill_(1);
    s2.labels.fill_(2);
    s2.attention_mask.fill_(1);
    samples.push_back(std::move(s2));

    BatchEx batch = collate_fn(samples);
    size_t max_len = batch.max_seq_len;

    // First sample labels: [20, 30, -100, -100, -100]
    if (batch.labels.data()[0 * max_len + 0] != 20) {
        std::cerr << "labels: position 0 should be 20" << std::endl;
        return false;
    }
    if (batch.labels.data()[0 * max_len + 1] != 30) {
        std::cerr << "labels: position 1 should be 30" << std::endl;
        return false;
    }
    // Padding positions should be -100 (IGNORE_INDEX)
    for (size_t i = 3; i < 5; ++i) {
        if (batch.labels.data()[0 * max_len + i] != IGNORE_INDEX) {
            std::cerr << "labels: padding position " << i << " should be " << IGNORE_INDEX << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_28_labels: PASSED" << std::endl;
    return true;
}

// Test 5: Causal mask
bool test_phase_28_causal_mask() {
    // Create simple causal mask for seq_len=4
    auto mask = create_causal_mask(4);

    // Should be lower triangular:
    // [1, 0, 0, 0]
    // [1, 1, 0, 0]
    // [1, 1, 1, 0]
    // [1, 1, 1, 1]
    if (mask.shape()[0] != 4 || mask.shape()[1] != 4) {
        std::cerr << "causal_mask: shape should be {4, 4}" << std::endl;
        return false;
    }

    // Check diagonal and below are 1
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            float expected = (j <= i) ? 1.0f : 0.0f;
            if (!float_eq(mask.data()[i * 4 + j], expected)) {
                std::cerr << "causal_mask: position [" << i << "," << j << "] should be "
                          << expected << ", got " << mask.data()[i * 4 + j] << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_28_causal_mask: PASSED" << std::endl;
    return true;
}

// Test 6: Causal mask with padding
bool test_phase_28_causal_with_padding() {
    // Create attention mask: batch=1, seq=4, with padding at position 3
    Tensor<int32_t> attn_mask({1, 4});
    attn_mask.data()[0] = 1;  // real
    attn_mask.data()[1] = 1;  // real
    attn_mask.data()[2] = 1;  // real
    attn_mask.data()[3] = 0;  // padding

    auto mask = create_causal_mask(attn_mask, 4);

    // Should be: (causal AND padding)
    // [1, 0, 0, 0]
    // [1, 1, 0, 0]
    // [1, 1, 1, 0]
    // [0, 0, 0, 0]  <- row 3 is all 0 because position 3 is padding

    // Check row 2 (last real token)
    if (!float_eq(mask.data()[2 * 4 + 0], 1.0f) ||
        !float_eq(mask.data()[2 * 4 + 1], 1.0f) ||
        !float_eq(mask.data()[2 * 4 + 2], 1.0f) ||
        !float_eq(mask.data()[2 * 4 + 3], 0.0f)) {
        std::cerr << "causal_with_padding: row 2 incorrect" << std::endl;
        return false;
    }

    // Check row 3 (padding position - all 0)
    for (size_t j = 0; j < 4; ++j) {
        if (!float_eq(mask.data()[3 * 4 + j], 0.0f)) {
            std::cerr << "causal_with_padding: row 3 should be all 0" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_28_causal_with_padding: PASSED" << std::endl;
    return true;
}

// Test 7: Attention mask to additive
bool test_phase_28_additive_mask() {
    Tensor<int32_t> mask({2, 3});
    mask.data()[0] = 1; mask.data()[1] = 1; mask.data()[2] = 0;  // [1,1,0]
    mask.data()[3] = 1; mask.data()[4] = 0; mask.data()[5] = 0;  // [1,0,0]

    auto additive = attention_mask_to_additive(mask, -1e9f);

    // Real tokens -> 0, padding -> -1e9
    if (!float_eq(additive.data()[0], 0.0f) ||
        !float_eq(additive.data()[1], 0.0f) ||
        additive.data()[2] > -1e8f) {
        std::cerr << "additive_mask: first row incorrect" << std::endl;
        return false;
    }

    if (!float_eq(additive.data()[3], 0.0f) ||
        additive.data()[4] > -1e8f ||
        additive.data()[5] > -1e8f) {
        std::cerr << "additive_mask: second row incorrect" << std::endl;
        return false;
    }

    std::cout << "test_phase_28_additive_mask: PASSED" << std::endl;
    return true;
}

// Test 8: Combine masks
bool test_phase_28_combine_masks() {
    // Causal mask: 3x3
    auto causal = create_causal_mask(3);

    // Padding mask: batch=1, seq=3, last position is padding
    Tensor<int32_t> padding({1, 3});
    padding.data()[0] = 1;
    padding.data()[1] = 1;
    padding.data()[2] = 0;  // padding

    auto combined = combine_masks(causal, padding, -1e9f);

    // Expected: causal AND (j is not padding)
    // Column 2 should be all masked (because position 2 is padding)
    // Row 0: [0, -inf, -inf]  (can only attend to self)
    // Row 1: [0, 0, -inf]     (can attend to 0 and 1, not 2)
    // Row 2: [0, 0, -inf]     (can attend to 0 and 1, not 2)

    // Check that column 2 is all masked
    for (size_t i = 0; i < 3; ++i) {
        if (combined.data()[i * 3 + 2] > -1e8f) {
            std::cerr << "combine_masks: column 2 should be masked" << std::endl;
            return false;
        }
    }

    // Check row 1 positions 0 and 1 are not masked
    if (!float_eq(combined.data()[1 * 3 + 0], 0.0f) ||
        !float_eq(combined.data()[1 * 3 + 1], 0.0f)) {
        std::cerr << "combine_masks: row 1 positions 0,1 should not be masked" << std::endl;
        return false;
    }

    std::cout << "test_phase_28_combine_masks: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 28: Batch Processing Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_28_pad_sequence()) ++failures;
    if (!test_phase_28_collate_shapes()) ++failures;
    if (!test_phase_28_attention_mask()) ++failures;
    if (!test_phase_28_labels()) ++failures;
    if (!test_phase_28_causal_mask()) ++failures;
    if (!test_phase_28_causal_with_padding()) ++failures;
    if (!test_phase_28_additive_mask()) ++failures;
    if (!test_phase_28_combine_masks()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 28 tests passed (8/8) ===" << std::endl;
    return 0;
}
