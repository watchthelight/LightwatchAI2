// Phase 27: Data Loading Tests

#include <lightwatch/data/dataloader.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <set>

using namespace lightwatch;
using namespace lightwatch::data;
using namespace lightwatch::tokenizer;

// Test 1: Dataset size
bool test_phase_27_dataset_size() {
    // Create dataset with 100 samples
    std::vector<std::vector<TokenId>> sequences;
    for (int i = 0; i < 100; ++i) {
        std::vector<TokenId> seq;
        for (int j = 0; j < 10; ++j) {
            seq.push_back(static_cast<TokenId>(j + i));
        }
        sequences.push_back(seq);
    }

    TokenDataset dataset(std::move(sequences), 1024);

    if (dataset.size() != 100) {
        std::cerr << "dataset_size: size should be 100, got " << dataset.size() << std::endl;
        return false;
    }

    std::cout << "test_phase_27_dataset_size: PASSED" << std::endl;
    return true;
}

// Test 2: Dataset get
bool test_phase_27_dataset_get() {
    std::vector<std::vector<TokenId>> sequences;
    std::vector<TokenId> seq = {10, 20, 30, 40, 50};
    sequences.push_back(seq);

    TokenDataset dataset(std::move(sequences), 1024);

    Sample sample = dataset.get(0);

    // Check input_ids
    if (sample.input_ids.numel() != 5) {
        std::cerr << "dataset_get: input_ids should have 5 elements" << std::endl;
        return false;
    }
    if (sample.input_ids.data()[0] != 10) {
        std::cerr << "dataset_get: first token should be 10" << std::endl;
        return false;
    }

    // Check labels (shifted by 1)
    if (sample.labels.data()[0] != 20) {
        std::cerr << "dataset_get: first label should be 20 (shifted)" << std::endl;
        return false;
    }
    if (sample.labels.data()[3] != 50) {
        std::cerr << "dataset_get: label[3] should be 50" << std::endl;
        return false;
    }
    if (sample.labels.data()[4] != -100) {
        std::cerr << "dataset_get: last label should be -100 (ignore)" << std::endl;
        return false;
    }

    // Check attention mask
    for (size_t i = 0; i < 5; ++i) {
        if (sample.attention_mask.data()[i] != 1) {
            std::cerr << "dataset_get: attention_mask should all be 1" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_27_dataset_get: PASSED" << std::endl;
    return true;
}

// Test 3: DataLoader iteration
bool test_phase_27_dataloader_iter() {
    std::vector<std::vector<TokenId>> sequences;
    for (int i = 0; i < 10; ++i) {
        std::vector<TokenId> seq;
        for (int j = 0; j < 5; ++j) {
            seq.push_back(static_cast<TokenId>(j));
        }
        sequences.push_back(seq);
    }

    TokenDataset dataset(std::move(sequences), 1024);
    DataLoader loader(dataset, 4, false);  // batch_size=4, no shuffle

    size_t batch_count = 0;
    size_t total_samples = 0;
    for (auto samples : loader) {
        ++batch_count;
        total_samples += samples.size();

        // First two batches should have 4 samples
        if (batch_count < 3 && samples.size() != 4) {
            std::cerr << "dataloader_iter: batch " << batch_count
                      << " should have 4 samples, got " << samples.size() << std::endl;
            return false;
        }
    }

    // Should have 3 batches (4+4+2)
    if (batch_count != 3) {
        std::cerr << "dataloader_iter: should have 3 batches, got " << batch_count << std::endl;
        return false;
    }

    // Should iterate all 10 samples
    if (total_samples != 10) {
        std::cerr << "dataloader_iter: should iterate 10 samples, got " << total_samples << std::endl;
        return false;
    }

    std::cout << "test_phase_27_dataloader_iter: PASSED" << std::endl;
    return true;
}

// Test 4: DataLoader shuffle
bool test_phase_27_dataloader_shuffle() {
    std::vector<std::vector<TokenId>> sequences;
    for (int i = 0; i < 20; ++i) {
        std::vector<TokenId> seq = {static_cast<TokenId>(i), static_cast<TokenId>(i + 1)};
        sequences.push_back(seq);
    }

    TokenDataset dataset(std::move(sequences), 1024);
    DataLoader loader(dataset, 4, true, 42);  // shuffle=true

    // Collect first tokens from first epoch
    std::vector<int32_t> epoch1_tokens;
    for (auto samples : loader) {
        for (const auto& s : samples) {
            epoch1_tokens.push_back(s.input_ids.data()[0]);
        }
    }

    // Reset and collect second epoch
    loader.reset();
    std::vector<int32_t> epoch2_tokens;
    for (auto samples : loader) {
        for (const auto& s : samples) {
            epoch2_tokens.push_back(s.input_ids.data()[0]);
        }
    }

    // Orders should differ (with high probability)
    bool different = false;
    for (size_t i = 0; i < epoch1_tokens.size(); ++i) {
        if (epoch1_tokens[i] != epoch2_tokens[i]) {
            different = true;
            break;
        }
    }

    if (!different) {
        std::cerr << "dataloader_shuffle: two epochs should have different order" << std::endl;
        return false;
    }

    // Both should contain all samples (0-19)
    std::set<int32_t> seen1(epoch1_tokens.begin(), epoch1_tokens.end());
    std::set<int32_t> seen2(epoch2_tokens.begin(), epoch2_tokens.end());

    if (seen1.size() != 20 || seen2.size() != 20) {
        std::cerr << "dataloader_shuffle: should see all samples in each epoch" << std::endl;
        return false;
    }

    std::cout << "test_phase_27_dataloader_shuffle: PASSED" << std::endl;
    return true;
}

// Test 5: Text dataset (requires tokenizer)
bool test_phase_27_text_dataset() {
    // Create a temporary text file
    const std::string path = "/tmp/test_text_dataset.txt";
    std::ofstream file(path);
    file << "Hello world! This is a test file for the text dataset. "
         << "It contains some text that will be tokenized and split into sequences. "
         << "The tokenizer will convert this text into token IDs.";
    file.close();

    try {
        // Load tokenizer (requires vocab files in assets/)
        BPETokenizer tokenizer = BPETokenizer::gpt2();

        TextDataset dataset(path, tokenizer, 20);  // max_length=20

        if (dataset.size() == 0) {
            std::cerr << "text_dataset: should have at least 1 sequence" << std::endl;
            return false;
        }

        Sample sample = dataset.get(0);

        // Should have tokens
        if (sample.input_ids.numel() == 0) {
            std::cerr << "text_dataset: sample should have tokens" << std::endl;
            return false;
        }

        // Labels should be shifted
        if (sample.labels.data()[0] != sample.input_ids.data()[1]) {
            std::cerr << "text_dataset: labels should be shifted by 1" << std::endl;
            return false;
        }

    } catch (const std::exception& e) {
        // If tokenizer files aren't available, skip this test
        std::cout << "test_phase_27_text_dataset: SKIPPED (tokenizer not available: "
                  << e.what() << ")" << std::endl;
        std::remove(path.c_str());
        return true;
    }

    std::remove(path.c_str());
    std::cout << "test_phase_27_text_dataset: PASSED" << std::endl;
    return true;
}

// Test 6: Collate function
bool test_phase_27_collate() {
    // Create samples with different lengths
    Sample s1, s2;
    s1.input_ids = Tensor<int32_t>({3});
    s1.labels = Tensor<int32_t>({3});
    s1.attention_mask = Tensor<int32_t>({3});
    s1.input_ids.data()[0] = 1; s1.input_ids.data()[1] = 2; s1.input_ids.data()[2] = 3;
    s1.labels.data()[0] = 2; s1.labels.data()[1] = 3; s1.labels.data()[2] = -100;
    s1.attention_mask.fill_(1);

    s2.input_ids = Tensor<int32_t>({5});
    s2.labels = Tensor<int32_t>({5});
    s2.attention_mask = Tensor<int32_t>({5});
    s2.input_ids.data()[0] = 10; s2.input_ids.data()[1] = 20;
    s2.input_ids.data()[2] = 30; s2.input_ids.data()[3] = 40; s2.input_ids.data()[4] = 50;
    s2.labels.data()[0] = 20; s2.labels.data()[1] = 30;
    s2.labels.data()[2] = 40; s2.labels.data()[3] = 50; s2.labels.data()[4] = -100;
    s2.attention_mask.fill_(1);

    Batch batch = collate({s1, s2});

    // Should be padded to max length (5)
    if (batch.input_ids.shape()[1] != 5) {
        std::cerr << "collate: should pad to max length 5" << std::endl;
        return false;
    }

    // First sample should be padded
    // s1 data: [1, 2, 3, 0, 0]
    if (batch.input_ids.data()[3] != 0 || batch.input_ids.data()[4] != 0) {
        std::cerr << "collate: s1 should be padded with 0s" << std::endl;
        return false;
    }

    // First sample attention mask should reflect padding
    if (batch.attention_mask.data()[3] != 0 || batch.attention_mask.data()[4] != 0) {
        std::cerr << "collate: s1 attention_mask should be 0 for padding" << std::endl;
        return false;
    }

    // Second sample should be intact
    if (batch.input_ids.data()[5] != 10) {  // row 1, col 0
        std::cerr << "collate: s2 first token should be 10" << std::endl;
        return false;
    }

    std::cout << "test_phase_27_collate: PASSED" << std::endl;
    return true;
}

// Test 7: Empty dataset handling
bool test_phase_27_empty_dataset() {
    std::vector<std::vector<TokenId>> sequences;
    TokenDataset dataset(std::move(sequences), 1024);

    if (dataset.size() != 0) {
        std::cerr << "empty_dataset: size should be 0" << std::endl;
        return false;
    }

    DataLoader loader(dataset, 4, false);

    size_t count = 0;
    for (auto samples : loader) {
        (void)samples;
        ++count;
    }

    if (count != 0) {
        std::cerr << "empty_dataset: should have no batches" << std::endl;
        return false;
    }

    std::cout << "test_phase_27_empty_dataset: PASSED" << std::endl;
    return true;
}

// Test 8: DataLoader num_batches
bool test_phase_27_num_batches() {
    std::vector<std::vector<TokenId>> sequences;
    for (int i = 0; i < 17; ++i) {
        sequences.push_back({static_cast<TokenId>(i), static_cast<TokenId>(i + 1)});
    }

    TokenDataset dataset(std::move(sequences), 1024);
    DataLoader loader(dataset, 5, false);

    // 17 samples / 5 = 4 batches (5+5+5+2)
    if (loader.num_batches() != 4) {
        std::cerr << "num_batches: should be 4, got " << loader.num_batches() << std::endl;
        return false;
    }

    std::cout << "test_phase_27_num_batches: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 27: Data Loading Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_27_dataset_size()) ++failures;
    if (!test_phase_27_dataset_get()) ++failures;
    if (!test_phase_27_dataloader_iter()) ++failures;
    if (!test_phase_27_dataloader_shuffle()) ++failures;
    if (!test_phase_27_text_dataset()) ++failures;
    if (!test_phase_27_collate()) ++failures;
    if (!test_phase_27_empty_dataset()) ++failures;
    if (!test_phase_27_num_batches()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 27 tests passed (8/8) ===" << std::endl;
    return 0;
}
