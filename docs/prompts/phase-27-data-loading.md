# Phase 27: Data Loading

## Objective
Implement Dataset and DataLoader for efficient training data access.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 07 | Vocabulary for tokenization |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 07 | include/lightwatch/tokenizer/vocabulary.hpp | TokenId |
| 06 | include/lightwatch/tokenizer/bpe.hpp | BPETokenizer |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/data/dataset.hpp | Dataset, TextDataset | Phase 28, 29 |
| include/lightwatch/data/dataloader.hpp | DataLoader | Phase 28, 29 |
| src/data/*.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::data {

struct Sample {
    Tensor<int32_t> input_ids;   // {seq_len}
    Tensor<int32_t> labels;       // {seq_len}
    Tensor<int32_t> attention_mask;  // {seq_len}
};

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual Sample get(size_t index) const = 0;
};

class TextDataset : public Dataset {
public:
    TextDataset(const std::string& path,
                const tokenizer::BPETokenizer& tokenizer,
                size_t max_length = 1024);

    size_t size() const override;
    Sample get(size_t index) const override;

private:
    std::vector<std::vector<TokenId>> sequences_;
    size_t max_length_;
};

class DataLoader {
public:
    DataLoader(Dataset& dataset,
               size_t batch_size,
               bool shuffle = true,
               size_t num_workers = 0);

    class Iterator {
    public:
        std::vector<Sample> operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;
    };

    Iterator begin();
    Iterator end();

    void reset();  // Re-shuffle for new epoch

private:
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::vector<size_t> indices_;
};

}  // namespace lightwatch::data
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **TextDataset**: Load text file, tokenize, chunk into sequences
2. **DataLoader**: Iterate in batches, optionally shuffle
3. **Shuffling**: Fisher-Yates shuffle at epoch start
4. **Memory**: Lazy loading for large datasets

### Performance Constraints
- Batch iteration: O(batch_size)
- Shuffle: O(dataset_size)
- Memory: Only current batch in memory

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_27_dataset_size` | 100 samples | size() == 100 |
| `test_phase_27_dataset_get` | get(0) | Valid Sample |
| `test_phase_27_dataloader_iter` | batch_size=4 | 4 samples per batch |
| `test_phase_27_dataloader_shuffle` | Two epochs | Different order |
| `test_phase_27_text_dataset` | Text file | Tokenized sequences |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_27" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/data/dataloader.hpp`
- [ ] DataLoader iterates all samples

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 4 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- For LM training: labels = input_ids shifted by 1
- Attention mask: 1 for real tokens, 0 for padding
- num_workers=0 means single-threaded (simplicity)
- Shuffle indices, not data (memory efficient)
