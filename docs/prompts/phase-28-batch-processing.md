# Phase 28: Batch Processing

## Objective
Implement batch collation, padding, and attention mask generation for variable-length sequences.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 27 | Dataset and DataLoader |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 27 | include/lightwatch/data/dataloader.hpp | Sample |
| 03 | include/lightwatch/tensor.hpp | Tensor |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/data/collate.hpp | collate_fn, pad_sequence | Phase 29 |
| src/data/collate.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::data {

struct Batch {
    Tensor<int32_t> input_ids;       // {batch, max_seq}
    Tensor<int32_t> labels;          // {batch, max_seq}
    Tensor<int32_t> attention_mask;  // {batch, max_seq}
    size_t batch_size;
    size_t max_seq_len;
};

// Pad sequence to given length
Tensor<int32_t> pad_sequence(
    const Tensor<int32_t>& seq,
    size_t target_length,
    TokenId pad_value = 50256);

// Collate samples into batch
Batch collate_fn(
    const std::vector<Sample>& samples,
    TokenId pad_id = 50256);

// Create causal attention mask for batch
Tensor<bool> create_attention_mask(
    const Tensor<int32_t>& attention_mask_1d,
    size_t seq_len);

}  // namespace lightwatch::data
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Padding**: Pad shorter sequences to max length in batch
2. **Attention mask 1D**: 1 for real tokens, 0 for padding
3. **Attention mask 2D**: Combine padding mask with causal mask
4. **Labels**: Copy input_ids, set padding positions to ignore_index

### Performance Constraints
- O(batch_size * max_seq_len) for collation
- Memory: One copy of padded batch

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_28_pad_sequence` | [1,2,3] to length 5 | [1,2,3,pad,pad] |
| `test_phase_28_collate_shapes` | 4 samples, varying len | Uniform batch shape |
| `test_phase_28_attention_mask` | Padded batch | 1s before padding, 0s after |
| `test_phase_28_labels` | Collated batch | Padding positions = -100 |
| `test_phase_28_causal_mask` | seq_len=4 | Lower triangular with padding |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_28" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/data/collate.hpp`
- [ ] Batches have correct shapes and masks

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-450 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 pad_id = eos_id = 50256
- ignore_index = -100 for loss computation
- Causal mask + padding mask combined with AND
- Left-padding vs right-padding: use right-padding
