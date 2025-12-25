# Phase 08: Embedding Layer

## Objective
Implement token embedding lookup with combined token and position embeddings for transformer input.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 03 | include/lightwatch/tensor.hpp |
| 07 | include/lightwatch/tokenizer/vocabulary.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 03 | include/lightwatch/tensor.hpp | Tensor<float> |
| 07 | include/lightwatch/tokenizer/vocabulary.hpp | TokenId |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/embedding.hpp | Embedding | Phase 10, 31 |
| src/nn/embedding.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Extends docs/contracts/module.hpp
namespace lightwatch::nn {

class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);

    // Lookup by indices tensor
    autograd::Variable forward(const Tensor<int32_t>& indices);

    // Required override (converts float tensor to int32 indices)
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;  // Shape: {num_embeddings, embedding_dim}

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
};

// Combined token + position embedding for GPT
class GPTEmbedding : public Module {
public:
    GPTEmbedding(size_t vocab_size, size_t max_seq_len, size_t embed_dim);

    // Takes token IDs, returns embeddings + positional
    autograd::Variable forward(const Tensor<int32_t>& token_ids);
    autograd::Variable forward(const autograd::Variable& input) override;

    Embedding wte;  // Token embeddings
    Embedding wpe;  // Position embeddings

private:
    size_t max_seq_len_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Index lookup**: Gather rows from weight matrix by index
2. **Position embedding**: Add learned position embeddings (not sinusoidal)
3. **Gradient flow**: Index_select backward scatters gradients
4. **Out of bounds**: Throw exception for invalid indices

### Performance Constraints
- Forward: O(seq_len * embed_dim) - simple gather
- Backward: O(seq_len * embed_dim) - scatter add

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_08_embed_shape` | indices {4} | Output shape {4, embed_dim} |
| `test_phase_08_embed_values` | Single index | Correct row from weight |
| `test_phase_08_embed_batch` | indices {2, 4} | Output shape {2, 4, embed_dim} |
| `test_phase_08_gpt_embed` | token_ids {1, 10} | Output {1, 10, 768} |
| `test_phase_08_embed_backward` | Backward pass | Weight gradients accumulated |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_08" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/embedding.hpp`
- [ ] Embedding lookup produces correct shapes

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses learned position embeddings (not sinusoidal)
- wte: {50257, 768}, wpe: {1024, 768} for GPT-2 Small
- Combined embedding: wte[token_ids] + wpe[positions]
- Position IDs are 0, 1, 2, ..., seq_len-1
