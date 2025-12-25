# Phase 31: GPT Architecture

## Objective
Assemble the complete GPT-2 model from transformer decoder blocks with proper weight tying.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 19 | TransformerDecoderBlock |
| 20 | Neural core tests passing |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 08 | include/lightwatch/nn/embedding.hpp | GPTEmbedding |
| 13 | include/lightwatch/nn/normalization.hpp | LayerNorm |
| 19 | include/lightwatch/nn/transformer.hpp | TransformerDecoderBlock |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/models/gpt.hpp | GPT2, GPT2Config | Phase 32-38 |
| src/models/gpt.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::models {

struct GPT2Config {
    size_t vocab_size = 50257;
    size_t max_seq_len = 1024;
    size_t embed_dim = 768;
    size_t num_heads = 12;
    size_t num_layers = 12;
    size_t ffn_dim = 3072;  // 4 * embed_dim
    float dropout_p = 0.1;
    float attn_dropout_p = 0.1;
    bool tie_weights = true;

    static GPT2Config gpt2_small();
    static GPT2Config gpt2_medium();
    static GPT2Config gpt2_large();
    static GPT2Config gpt2_xl();
};

class GPT2 : public nn::Module {
public:
    explicit GPT2(GPT2Config config = GPT2Config::gpt2_small());

    // Forward pass: logits = model(input_ids)
    autograd::Variable forward(const Tensor<int32_t>& input_ids);
    autograd::Variable forward(const autograd::Variable& input) override;

    // Get hidden states (for analysis)
    autograd::Variable get_hidden_states(const Tensor<int32_t>& input_ids);

    GPT2Config config() const;
    size_t num_parameters() const;

    nn::GPTEmbedding embedding;
    std::vector<std::shared_ptr<nn::TransformerDecoderBlock>> layers;
    nn::LayerNorm ln_f;  // Final layer norm
    nn::Linear lm_head;  // Language model head

private:
    GPT2Config config_;
};

}  // namespace lightwatch::models
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Forward pass**:
   - x = embedding(input_ids)  // token + position
   - for layer in layers: x = layer(x)
   - x = ln_f(x)
   - logits = lm_head(x)
2. **Weight tying**: lm_head.weight = embedding.wte.weight.T
3. **Pre-norm**: LayerNorm before attention/FFN in each block

### Performance Constraints
- Memory: ~500MB for GPT-2 Small weights
- Forward: O(seq² * embed + seq * vocab) for attention + projection

## Required Tests
See `docs/test_specs/phase-31-gpt.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_31_gpt2_shape` | {1, 16} input_ids | logits {1, 16, 50257} |
| `test_phase_31_gpt2_params` | Count parameters | ~124M for small |
| `test_phase_31_weight_tying` | Check weights | lm_head.weight == wte.weight.T |
| `test_phase_31_causal` | Attention check | No future leakage |
| `test_phase_31_forward_backward` | Full pass | Gradients exist |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_31" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/models/gpt.hpp`
- [ ] Parameter count matches GPT-2 Small (~124M)

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 1 |
| Complexity | HIGH |

## Notes
- GPT-2 Small: 12 layers, 768 hidden, 12 heads
- Weight tying reduces parameters by ~vocab_size * embed_dim
- Final LayerNorm (ln_f) is GPT-2 specific
- Pre-norm architecture (norm before attention)
