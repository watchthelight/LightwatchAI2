# Phase 35: Sampling

## Objective
Implement temperature, top-k, and top-p (nucleus) sampling for diverse text generation.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 34 | Greedy generation infrastructure |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 34 | include/lightwatch/generate.hpp | GenerateConfig |
| 31 | include/lightwatch/models/gpt.hpp | GPT2 |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/generate.hpp | generate_sample (extended) | Phase 36, 38 |
| src/generate.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

struct SamplingConfig : GenerateConfig {
    float temperature = 1.0;    // Logit scaling
    int top_k = 0;              // 0 = disabled
    float top_p = 1.0;          // 1.0 = disabled
    bool do_sample = true;      // false = greedy
    unsigned int seed = 0;      // 0 = random seed
};

// Sample-based generation
std::vector<TokenId> generate_sample(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    SamplingConfig config = {});

// Apply sampling transformations to logits
Tensor<float> apply_temperature(const Tensor<float>& logits, float temperature);
Tensor<float> apply_top_k(const Tensor<float>& logits, int k);
Tensor<float> apply_top_p(const Tensor<float>& logits, float p);

// Sample from probability distribution
TokenId sample_token(const Tensor<float>& probs, std::mt19937& rng);

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Temperature**: logits = logits / temperature
2. **Top-k**: Zero out all except top k logits
3. **Top-p**: Zero out tokens with cumulative prob > p
4. **Sample**: Random sample from softmax(logits)
5. **Order**: temperature -> top_k -> top_p -> sample

### Performance Constraints
- Top-k: O(vocab * log(k)) with partial sort
- Top-p: O(vocab * log(vocab)) for full sort

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_35_temperature_0` | temp=0.0001 | Nearly greedy |
| `test_phase_35_temperature_high` | temp=2.0 | More diverse |
| `test_phase_35_top_k` | k=10 | Only top 10 possible |
| `test_phase_35_top_p` | p=0.9 | Nucleus subset |
| `test_phase_35_seed` | Same seed | Reproducible |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_35" --output-on-failure` exits 0
- [ ] Temperature/top-k/top-p produce expected diversity
- [ ] Sampling is reproducible with fixed seed

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-450 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- temperature=0 equivalent to greedy (use very small value)
- top_k=40 is a common setting
- top_p=0.9 (nucleus sampling) often preferred
- Seed for reproducibility in tests/demos
