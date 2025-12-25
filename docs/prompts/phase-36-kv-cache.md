# Phase 36: KV-Cache

## Objective
Implement key-value caching for efficient incremental generation.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 35 | Sampling generation |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 35 | include/lightwatch/generate.hpp | SamplingConfig |
| 16 | include/lightwatch/nn/attention.hpp | MultiHeadAttention |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/cache.hpp | KVCache | Phase 38 |
| include/lightwatch/generate.hpp | generate_with_cache (extended) | Phase 38 |
| src/cache.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

class KVCache {
public:
    KVCache(size_t num_layers, size_t num_heads, size_t head_dim,
            size_t max_seq_len, size_t batch_size = 1);

    // Get cached K, V for a layer
    std::pair<Tensor<float>, Tensor<float>> get(size_t layer) const;

    // Update cache with new K, V
    void update(size_t layer, const Tensor<float>& new_k,
                const Tensor<float>& new_v);

    // Current sequence length in cache
    size_t seq_len() const;

    // Clear cache for new generation
    void reset();

private:
    std::vector<Tensor<float>> k_cache_;  // {num_layers}
    std::vector<Tensor<float>> v_cache_;  // Each: {batch, heads, seq, head_dim}
    size_t current_len_ = 0;
};

// Generation with KV cache (incremental)
std::vector<TokenId> generate_with_cache(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    KVCache& cache,
    SamplingConfig config = {});

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **First token**: Full forward pass, cache all K, V
2. **Subsequent tokens**: Only process new token, use cached K, V
3. **Attention**: Query new, Key/Value concatenated with cache
4. **Memory**: Pre-allocate for max_seq_len to avoid reallocation

### Performance Constraints
- With cache: O(seq) per token instead of O(seq²)
- Memory: O(layers * seq * head_dim) per batch

## Required Tests
See `docs/test_specs/phase-36-kvcache.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_36_cache_update` | Add K, V | seq_len increases |
| `test_phase_36_cache_equivalence` | With vs without cache | Same output |
| `test_phase_36_cache_speed` | 100 tokens | Faster with cache |
| `test_phase_36_cache_memory` | Check memory | O(seq) not O(seq²) |
| `test_phase_36_cache_reset` | Generate twice | Independent |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_36" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/cache.hpp`
- [ ] Cached generation matches non-cached output

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 1 |
| Complexity | HIGH |

## Notes
- KV cache is critical for fast autoregressive generation
- Memory budget: ~75MB for full context (1024 tokens)
- Cache must be reset between independent generations
- Attention layer needs modification to accept cached K, V
