# Phase 36: KV-Cache - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 5

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_kv_cache_incremental` | Generate 10 tokens | Same result as full recompute |
| `test_kv_cache_shape` | After 5 tokens | Cache shape `[B, H, 5, D]` |
| `test_kv_cache_reset` | New sequence | Cache cleared |
| `test_kv_cache_memory` | 1024 tokens | Memory ~75MB per batch |
| `test_kv_cache_speedup` | Cached vs uncached | Cached ≥2x faster |

## Implementation Notes

- KV-cache stores key/value tensors for all previous positions
- Memory per layer: 2 * seq_len * n_heads * head_dim * sizeof(float)
- Full context (1024 tokens, 12 layers): ~75MB
- Cache must be cleared when starting new sequence
- Incremental generation should produce identical results to full recompute
