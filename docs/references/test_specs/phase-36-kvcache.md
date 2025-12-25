<!-- File: docs/references/test_specs/phase-36-kvcache.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 36: KV-Cache - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 5

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_36_kvcache_shape` | After 10 tokens | Cache shape `{12, 2, 10, 64}` (layers, kv, seq, head_dim) |
| `test_phase_36_kvcache_equivalence` | Generate 20 tokens with/without cache | Identical output tokens |
| `test_phase_36_kvcache_speedup` | 128 tokens | Cached ≥2x faster than uncached |
| `test_phase_36_kvcache_memory` | 1024 tokens | Cache uses ~75MB (12 layers × 2 × 1024 × 768 × 4 bytes) |
| `test_phase_36_kvcache_clear` | Generate, clear, generate | Second generation starts fresh |

## Implementation Notes

- Cache stores K and V for all attention layers
- Memory: 12 layers × 2 (K,V) × seq_len × 768 × 4 bytes = 75MB at 1024 tokens
- Incremental: only compute attention for new token, use cached K,V for context
