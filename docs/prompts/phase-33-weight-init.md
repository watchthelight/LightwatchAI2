# Phase 33: Weight Init

## Objective
Implement GPT-2 specific weight initialization scheme.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 31 | GPT2 model |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 31 | include/lightwatch/models/gpt.hpp | GPT2 |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/init.hpp | init_gpt2_weights, InitMethod | Phase 38 |
| src/init.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::init {

enum class InitMethod {
    NORMAL,      // N(0, std)
    XAVIER,      // Xavier/Glorot
    GPT2         // GPT-2 specific
};

struct InitConfig {
    float std = 0.02;           // Base std for embeddings
    float residual_std = 0.02;  // For residual projections
    bool scale_by_depth = true; // Scale residual by 1/sqrt(2*n_layers)
};

// Initialize all weights in model
void init_gpt2_weights(models::GPT2& model, InitConfig config = {});

// Individual layer initialization
void init_linear(nn::Linear& layer, float std);
void init_embedding(nn::Embedding& layer, float std);
void init_layer_norm(nn::LayerNorm& layer);

}  // namespace lightwatch::init
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Embeddings**: N(0, 0.02)
2. **Linear layers**: N(0, 0.02) for weights, zeros for bias
3. **Residual projections**: N(0, 0.02 / sqrt(2 * n_layers))
4. **LayerNorm**: weight = 1, bias = 0

### Performance Constraints
- O(total_parameters) initialization time
- Random seed for reproducibility

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_33_init_std` | Check weight std | ~0.02 |
| `test_phase_33_init_residual` | Residual layer | Scaled by depth |
| `test_phase_33_init_layernorm` | LN weights | weight=1, bias=0 |
| `test_phase_33_init_seed` | Same seed twice | Identical weights |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_33" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/init.hpp`
- [ ] Weights have correct distribution

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 150-250 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 uses std=0.02 for most weights
- Residual scaling prevents exploding activations
- Seed configuration for reproducible experiments
- init_gpt2_weights applies correct init to each layer type
