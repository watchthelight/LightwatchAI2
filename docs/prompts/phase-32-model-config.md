# Phase 32: Model Config

## Objective
Implement JSON-based model configuration with presets and validation.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 31 | GPT2, GPT2Config |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 31 | include/lightwatch/models/gpt.hpp | GPT2Config |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/config.hpp | load_config, save_config, ModelConfig | Phase 38 |
| configs/gpt2-small.json | Preset configuration | Phase 38 |
| src/config.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

struct ModelConfig {
    std::string model_type;  // "gpt2"
    models::GPT2Config gpt2;

    // I/O
    static ModelConfig from_json(const std::string& json);
    std::string to_json() const;

    static ModelConfig load(const std::string& path);
    void save(const std::string& path) const;
};

// Config validation
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void validate_config(const ModelConfig& config);

}  // namespace lightwatch
```

### Function Signatures
```cpp
// JSON format (using nlohmann/json)
{
    "model_type": "gpt2",
    "vocab_size": 50257,
    "max_seq_len": 1024,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "ffn_dim": 3072,
    "dropout_p": 0.1,
    "attn_dropout_p": 0.1,
    "tie_weights": true
}
```

### Algorithmic Requirements
1. **Parse**: JSON to GPT2Config struct
2. **Validate**: Check constraints (head_dim divides embed_dim, etc.)
3. **Presets**: Load from named configs (gpt2-small, gpt2-medium, etc.)
4. **Serialize**: Config to JSON for saving

### Performance Constraints
- Config load: O(1) (small file)
- Validation: O(1)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_32_load_preset` | "gpt2-small" | Correct config |
| `test_phase_32_json_roundtrip` | Save + load | Config matches |
| `test_phase_32_validation` | Invalid config | Throws ConfigError |
| `test_phase_32_custom_config` | Modified JSON | Parsed correctly |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_32" --output-on-failure` exits 0
- [ ] `test -f configs/gpt2-small.json`
- [ ] Config files parse correctly

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 200-350 |
| New source files | 2 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- Uses nlohmann/json for parsing
- Presets in configs/ directory
- Validation catches common errors early
- Config is passed to GPT2 constructor
