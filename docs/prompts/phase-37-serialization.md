# Phase 37: Serialization

## Objective
Implement model weight serialization in .lwbin format with HuggingFace conversion scripts.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 31 | GPT2 model with state_dict |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 31 | include/lightwatch/models/gpt.hpp | GPT2 |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/serialize.hpp | save_weights, load_weights | Phase 38 |
| scripts/convert_hf_to_lwbin.py | HF -> lwbin converter | External |
| scripts/convert_lwbin_to_hf.py | lwbin -> HF converter | External |
| src/serialize.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

// Native .lwbin format
// Header (64 bytes):
//   Magic: "LWAI" (4 bytes)
//   Version: uint32_t (4 bytes)
//   Tensor count: uint32_t (4 bytes)
//   Reserved: 52 bytes

struct TensorMetadata {
    std::string name;
    std::vector<int64_t> shape;
    uint8_t dtype;  // 0=float32, 1=float16, 2=int32
};

void save_weights(const std::string& path, const nn::Module& model);
void load_weights(const std::string& path, nn::Module& model);

// Utilities
std::vector<TensorMetadata> inspect_weights(const std::string& path);
bool validate_weights(const std::string& path, const nn::Module& model);

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Save**: Write header, then for each tensor: name, shape, data
2. **Load**: Read header, verify magic/version, load tensors by name
3. **Endianness**: Little-endian for portability
4. **Validation**: Check shapes match model architecture

### Performance Constraints
- Save/load: O(model_size) - limited by I/O
- Memory: Stream tensors, don't load all into memory

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_37_save_load` | Save + load | State matches |
| `test_phase_37_header` | Check header | Magic, version correct |
| `test_phase_37_validate` | Wrong shape | Returns false |
| `test_phase_37_inspect` | List tensors | Names and shapes |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_37" --output-on-failure` exits 0
- [ ] `test -f scripts/convert_hf_to_lwbin.py`
- [ ] Roundtrip preserves all weights

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- .lwbin is native format, optimized for fast loading
- Conversion scripts enable using HuggingFace pretrained weights
- Version number allows format evolution
- Strict mode rejects shape mismatches; lenient mode warns
