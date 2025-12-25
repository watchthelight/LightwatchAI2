# Phase 26: Checkpointing

## Objective
Implement training checkpoint save/load for model state, optimizer state, and training progress.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 05 | Autograd with Variable |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable |
| 22 | include/lightwatch/optim/optimizer.hpp | Optimizer::state_dict |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/checkpoint.hpp | save_checkpoint, load_checkpoint | Phase 29, 37 |
| src/checkpoint.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

struct Checkpoint {
    std::unordered_map<std::string, Tensor<float>> model_state;
    std::unordered_map<std::string, Tensor<float>> optimizer_state;
    int epoch;
    int step;
    float loss;
    std::string config_json;
};

// Save checkpoint to file
void save_checkpoint(
    const std::string& path,
    const nn::Module& model,
    const optim::Optimizer& optimizer,
    int epoch,
    int step,
    float loss,
    const std::string& config = "");

// Load checkpoint from file
Checkpoint load_checkpoint(const std::string& path);

// Apply checkpoint to model and optimizer
void restore_checkpoint(
    const Checkpoint& ckpt,
    nn::Module& model,
    optim::Optimizer& optimizer);

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Save**: Serialize model state_dict, optimizer state_dict, training metadata
2. **Load**: Deserialize and validate tensor shapes
3. **Format**: Use binary format with header for fast I/O
4. **Atomic writes**: Write to temp file, then rename

### Performance Constraints
- Save: O(model_size) time
- Load: O(model_size) time
- File size: ~2x model size (model + optimizer states)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_26_save_load` | Save and reload | States match |
| `test_phase_26_optimizer_state` | Save optimizer | Momentum buffers preserved |
| `test_phase_26_metadata` | epoch, step, loss | Correctly restored |
| `test_phase_26_atomic` | Interrupt during save | No corrupt file |
| `test_phase_26_missing_key` | Partial state dict | Warning + skip |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_26" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/checkpoint.hpp`
- [ ] Checkpoint roundtrip preserves all state

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Checkpoint file extension: .ckpt
- Include version number for format compatibility
- Atomic write prevents corruption on crash
- strict=false allows loading partial state dicts
