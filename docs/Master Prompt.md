# LightwatchAI2 - C++ Text Generation Model Build System

## Claude Code Master Orchestration Prompt

---

## PREAMBLE

You are building a **minimal-dependency C++ text generation model** from scratch. The only allowed external dependency is a JSON parsing library (nlohmann/json or similar). No Eigen, OpenBLAS, PyTorch, TensorFlow, or other ML frameworks.

**Repository:** https://github.com/watchthelight/LightwatchAI2

---

## CRITICAL CONSTRAINTS

### Commit Authorship Policy
All commits MUST be authored by watchthelight, not Claude. This is configured in CLAUDE.md and enforced via:
```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" commit -m "message"
```

### Dependencies
- **REQUIRED:** C++17 or later, CMake 3.16+
- **ALLOWED:** nlohmann/json (header-only), standard library only
- **FORBIDDEN:** Eigen, OpenBLAS, BLAS, LAPACK, any ML framework

### Performance Target
- Inference: >100 tokens/second on modern CPU (with SIMD)
- If AVX2 implementation achieves >50 tok/s → proceed
- If <50 tok/s → document bottleneck, add OPTIONAL OpenBLAS path behind compile flag
- External BLAS must NEVER be required

---

## TARGET MODEL SPECIFICATION

This project implements **GPT-2 Small (124M parameters)**.

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Vocab size | 50257 |
| Context length | 1024 |
| FFN hidden | 3072 (4x hidden) |
| Parameters | ~124M |

All architecture decisions target this configuration. Do not over-engineer for larger models.

### Weight Compatibility
The serialization format (Phase 37) must be compatible with HuggingFace GPT-2 weights for optional pretrained weight loading. This enables:
1. Validation against reference implementation
2. Inference without training
3. Fine-tuning from pretrained weights

### Memory Budget
| Component | Size (fp32) |
|-----------|-------------|
| Embeddings (wte + wpe) | ~154 MB |
| Attention weights (per layer) | ~9.4 MB |
| FFN weights (per layer) | ~18.9 MB |
| Total model | ~497 MB |
| KV-cache (full context) | ~75 MB |

---

## TOOLCHAIN REQUIREMENTS

| Tool | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| CMake | 3.16 | 3.28+ | For FetchContent, presets |
| GCC | 10 | 13+ | C++17 constexpr support |
| Clang | 12 | 17+ | Better diagnostics |
| MSVC | 19.29 | 19.38+ | VS 2022 recommended |
| Python | 3.9 | 3.11+ | For validation scripts |
| jq | 1.6 | 1.7+ | JSON parsing in scripts |
| valgrind | 3.15 | 3.21+ | Memory leak detection (Linux) |
| curl | 7.68 | 8.0+ | Asset downloads |

### Version Check Script
Create during Phase 0 at `scripts/check_toolchain.sh`:
```bash
#!/bin/bash
set -e
echo "=== Toolchain Versions ==="
cmake --version | head -1
${CXX:-g++} --version | head -1
python3 --version
jq --version
curl --version | head -1
valgrind --version 2>/dev/null || echo "valgrind: not installed (optional on macOS)"
echo "=== Check Complete ==="
```

Run during Phase 0 and document results in `docs/architecture/TOOLCHAIN.md`.

---

## STATE MANAGEMENT

### State File: .lightwatch_state.json
```json
{
  "project_status": "IN_PROGRESS",
  "generation_complete": false,
  "current_phase": 0,
  "phase_status": "NOT_STARTED",
  "current_branch": "main",
  "completed_phases": [],
  "failed_phases": [],
  "completed_deliverables": [],
  "pending_deliverables": [],
  "last_commit": "",
  "validation_passed": false,
  "checkpoints_passed": [],
  "escalations": []
}
```

**Project status values:**
- `"IN_PROGRESS"` — Normal execution
- `"COMPLETE"` — All 40 phases done, verify_complete.sh passed
- `"ESCALATED"` — Waiting for human input

**On session start:**
1. Read `.lightwatch_state.json` (create if missing)
2. If `project_status == "COMPLETE"` → report completion, no action needed
3. If `project_status == "ESCALATED"` → STOP, wait for human input
4. If `generation_complete == false` → resume/start Phase 0.5 (prompt generation)
5. If `phase_status == "EXECUTING"` → resume from `pending_deliverables`
6. If `phase_status == "VERIFYING"` → re-run verification
7. If `phase_status == "FAILED"` → read error, attempt fix or escalate

**On phase completion:**
1. Update state file with completed deliverables
2. Set `phase_status = "COMPLETE"`
3. Add phase number to `completed_phases`
4. Commit state file

**On phase failure:**
1. Set `phase_status = "FAILED"`
2. Add phase number to `failed_phases`
3. Document error in state file
4. Do NOT proceed to dependent phases

**On partial phase failure (session interrupted mid-phase):**
1. Set `phase_status = "PARTIAL_FAILURE"`
2. Record completed deliverables in `completed_deliverables` array
3. Record remaining items in `pending_deliverables` array
4. Commit state file with partial progress
5. On resume:
   - `git status` to check for uncommitted changes
   - If changes exist: review and commit or discard
   - `git reset --hard` to last passing commit if needed
   - Continue from `pending_deliverables[0]`
   - Re-run acceptance criteria for completed items to verify

**On project completion:**
1. Run `scripts/verify_complete.sh`
2. If exits 0: Set `project_status = "COMPLETE"`
3. Commit final state

---

## PHASE EXECUTION ORDER

```
Phase 0: Project Bootstrap
    ↓
Phase 0.3: Create API Contract Files
    ↓
Phase 0.7: Acquire External Assets
    ↓
Phase 0.5: Generate All 40 Phase Prompts
    ↓
Phase 0.9: Toolchain Smoke Test
    ↓
[Validation: scripts/validate_prompts.py must exit 0]
    ↓
Phases 1-40: Implementation (respecting dependency graph)
    ↓
[Checkpoint validations at Phases 10, 20, 30, 40]
    ↓
[Verification: scripts/verify_complete.sh must exit 0]
```

**Critical:** Do NOT proceed to Phase 1 until:
1. All Phase 0.x phases are complete
2. `scripts/validate_prompts.py` exits 0
3. Phase 0.9 smoke test passes

> **Note:** Phase 0.5 (prompt generation) comes AFTER Phase 0.7 (asset acquisition)
> because prompts may reference downloaded assets for validation. The numbering
> is non-sequential by design: 0 → 0.3 → 0.7 → 0.5 → 0.9.

---

## PHASE 0: PROJECT BOOTSTRAP

### Objective
Create the project skeleton, build system, configuration files, and infrastructure scripts.

### Deliverables
```
LightwatchAI2/
├── .gitignore
├── .lightwatch_state.json
├── CMakeLists.txt
├── CLAUDE.md
├── README.md
├── docs/
│   ├── prompts/              # Generated phase prompts
│   ├── contracts/            # API contract headers
│   ├── architecture/
│   │   ├── DECISIONS.md      # Architectural decision log
│   │   ├── TOOLCHAIN.md      # Toolchain versions used
│   │   └── ESCALATIONS.md    # Escalation log (if any)
│   └── plans/                # Phase implementation plans
├── scripts/
│   ├── check_toolchain.sh
│   ├── validate_prompts.py
│   ├── verify_complete.sh
│   └── acquire_assets.sh
├── include/
│   └── lightwatch/
├── src/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── benchmarks/
├── configs/
└── data/
    ├── vocab/                # Tokenizer assets
    └── weights/              # Optional pretrained weights
```

### CLAUDE.md Content
```markdown
# Claude Code Configuration

## Commit Authorship
ALL commits must use:
- Author: watchthelight <buteverythingisnormal@gmail.com>
- Command: git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" commit

Claude MUST NOT be listed as commit author under any circumstances.

## Code Style
- C++17 standard
- 4-space indentation
- snake_case for functions/variables
- PascalCase for classes/types
- UPPER_CASE for constants

## Testing
- All public APIs must have unit tests
- Integration tests for cross-component functionality
- Benchmarks for performance-critical code

## Model Target
- GPT-2 Small (124M parameters)
- 12 layers, 768 hidden, 12 heads
- 50257 vocab, 1024 context
```

### docs/architecture/DECISIONS.md Template
```markdown
# Architectural Decisions

This document records significant architectural decisions made during development.

## Format

Each decision follows this template:

### [YYYY-MM-DD] Decision Title

**Phase:** N (where the decision was made)

**Context:**
What situation prompted this decision? What constraints existed?

**Options Considered:**
1. Option A — pros/cons
2. Option B — pros/cons
3. Option C — pros/cons

**Decision:**
What was decided and why?

**Consequences:**
- Positive: What benefits does this bring?
- Negative: What tradeoffs were accepted?
- Phases Affected: Which phases are impacted?

---

## Decision Log

(Entries will be added as decisions are made)
```

### Example Decision Entry
```markdown
### [2025-01-15] Use Row-Major Tensor Layout

**Phase:** 03 (Tensor Core)

**Context:**
Needed to decide between row-major (C-style) and column-major (Fortran-style) memory layout for tensors. This affects cache performance and compatibility with external libraries.

**Options Considered:**
1. Row-major — Matches C++ array semantics, better for batch processing, standard in PyTorch
2. Column-major — Better for linear algebra, standard in NumPy/BLAS
3. Configurable — Maximum flexibility but implementation complexity

**Decision:**
Row-major layout. Rationale:
- Matches C++ native arrays
- PyTorch compatibility (for weight loading)
- Batch dimension is first, which is common access pattern
- SIMD vectorization works well with contiguous rows

**Consequences:**
- Positive: Simpler implementation, good cache locality for inference
- Negative: May need transpose for some BLAS operations if optional BLAS is enabled
- Phases Affected: 04 (SIMD must respect layout), 37 (serialization must match PyTorch)
```

### Acceptance Criteria
- [ ] `cmake -B build -S .` exits 0
- [ ] `cmake --build build` exits 0
- [ ] `.lightwatch_state.json` exists and `jq . .lightwatch_state.json` exits 0
- [ ] `git log -1 --format='%ae'` outputs `buteverythingisnormal@gmail.com`
- [ ] `test -d docs/contracts && test -d docs/prompts && test -d scripts`
- [ ] `bash scripts/check_toolchain.sh` exits 0
- [ ] `test -f docs/architecture/DECISIONS.md`

---

## PHASE 0.3: CREATE API CONTRACT FILES

### Objective
Create authoritative API contract files that define cross-phase interfaces. These are the source of truth for all generated prompts.

### Deliverables

#### docs/contracts/tensor.hpp
```cpp
// LightwatchAI2 API Contract: Tensor
// Defined by: Phase 03
// Consumers: 04, 05, 08, 09, 11-19, 21-25, 31-36
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include <vector>
#include <initializer_list>
#include <memory>
#include <cstddef>

namespace lightwatch {

using Shape = std::vector<size_t>;

template<typename T>
class Tensor {
public:
    // Construction
    Tensor();
    explicit Tensor(const Shape& shape);
    Tensor(const Shape& shape, const T* data);
    Tensor(const Shape& shape, std::initializer_list<T> data);

    // Static factories
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor full(const Shape& shape, T value);
    static Tensor randn(const Shape& shape);  // Normal distribution N(0,1)
    static Tensor rand(const Shape& shape);   // Uniform [0,1)

    // Element access
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
    T& at(const std::vector<size_t>& indices);
    const T& at(const std::vector<size_t>& indices) const;

    // Properties
    const Shape& shape() const;
    size_t size(int dim) const;  // Negative dims count from end
    size_t numel() const;        // Total elements
    size_t ndim() const;
    T* data();
    const T* data() const;

    // Shape operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const;  // May share data
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    Tensor contiguous() const;
    bool is_contiguous() const;

    // Reductions
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim = -1, bool keepdim = false) const;
    Tensor min(int dim = -1, bool keepdim = false) const;
    Tensor var(int dim = -1, bool keepdim = false) const;
    T item() const;  // For scalar tensors

    // Element-wise ops (return new tensor)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Hadamard product
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;
    Tensor abs() const;
    Tensor sqrt() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(T exponent) const;

    // Scalar ops
    Tensor operator+(T scalar) const;
    Tensor operator-(T scalar) const;
    Tensor operator*(T scalar) const;
    Tensor operator/(T scalar) const;

    // In-place ops (return reference for chaining)
    Tensor& fill_(T value);
    Tensor& zero_();
    Tensor& add_(const Tensor& other);
    Tensor& sub_(const Tensor& other);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);

    // Comparison (return boolean tensor)
    Tensor<bool> operator==(const Tensor& other) const;
    Tensor<bool> operator!=(const Tensor& other) const;
    Tensor<bool> operator<(const Tensor& other) const;
    Tensor<bool> operator<=(const Tensor& other) const;
    Tensor<bool> operator>(const Tensor& other) const;
    Tensor<bool> operator>=(const Tensor& other) const;

    // Utilities
    Tensor clone() const;

private:
    Shape shape_;
    std::shared_ptr<T[]> data_;
    std::vector<size_t> strides_;
    size_t offset_ = 0;
};

// Free functions
template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);

template<typename T>
Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int dim);

template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int dim);

template<typename T>
Tensor<T> where(const Tensor<bool>& condition, const Tensor<T>& x, const Tensor<T>& y);

}  // namespace lightwatch
```

#### docs/contracts/autograd.hpp
```cpp
// LightwatchAI2 API Contract: Autograd
// Defined by: Phase 05
// Consumers: 08, 11-19, 21-25, 31
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace lightwatch::autograd {

// Forward declarations
class Function;
class Variable;

class Variable {
public:
    Variable();
    explicit Variable(Tensor<float> data, bool requires_grad = false);

    // Access underlying tensor
    Tensor<float>& data();
    const Tensor<float>& data() const;

    // Gradient access
    Tensor<float>& grad();
    const Tensor<float>& grad() const;
    bool has_grad() const;
    bool requires_grad() const;
    void set_requires_grad(bool requires);
    void zero_grad();

    // Shape delegation
    const Shape& shape() const;
    size_t size(int dim) const;
    size_t numel() const;
    size_t ndim() const;

    // Backward pass
    void backward();
    void backward(const Tensor<float>& grad_output);

    // Computation graph
    void set_grad_fn(std::shared_ptr<Function> fn);
    std::shared_ptr<Function> grad_fn() const;

    // Detach from graph (returns new Variable with no grad_fn)
    Variable detach() const;

    // Retain gradient for non-leaf variables
    void retain_grad();

private:
    Tensor<float> data_;
    Tensor<float> grad_;
    bool requires_grad_ = false;
    bool has_grad_ = false;
    bool retain_grad_ = false;
    std::shared_ptr<Function> grad_fn_;
};

class Function {
public:
    virtual ~Function() = default;

    // Compute gradients given upstream gradient
    virtual std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) = 0;

    // Access saved tensors/variables
    const std::vector<Variable>& saved_variables() const;

protected:
    // Context for saving values needed in backward
    void save_for_backward(const Variable& v);
    void save_for_backward(const Tensor<float>& t);

    std::vector<Variable> saved_variables_;
    std::vector<Tensor<float>> saved_tensors_;
};

// Differentiable operations (return Variable with grad tracking)
namespace ops {
    // Arithmetic
    Variable add(const Variable& a, const Variable& b);
    Variable sub(const Variable& a, const Variable& b);
    Variable mul(const Variable& a, const Variable& b);  // Element-wise
    Variable div(const Variable& a, const Variable& b);
    Variable neg(const Variable& x);

    // Matrix operations
    Variable matmul(const Variable& a, const Variable& b);
    Variable transpose(const Variable& x, int dim0, int dim1);

    // Activations
    Variable relu(const Variable& x);
    Variable gelu(const Variable& x);
    Variable silu(const Variable& x);  // Swish
    Variable sigmoid(const Variable& x);
    Variable tanh(const Variable& x);
    Variable softmax(const Variable& x, int dim);
    Variable log_softmax(const Variable& x, int dim);

    // Reductions
    Variable sum(const Variable& x, int dim = -1, bool keepdim = false);
    Variable mean(const Variable& x, int dim = -1, bool keepdim = false);

    // Shape operations
    Variable reshape(const Variable& x, const Shape& new_shape);
    Variable squeeze(const Variable& x, int dim = -1);
    Variable unsqueeze(const Variable& x, int dim);

    // Indexing
    Variable slice(const Variable& x, int dim, size_t start, size_t end);
    Variable index_select(const Variable& x, int dim, const Tensor<int32_t>& indices);

    // Misc
    Variable dropout(const Variable& x, float p, bool training);
    Variable layer_norm(const Variable& x, const Variable& weight, const Variable& bias, float eps);
}

// No-grad context (RAII)
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    static thread_local int guard_count_;
};

bool is_grad_enabled();

}  // namespace lightwatch::autograd
```

#### docs/contracts/tokenizer.hpp
```cpp
// LightwatchAI2 API Contract: Tokenizer
// Defined by: Phases 06-07
// Consumers: 08, 27, 38
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace lightwatch::tokenizer {

using TokenId = int32_t;

// GPT-2 special token IDs (from official tokenizer)
struct SpecialTokens {
    static constexpr TokenId PAD = 50256;  // Same as EOS for GPT-2
    static constexpr TokenId UNK = 50256;  // GPT-2 has no UNK, uses byte fallback
    static constexpr TokenId BOS = 50256;  // Not used in GPT-2
    static constexpr TokenId EOS = 50256;  // <|endoftext|>
};

class Vocabulary {
public:
    Vocabulary();

    // Token operations
    TokenId add_token(const std::string& token);
    TokenId token_to_id(const std::string& token) const;
    std::string id_to_token(TokenId id) const;

    bool contains(const std::string& token) const;
    bool contains(TokenId id) const;
    size_t size() const;

    // Special tokens
    TokenId pad_id() const;
    TokenId eos_id() const;
    bool is_special_token(TokenId id) const;

    // Serialization
    void save(const std::string& path) const;
    static Vocabulary load(const std::string& path);

    // Load from GPT-2 format
    static Vocabulary from_encoder_json(const std::string& path);

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;
};

class BPETokenizer {
public:
    BPETokenizer();

    // Encode text to token IDs
    std::vector<TokenId> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<TokenId>& tokens) const;

    // Batch operations
    std::vector<std::vector<TokenId>> encode_batch(
        const std::vector<std::string>& texts) const;
    std::vector<std::string> decode_batch(
        const std::vector<std::vector<TokenId>>& token_batches) const;

    // Vocabulary access
    const Vocabulary& vocab() const;
    size_t vocab_size() const;

    // Special tokens
    TokenId pad_id() const;
    TokenId eos_id() const;

    // Factory methods
    static BPETokenizer from_files(
        const std::string& vocab_path,    // encoder.json
        const std::string& merges_path);  // vocab.bpe

    static BPETokenizer gpt2(const std::string& vocab_dir = "data/vocab");

    // Serialization
    void save(const std::string& path) const;
    static BPETokenizer load(const std::string& path);

private:
    Vocabulary vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    // Hash function for string pairs
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;
};

}  // namespace lightwatch::tokenizer
```

#### docs/contracts/module.hpp
```cpp
// LightwatchAI2 API Contract: Module
// Defined by: Phase 11
// Consumers: 12-19, 31
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "autograd.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <ostream>
#include <istream>

namespace lightwatch::nn {

class Module {
public:
    virtual ~Module() = default;

    // Forward pass - derived classes implement this
    virtual autograd::Variable forward(const autograd::Variable& input) = 0;

    // Multi-input forward (for attention, etc.)
    virtual autograd::Variable forward(
        const autograd::Variable& input,
        const autograd::Variable& other) {
        (void)other;
        return forward(input);
    }

    // Parameter access
    std::vector<autograd::Variable*> parameters();
    std::vector<std::pair<std::string, autograd::Variable*>> named_parameters();
    size_t num_parameters() const;

    // Submodule access
    std::vector<Module*> modules();
    std::vector<std::pair<std::string, Module*>> named_modules();

    // Training mode
    void train(bool mode = true);
    void eval();
    bool is_training() const;

    // Gradient control
    void zero_grad();
    void requires_grad_(bool requires_grad);

    // Serialization
    virtual void save_state(std::ostream& os) const;
    virtual void load_state(std::istream& is);

    // State dict (for HuggingFace compatibility)
    std::unordered_map<std::string, Tensor<float>> state_dict() const;
    void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict);

protected:
    bool training_ = true;

    // Registration
    void register_parameter(const std::string& name, autograd::Variable& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);
    void register_buffer(const std::string& name, Tensor<float>& buffer);

private:
    std::vector<std::pair<std::string, autograd::Variable*>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
    std::vector<std::pair<std::string, Tensor<float>*>> buffers_;
};

// Common layer types (signatures only - implementations in respective phases)

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;
    autograd::Variable bias;

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
};

class LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;
    autograd::Variable bias;

private:
    size_t normalized_shape_;
    float eps_;
};

class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);
    autograd::Variable forward(const Tensor<int32_t>& indices);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
};

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1);
    autograd::Variable forward(const autograd::Variable& input) override;

private:
    float p_;
};

}  // namespace lightwatch::nn
```

#### docs/contracts/optimizer.hpp
```cpp
// LightwatchAI2 API Contract: Optimizer
// Defined by: Phase 22
// Consumers: 23-26, 29
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "autograd.hpp"
#include <vector>
#include <unordered_map>
#include <string>

namespace lightwatch::optim {

struct OptimizerOptions {
    float lr = 1e-3;
    float weight_decay = 0.0;
};

class Optimizer {
public:
    explicit Optimizer(std::vector<autograd::Variable*> params, OptimizerOptions options = {});
    virtual ~Optimizer() = default;

    // Perform one optimization step
    virtual void step() = 0;

    // Zero all parameter gradients
    virtual void zero_grad();

    // Parameter group management
    void add_param_group(std::vector<autograd::Variable*> params, OptimizerOptions options = {});

    // Learning rate access
    float get_lr() const;
    void set_lr(float lr);

    // State access (for checkpointing)
    virtual std::unordered_map<std::string, Tensor<float>> state_dict() const;
    virtual void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict);

protected:
    struct ParamGroup {
        std::vector<autograd::Variable*> params;
        OptimizerOptions options;
    };

    std::vector<ParamGroup> param_groups_;

    // Per-parameter state (momentum buffers, Adam moments, etc.)
    std::unordered_map<autograd::Variable*, std::unordered_map<std::string, Tensor<float>>> state_;
};

// SGD with momentum
struct SGDOptions : OptimizerOptions {
    float momentum = 0.0;
    bool nesterov = false;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<autograd::Variable*> params, SGDOptions options = {});
    void step() override;

private:
    SGDOptions options_;
};

// Adam / AdamW
struct AdamOptions : OptimizerOptions {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    bool amsgrad = false;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;

private:
    AdamOptions options_;
    int step_count_ = 0;
};

class AdamW : public Adam {
public:
    AdamW(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;
};

// Learning rate schedulers
class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;

    virtual void step() = 0;
    float get_last_lr() const;

protected:
    Optimizer& optimizer_;
    int step_count_ = 0;
    float last_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0);
    void step() override;

private:
    int T_max_;
    float eta_min_;
    float base_lr_;
};

class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer, int warmup_steps, float start_factor = 0.0);
    void step() override;

private:
    int warmup_steps_;
    float start_factor_;
    float base_lr_;
};

}  // namespace lightwatch::optim
```

### Acceptance Criteria
- [ ] `test -f docs/contracts/tensor.hpp`
- [ ] `test -f docs/contracts/autograd.hpp`
- [ ] `test -f docs/contracts/tokenizer.hpp`
- [ ] `test -f docs/contracts/module.hpp`
- [ ] `test -f docs/contracts/optimizer.hpp`
- [ ] `grep -q "namespace lightwatch" docs/contracts/*.hpp`

---

## PHASE 0.7: ACQUIRE EXTERNAL ASSETS

### Objective
Download required tokenizer assets before implementation begins.

### Deliverables
| File | Source | Purpose |
|------|--------|---------|
| `data/vocab/vocab.bpe` | HuggingFace gpt2 | BPE merge rules (50000 merges) |
| `data/vocab/encoder.json` | HuggingFace gpt2 | Token-to-ID mapping (50257 entries) |

### Fallback URLs

If primary HuggingFace URLs fail, try in order:

| Asset | Primary | Fallback 1 |
|-------|---------|------------|
| vocab.bpe | huggingface.co/gpt2/raw/main/vocab.bpe | openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe |
| encoder.json | huggingface.co/gpt2/raw/main/encoder.json | openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json |

### Acquisition Script: scripts/acquire_assets.sh
```bash
#!/bin/bash
set -e

echo "=== Acquiring GPT-2 Tokenizer Assets ==="

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed"
    exit 1
fi

mkdir -p data/vocab

try_download() {
    local name=$1
    shift
    for url in "$@"; do
        echo "Trying $url..."
        if curl -fsSL --connect-timeout 10 "$url" -o "data/vocab/$name"; then
            echo "Downloaded $name"
            return 0
        fi
        echo "Failed, trying next URL..."
    done
    echo "ERROR: Failed to download $name from all sources"
    return 1
}

try_download "vocab.bpe" \
    "https://huggingface.co/gpt2/raw/main/vocab.bpe" \
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"

try_download "encoder.json" \
    "https://huggingface.co/gpt2/raw/main/encoder.json" \
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"

# Validate
echo ""
echo "=== Validating Assets ==="

if [ ! -s data/vocab/vocab.bpe ]; then
    echo "ERROR: vocab.bpe is empty or missing"
    exit 1
fi

VOCAB_SIZE=$(jq 'length' data/vocab/encoder.json)
if [ "$VOCAB_SIZE" -ne 50257 ]; then
    echo "ERROR: encoder.json has $VOCAB_SIZE tokens, expected 50257"
    exit 1
fi

MERGE_COUNT=$(wc -l < data/vocab/vocab.bpe | tr -d ' ')
echo "vocab.bpe: $MERGE_COUNT lines"
echo "encoder.json: $VOCAB_SIZE tokens"

echo ""
echo "=== Assets Acquired Successfully ==="
```

### Acceptance Criteria
- [ ] `test -f data/vocab/vocab.bpe && test -s data/vocab/vocab.bpe`
- [ ] `test -f data/vocab/encoder.json && test -s data/vocab/encoder.json`
- [ ] `jq 'length' data/vocab/encoder.json | grep -q 50257`
- [ ] `wc -l < data/vocab/vocab.bpe | awk '{exit ($1 < 50000)}'`

### If All URLs Fail
This is an **ESCALATION TRIGGER**. Do not proceed:
1. Document the failure in `docs/architecture/ESCALATIONS.md`
2. Update `.lightwatch_state.json` with `phase_status: "ESCALATED"`
3. STOP and wait for human input

### Optional: Pretrained Weights
For testing inference without training:
```bash
mkdir -p data/weights
# Option 1: Download via Python script
python3 scripts/download_weights.py --model gpt2 --output data/weights/

# Option 2: Convert from HuggingFace (requires transformers)
python3 scripts/convert_hf_weights.py --model gpt2 --output data/weights/gpt2.bin
```

This is OPTIONAL. The model must be trainable from scratch. Weight loading is for validation only.

---

## PHASE 0.5: GENERATE ALL PHASE PROMPTS

### Objective
Before any implementation begins, generate all 40 phase prompt files. This ensures architectural coherence across the entire project.

### Phase Prompt Template (MANDATORY)

Every generated prompt MUST follow this exact structure:

~~~markdown
# Phase XX: <Title>

## Objective
<1-2 sentences: what this phase delivers>

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| NN    | Specific files/symbols needed from that phase |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| NN           | path/to/file.hpp | ClassName, function_name |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| path/to/new/file.hpp | ClassName | Phase NN, MM |

## Specification

### Data Structures
```cpp
// Exact class/struct definitions
```

### Function Signatures
```cpp
// Exact C++ signatures for ALL public API
```

### Algorithmic Requirements
<Pseudocode or step-by-step logic>

### Performance Constraints
<Measurable targets if applicable>

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_function_name` | Concrete input | Concrete assertion |

## Acceptance Criteria
- [ ] `<shell command that exits 0 on success>`
- [ ] `<another shell command>`

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | XXX-XXX |
| New source files | N |
| New test files | N |
| Complexity | LOW/MEDIUM/HIGH |

## Notes
<Edge cases, risks, design decisions>
~~~

### Template Rules
1. **Prerequisites**: Must exactly match `PHASE_DEPS[N]` from the dependency table
2. **Inputs**: Copy signatures verbatim from contract files or prior phase Outputs
3. **Outputs**: Must satisfy all forward dependencies (phases that list N as prerequisite)
4. **Required Tests**: Minimum counts based on complexity:
   - HIGH: ≥6 tests
   - MEDIUM: ≥4 tests
   - LOW: ≥2 tests
5. **Acceptance Criteria**: Only executable shell commands (cmake, ctest, test, grep, jq, etc.)

### Prompt Generation Procedure

For each phase N from 1 to 40:

**Step 1: Gather Dependencies**
```python
deps = PHASE_DEPS[N]  # Phases this phase depends on
consumers = [p for p in range(N+1, 41) if N in PHASE_DEPS[p]]  # Phases that depend on this
```

**Step 2: Collect Inputs**
- For each phase in `deps`, identify the Outputs that this phase needs
- If a contract file exists (tensor.hpp, autograd.hpp, etc.), copy relevant signatures verbatim
- List in Inputs table with exact file paths and symbol names

**Step 3: Define Outputs**
- Determine what files this phase creates
- For each file, list all public symbols (classes, functions, constants)
- Cross-reference `consumers` to ensure all their Input needs are satisfied
- If this phase defines a contract, the Output signatures must match the contract file exactly

**Step 4: Write Specification**
- Data structures: Full class definitions with member types
- Function signatures: Exact C++ declarations
- Algorithmic requirements: Pseudocode or numbered steps
- Performance constraints: Measurable targets (e.g., "O(n) time", ">1M ops/sec")

**Step 5: Define Tests**
- Consult TEST SPECIFICATIONS section for required test cases
- If phase has predefined tests, copy them exactly
- If not predefined, create tests following the pattern:
  - Test name: `test_<component>_<behavior>`
  - Input: Concrete values, not placeholders
  - Expected: Specific assertion, not "works correctly"

**Step 6: Write Acceptance Criteria**
- Build command: `cmake --build build --target <target>`
- Test command: `ctest -R <pattern> --output-on-failure`
- Any phase-specific validation commands

**Step 7: Save and Validate**
```bash
# Save to docs/prompts/phase-{N:02d}-{slug}.md
# Validate immediately:
python3 scripts/validate_prompts.py
```

**Step 8: Commit**
```bash
git add docs/prompts/phase-{N:02d}-*.md
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-00] docs: Generate prompt for Phase $N"
```

---

## PHASE 0.9: TOOLCHAIN SMOKE TEST

### Objective
Validate that the entire toolchain works before starting real implementation. Catch environment issues early.

### Procedure

1. **Create minimal test project:**
```cpp
// tests/smoke/smoke_test.cpp
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    // Test basic C++17 features
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};

    float sum = 0;
    for (auto x : v) sum += x;

    // Test floating point operations
    float expected = 10.0f;
    if (std::abs(sum - expected) > 1e-6f) {
        std::cerr << "ERROR: sum = " << sum << ", expected " << expected << std::endl;
        return 1;
    }

    std::cout << "Smoke test passed: sum = " << sum << std::endl;
    return 0;
}
```

2. **Add to CMakeLists.txt:**
```cmake
# Smoke test target
add_executable(smoke_test tests/smoke/smoke_test.cpp)
```

3. **Build and run:**
```bash
cmake -B build -S .
cmake --build build --target smoke_test
./build/smoke_test
```

4. **Test git authorship:**
```bash
git add tests/smoke/
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-00] test: Add smoke test

Validates toolchain before main implementation.

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"

# Verify authorship
AUTHOR_EMAIL=$(git log -1 --format='%ae')
AUTHOR_NAME=$(git log -1 --format='%an')

if [ "$AUTHOR_EMAIL" != "buteverythingisnormal@gmail.com" ]; then
    echo "ERROR: Commit author email is $AUTHOR_EMAIL, expected buteverythingisnormal@gmail.com"
    exit 1
fi

if [ "$AUTHOR_NAME" != "watchthelight" ]; then
    echo "ERROR: Commit author name is $AUTHOR_NAME, expected watchthelight"
    exit 1
fi

echo "Git authorship verified"
```

### Acceptance Criteria
- [ ] `cmake --build build --target smoke_test` exits 0
- [ ] `./build/smoke_test` prints "Smoke test passed" and exits 0
- [ ] `git log -1 --format='%ae'` outputs `buteverythingisnormal@gmail.com`
- [ ] `git log -1 --format='%an'` outputs `watchthelight`

### On Failure
If any step fails:
1. Document the error in `docs/architecture/TOOLCHAIN.md`
2. Fix the environment issue
3. Re-run from step 1
4. Do NOT proceed to Phase 1 until all criteria pass

---

## CROSS-PHASE API CONTRACTS

Canonical API signatures are defined in separate files. These are the AUTHORITATIVE definitions — generated prompts must copy them verbatim.

| Contract | File | Defined In | Consumers |
|----------|------|------------|-----------|
| Tensor | `docs/contracts/tensor.hpp` | Phase 03 | 04,05,08,09,11-19,21-25,31-36 |
| Autograd | `docs/contracts/autograd.hpp` | Phase 05 | 08,11-19,21-25,31 |
| Tokenizer | `docs/contracts/tokenizer.hpp` | Phase 06-07 | 08,27,38 |
| Module | `docs/contracts/module.hpp` | Phase 11 | 12-19,31 |
| Optimizer | `docs/contracts/optimizer.hpp` | Phase 22 | 23-26,29 |

### Contract Usage Rules
1. When generating Phase N's prompt, if N defines a contract, copy the contract file verbatim into Outputs
2. When Phase N consumes a contract, copy the relevant signatures into Inputs
3. Implementation MUST match contract signatures exactly (character-for-character for public API)
4. Contract modifications require updating ALL consumer phases

---

## PHASE DEPENDENCIES

### Table Format

| Phase | Title | Depends On | Unlocks | Track |
|-------|-------|------------|---------|-------|
| 01 | Build System | - | 02,06 | - |
| 02 | Memory Management | 01 | 03,06 | A |
| 03 | Tensor Core | 02 | 04,05,08,09 | A |
| 04 | SIMD Operations | 03 | 05 | A |
| 05 | Autograd Engine | 03,04 | 11,22,25,26 | A |
| 06 | BPE Tokenizer | 01,02 | 07 | B |
| 07 | Vocabulary | 06 | 08,27 | B |
| 08 | Embedding Layer | 03,07 | 10 | Merge |
| 09 | Positional Encoding | 03 | 10 | A |
| 10 | Foundation Tests | 01-09 | 11-19 | **CP** |
| 11 | Dense Layer | 05 | 12,13,14,15,17 | C |
| 12 | Activations | 11 | 15,17,21 | C |
| 13 | Layer Normalization | 11 | 15,17,21 | C |
| 14 | Dropout | 11 | 15,17 | C |
| 15 | Single-Head Attention | 11,12,13,14 | 16 | C |
| 16 | Multi-Head Attention | 15 | 18,19 | C |
| 17 | FFN Block | 11,12,13,14 | 18,19 | C |
| 18 | Transformer Encoder | 16,17 | 20 | C |
| 19 | Transformer Decoder | 16,17 | 20,31 | C |
| 20 | Neural Core Tests | 11-19 | 31 | **CP** |
| 21 | Loss Functions | 12,13 | 29 | D |
| 22 | SGD Optimizer | 05 | 23,24 | D |
| 23 | Adam Optimizer | 22 | 29 | D |
| 24 | LR Schedulers | 22 | 29 | D |
| 25 | Gradient Clipping | 05 | 29 | D |
| 26 | Checkpointing | 05 | 29 | D |
| 27 | Data Loading | 07 | 28 | D |
| 28 | Batch Processing | 27 | 29 | D |
| 29 | Training Loop | 21,23,24,25,26,28 | 30 | D |
| 30 | Training Tests | 21-29 | 31 | **CP** |
| 31 | GPT Architecture | 19,20 | 32,33,34,37 | E |
| 32 | Model Config | 31 | 38 | E |
| 33 | Weight Init | 31 | 38 | E |
| 34 | Greedy Decode | 31 | 35 | E |
| 35 | Sampling | 34 | 36,38 | E |
| 36 | KV-Cache | 35 | 38 | E |
| 37 | Serialization | 31 | 38 | E |
| 38 | CLI/REPL | 31,35,37 | 39 | E |
| 39 | Profiling | 38 | 40 | E |
| 40 | Final Integration | 01-39 | - | **CP** |

**Track Legend:**
- **A:** Compute foundation (tensors, math, autograd)
- **B:** Text processing (tokenizer, vocabulary)
- **C:** Neural network layers
- **D:** Training infrastructure
- **E:** Model assembly and tooling
- **CP:** Checkpoint (extended validation required)
- **Merge:** Requires multiple tracks

### Machine-Readable Dependency Dict
```python
PHASE_DEPS = {
    1: [], 2: [1], 3: [2], 4: [3], 5: [3,4],
    6: [1,2], 7: [6], 8: [3,7], 9: [3], 10: list(range(1,10)),
    11: [5], 12: [11], 13: [11], 14: [11], 15: [11,12,13,14],
    16: [15], 17: [11,12,13,14], 18: [16,17], 19: [16,17], 20: list(range(11,20)),
    21: [12,13], 22: [5], 23: [22], 24: [22], 25: [5],
    26: [5], 27: [7], 28: [27], 29: [21,23,24,25,26,28], 30: list(range(21,30)),
    31: [19,20], 32: [31], 33: [31], 34: [31], 35: [34],
    36: [35], 37: [31], 38: [31,35,37], 39: [38], 40: list(range(1,40))
}
```

---

## PHASE SPECIFICATION TABLE

| Phase | Title | Expected LOC | Files | Complexity | Key Outputs |
|-------|-------|--------------|-------|------------|-------------|
| 01 | Build System | 100-200 | 3 | LOW | CMakeLists.txt, directory structure |
| 02 | Memory Management | 400-600 | 4 | MEDIUM | Arena allocator, pool allocator, aligned_alloc |
| 03 | Tensor Core | 1500-2500 | 6 | HIGH | Tensor<T>, Shape, basic ops |
| 04 | SIMD Operations | 800-1200 | 4 | HIGH | AVX2/SSE dispatch, vectorized ops |
| 05 | Autograd Engine | 1000-1500 | 5 | HIGH | Variable, Function, backward graph |
| 06 | BPE Tokenizer | 600-900 | 3 | MEDIUM | BPE encode/decode, merge learning |
| 07 | Vocabulary | 400-600 | 3 | LOW | Token↔ID maps, special tokens |
| 08 | Embedding Layer | 300-500 | 3 | MEDIUM | Token + position lookup |
| 09 | Positional Encoding | 400-600 | 4 | MEDIUM | Sinusoidal, Learned, RoPE, ALiBi |
| 10 | Foundation Tests | 500-800 | 5 | MEDIUM | Integration tests, benchmarks |
| 11 | Dense Layer | 300-500 | 3 | MEDIUM | Linear, weight init |
| 12 | Activations | 300-500 | 2 | LOW | ReLU, GELU, SiLU, Softmax |
| 13 | Layer Normalization | 250-400 | 2 | MEDIUM | LayerNorm, RMSNorm |
| 14 | Dropout | 150-250 | 2 | LOW | Dropout, DropPath |
| 15 | Single-Head Attention | 400-600 | 3 | HIGH | Scaled dot-product, causal mask |
| 16 | Multi-Head Attention | 300-500 | 2 | MEDIUM | Head split/merge, projections |
| 17 | FFN Block | 200-350 | 2 | LOW | FFN, SwiGLU variant |
| 18 | Transformer Encoder | 250-400 | 2 | MEDIUM | Pre-norm encoder block |
| 19 | Transformer Decoder | 300-450 | 2 | MEDIUM | Causal decoder block |
| 20 | Neural Core Tests | 400-600 | 4 | MEDIUM | Stack tests, gradient checks |
| 21 | Loss Functions | 300-500 | 2 | MEDIUM | CrossEntropy, label smoothing |
| 22 | SGD Optimizer | 200-350 | 2 | MEDIUM | Momentum, weight decay |
| 23 | Adam Optimizer | 250-400 | 2 | MEDIUM | Adam, AdamW |
| 24 | LR Schedulers | 200-350 | 2 | LOW | Cosine, warmup |
| 25 | Gradient Clipping | 150-250 | 2 | LOW | Norm clip, value clip |
| 26 | Checkpointing | 300-500 | 2 | MEDIUM | Save/load state |
| 27 | Data Loading | 400-600 | 4 | MEDIUM | Dataset, DataLoader |
| 28 | Batch Processing | 300-450 | 3 | MEDIUM | Collation, padding |
| 29 | Training Loop | 500-800 | 3 | HIGH | Trainer, callbacks |
| 30 | Training Tests | 400-600 | 4 | MEDIUM | Overfit tests |
| 31 | GPT Architecture | 400-600 | 3 | HIGH | Complete GPT model |
| 32 | Model Config | 200-350 | 2 | LOW | JSON config, presets |
| 33 | Weight Init | 150-250 | 2 | LOW | GPT-2 init scheme |
| 34 | Greedy Decode | 250-400 | 2 | MEDIUM | Argmax generation |
| 35 | Sampling | 300-450 | 2 | MEDIUM | Temperature, top-k, top-p |
| 36 | KV-Cache | 400-600 | 3 | HIGH | Incremental attention |
| 37 | Serialization | 400-600 | 3 | MEDIUM | Save/load weights |
| 38 | CLI/REPL | 400-600 | 3 | MEDIUM | Commands, streaming, JSON output |
| 39 | Profiling | 300-450 | 3 | MEDIUM | Timer, profiler |
| 40 | Final Integration | 300-500 | 5 | MEDIUM | Docs, examples, CI |

---

## TEST SPECIFICATIONS

Each phase must include specific test cases. The prompt's "Required Tests" section defines concrete tests.

### Core Test Cases by Phase

#### Phase 03: Tensor (HIGH complexity - 12 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_tensor_construction` | `Shape{2,3,4}` | `numel()==24`, `ndim()==3` |
| `test_tensor_zeros` | `Shape{3,3}` | All elements == 0.0 |
| `test_tensor_ones` | `Shape{2,2}` | All elements == 1.0 |
| `test_tensor_randn` | `Shape{100}` | Mean ≈ 0.0 (±0.3), Std ≈ 1.0 (±0.3) |
| `test_tensor_matmul_2d` | `A[2,3] @ B[3,4]` | Result shape `[2,4]`, values correct |
| `test_tensor_matmul_batch` | `A[5,2,3] @ B[5,3,4]` | Result shape `[5,2,4]` |
| `test_tensor_broadcast_add` | `A[2,3] + B[3]` | Result shape `[2,3]`, values correct |
| `test_tensor_broadcast_mul` | `A[2,3,4] * B[1,4]` | Result shape `[2,3,4]` |
| `test_tensor_slice` | `T[10,20].slice(0,2,5)` | Result shape `[3,20]` |
| `test_tensor_transpose` | `T[2,3].transpose(0,1)` | Result shape `[3,2]` |
| `test_tensor_contiguous` | Non-contiguous slice | `is_contiguous()==true` after `.contiguous()` |
| `test_tensor_reduction_sum` | `T[2,3].sum(1)` | Result shape `[2]`, values correct |

#### Phase 04: SIMD Operations (HIGH complexity - 6 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_simd_add` | `A[1024], B[1024]` | Element-wise sum matches scalar |
| `test_simd_mul` | `A[1024], B[1024]` | Element-wise product matches scalar |
| `test_simd_dot` | `A[1024], B[1024]` | Dot product matches scalar (tol 1e-5) |
| `test_simd_matmul` | `A[64,64] @ B[64,64]` | Matches scalar implementation |
| `test_simd_exp` | `A[1024]` in range [-5,5] | Matches std::exp (tol 1e-5) |
| `test_simd_alignment` | Unaligned data | No crash, correct results |

#### Phase 05: Autograd (HIGH complexity - 10 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_autograd_add` | `c = a + b; c.backward()` | `a.grad == 1`, `b.grad == 1` |
| `test_autograd_mul` | `c = a * b; c.backward()` | `a.grad == b.data`, `b.grad == a.data` |
| `test_autograd_matmul` | `C = A @ B; C.sum().backward()` | Gradients match numerical diff (tol 1e-5) |
| `test_autograd_chain` | `d = relu(a @ b + c)` | All gradients computed |
| `test_autograd_no_grad` | `requires_grad=false` | `has_grad()==false` |
| `test_autograd_accumulation` | Two backward passes | Gradients sum |
| `test_autograd_detach` | `b = a.detach()` | `b.grad_fn() == nullptr` |
| `test_autograd_relu_grad` | `y = relu(x); y.backward()` | grad = 1 if x > 0, else 0 |
| `test_autograd_softmax_grad` | `y = softmax(x); y.backward()` | Jacobian matches numerical diff |
| `test_autograd_no_grad_guard` | Inside NoGradGuard | No graph built |

#### Phase 06: Tokenizer (MEDIUM complexity - 8 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_tokenizer_roundtrip` | `"Hello, world!"` | `decode(encode(x)) == x` |
| `test_tokenizer_special` | EOS token | ID == 50256 |
| `test_tokenizer_unicode` | `"日本語"` | No crash, tokens produced |
| `test_tokenizer_empty` | `""` | Returns empty vector |
| `test_tokenizer_vocab_size` | Load GPT-2 vocab | `vocab_size() == 50257` |
| `test_tokenizer_whitespace` | `"  leading and trailing  "` | Roundtrip exact match |
| `test_tokenizer_numbers` | `"12345"` | Tokenizes correctly |
| `test_tokenizer_long_text` | 2000 random tokens | No crash, valid IDs |
| `test_tokenizer_emoji` | `"Hello 🌍 World"` | Roundtrip preserves emoji |
| `test_tokenizer_newlines` | `"line1\nline2\r\nline3"` | Roundtrip exact match |

#### Phase 15: Attention (HIGH complexity - 8 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_attention_shape` | `Q,K,V [B,H,S,D]` | Output `[B,H,S,D]` |
| `test_attention_causal` | Position i query | Weights 0 for j > i |
| `test_attention_softmax` | Any input | Weights sum to 1.0 per query |
| `test_attention_gradient` | Backward pass | Matches numerical gradient (tol 1e-4) |
| `test_attention_scale` | `d_k=64` | Scaled by `1/sqrt(64)` |
| `test_attention_mask_inf` | Masked positions | Output unaffected by masked values |
| `test_attention_single_token` | `S=1` | No crash, correct output |
| `test_attention_long_sequence` | `S=1024` | Memory efficient, no OOM |

#### Phase 16: Multi-Head Attention (MEDIUM complexity - 6 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_mha_shape` | `x [B,S,D]`, 12 heads | Output `[B,S,D]` |
| `test_mha_head_split` | 12 heads, D=768 | Each head gets 64 dims |
| `test_mha_projection` | Input/output | Wq, Wk, Wv, Wo correctly sized |
| `test_mha_gradient` | Full backward | All projections have gradients |
| `test_mha_causal` | Decoder attention | Causal mask applied |
| `test_mha_cross_attention` | Encoder-decoder | Cross-attention works |

#### Phase 19: Transformer Decoder (MEDIUM complexity - 5 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_decoder_block_shape` | `x [B,S,D]` | Output `[B,S,D]` |
| `test_decoder_block_causal` | Full sequence | Position i uses only 0..i |
| `test_decoder_block_gradient` | Backward pass | All parameters have gradients |
| `test_decoder_residual` | Skip connections | Output = input + sublayer(input) |
| `test_decoder_prenorm` | LayerNorm position | Applied before attention/FFN |

#### Phase 29: Training Loop (HIGH complexity - 6 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_training_overfit` | 10 samples, 1000 epochs | Loss < 0.01 |
| `test_training_gradient_clip` | Large gradients | Norm <= clip_value |
| `test_training_lr_schedule` | Warmup + cosine | LR follows schedule |
| `test_training_checkpoint` | Save/load mid-training | Training resumes correctly |
| `test_training_loss_decreases` | 100 steps | Loss at step 100 < loss at step 1 |
| `test_training_nan_detection` | Bad learning rate | NaN detected and reported |

#### Phase 31: GPT Architecture (HIGH complexity - 6 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_gpt_forward_shape` | `[B, S]` input | Output `[B, S, V]` |
| `test_gpt_causal` | Sequence | Position i output depends only on 0..i |
| `test_gpt_parameter_count` | GPT-2 Small config | ~124M parameters (±5%) |
| `test_gpt_gradient` | Full backward | All parameters have gradients |
| `test_gpt_embedding_tied` | wte and lm_head | Share same weight matrix |
| `test_gpt_layer_order` | 12 layers | Layers execute in correct order |

#### Phase 36: KV-Cache (HIGH complexity - 5 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_kv_cache_incremental` | Generate 10 tokens | Same result as full recompute |
| `test_kv_cache_shape` | After 5 tokens | Cache shape `[B, H, 5, D]` |
| `test_kv_cache_reset` | New sequence | Cache cleared |
| `test_kv_cache_memory` | 1024 tokens | Memory ~75MB per batch |
| `test_kv_cache_speedup` | Cached vs uncached | Cached ≥2x faster |

#### Phase 38: CLI/REPL (MEDIUM complexity - 8 test cases required)
| Test | Input | Expected |
|------|-------|----------|
| `test_cli_generate_basic` | `--prompt "Hello"` | Non-empty output, exit 0 |
| `test_cli_generate_json` | `--prompt "Hi" --json` | Valid JSON with required fields |
| `test_cli_benchmark_json` | `benchmark --json` | Valid JSON with tokens_per_second |
| `test_cli_json_schema` | `--json` output | Contains: command, prompt_tokens, tokens_per_second |
| `test_cli_help` | `--help` | Shows usage, exit 0 |
| `test_cli_invalid_flag` | `--invalid` | Error message, exit non-zero |
| `test_cli_max_tokens` | `--max-tokens 10` | Output ≤ 10 tokens |
| `test_cli_seed` | `--seed 42` twice | Same output both times |

---

## BENCHMARK SPECIFICATION

### Hardware Baseline
- **CPU:** Any x86-64 with AVX2 (Intel 8th gen+ / AMD Zen 2+)
- **RAM:** 8GB minimum available
- **OS:** Linux preferred, macOS/Windows supported

### Benchmark Parameters
| Parameter | Value |
|-----------|-------|
| Batch size | 1 (single sequence) |
| Prompt length | 128 tokens |
| Generation length | 128 tokens |
| Warmup iterations | 5 (discarded) |
| Measured iterations | 100 |
| Metric | Generated tokens / wall-clock seconds |

### Measurement Method
```cpp
// Pseudocode
for (int i = 0; i < warmup; i++) generate(prompt, 128);  // Discard

auto start = high_resolution_clock::now();
for (int i = 0; i < iterations; i++) {
    generate(prompt, 128);
}
auto end = high_resolution_clock::now();

double seconds = duration<double>(end - start).count();
double tokens_per_second = (iterations * 128) / seconds;
```

### Performance Thresholds
| Level | tok/s | Action |
|-------|-------|--------|
| PASS | ≥50 | Proceed |
| MARGINAL | 25-49 | Document in DECISIONS.md, consider optional BLAS |
| FAIL | <25 | **ESCALATION TRIGGER** — stop and request input |

### Benchmark CLI
```bash
./build/bin/lightwatch benchmark \
    --prompt-tokens 128 \
    --generate-tokens 128 \
    --warmup 5 \
    --iterations 100 \
    --json
```

Expected JSON output:
```json
{
    "prompt_tokens": 128,
    "generated_tokens": 128,
    "iterations": 100,
    "total_seconds": 2.56,
    "tokens_per_second": 50.0,
    "tokens_per_iteration": 128,
    "hardware": "x86-64 AVX2"
}
```

---

## PHASE 38 OBSERVABILITY NOTE

The CLI's `--json` output mode serves as the foundation for structured observability. When `--json` is specified:

### generate command output:
```json
{
    "command": "generate",
    "prompt": "<input>",
    "prompt_tokens": 128,
    "generated_tokens": 64,
    "total_duration_ms": 1240,
    "first_token_ms": 89,
    "tokens_per_second": 51.6,
    "stop_reason": "eos",
    "seed": 42
}
```

### benchmark command output:
```json
{
    "command": "benchmark",
    "prompt_tokens": 128,
    "generated_tokens": 128,
    "iterations": 100,
    "warmup_iterations": 5,
    "total_seconds": 2.48,
    "tokens_per_second": 51.6,
    "p50_latency_ms": 24.1,
    "p99_latency_ms": 31.2
}
```

This structured output enables:
1. Scripted testing and CI integration
2. Performance tracking over time
3. Future observability infrastructure to build on

---

## PROMPT VALIDATION

After generating all 40 prompts, run the validation script:

```bash
python3 scripts/validate_prompts.py
```

### scripts/validate_prompts.py
```python
#!/usr/bin/env python3
"""
Validates all 40 phase prompts for consistency.
Exit 0 = valid, Exit 1 = errors found.
"""

import sys
import re
import shutil
from pathlib import Path

# Check dependencies first
def check_dependencies():
    """Verify required tools are available."""
    required = ['jq']
    for tool in required:
        if not shutil.which(tool):
            print(f"ERROR: Required tool '{tool}' not found", file=sys.stderr)
            sys.exit(1)

check_dependencies()

PHASE_DEPS = {
    1: [], 2: [1], 3: [2], 4: [3], 5: [3,4],
    6: [1,2], 7: [6], 8: [3,7], 9: [3], 10: list(range(1,10)),
    11: [5], 12: [11], 13: [11], 14: [11], 15: [11,12,13,14],
    16: [15], 17: [11,12,13,14], 18: [16,17], 19: [16,17], 20: list(range(11,20)),
    21: [12,13], 22: [5], 23: [22], 24: [22], 25: [5],
    26: [5], 27: [7], 28: [27], 29: [21,23,24,25,26,28], 30: list(range(21,30)),
    31: [19,20], 32: [31], 33: [31], 34: [31], 35: [34],
    36: [35], 37: [31], 38: [31,35,37], 39: [38], 40: list(range(1,40))
}

REQUIRED_SECTIONS = [
    'Objective',
    'Prerequisites',
    'Inputs',
    'Outputs',
    'Specification',
    'Required Tests',
    'Acceptance Criteria',
    'Estimated Scope'
]

# Minimum test counts for HIGH/MEDIUM complexity phases
MIN_TESTS = {
    3: 10, 4: 5, 5: 8, 6: 6, 15: 6, 16: 4, 19: 4, 29: 4, 31: 4, 36: 4, 38: 6
}

def find_prompt(phase_num: int) -> Path | None:
    """Find the prompt file for a given phase number."""
    matches = list(Path("docs/prompts").glob(f"phase-{phase_num:02d}-*.md"))
    return matches[0] if matches else None

def extract_prerequisites(content: str) -> list[int]:
    """Extract phase numbers from Prerequisites table."""
    prereq_section = re.search(r'## Prerequisites\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if not prereq_section:
        return []

    phases = []
    for match in re.finditer(r'\|\s*(\d{1,2})\s*\|', prereq_section.group(1)):
        phases.append(int(match.group(1)))
    return phases

def extract_sections(content: str) -> dict[str, str]:
    """Extract major sections from prompt."""
    sections = {}
    current = None
    lines = []

    for line in content.split('\n'):
        if line.startswith('## '):
            if current:
                sections[current] = '\n'.join(lines)
            current = line[3:].strip()
            lines = []
        else:
            lines.append(line)

    if current:
        sections[current] = '\n'.join(lines)
    return sections

def count_test_rows(content: str) -> int:
    """Count test rows in Required Tests table."""
    tests_section = re.search(r'## Required Tests\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if not tests_section:
        return 0

    # Count table rows (lines starting with |, excluding header and separator)
    rows = [l for l in tests_section.group(1).split('\n')
            if l.strip().startswith('|') and not l.strip().startswith('|---')]
    return max(0, len(rows) - 1)  # Subtract header row

def validate_acceptance_criteria(content: str) -> list[str]:
    """Check that acceptance criteria are executable commands."""
    errors = []
    valid_prefixes = [
        'cmake', 'ctest', 'test', './build', 'grep', 'wc',
        'jq', 'valgrind', 'python3', 'bash', 'diff'
    ]

    for match in re.finditer(r'- \[ \] `([^`]+)`', content):
        cmd = match.group(1)
        first_word = cmd.split()[0]

        if not any(first_word.startswith(v) for v in valid_prefixes):
            if not first_word.startswith('./'):
                errors.append(f"Unknown command prefix: {cmd[:50]}...")

    return errors

def validate_required_tests(content: str) -> list[str]:
    """Check that Required Tests section has proper table format."""
    errors = []
    tests_section = re.search(r'## Required Tests\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)

    if not tests_section:
        errors.append("Missing 'Required Tests' section")
        return errors

    table_match = re.search(r'\|.*\|.*\|.*\|', tests_section.group(1))
    if not table_match:
        errors.append("Required Tests section missing table format")

    return errors

def main():
    errors = []
    prompts_dir = Path("docs/prompts")

    if not prompts_dir.exists():
        print("ERROR: docs/prompts directory does not exist", file=sys.stderr)
        sys.exit(1)

    # Check all files exist
    for i in range(1, 41):
        path = find_prompt(i)
        if not path:
            errors.append(f"Phase {i:02d}: Prompt file missing")
            continue

        content = path.read_text()

        # Check prerequisites match PHASE_DEPS
        prereqs = set(extract_prerequisites(content))
        expected = set(PHASE_DEPS[i])
        if prereqs != expected:
            missing = expected - prereqs
            extra = prereqs - expected
            if missing:
                errors.append(f"Phase {i:02d}: Missing prerequisites {sorted(missing)}")
            if extra:
                errors.append(f"Phase {i:02d}: Extra prerequisites {sorted(extra)}")

        # Check required sections exist
        sections = extract_sections(content)
        for sec in REQUIRED_SECTIONS:
            found = sec in sections or any(sec.lower() in s.lower() for s in sections)
            if not found:
                errors.append(f"Phase {i:02d}: Missing section '{sec}'")

        # Check acceptance criteria are executable
        cmd_errors = validate_acceptance_criteria(content)
        for e in cmd_errors:
            errors.append(f"Phase {i:02d}: {e}")

        # Check Required Tests format
        test_errors = validate_required_tests(content)
        for e in test_errors:
            errors.append(f"Phase {i:02d}: {e}")

        # Check minimum test count for complex phases
        if i in MIN_TESTS:
            test_count = count_test_rows(content)
            if test_count < MIN_TESTS[i]:
                errors.append(f"Phase {i:02d}: Has {test_count} tests, requires ≥{MIN_TESTS[i]}")

    # Report results
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for e in sorted(errors):
            print(f"  ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} errors found.", file=sys.stderr)
        sys.exit(1)

    print(f"Validation passed: 40 prompts verified")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**IF VALIDATION FAILS:** Fix the indicated prompts and re-run. Do NOT proceed to implementation.

---

## HOW TO READ A PHASE PROMPT

When beginning Phase N, follow this exact procedure:

### Step 1: Open the Prompt
```bash
cat docs/prompts/phase-$(printf "%02d" N)-*.md
```

### Step 2: Parse Prerequisites
The Prerequisites table lists phases that must be complete. Verify:
```bash
for dep in <list-of-deps>; do
    jq -e ".completed_phases | index($dep)" .lightwatch_state.json > /dev/null || {
        echo "ERROR: Prerequisite phase $dep not complete"
        exit 1
    }
done
```

### Step 3: Parse Inputs
The Inputs table lists APIs you MUST use. These come from contract files or prior phase outputs.
- DO use these exact signatures
- DO NOT modify input APIs
- DO NOT substitute alternative implementations

### Step 4: Parse Outputs
The Outputs table lists files you MUST create. Each entry specifies:
- File path
- Public symbols (classes, functions)
- Which future phases consume this output

### Step 5: Read Specification
The Specification section contains:
- Data structures (exact class definitions)
- Function signatures (exact C++ signatures)
- Algorithmic requirements (logic to implement)
- Performance constraints (measurable targets)

### Step 6: Read Test Requirements
The Required Tests table lists exact test cases:
- Test name
- Input values
- Expected output

Implement ALL listed tests. Do not improvise different tests.

### Step 7: Read Acceptance Criteria
These are shell commands that MUST exit 0. Run them ALL before considering the phase complete.

### What NOT to Do
- Do not add deliverables not in Outputs table
- Do not skip tests listed in Required Tests
- Do not modify Input APIs
- Do not ignore Performance Constraints
- Do not proceed if ANY acceptance criterion fails

---

## EXECUTION LOOP

### For Each Phase N

```
1. VERIFY DEPENDENCIES
   - For each dep in PHASE_DEPS[N]:
     - Check that docs/prompts/phase-{dep:02d}-*.md exists
     - Check that {dep} is in .lightwatch_state.json completed_phases
   - IF ANY MISSING: Complete missing dependencies first

2. BRANCH
   - git checkout main
   - git pull origin main (if remote exists)
   - git checkout -b phase-{N:02d}-{slug}
   - Update .lightwatch_state.json: current_phase=N, phase_status="EXECUTING"
   - Commit state change

3. READ PROMPT
   - Read docs/prompts/phase-{N:02d}-*.md
   - Parse Inputs table to identify required APIs
   - Parse Outputs table to identify deliverables
   - Parse Required Tests table

4. PLAN
   - Create docs/plans/phase-{N:02d}-plan.md with:
     * Objectives (from prompt)
     * Risks identified
     * Subtask breakdown
     * File creation order
     * Test implementation order

5. IMPLEMENT
   - Create files listed in Outputs
   - Implement according to Specification
   - Match contract signatures EXACTLY
   - Write all tests from Required Tests table
   - Commit frequently with descriptive messages

6. VERIFY
   - Update .lightwatch_state.json: phase_status="VERIFYING"
   - Run ALL acceptance criteria commands
   - ALL must exit 0
   - IF CHECKPOINT PHASE: Run checkpoint verification

7. COMPLETE
   - Update .lightwatch_state.json:
     * Add N to completed_phases
     * phase_status="COMPLETE"
     * Clear pending_deliverables
     * If checkpoint: add to checkpoints_passed
   - git checkout main
   - git merge phase-{N:02d}-{slug} --no-ff
   - git push origin main (if remote exists)

8. COMMIT STATE
   - git add .lightwatch_state.json
   - git commit with authorship
```

---

## CHECKPOINT PHASES

These phases require extended validation before proceeding. Do not advance past a checkpoint until all verification passes.

| Checkpoint | Phase | Verification |
|------------|-------|--------------|
| Foundation | 10 | Tensor ops, tokenizer, and autograd integrate correctly |
| Neural Core | 20 | Can build and forward-pass a multi-layer network |
| Training | 30 | Can overfit a 10-sample dataset to <0.01 loss |
| Complete | 40 | Full end-to-end generation works |

### Checkpoint 10: Foundation Verification
```bash
# Run foundation integration tests
ctest -R "integration_foundation" --output-on-failure

# Verify tensor-autograd integration
./build/bin/test_tensor_autograd

# Verify tokenizer produces valid token IDs
./build/bin/test_tokenizer --vocab data/vocab/encoder.json

# Memory check (if valgrind available)
if command -v valgrind &> /dev/null; then
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/bin/test_tensor_basic
fi
```

### Checkpoint 20: Neural Core Verification
```bash
# Build a 2-layer MLP and verify forward/backward
./build/bin/test_mlp_integration

# Gradient check: numerical vs analytical (tolerance 1e-5)
./build/bin/test_gradient_check --tolerance 1e-5

# Attention mask verification
./build/bin/test_attention_causal
```

### Checkpoint 30: Training Verification
```bash
# Overfit test: 10 samples, should reach <0.01 loss
./build/bin/test_overfit --samples 10 --max-epochs 1000 --target-loss 0.01

# Checkpoint save/load
./build/bin/test_checkpoint_roundtrip

# LR scheduler verification
./build/bin/test_lr_schedule --warmup 100 --total 1000
```

### Checkpoint 40: Final Verification
Run the full completion criteria script (see COMPLETION CRITERIA).

---

## REFACTORING PROTOCOL

If Phase N reveals architectural issues in Phase M (where M < N):

### 1. Complete Current Phase to Stable State
All tests must pass before refactoring.

### 2. Document the Issue
Add entry to `docs/architecture/DECISIONS.md`:
```markdown
## Decision: [YYYY-MM-DD] - Refactoring Phase M from Phase N

### Problem
<What was discovered during Phase N implementation>

### Impact
- Phases affected: M, [list any phases between M and N that depend on M]
- Breaking changes: [list signature changes]

### Solution
<What will change in Phase M>

### Migration
<Steps to update dependent code>
```

### 3. Create Refactoring Branch
```bash
git checkout main
git checkout -b refactor-phase-{M:02d}-from-{N:02d}
```

### 4. Apply Fix
- Fix Phase M code
- Update any contract files if signatures change
- Update all affected prompts

### 5. Validate
```bash
# Re-run tests for phases M through N (inclusive)
for p in $(seq M N); do
    ctest -R "phase_${p}" --output-on-failure || exit 1
done
```

### 6. Merge
```bash
git checkout main
git merge refactor-phase-{M:02d}-from-{N:02d} --no-ff
```

### 7. Update State
```bash
# State file should reflect the refactoring
jq '.last_refactor = "Phase M from Phase N"' .lightwatch_state.json > tmp.json
mv tmp.json .lightwatch_state.json
```

---

## ERROR RECOVERY

### Build Failure
```
1. Read error message carefully
2. Identify root cause:
   - Missing include → add #include
   - Wrong signature → check contract file
   - Undefined symbol → check namespace, link libraries
3. Fix the immediate error
4. Re-run cmake --build build
5. If persistent: Check if API contract was violated
6. If contract violated: Follow REFACTORING PROTOCOL
```

### Test Failure
```
1. Identify which test failed
2. Run test in isolation: ctest -R <test_name> -V
3. Check if test expectation matches specification
4. Add debug output if needed
5. Fix implementation OR fix test if spec was misread
6. Re-run full test suite for affected phase
```

### Session Interrupted
```
On resume:
1. Read .lightwatch_state.json
2. Check phase_status:
   - "EXECUTING" → Read pending_deliverables, check which files exist, resume
   - "VERIFYING" → Re-run all acceptance criteria
   - "FAILED" → Read error, assess if fixable, continue or escalate
   - "ESCALATED" → STOP, wait for human input
3. Check git status for uncommitted changes
4. If clean: continue from last known state
5. If dirty: assess changes, commit or stash as appropriate
```

---

## ROLLBACK PROCEDURE

If a merge to main breaks the build or tests:

### Identify the Breaking Commit
```bash
# Quick check: last few commits
git log --oneline -10

# Binary search if needed
git bisect start HEAD <last-known-good-commit>
git bisect run sh -c "cmake --build build && ctest --test-dir build"
git bisect reset
```

### Revert
```bash
git revert <breaking-commit-hash> --no-commit
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[HOTFIX] revert: Undo breaking change from Phase XX

Reverts commit <hash> which broke <description>.

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"
```

### Fix Forward
1. Create hotfix branch: `git checkout -b hotfix-phase-XX-description`
2. Identify and fix the root cause
3. Add regression test that catches this specific failure
4. Run full test suite
5. Merge hotfix to main
6. Document in `docs/architecture/DECISIONS.md`

### State Recovery
After rollback, update `.lightwatch_state.json`:
```bash
jq '.completed_phases -= [XX] | .failed_phases += [XX] | .phase_status = "FAILED"' \
    .lightwatch_state.json > tmp.json && mv tmp.json .lightwatch_state.json
```

---

## KNOWN TENSIONS

### Performance vs. Dependencies
- **Target:** 100 tok/s on CPU with no BLAS libraries
- **Reality:** This requires hand-tuned AVX2/AVX-512 intrinsics
- **Decision Tree:**
  - If AVX2 implementation achieves >50 tok/s → proceed
  - If <50 tok/s → document bottleneck in DECISIONS.md
  - Add OPTIONAL OpenBLAS path behind compile flag: `-DLIGHTWATCH_USE_BLAS=ON`
  - External BLAS must NEVER be required for basic functionality

### Memory vs. Speed
- KV-cache (Phase 36) trades memory for speed
- Full context (1024 tokens): ~75 MB cache
- Document memory requirements for different sequence lengths
- Provide configuration to limit cache size

### Generality vs. Simplicity
- The architecture targets GPT-2 Small specifically
- Avoid over-engineering for hypothetical larger models
- If a feature isn't needed for 124M parameters, don't build it

### Precision vs. Speed
- Use fp32 throughout (no fp16/bf16 for simplicity)
- SIMD operations may lose precision at edges
- Document any precision-sensitive code

### Observability vs. Simplicity
- Production systems benefit from "wide events" (one structured event per operation with full context)
- For initial build: Simple stdout/stderr logging is acceptable
- The CLI (Phase 38) should emit structured JSON for `--json` flag, which provides a foundation
- Future enhancement: Add `InferenceEvent`, `TrainingStepEvent` structs in a Phase 41+
- Reference: loggingsucks.com explains the wide events philosophy

---

## ANTI-PATTERNS (DO NOT DO)

These are common failure modes. Avoid them.

### 1. Premature Generalization
❌ Building a general-purpose tensor library with int8, bf16, sparse support
✓ Building exactly what GPT-2 Small needs: fp32 dense tensors

### 2. Premature Optimization
❌ Hand-tuning SIMD in Phase 03 before correctness is proven
✓ Get correct implementation first, optimize in Phase 39

### 3. Skipping Tests to Move Faster
❌ "Tests pass locally, I'll add more later"
✓ Phase is not complete until ALL acceptance criteria pass

### 4. Modifying Contract Signatures
❌ Changing `Tensor::shape()` return type in Phase 15
✓ Contracts are frozen after Phase 0.3. If change is essential, follow REFACTORING PROTOCOL.

### 5. Silent Dependency Addition
❌ Adding `#include <Eigen/Dense>` to solve a matrix problem
✓ If tempted to add a dependency, STOP and document in DECISIONS.md. Evaluate alternatives.

### 6. Skipping Checkpoints
❌ Rushing past Phase 10 without running integration tests
✓ Checkpoint phases require EXTENDED validation. No exceptions.

### 7. Ignoring Failing Tests
❌ Commenting out a failing test to proceed
✓ Fix the test or fix the code. Never disable tests.

### 8. Improvising Deliverables
❌ Adding `src/utils/string_utils.hpp` that wasn't in the prompt
✓ If additional files are needed, document in DECISIONS.md with justification.

### 9. Diverging from Phase Prompt
❌ Implementing a different API than specified in Outputs table
✓ Phase prompts are specifications. Follow them exactly.

### 10. Proceeding After Escalation Trigger
❌ Continuing after performance is <25 tok/s
✓ Stop and request human input when escalation triggers fire.

### 11. Guessing at Ambiguity
❌ "I think they meant X, I'll just do that"
✓ If specification is unclear, check contract files. If still unclear, document assumption in DECISIONS.md.

### 12. Bulk Commits
❌ One giant commit at the end of a phase
✓ Commit after each logical unit of work with descriptive messages.

### 13. Scattered Debug Logging
❌ 20 `std::cerr << "DEBUG: ..."` statements per inference request
✓ Structured output via `--json` flag; minimal stderr logging otherwise

### 14. Logging Without Context
❌ `std::cerr << "Error: generation failed" << std::endl;`
✓ `std::cerr << "Error: generation failed: " << e.what() << " (prompt_tokens=" << n << ")" << std::endl;`

---

## ESCALATION TRIGGERS

Stop execution and request human input if ANY of these conditions occur:

| Trigger | Condition | Why |
|---------|-----------|-----|
| Contract Change Required | A contract signature must change after Phase 10 | Ripple effects across many phases |
| Performance Floor Breach | <25 tok/s after Phase 39 optimization attempts | May require architectural rethink |
| Dependency Unavoidable | External library (non-nlohmann/json) seems required | Core constraint violation |
| Memory Ceiling Breach | Inference uses >8GB RAM | Likely implementation bug |
| Consecutive Failures | Two phases in a row fail all retry attempts | Systemic issue |
| Asset Unavailable | All fallback URLs for vocab/weights fail | Cannot proceed without assets |
| Test Impossibility | A required test cannot be written as specified | Spec may be wrong |
| Infinite Loop | Same error recurs after 3 fix attempts | Need different approach |

### Escalation Procedure
1. Update `.lightwatch_state.json`:
   ```json
   {
     "project_status": "ESCALATED",
     "phase_status": "ESCALATED",
     "escalations": ["Description of trigger"]
   }
   ```
2. Document the issue in `docs/architecture/ESCALATIONS.md`:
   - What triggered escalation
   - What was attempted
   - What options exist
3. **STOP execution**
4. Wait for human input before proceeding

### DO NOT:
- Silently work around the issue
- Disable tests to proceed
- Add forbidden dependencies
- Modify contracts without authorization
- Continue after escalation

---

## COMPLETION CRITERIA

The project is complete when ALL of the following commands exit 0:

### scripts/verify_complete.sh
```bash
#!/bin/bash
set -e

echo "=== LightwatchAI2 Completion Verification ==="

# Check required tools
echo "Checking required tools..."
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: Required tool '$1' not found"
        exit 1
    fi
}

check_tool cmake
check_tool jq
check_tool curl

# Optional tools (warn but don't fail)
SKIP_VALGRIND=0
if ! command -v valgrind &> /dev/null; then
    echo "WARNING: valgrind not found, memory checks will be skipped"
    SKIP_VALGRIND=1
fi

echo "All required tools present"
echo ""

# 1. Build succeeds with strict warnings
echo "[1/7] Building with strict warnings..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wall -Werror -Wextra"
cmake --build build --parallel

# 2. All tests pass
echo "[2/7] Running tests..."
ctest --test-dir build --output-on-failure

# 3. Generation produces valid output
echo "[3/7] Testing generation..."
./build/bin/lightwatch generate --prompt "The" --max-tokens 20 --seed 42 > /tmp/gen.txt
test -s /tmp/gen.txt  # File is non-empty
WORDS=$(wc -w < /tmp/gen.txt)
if [ "$WORDS" -lt 5 ]; then
    echo "ERROR: Generated only $WORDS words, expected >= 5"
    exit 1
fi
echo "Generated: $(cat /tmp/gen.txt)"

# 4. Performance meets minimum threshold
echo "[4/7] Running benchmark..."
./build/bin/lightwatch benchmark --iterations 100 --json > /tmp/bench.json
TPS=$(jq '.tokens_per_second' /tmp/bench.json)
if ! jq -e '.tokens_per_second > 50' /tmp/bench.json > /dev/null; then
    echo "ERROR: Performance $TPS tok/s below 50 tok/s threshold"
    exit 1
fi
echo "Performance: $TPS tokens/second"

# 5. Memory leak check (skip if valgrind unavailable)
echo "[5/8] Checking memory leaks..."
if [ "$SKIP_VALGRIND" -eq 0 ]; then
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/bin/lightwatch generate --prompt "Test" --max-tokens 10 2>&1 | tail -20
    echo "Memory leak check passed"
else
    echo "Valgrind not available, skipping memory leak check"
fi

# 6. Memory budget check (model should use < 2GB)
echo "[6/8] Checking memory budget..."
MAX_RSS_KB=2097152  # 2GB in KB
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MEMORY_OUTPUT=$(/usr/bin/time -v ./build/bin/lightwatch generate --prompt "Test" --max-tokens 100 2>&1)
    MAX_RSS=$(echo "$MEMORY_OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
    if [ -n "$MAX_RSS" ] && [ "$MAX_RSS" -gt "$MAX_RSS_KB" ]; then
        echo "ERROR: Memory usage ${MAX_RSS}KB exceeds 2GB limit"
        exit 1
    fi
    echo "Memory usage: ${MAX_RSS}KB (limit: ${MAX_RSS_KB}KB)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: /usr/bin/time -l reports bytes
    MEMORY_OUTPUT=$(/usr/bin/time -l ./build/bin/lightwatch generate --prompt "Test" --max-tokens 100 2>&1)
    MAX_RSS_BYTES=$(echo "$MEMORY_OUTPUT" | grep "maximum resident set size" | awk '{print $1}')
    MAX_RSS=$((MAX_RSS_BYTES / 1024))
    if [ -n "$MAX_RSS" ] && [ "$MAX_RSS" -gt "$MAX_RSS_KB" ]; then
        echo "ERROR: Memory usage ${MAX_RSS}KB exceeds 2GB limit"
        exit 1
    fi
    echo "Memory usage: ${MAX_RSS}KB (limit: ${MAX_RSS_KB}KB)"
else
    echo "Memory budget check skipped (unsupported platform)"
fi

# 7. State file shows all phases complete
echo "[7/8] Checking state file..."
COMPLETED=$(jq '.completed_phases | length' .lightwatch_state.json)
if [ "$COMPLETED" -ne 40 ]; then
    echo "ERROR: Only $COMPLETED/40 phases completed"
    exit 1
fi
echo "All 40 phases completed"

# 8. Documentation exists
echo "[8/8] Checking documentation..."
test -f README.md && test -s README.md
test -f docs/architecture/DECISIONS.md

echo ""
echo "=== VERIFICATION PASSED ==="
echo "LightwatchAI2 build is complete and functional."

# Update project status
jq '.project_status = "COMPLETE"' .lightwatch_state.json > tmp.json
mv tmp.json .lightwatch_state.json
```

Make executable: `chmod +x scripts/verify_complete.sh`

---

## INITIAL COMMIT

After Phase 0, 0.3, 0.7, and 0.9 are complete:

```bash
cd ~/Projects/LightwatchAI2  # or wherever the project lives

git add .
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-00] feat: Bootstrap project structure

- Create directory structure for C++ text generation model
- Add CMake build system
- Add CLAUDE.md with commit authorship policy
- Add API contract files in docs/contracts/
- Add validation and verification scripts
- Download GPT-2 tokenizer assets
- Add smoke test
- Initialize state tracking

Target: GPT-2 Small (124M parameters)
- 12 layers, 768 hidden, 12 heads
- 50257 vocab, 1024 context

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"

git branch -M main
git remote add origin https://github.com/watchthelight/LightwatchAI2.git
git push -u origin main
```

---

## SUMMARY

This prompt orchestrates building a minimal-dependency C++ GPT-2 inference engine through 40 phases. Key principles:

1. **Generate all prompts first** (Phase 0.5) — ensures architectural coherence
2. **API contracts in separate files** — single source of truth
3. **Validate prompt consistency** before implementation
4. **Respect dependency graph** — parallel work where possible
5. **Executable acceptance criteria** — no subjective judgments
6. **Concrete test specifications** — exact inputs and expected outputs
7. **State tracking** — resilient to session interruption
8. **Checkpoint phases** — extended validation at integration points
9. **Rollback procedure** — handle broken merges
10. **Escalation triggers** — know when to stop and ask
11. **Anti-patterns documented** — avoid common failure modes
12. **Known tensions documented** — no surprises

### Quick Reference

| Item | Location |
|------|----------|
| State file | `.lightwatch_state.json` |
| API contracts | `docs/contracts/*.hpp` |
| Phase prompts | `docs/prompts/phase-XX-*.md` |
| Architecture decisions | `docs/architecture/DECISIONS.md` |
| Escalation log | `docs/architecture/ESCALATIONS.md` |
| Toolchain info | `docs/architecture/TOOLCHAIN.md` |
| Asset acquisition | `scripts/acquire_assets.sh` |
| Validation script | `scripts/validate_prompts.py` |
| Completion check | `scripts/verify_complete.sh` |
| Toolchain check | `scripts/check_toolchain.sh` |

### Phase Order Quick Reference
```
0 → 0.3 → 0.7 → 0.5 → 0.9 → validate → 1..40 → verify
```
