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

| Performance | Status | Action |
|-------------|--------|--------|
| ≥100 tok/s | EXCELLENT | Proceed, no action needed |
| 50-99 tok/s | PASS | Proceed, document in DECISIONS.md |
| 25-49 tok/s | MARGINAL | Document bottleneck, add OPTIONAL `-DLIGHTWATCH_USE_BLAS=ON` path, proceed |
| <25 tok/s | FAIL | **ESCALATION TRIGGER** — stop and request human input |

External BLAS must NEVER be required for the default build.

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

#### Native Format: `.lwbin`

```
HEADER (64 bytes):
  - Magic: "LWAI" (4 bytes)
  - Version: uint32_t (4 bytes) - currently 1
  - Tensor count: uint32_t (4 bytes)
  - Reserved: 52 bytes (zero-filled)

For each tensor:
  - Name length: uint32_t
  - Name: char[name_length] (UTF-8, no null terminator)
  - Dtype: uint8_t (0=float32, 1=float16, 2=int32)
  - Ndim: uint8_t
  - Shape: int64_t[ndim]
  - Data: dtype[prod(shape)] (little-endian)
```

**Conversion scripts** (created in Phase 37):
- `scripts/convert_hf_to_lwbin.py` - HuggingFace `.bin` → `.lwbin`
- `scripts/convert_lwbin_to_hf.py` - `.lwbin` → HuggingFace `.bin`
- `scripts/validate_weights.py` - Verify weight compatibility

**Weight loading policy:**
- Weights are NOT automatically downloaded during build
- Tests use small random weights (seed 42 for reproducibility)
- Full GPT-2 weights are optional for inference validation
- Download command: `./build/bin/lightwatch download-weights --model gpt2`

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

### Quick Install (if tools missing)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y cmake g++ python3 python3-pip jq curl valgrind
pip3 install --user numpy  # For validation scripts if needed
```

**macOS (Homebrew):**
```bash
brew install cmake jq curl
# valgrind not available on macOS ARM; memory checks will be skipped
```

**Windows (with chocolatey):**
```powershell
choco install cmake jq curl python3
# Build with Visual Studio 2022 Developer Command Prompt
```

**Verify installation:**
```bash
bash scripts/check_toolchain.sh
```

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

## EXTERNAL DEPENDENCIES

### Allowed Dependencies
| Library | Version | Acquisition | Purpose |
|---------|---------|-------------|---------|
| nlohmann/json | 3.11.3+ | CMake FetchContent | JSON parsing for configs, state |

### Acquisition Method
All external dependencies are acquired via CMake FetchContent. Do NOT use:
- Git submodules
- Manual downloads
- System packages

This ensures reproducible builds across environments.

### CMake FetchContent Pattern
```cmake
include(FetchContent)
FetchContent_Declare(
    <name>
    GIT_REPOSITORY <url>
    GIT_TAG <version>
)
FetchContent_MakeAvailable(<name>)
```

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

**Session recovery checklist (run on every session start):**
```bash
# 0. Check for concurrent execution (lock file with staleness detection)
LOCK_FILE=".lightwatch.lock"
STALE_HOURS=4  # Lock files older than 4 hours are considered stale

if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)

    # Check if PID is still running
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: Another session is actively running (PID $LOCK_PID)"
        exit 1
    fi

    # Check lock file age (staleness detection)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LOCK_AGE=$(( ($(date +%s) - $(stat -f %m "$LOCK_FILE")) / 3600 ))
    else
        LOCK_AGE=$(( ($(date +%s) - $(stat -c %Y "$LOCK_FILE")) / 3600 ))
    fi

    if [ "$LOCK_AGE" -ge "$STALE_HOURS" ]; then
        echo "WARNING: Stale lock file detected (${LOCK_AGE}h old, PID $LOCK_PID not running)"
        echo "Removing stale lock and continuing..."
        rm -f "$LOCK_FILE"
    else
        echo "WARNING: Lock file exists (PID $LOCK_PID not running, ${LOCK_AGE}h old)"
        echo "If certain no other session is active: rm $LOCK_FILE"
        exit 1
    fi
fi
echo $$ > "$LOCK_FILE"  # Create lock with current PID

# 1. Check state file exists and is valid JSON
jq . .lightwatch_state.json > /dev/null || echo "ERROR: Invalid state file"

# 2. Check current branch matches state
CURRENT=$(git branch --show-current)
EXPECTED=$(jq -r '.current_branch' .lightwatch_state.json)
if [ "$CURRENT" != "$EXPECTED" ]; then
    echo "WARNING: On branch $CURRENT, state expects $EXPECTED"
fi

# 3. Check for uncommitted changes
if ! git diff --quiet; then
    echo "WARNING: Uncommitted changes detected"
    git status --short
fi

# 4. Verify last commit matches state
LAST_COMMIT=$(git log -1 --format='%H')
STATE_COMMIT=$(jq -r '.last_commit' .lightwatch_state.json)
if [ "$LAST_COMMIT" != "$STATE_COMMIT" ] && [ "$STATE_COMMIT" != "" ]; then
    echo "WARNING: HEAD ($LAST_COMMIT) differs from state ($STATE_COMMIT)"
fi

# 5. If phase was EXECUTING, verify what's done
if [ "$(jq -r '.phase_status' .lightwatch_state.json)" == "EXECUTING" ]; then
    echo "Resuming phase $(jq '.current_phase' .lightwatch_state.json)"
    echo "Completed: $(jq '.completed_deliverables' .lightwatch_state.json)"
    echo "Pending: $(jq '.pending_deliverables' .lightwatch_state.json)"
fi
```
Run this checklist before taking any action. Address warnings before proceeding.

**Lock file notes:**
- Created on session start: `echo $$ > .lightwatch.lock`
- Removed on phase completion: `rm -f .lightwatch.lock`
- Staleness detection: locks older than 4 hours with dead PID are auto-removed
- If PID is dead but lock is <4h old, user must manually confirm removal
- Manual removal: `rm .lightwatch.lock` (only if certain no session is active)

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
│   ├── MASTER_PROMPT.md      # Full orchestration prompt (copy of original)
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
# Claude Code Configuration - LightwatchAI2

## ⚠️ CRITICAL: Read This First

**Before taking ANY action on session start, you MUST:**
1. Read the full orchestration prompt:
   ```bash
   if [ -f docs/MASTER_PROMPT.md ]; then
       cat docs/MASTER_PROMPT.md
   else
       echo "ERROR: docs/MASTER_PROMPT.md not found"
       echo "This file should have been created during Phase 0"
       echo "If starting fresh, copy the original Master_Prompt.md to docs/MASTER_PROMPT.md"
       exit 1
   fi
   ```
2. Run the session recovery checklist (see below)
3. Check `.lightwatch_state.json` for current state

The Master Prompt contains all orchestration logic, phase specifications, contracts, and procedures. This CLAUDE.md file contains quick reference only.

---

## Session Recovery Checklist

Run this on EVERY session start before taking any action:

```bash
#!/bin/bash
# 1. Check state file exists and is valid JSON
jq . .lightwatch_state.json > /dev/null || { echo "ERROR: Invalid state file"; exit 1; }

# 2. Check for lock file (concurrent execution)
if [ -f ".lightwatch.lock" ]; then
    PID=$(cat .lightwatch.lock)
    if kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Another session is running (PID $PID)"
        exit 1
    fi
    rm -f .lightwatch.lock
fi

# 3. Check current branch matches state
CURRENT=$(git branch --show-current)
EXPECTED=$(jq -r '.current_branch' .lightwatch_state.json)
if [ "$CURRENT" != "$EXPECTED" ]; then
    echo "WARNING: On branch $CURRENT, state expects $EXPECTED"
fi

# 4. Check for uncommitted changes
if ! git diff --quiet; then
    echo "WARNING: Uncommitted changes detected"
    git status --short
fi

# 5. Report current state
echo "=== Current State ==="
echo "Phase: $(jq '.current_phase' .lightwatch_state.json)"
echo "Status: $(jq -r '.phase_status' .lightwatch_state.json)"
echo "Completed: $(jq '.completed_phases | length' .lightwatch_state.json)/40 phases"
```

---

## Quick Reference

### Commit Authorship (MANDATORY)
ALL commits must use:
```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" commit -m "message"
```
Claude MUST NOT be listed as commit author under any circumstances.

### Commit Message Format
See detailed format specification below. Summary: `[PHASE-XX] <type>: <subject>` with Signed-off-by line.

### Code Style
- C++17 standard
- 4-space indentation
- snake_case for functions/variables
- PascalCase for classes/types
- UPPER_CASE for constants

### Test Naming Convention
All test names MUST follow: `test_phase_XX_<component>_<behavior>`

### Pre-Commit Testing (MANDATORY)
Before EVERY commit:
```bash
cmake --build build
ctest --test-dir build -R "phase_$(printf '%02d' $CURRENT_PHASE)" --output-on-failure
```
If either fails, fix before committing.

### Escalation Triggers (STOP and wait for human)
- Contract change required after Phase 10
- Performance < 25 tok/s after optimization
- External dependency seems required
- Memory > 8GB RAM
- Two consecutive phase failures
- Asset download fails all fallbacks
- Same error 3 times after fix attempts
- Git push fails 3 consecutive times

### Model Target
- GPT-2 Small (124M parameters)
- 12 layers, 768 hidden, 12 heads, 64 head dim
- 50257 vocab, 1024 context, 3072 FFN

### Key Files
| Purpose | Location |
|---------|----------|
| Full orchestration | `docs/MASTER_PROMPT.md` |
| Current state | `.lightwatch_state.json` |
| API contracts | `docs/contracts/*.hpp` |
| Phase prompts | `docs/prompts/phase-XX-*.md` |
| Test specs | `docs/test_specs/phase-XX-*.md` |
| Decisions log | `docs/architecture/DECISIONS.md` |
| Escalations | `docs/architecture/ESCALATIONS.md` |

### Phase Order
```
0 → 0.3 → 0.7 → 0.5 → 0.9 → validate → 1..40 → verify
```

### State Values
- `project_status`: IN_PROGRESS | COMPLETE | ESCALATED
- `phase_status`: NOT_STARTED | EXECUTING | VERIFYING | COMPLETE | FAILED | PARTIAL_FAILURE | ESCALATED
```

### Setup Instructions

During Phase 0 bootstrap:
1. Copy the Master Prompt into the repository:
   ```bash
   cp /path/to/Master_Prompt.md docs/MASTER_PROMPT.md
   ```
2. This ensures the full orchestration logic is always available in the repo
3. On session resume, Claude Code should read `docs/MASTER_PROMPT.md` first

### Commit Message Format

All commits MUST follow this format:

```
[PHASE-XX] <type>: <subject>

<body>

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>
```

**Type values:**
- `feat` — New feature or functionality
- `fix` — Bug fix
- `test` — Adding or updating tests
- `docs` — Documentation changes
- `refactor` — Code restructuring without behavior change
- `perf` — Performance improvement
- `chore` — Build system, dependencies, tooling

**Examples:**
```bash
# Feature addition
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-03] feat: Implement Tensor reshape and view operations

Add reshape() for copying data to new shape and view() for zero-copy
reshape when strides permit. Both validate that total elements match.

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"

# Bug fix
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-04] fix: Handle unaligned memory in SIMD dot product

AVX2 load instructions require 32-byte alignment. Added scalar fallback
for the first N elements until aligned, then SIMD for remainder.

Fixes: test_simd_alignment

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"

# Test addition
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-06] test: Add tokenizer edge case tests

- test_tokenizer_empty: empty string handling
- test_tokenizer_emoji: Unicode emoji roundtrip
- test_tokenizer_long: 2000 token stress test

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"
```

**Rules:**
1. Subject line ≤ 72 characters
2. Body wrapped at 72 characters
3. Blank line between subject and body
4. Reference test names when fixing test failures
5. ALWAYS include Signed-off-by line

### CMakeLists.txt Template
```cmake
cmake_minimum_required(VERSION 3.16)
project(lightwatch VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# JSON library (header-only)
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files (populated by later phases)
# add_subdirectory(src)

# Testing
enable_testing()
include(CTest)
add_subdirectory(tests)

# Installation (optional)
install(DIRECTORY include/ DESTINATION include)
```

This template provides:
- C++17 with no extensions
- Consistent output directories
- Compiler warnings enabled
- nlohmann/json via FetchContent
- Testing infrastructure
- IDE support via compile_commands.json

### .gitignore Content
```gitignore
# Build directories
build/
cmake-build-*/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# Compiled objects
*.o
*.obj
*.a
*.lib
*.so
*.dylib
*.dll

# Executables
*.exe
*.out

# CMake generated
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
compile_commands.json
CTestTestfile.cmake
Testing/

# Session lock
.lightwatch.lock

# Large binary files (if weights are downloaded)
data/weights/*.bin
data/weights/*.pt
data/weights/*.safetensors

# macOS
.DS_Store

# Windows
Thumbs.db
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

| Contract File | Defined By | Key Types | Spec File |
|---------------|------------|-----------|-----------|
| `docs/contracts/tensor.hpp` | Phase 03 | Shape, Tensor<T>, matmul, concat, stack | `tensor.hpp.spec` |
| `docs/contracts/autograd.hpp` | Phase 05 | Variable, Function, ops namespace | `autograd.hpp.spec` |
| `docs/contracts/tokenizer.hpp` | Phase 06-07 | TokenId, Vocabulary, BPETokenizer | `tokenizer.hpp.spec` |
| `docs/contracts/module.hpp` | Phase 11 | Module, Linear, LayerNorm, Embedding, Dropout | `module.hpp.spec` |
| `docs/contracts/optimizer.hpp` | Phase 22 | Optimizer, SGD, Adam, AdamW, LRScheduler | `optimizer.hpp.spec` |

### Contract Specifications

Full contract specifications are in `docs/contracts/*.hpp.spec` files. During Phase 0.3:
1. Read each `.spec` file
2. Create the corresponding `.hpp` file with exact signatures
3. Add header comment with defined-by/consumers info

### Key Design Decisions (apply to all contracts)
- **Namespace:** `lightwatch` (sub-namespaces: `autograd`, `tokenizer`, `nn`, `optim`)
- **Memory:** Row-major layout, `std::shared_ptr<T[]>` for data
- **Style:** Return new objects for const methods, reference for in-place (`_` suffix)

### Acceptance Criteria
- [ ] `test -f docs/contracts/tensor.hpp`
- [ ] `test -f docs/contracts/autograd.hpp`
- [ ] `test -f docs/contracts/tokenizer.hpp`
- [ ] `test -f docs/contracts/module.hpp`
- [ ] `test -f docs/contracts/optimizer.hpp`
- [ ] `grep -q "namespace lightwatch" docs/contracts/*.hpp`

---

## CONTRACT SPECIFICATIONS

Complete API specifications for each contract file. Copy these to `docs/contracts/*.hpp` during Phase 0.3.

### tensor.hpp

```cpp
// LightwatchAI2 API Contract: Tensor
// Defined by: Phase 03 | Consumers: 04, 05, 08, 09, 11-19, 21-25, 31-36
#pragma once
#include <vector>
#include <memory>
#include <cstddef>
#include <initializer_list>

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
    static Tensor randn(const Shape& shape);  // N(0,1)
    static Tensor rand(const Shape& shape);   // U[0,1)

    // Properties
    const Shape& shape() const;
    size_t size(int dim) const;   // Negative dims count from end
    size_t numel() const;
    size_t ndim() const;
    T* data();
    const T* data() const;
    bool is_contiguous() const;

    // Element access
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    // Shape operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    Tensor contiguous() const;

    // Arithmetic (return new tensor)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    // Scalar arithmetic
    Tensor operator+(T scalar) const;
    Tensor operator*(T scalar) const;

    // In-place (return reference)
    Tensor& fill_(T value);
    Tensor& zero_();
    Tensor& add_(const Tensor& other);
    Tensor& mul_(const Tensor& other);

    // Reductions
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim = -1, bool keepdim = false) const;
    T item() const;  // Scalar tensors only

    // Math functions
    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor abs() const;
    Tensor pow(T exponent) const;

    Tensor clone() const;

private:
    Shape shape_;
    std::vector<size_t> strides_;
    std::shared_ptr<T[]> data_;
    size_t offset_ = 0;
};

// Free functions
template<typename T> Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b);
template<typename T> Tensor<T> concat(const std::vector<Tensor<T>>& tensors, int dim);
template<typename T> Tensor<T> stack(const std::vector<Tensor<T>>& tensors, int dim);

}  // namespace lightwatch
```

### autograd.hpp

```cpp
// LightwatchAI2 API Contract: Autograd
// Defined by: Phase 05 | Consumers: 08, 11-19, 21-25, 31
#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>

namespace lightwatch::autograd {

class Function;

class Variable {
public:
    Variable();
    explicit Variable(Tensor<float> data, bool requires_grad = false);

    // Data access
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
    size_t numel() const;
    size_t ndim() const;

    // Backpropagation
    void backward();
    void backward(const Tensor<float>& grad_output);

    // Graph management
    void set_grad_fn(std::shared_ptr<Function> fn);
    std::shared_ptr<Function> grad_fn() const;
    Variable detach() const;
    void retain_grad();

private:
    Tensor<float> data_;
    Tensor<float> grad_;
    bool requires_grad_ = false;
    bool has_grad_ = false;
    std::shared_ptr<Function> grad_fn_;
};

class Function {
public:
    virtual ~Function() = default;
    virtual std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) = 0;

protected:
    void save_for_backward(const Variable& v);
    std::vector<Variable> saved_variables_;
};

// Differentiable operations
namespace ops {
    Variable add(const Variable& a, const Variable& b);
    Variable sub(const Variable& a, const Variable& b);
    Variable mul(const Variable& a, const Variable& b);
    Variable div(const Variable& a, const Variable& b);
    Variable neg(const Variable& x);
    Variable matmul(const Variable& a, const Variable& b);
    Variable transpose(const Variable& x, int dim0, int dim1);

    // Activations
    Variable relu(const Variable& x);
    Variable gelu(const Variable& x);
    Variable sigmoid(const Variable& x);
    Variable tanh(const Variable& x);
    Variable softmax(const Variable& x, int dim);
    Variable log_softmax(const Variable& x, int dim);

    // Reductions
    Variable sum(const Variable& x, int dim = -1, bool keepdim = false);
    Variable mean(const Variable& x, int dim = -1, bool keepdim = false);

    // Shape
    Variable reshape(const Variable& x, const Shape& new_shape);
    Variable squeeze(const Variable& x, int dim = -1);
    Variable unsqueeze(const Variable& x, int dim);
    Variable slice(const Variable& x, int dim, size_t start, size_t end);

    // Misc
    Variable dropout(const Variable& x, float p, bool training);
    Variable layer_norm(const Variable& x, const Variable& weight,
                        const Variable& bias, float eps = 1e-5);
}

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

### tokenizer.hpp

```cpp
// LightwatchAI2 API Contract: Tokenizer
// Defined by: Phase 06-07 | Consumers: 08, 27, 38
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace lightwatch::tokenizer {

using TokenId = int32_t;

struct SpecialTokens {
    static constexpr TokenId EOS = 50256;  // <|endoftext|>
    static constexpr TokenId PAD = 50256;  // Same as EOS for GPT-2
};

class Vocabulary {
public:
    Vocabulary();

    TokenId token_to_id(const std::string& token) const;
    std::string id_to_token(TokenId id) const;
    bool contains(const std::string& token) const;
    size_t size() const;

    TokenId eos_id() const;
    TokenId pad_id() const;

    static Vocabulary from_json(const std::string& path);  // encoder.json

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;
};

class BPETokenizer {
public:
    BPETokenizer();

    std::vector<TokenId> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenId>& tokens) const;

    std::vector<std::vector<TokenId>> encode_batch(
        const std::vector<std::string>& texts) const;

    const Vocabulary& vocab() const;
    size_t vocab_size() const;  // Returns 50257
    TokenId eos_id() const;
    TokenId pad_id() const;

    static BPETokenizer from_files(const std::string& vocab_path,
                                    const std::string& merges_path);
    static BPETokenizer gpt2(const std::string& vocab_dir = "data/vocab");

private:
    Vocabulary vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const;
    };
    std::unordered_map<std::pair<std::string,std::string>, int, PairHash> merge_ranks_;
};

}  // namespace lightwatch::tokenizer
```

### module.hpp

```cpp
// LightwatchAI2 API Contract: Module
// Defined by: Phase 11 | Consumers: 12-19, 31
#pragma once
#include "autograd.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace lightwatch::nn {

using autograd::Variable;

class Module {
public:
    virtual ~Module() = default;

    virtual Variable forward(const Variable& input) = 0;

    std::vector<Variable*> parameters();
    std::vector<std::pair<std::string, Variable*>> named_parameters();
    size_t num_parameters() const;

    void train(bool mode = true);
    void eval();
    bool is_training() const;

    void zero_grad();

    std::unordered_map<std::string, Tensor<float>> state_dict() const;
    void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict);

protected:
    bool training_ = true;
    void register_parameter(const std::string& name, Variable& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

private:
    std::vector<std::pair<std::string, Variable*>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
};

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Variable forward(const Variable& input) override;

    Variable weight;  // Shape: [out_features, in_features]
    Variable bias;    // Shape: [out_features]
};

class LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5);
    Variable forward(const Variable& input) override;

    Variable weight;  // Shape: [normalized_shape]
    Variable bias;    // Shape: [normalized_shape]
private:
    float eps_;
};

class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);
    Variable forward(const Variable& input) override;
    Variable forward(const Tensor<int32_t>& indices);

    Variable weight;  // Shape: [num_embeddings, embedding_dim]
};

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1);
    Variable forward(const Variable& input) override;
private:
    float p_;
};

}  // namespace lightwatch::nn
```

### optimizer.hpp

```cpp
// LightwatchAI2 API Contract: Optimizer
// Defined by: Phase 22 | Consumers: 23-26, 29
#pragma once
#include "autograd.hpp"
#include <vector>
#include <unordered_map>

namespace lightwatch::optim {

using autograd::Variable;

struct OptimizerOptions {
    float lr = 1e-3;
    float weight_decay = 0.0;
};

class Optimizer {
public:
    explicit Optimizer(std::vector<Variable*> params, OptimizerOptions opts = {});
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void zero_grad();

    float get_lr() const;
    void set_lr(float lr);

protected:
    std::vector<Variable*> params_;
    OptimizerOptions options_;
    std::unordered_map<Variable*, std::unordered_map<std::string, Tensor<float>>> state_;
};

struct SGDOptions : OptimizerOptions {
    float momentum = 0.0;
    bool nesterov = false;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Variable*> params, SGDOptions opts = {});
    void step() override;
private:
    SGDOptions opts_;
};

struct AdamOptions : OptimizerOptions {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Variable*> params, AdamOptions opts = {});
    void step() override;
private:
    AdamOptions opts_;
    int step_count_ = 0;
};

class AdamW : public Adam {
public:
    AdamW(std::vector<Variable*> params, AdamOptions opts = {});
    void step() override;
};

class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;
    virtual void step() = 0;
    float get_last_lr() const;
protected:
    Optimizer& optimizer_;
    int step_count_ = 0;
    float base_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0);
    void step() override;
private:
    int T_max_;
    float eta_min_;
};

class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer, int warmup_steps, float start_factor = 0.0);
    void step() override;
private:
    int warmup_steps_;
    float start_factor_;
};

}  // namespace lightwatch::optim
```

---

## PHASE 0.7: ACQUIRE EXTERNAL ASSETS

### Objective
Download required tokenizer assets before implementation begins.

### Deliverables
| File | Source | Purpose | Required |
|------|--------|---------|----------|
| `data/vocab/vocab.bpe` | HuggingFace gpt2 | BPE merge rules (50000 merges) | **YES** |
| `data/vocab/encoder.json` | HuggingFace gpt2 | Token-to-ID mapping (50257 entries) | **YES** |
| `data/weights/pytorch_model.bin` | HuggingFace gpt2 | Pretrained weights (~500MB) | **NO** (optional) |

### Weight Download Policy
- Tokenizer assets (`vocab.bpe`, `encoder.json`) are **required** for any operation
- Pretrained weights are **optional** and only needed for:
  - Validation against reference implementation
  - Inference without training
  - Fine-tuning from pretrained checkpoint

If pretrained weights are needed later (e.g., Phase 37 validation), download them then:
```bash
# Optional: Download pretrained weights when needed
curl -L -o data/weights/pytorch_model.bin \
    "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin"
```

Do NOT fail Phase 0.7 if weight download fails. Only tokenizer assets are blocking.

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

### Test Spec File Creation

During Phase 0.5, create the test spec files in `docs/test_specs/` BEFORE generating phase prompts. The test specs are defined in the TEST SPEC FILE TEMPLATES section of this document.

**Creation order within Phase 0.5:**
1. Create `docs/test_specs/README.md` with usage instructions
2. Create all `docs/test_specs/phase-XX-*.md` files from templates in TEST SPEC FILE TEMPLATES section
3. Generate phase prompts that reference these test specs
4. Run `python3 scripts/validate_prompts.py` to validate

```bash
# Step 1: Create test_specs directory and README
mkdir -p docs/test_specs
# (copy README content from TEST SPEC FILE TEMPLATES section)

# Step 2: Create test spec files for phases 03, 05, 06, 15, 31, etc.
# (copy each template from TEST SPEC FILE TEMPLATES section)

# Step 3: Generate phase prompts (using the procedure below)

# Step 4: Validate
python3 scripts/validate_prompts.py
```

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

**Note:** Detailed test specifications for complex phases have been extracted to `docs/test_specs/` to reduce context pressure. Reference these files when generating phase prompts:

| Phase | File | Tests |
|-------|------|-------|
| 03 | `docs/test_specs/phase-03-tensor.md` | 12 |
| 04 | `docs/test_specs/phase-04-simd.md` | 6 |
| 05 | `docs/test_specs/phase-05-autograd.md` | 10 |
| 06 | `docs/test_specs/phase-06-tokenizer.md` | 10 |
| 15 | `docs/test_specs/phase-15-attention.md` | 8 |
| 16 | `docs/test_specs/phase-16-mha.md` | 6 |
| 19 | `docs/test_specs/phase-19-decoder.md` | 5 |
| 29 | `docs/test_specs/phase-29-training.md` | 6 |
| 31 | `docs/test_specs/phase-31-gpt.md` | 6 |
| 36 | `docs/test_specs/phase-36-kvcache.md` | 5 |
| 38 | `docs/test_specs/phase-38-cli.md` | 8 |

See `docs/test_specs/README.md` for usage instructions.

### Test Naming Convention

All tests MUST follow this format: `test_phase_XX_<component>_<behavior>`

Example: `test_phase_03_tensor_matmul_2d`

This enables phase-specific filtering: `ctest -R "phase_03"`

### Minimum Test Requirements

| Complexity | Minimum Tests |
|------------|---------------|
| HIGH | ≥6 tests |
| MEDIUM | ≥4 tests |
| LOW | ≥2 tests |

---

## TEST SPEC FILE TEMPLATES

These templates define the complete test specifications for complex phases. During Phase 0.5, copy these to `docs/test_specs/phase-XX-*.md` files BEFORE generating phase prompts.

### docs/test_specs/phase-03-tensor.md

```markdown
# Phase 03: Tensor Core - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 12

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_03_tensor_construction` | `Tensor<float> t({2,3,4})` | `t.numel()==24`, `t.ndim()==3`, `t.shape()=={2,3,4}` |
| `test_phase_03_tensor_zeros` | `Tensor<float>::zeros({3,3})` | All 9 elements == 0.0f |
| `test_phase_03_tensor_ones` | `Tensor<float>::ones({2,2})` | All 4 elements == 1.0f |
| `test_phase_03_tensor_randn` | `Tensor<float>::randn({1000})` | Mean ∈ [-0.1, 0.1], Std ∈ [0.9, 1.1] |
| `test_phase_03_tensor_matmul_2d` | `A{2,3}=[1..6], B{3,4}=[1..12]` | Result shape `{2,4}`, `C[0,0]==38.0f` |
| `test_phase_03_tensor_matmul_batch` | `A{2,2,3}, B{2,3,4}` randn | Result shape `{2,2,4}`, matches loop impl |
| `test_phase_03_tensor_broadcast_add` | `A{2,3}=[1..6] + B{3}=[10,20,30]` | `C[0,:]={11,22,33}`, `C[1,:]={14,25,36}` |
| `test_phase_03_tensor_broadcast_mul` | `A{2,1,4}=1.0 * B{3,1}=[2,3,4]` | Result shape `{2,3,4}`, all rows scaled |
| `test_phase_03_tensor_slice` | `T{10,20}.slice(0, 2, 5)` | Result shape `{3,20}`, `R[0,:]==T[2,:]` |
| `test_phase_03_tensor_transpose` | `T{2,3}=[1..6].transpose(0,1)` | Result shape `{3,2}`, `R[0,1]==T[1,0]==4` |
| `test_phase_03_tensor_contiguous` | `T{4,4}.slice(0,1,3)` (non-contig) | After `.contiguous()`: `is_contiguous()==true` |
| `test_phase_03_tensor_reduction_sum` | `T{2,3}=[1,2,3,4,5,6].sum(1)` | Result `{2}`, values `[6.0, 15.0]` |

## Implementation Notes

- All operations must handle edge cases: empty tensors, single-element tensors
- Broadcasting follows NumPy rules (right-align shapes, expand dims of size 1)
- Tolerance for floating-point comparisons: 1e-5
```

### docs/test_specs/phase-05-autograd.md

```markdown
# Phase 05: Autograd Engine - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 10

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_05_autograd_add` | `a=Var(2.0,grad=T), b=Var(3.0,grad=T), c=a+b, c.backward()` | `a.grad==1.0`, `b.grad==1.0` |
| `test_phase_05_autograd_mul` | `a=Var(2.0,grad=T), b=Var(3.0,grad=T), c=a*b, c.backward()` | `a.grad==3.0`, `b.grad==2.0` |
| `test_phase_05_autograd_matmul` | `A{2,3}=randn, B{3,4}=randn, C=A@B, C.sum().backward()` | `A.grad.shape=={2,3}`, numerical grad check (tol 1e-4) |
| `test_phase_05_autograd_chain` | `a{2,2}=randn, b{2,2}=randn, d=relu(a@b+1.0), d.sum().backward()` | All inputs have `.has_grad()==true` |
| `test_phase_05_autograd_no_grad` | `a=Var(2.0,grad=F), b=Var(3.0,grad=T), c=a*b` | `a.has_grad()==false`, `b.has_grad()==true` |
| `test_phase_05_autograd_accumulation` | `a=Var(2.0,grad=T), b=a*3, c=a*4, (b+c).backward()` | `a.grad==7.0` (gradients accumulate) |
| `test_phase_05_autograd_detach` | `a=Var(2.0,grad=T), b=a.detach()` | `b.grad_fn()==nullptr`, `b.data()==a.data()` |
| `test_phase_05_autograd_relu_grad` | `x=Var([-1,0,1,2],grad=T), y=relu(x), y.sum().backward()` | `x.grad==[0,0,1,1]` |
| `test_phase_05_autograd_softmax_grad` | `x=Var([1,2,3],grad=T), y=softmax(x,0), y[1].backward()` | Jacobian matches numerical diff (tol 1e-4) |
| `test_phase_05_autograd_no_grad_guard` | Inside `NoGradGuard{}`, create `c=a*b` | `c.grad_fn()==nullptr` regardless of input grad flags |

## Implementation Notes

- Numerical gradient checking: `(f(x+h) - f(x-h)) / 2h` with `h=1e-4`
- Gradient accumulation is the default (call `zero_grad()` to reset)
- NoGradGuard must be thread-safe (use thread_local counter)
```

### docs/test_specs/phase-06-tokenizer.md

```markdown
# Phase 06: BPE Tokenizer - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 10

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_06_tokenizer_roundtrip` | `"Hello, world!"` | `decode(encode(x)) == "Hello, world!"` |
| `test_phase_06_tokenizer_special` | `tokenizer.eos_id()` | Returns `50256` |
| `test_phase_06_tokenizer_unicode` | `"日本語テスト"` | Non-empty token vector, no crash |
| `test_phase_06_tokenizer_empty` | `""` | Returns empty `vector<TokenId>{}` |
| `test_phase_06_tokenizer_vocab_size` | `tokenizer.vocab_size()` | Returns `50257` |
| `test_phase_06_tokenizer_whitespace` | `"  hello   world  "` | Roundtrip preserves exact whitespace |
| `test_phase_06_tokenizer_numbers` | `"12345 67890"` | Roundtrip exact match |
| `test_phase_06_tokenizer_long_text` | 10KB random ASCII text | No crash, all token IDs in `[0, 50256]` |
| `test_phase_06_tokenizer_emoji` | `"Hello 🌍🚀 World"` | Roundtrip preserves emojis exactly |
| `test_phase_06_tokenizer_newlines` | `"line1\nline2\r\nline3"` | Roundtrip exact match |

## Implementation Notes

- GPT-2 uses byte-level BPE (handles arbitrary UTF-8)
- Vocab files: `encoder.json` (50257 entries), `vocab.bpe` (50000 merges)
- No UNK token — unknown bytes encoded as byte tokens (e.g., `<0xFF>`)
```

### docs/test_specs/phase-15-attention.md

```markdown
# Phase 15: Single-Head Attention - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 8

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_15_attention_shape` | `Q,K,V all {2,12,64,64}` | Output shape `{2,12,64,64}` |
| `test_phase_15_attention_causal` | `S=4, query at pos 2` | Attention weights: `w[3]==0.0` (future masked) |
| `test_phase_15_attention_softmax` | `S=8, any Q,K,V` | Each row of attention weights sums to 1.0 (tol 1e-5) |
| `test_phase_15_attention_gradient` | `Q,K,V {1,1,4,8} randn` | Numerical gradient check passes (tol 1e-3) |
| `test_phase_15_attention_scale` | `d_k=64` | Pre-softmax scores scaled by `1/8.0` (sqrt(64)) |
| `test_phase_15_attention_mask_inf` | Masked position `K[0,0,3,:]=999.0` | Output unaffected by masked position values |
| `test_phase_15_attention_single_token` | `S=1, Q,K,V {1,1,1,64}` | Output shape `{1,1,1,64}`, no NaN |
| `test_phase_15_attention_long_sequence` | `S=1024, B=1, H=1, D=64` | Completes without OOM, output shape correct |

## Implementation Notes

- Attention formula: `softmax(Q @ K.T / sqrt(d_k) + mask) @ V`
- Causal mask: `-inf` for positions where `j > i`
- Use `-1e9` instead of `-inf` to avoid NaN in softmax gradients
```

### docs/test_specs/phase-31-gpt.md

```markdown
# Phase 31: GPT Architecture - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_31_gpt_forward_shape` | `input_ids {2, 16}` (batch=2, seq=16) | Output logits shape `{2, 16, 50257}` |
| `test_phase_31_gpt_causal` | Seq `[A,B,C,D]`, compute logits | `logits[2]` (for C) identical whether D present or not |
| `test_phase_31_gpt_parameter_count` | GPT-2 Small config | Total params in `[118M, 130M]` (~124M ± 5%) |
| `test_phase_31_gpt_gradient` | Forward + backward on random input | All named parameters have non-zero `.grad` |
| `test_phase_31_gpt_embedding_tied` | Check `wte.weight` and `lm_head.weight` | Same underlying data pointer |
| `test_phase_31_gpt_layer_order` | Hook each layer, forward pass | Layers execute in order 0,1,2,...,11 |

## Implementation Notes

- GPT-2 Small config: 12 layers, 768 hidden, 12 heads, 50257 vocab, 1024 ctx
- Parameter count breakdown: wte(38.6M) + wpe(0.8M) + 12*layer(7.1M) + ln_f(1.5K) ≈ 124M
- Embedding tying: `lm_head.weight = wte.weight` (shared, not copied)
```

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

Validates all 40 phase prompts for consistency. Full script: `scripts/validate_prompts.py.template`

**Key validations:**
- All 40 prompt files exist (`docs/prompts/phase-XX-*.md`)
- Prerequisites match `PHASE_DEPS` dictionary
- Required sections present: Objective, Prerequisites, Inputs, Outputs, Specification, Required Tests, Acceptance Criteria
- Acceptance criteria use valid command prefixes
- Minimum test counts for complex phases (03: 10, 04: 5, 05: 8, etc.)
- Test spec files exist in `docs/test_specs/`

**Usage:**
```bash
python3 scripts/validate_prompts.py
# Exit 0 = all valid, Exit 1 = errors found
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
   - **Before EVERY commit:**
     1. Run `cmake --build build` — must exit 0
     2. Run `ctest --test-dir build -R "phase_$(printf '%02d' $N)" --output-on-failure` — must exit 0
        - This matches all tests named `test_phase_XX_*` for current phase
     3. If either fails, fix before committing
   - Commit after each logical unit of work with descriptive messages
   - If tests don't exist yet (early in phase), at minimum ensure build succeeds

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
     * Update last_commit with new HEAD
     * If checkpoint: add to checkpoints_passed
   - Remove lock file: `rm -f .lightwatch.lock`
   - git checkout main
   - git merge phase-{N:02d}-{slug} --no-ff
   - git push origin main (if remote exists)

8. COMMIT STATE
   - git add .lightwatch_state.json
   - git commit with authorship
```

---

## BUILD FAILURE RECOVERY

When `cmake --build build` fails during implementation:

### 1. Capture the Error
```bash
cmake --build build 2>&1 | tee /tmp/build_error.txt
```

### 2. Classify the Error

| Error Type | Example | Action |
|------------|---------|--------|
| Syntax error | `error: expected ';'` | Fix the syntax in indicated file:line |
| Missing include | `fatal error: 'foo.hpp' not found` | Add include or create missing file |
| Undefined reference | `undefined reference to 'Bar::baz()'` | Implement missing function or fix signature |
| Type mismatch | `cannot convert 'X' to 'Y'` | Check contract, fix types |
| Contract violation | Signature doesn't match contract | DO NOT change contract; fix implementation |

### 3. Fix and Verify
```bash
# Make fix
# Rebuild
cmake --build build 2>&1 | tee /tmp/build_error.txt

# If same error: re-read error message carefully
# If new error: address new error
# If success: run tests before committing
```

### 4. Escalate if Stuck
If the same build error persists after 3 fix attempts:
1. Document in `docs/architecture/ESCALATIONS.md`
2. Include full error text
3. Include what was attempted
4. Set `project_status = "ESCALATED"`
5. STOP and wait for human input

---

## GIT PUSH FAILURE HANDLING

When `git push` fails during the COMPLETE step:

### 1. Identify the Error
```bash
git push origin main 2>&1 | tee /tmp/push_error.txt
```

### 2. Common Errors and Actions

| Error | Cause | Action |
|-------|-------|--------|
| `Permission denied (publickey)` | SSH key not configured | Escalate — requires human SSH setup |
| `fatal: Authentication failed` | Bad credentials | Escalate — requires human auth setup |
| `! [rejected]` (non-fast-forward) | Remote has commits not in local | `git pull --rebase origin main` then retry |
| `remote: Repository not found` | Wrong remote URL or no access | Escalate — requires human intervention |
| Connection timeout | Network issue | Retry up to 3 times with 10s delay |

### 3. Retry Logic
```bash
push_with_retry() {
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "Push attempt $attempt/$max_attempts..."
        if git push origin main 2>&1; then
            echo "Push succeeded"
            return 0
        fi

        echo "Push failed, waiting 60s before retry..."
        sleep 60
        attempt=$((attempt + 1))
    done

    echo "ERROR: Push failed after $max_attempts attempts"
    return 1
}
```

### 4. Escalation
If push fails after 3 retries OR encounters auth error:
1. Document error in `docs/architecture/ESCALATIONS.md`
2. Include full error text from `/tmp/push_error.txt`
3. Set `project_status = "ESCALATED"` in state file
4. Commit state locally (it won't push, but preserves progress)
5. **STOP** and wait for human to fix auth/network

### 5. Local Progress is NOT Lost
All commits exist locally. Once push access is restored:
```bash
git push origin main  # Will push all pending commits
```

> **Note:** Git push failures do NOT block local phase completion. The phase is complete when code is committed locally. Push is for backup/sync only.

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

# Memory baseline test (Linux/macOS)
# Creates a 512x768 tensor, performs matmul, checks RSS < 50MB
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MEMORY_OUTPUT=$(/usr/bin/time -v ./build/bin/test_tensor_memory_baseline 2>&1)
    MAX_RSS=$(echo "$MEMORY_OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
    if [ "$MAX_RSS" -gt 51200 ]; then
        echo "ERROR: Memory baseline ${MAX_RSS}KB exceeds 50MB limit"
        exit 1
    fi
    echo "Memory baseline: ${MAX_RSS}KB (limit: 51200KB)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MEMORY_OUTPUT=$(/usr/bin/time -l ./build/bin/test_tensor_memory_baseline 2>&1)
    MAX_RSS_BYTES=$(echo "$MEMORY_OUTPUT" | grep "maximum resident set size" | awk '{print $1}')
    MAX_RSS=$((MAX_RSS_BYTES / 1024))
    if [ "$MAX_RSS" -gt 51200 ]; then
        echo "ERROR: Memory baseline ${MAX_RSS}KB exceeds 50MB limit"
        exit 1
    fi
    echo "Memory baseline: ${MAX_RSS}KB (limit: 51200KB)"
fi

# Memory leak check (if valgrind available)
if command -v valgrind &> /dev/null; then
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/bin/test_tensor_basic 2>&1 | tail -20
    echo "Valgrind memory leak check passed"
fi
```

**Memory baseline test requirements (Phase 05):**
- `test_tensor_memory_baseline` creates 512x768 float32 tensor (~1.5MB)
- Performs matmul with transposed self: O(512×768×768) = ~302M FLOPs
- Peak RSS must stay under 50MB (allows for temporary allocations)
- No external BLAS calls in this test

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

### 15. Committing Without Testing
❌ `git commit` before running `cmake --build` and `ctest`
✓ Every commit should represent a buildable, tested state. Run tests before every commit.

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
| Git Push Failure | `git push` fails after 3 retry attempts | Auth/permission issue needs human |

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

Final verification that project is complete. Full script: `scripts/verify_complete.sh.template`

**Checks (all must pass):**
1. Build with strict warnings (`-Wall -Werror -Wextra`)
2. All tests pass (`ctest`)
3. Generation produces valid output (≥5 words from "The")
4. Performance ≥50 tok/s
5. No memory leaks (valgrind, if available)
6. Memory usage <2GB
7. State file shows 40/40 phases complete
8. Documentation exists (README.md, DECISIONS.md)

**Usage:**
```bash
chmod +x scripts/verify_complete.sh
./scripts/verify_complete.sh
# Exit 0 = COMPLETE, Exit 1 = failed
```

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

## CONTEXT MANAGEMENT

This prompt is large (~2500 lines). Follow these guidelines to manage context efficiently:

### Reading Strategy
1. **Do NOT read entire Master Prompt at session start** — rely on state file for current phase
2. **Read only relevant sections** when needed:
   - Starting a phase → read its prompt file from `docs/prompts/phase-XX-*.md`
   - Need contract → read from `docs/contracts/*.hpp.spec`
   - Need test spec → read from `docs/test_specs/phase-XX-*.md`
3. **Use state file** for navigation — `current_phase` tells you where to focus

### Context Budget
| Content | When to Load |
|---------|--------------|
| State file | Always (first read each session) |
| Current phase prompt | When starting implementation |
| Dependency phase outputs | Only if needed for inputs |
| Contract files | When implementing module boundaries |
| Test specs | When writing tests |
| Master Prompt sections | Only for reference (checkpoints, anti-patterns, etc.) |

### Avoiding Context Overflow
- Complete and commit each phase before reading the next phase prompt
- If context becomes constrained, summarize completed work in state file
- Use `jq` to query state file rather than holding full state in context
- Reference external files by path rather than copying content into chat

### Session Recovery
If resuming after interruption:
```bash
# 1. Read state file
cat .lightwatch_state.json | jq '.current_phase, .completed_phases'

# 2. Check for uncommitted work
git status

# 3. Resume from current phase prompt
cat docs/prompts/phase-$(jq -r '.current_phase' .lightwatch_state.json | xargs printf "%02d")-*.md
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
