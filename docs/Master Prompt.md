# LightwatchAI2 - C++ Text Generation Model Build System

## Claude Code Master Orchestration Prompt

---

## PREAMBLE

You are building a **minimal-dependency C++ text generation model** from scratch. The only allowed external dependency is a JSON parsing library (nlohmann/json or similar). No Eigen, OpenBLAS, PyTorch, TensorFlow, or other ML frameworks.

**Repository:** https://github.com/watchthelight/LightwatchAI2

---

## REFERENCE FILES

Code templates, scripts, and specifications have been extracted to `docs/references/` to reduce context pressure. See `docs/references/README.md` for the complete index.

```
docs/references/
├── README.md                     # Index of all reference files
├── scripts/                      # Shell/Python scripts
├── templates/                    # Configuration file templates
├── contracts/                    # C++ API specifications
├── test_specs/                   # Detailed test specifications
├── examples/                     # Code examples
└── schemas/                      # JSON schemas
```

**Usage:** Read files during phase execution:
```bash
cat docs/references/scripts/acquire_assets.sh
cat docs/references/contracts/tensor.hpp.spec
cat docs/references/test_specs/phase-03-tensor.md
```

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

**File:** `docs/references/scripts/check_toolchain.sh`

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

**File:** `docs/references/scripts/session_recovery.sh`

Key checks:
- Lock file with 4-hour staleness detection
- State file JSON validation
- Branch/commit verification
- Uncommitted changes detection
- Phase resumption status

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

**File:** `docs/references/templates/CLAUDE.md.template`

Key sections:
- Critical first steps (read Master Prompt, run recovery checklist)
- Session recovery checklist (lock file, state validation)
- Quick reference (authorship, code style, escalation triggers)
- Commit message format and examples

See also: `docs/references/examples/commit_messages.md` for commit examples

### CMakeLists.txt Template

**File:** `docs/references/templates/CMakeLists.txt.template`

Provides: C++17, output directories, warnings, nlohmann/json via FetchContent, testing infrastructure, IDE support.

### .gitignore Content

**File:** `docs/references/templates/gitignore.template`

### docs/architecture/DECISIONS.md Template

**File:** `docs/references/templates/DECISIONS.md.template`

### Example Decision Entry

**File:** `docs/references/examples/decision_entry.md`

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

Complete API specifications are in `docs/references/contracts/`:

| Contract | File | Defined In | Consumers |
|----------|------|------------|-----------|
| Tensor | `tensor.hpp.spec` | Phase 03 | 04, 05, 08, 09, 11-19, 21-25, 31-36 |
| Autograd | `autograd.hpp.spec` | Phase 05 | 08, 11-19, 21-25, 31 |
| Tokenizer | `tokenizer.hpp.spec` | Phase 06-07 | 08, 27, 38 |
| Module | `module.hpp.spec` | Phase 11 | 12-19, 31 |
| Optimizer | `optimizer.hpp.spec` | Phase 22 | 23-26, 29 |

During Phase 0.3, read each `.spec` file and create the corresponding `.hpp` file in `docs/contracts/`.

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

**File:** `docs/references/scripts/acquire_assets.sh`

Features: Tool checking (curl, jq), fallback URLs, asset validation.

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

**File:** `docs/references/templates/phase_prompt.md.template`

Every generated prompt MUST follow the exact structure in the template file.

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
> 📁 **Reference:** [`docs/references/examples/smoke_test.cpp`](references/examples/smoke_test.cpp)
> Copy this file to `tests/smoke/smoke_test.cpp`

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

Test specifications for complex phases are in `docs/references/test_specs/`:

| Phase | File | Tests |
|-------|------|-------|
| 03 | `phase-03-tensor.md` | 12 |
| 04 | `phase-04-simd.md` | 6 |
| 05 | `phase-05-autograd.md` | 10 |
| 06 | `phase-06-tokenizer.md` | 10 |
| 15 | `phase-15-attention.md` | 8 |
| 16 | `phase-16-mha.md` | 6 |
| 19 | `phase-19-decoder.md` | 5 |
| 29 | `phase-29-training.md` | 6 |
| 31 | `phase-31-gpt.md` | 6 |
| 36 | `phase-36-kvcache.md` | 5 |
| 38 | `phase-38-cli.md` | 8 |

During Phase 0.5, copy these to `docs/test_specs/` BEFORE generating phase prompts.

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
> 📁 **Reference:** [`docs/references/examples/benchmark_loop.cpp`](references/examples/benchmark_loop.cpp)

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

Expected JSON output follows [`docs/references/schemas/cli_output.schema.json`](references/schemas/cli_output.schema.json)

---

## PHASE 38 OBSERVABILITY NOTE

The CLI's `--json` output mode serves as the foundation for structured observability. Output formats are defined in:

> 📁 **Reference:** [`docs/references/schemas/cli_output.schema.json`](references/schemas/cli_output.schema.json)
> Defines `generate_output` and `benchmark_output` JSON structures.

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

**File:** `docs/references/scripts/push_with_retry.sh`

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
> 📁 **Reference:** [`docs/references/scripts/checkpoint_10.sh`](references/scripts/checkpoint_10.sh)
> Tests: tensor-autograd integration, tokenizer, memory baseline (<50MB RSS), valgrind leak check

**Memory baseline test requirements (Phase 05):**
- `test_tensor_memory_baseline` creates 512x768 float32 tensor (~1.5MB)
- Performs matmul with transposed self: O(512×768×768) = ~302M FLOPs
- Peak RSS must stay under 50MB (allows for temporary allocations)
- No external BLAS calls in this test

### Checkpoint 20: Neural Core Verification
> 📁 **Reference:** [`docs/references/scripts/checkpoint_20.sh`](references/scripts/checkpoint_20.sh)
> Tests: MLP integration, gradient check (tolerance 1e-5), causal attention mask

### Checkpoint 30: Training Verification
> 📁 **Reference:** [`docs/references/scripts/checkpoint_30.sh`](references/scripts/checkpoint_30.sh)
> Tests: overfit (10 samples → <0.01 loss), checkpoint roundtrip, LR scheduler

### Checkpoint 40: Final Verification
Run the full completion criteria script (see COMPLETION CRITERIA).

---

## REFACTORING PROTOCOL

If Phase N reveals architectural issues in Phase M (where M < N):

### 1. Complete Current Phase to Stable State
All tests must pass before refactoring.

### 2. Document the Issue
Add entry to `docs/architecture/DECISIONS.md` using the decision entry format:
> 📁 **Reference:** [`docs/references/examples/decision_entry.md`](references/examples/decision_entry.md)

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
