# LightwatchAI2 Reference Files

This directory contains code templates, scripts, and specifications extracted from the Master Prompt to reduce context pressure and enable independent editing.

## Directory Structure

| Directory | Contents |
|-----------|----------|
| `scripts/` | Shell and Python scripts for build, validation, and asset acquisition |
| `templates/` | Configuration file templates (CMake, gitignore, CLAUDE.md, etc.) |
| `contracts/` | C++ API contract specifications (.hpp.spec files) |
| `test_specs/` | Detailed test specifications for complex phases |
| `examples/` | Example code snippets and templates |
| `schemas/` | JSON schemas for CLI output and state files |

## Usage

During phase execution, read relevant files with:

```bash
# Read a script
cat docs/references/scripts/acquire_assets.sh

# Read a contract
cat docs/references/contracts/tensor.hpp.spec

# Read test specs for a phase
cat docs/references/test_specs/phase-03-tensor.md
```

## Scripts

| File | Used In | Purpose |
|------|---------|---------|
| `check_toolchain.sh` | Phase 0 | Verify toolchain versions |
| `session_recovery.sh` | Every session | Lock file and state validation |
| `acquire_assets.sh` | Phase 0.7 | Download GPT-2 tokenizer assets |
| `push_with_retry.sh` | Git operations | Retry git push with backoff |
| `checkpoint_10.sh` | Checkpoint 10 | Foundation verification (tensor, tokenizer, autograd) |
| `checkpoint_20.sh` | Checkpoint 20 | Neural core verification (MLP, gradients, attention) |
| `checkpoint_30.sh` | Checkpoint 30 | Training verification (overfit, checkpoint, LR) |

## Templates

| File | Created In | Purpose |
|------|------------|---------|
| `lightwatch_state.json.template` | Phase 0 | Initial state file structure |
| `CMakeLists.txt.template` | Phase 0 | Base CMake configuration |
| `gitignore.template` | Phase 0 | Git ignore patterns |
| `CLAUDE.md.template` | Phase 0 | Claude Code configuration |
| `DECISIONS.md.template` | Phase 0 | Architectural decision record format |
| `phase_prompt.md.template` | Phase 0.5 | Mandatory phase prompt structure |

## Contracts

| File | Defined In | Consumers |
|------|------------|-----------|
| `tensor.hpp.spec` | Phase 03 | 04, 05, 08, 09, 11-19, 21-25, 31-36 |
| `autograd.hpp.spec` | Phase 05 | 08, 11-19, 21-25, 31 |
| `tokenizer.hpp.spec` | Phase 06-07 | 08, 27, 38 |
| `module.hpp.spec` | Phase 11 | 12-19, 31 |
| `optimizer.hpp.spec` | Phase 22 | 23-26, 29 |

## Test Specifications

| File | Complexity | Min Tests |
|------|------------|-----------|
| `phase-03-tensor.md` | HIGH | 12 |
| `phase-04-simd.md` | HIGH | 6 |
| `phase-05-autograd.md` | HIGH | 10 |
| `phase-06-tokenizer.md` | MEDIUM | 10 |
| `phase-15-attention.md` | HIGH | 8 |
| `phase-16-mha.md` | MEDIUM | 6 |
| `phase-19-decoder.md` | MEDIUM | 5 |
| `phase-29-training.md` | HIGH | 6 |
| `phase-31-gpt.md` | HIGH | 6 |
| `phase-36-kvcache.md` | HIGH | 5 |
| `phase-38-cli.md` | MEDIUM | 8 |

## Examples

| File | Purpose |
|------|---------|
| `commit_messages.md` | Commit message format and examples |
| `decision_entry.md` | How to document architectural decisions |
| `smoke_test.cpp` | Phase 0.9 toolchain validation |
| `benchmark_loop.cpp` | Benchmark measurement pseudocode |

## Schemas

| File | Purpose |
|------|---------|
| `cli_output.schema.json` | JSON schema for `--json` CLI output |
| `state_file.schema.json` | JSON schema for `.lightwatch_state.json` |
