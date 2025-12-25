# Phase 40: Final Integration (Checkpoint)

## Objective
Complete integration testing, documentation, CI setup, and verification that all requirements are met.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 01-39 | All prior phases complete |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| All | All headers | Full API |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| docs/API.md | API documentation | Users |
| docs/BUILDING.md | Build instructions | Users |
| .github/workflows/ci.yml | CI configuration | GitHub |
| examples/*.cpp | Usage examples | Users |

## Specification

### Data Structures
N/A (integration phase)

### Function Signatures
N/A (integration phase)

### Algorithmic Requirements
1. **Full test suite**: All phases pass
2. **Performance check**: >= 50 tok/s on benchmark
3. **Memory check**: < 2GB for inference
4. **Documentation**: Complete API docs
5. **Examples**: Working code examples

### Performance Constraints
- Full test suite: < 10 minutes
- Benchmark: >= 50 tok/s

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_40_full_pipeline` | Load, generate, decode | Valid text |
| `test_phase_40_performance` | Benchmark | >= 50 tok/s |
| `test_phase_40_memory` | Monitor RSS | < 2GB |
| `test_phase_40_examples_compile` | Compile examples | All succeed |
| `test_phase_40_examples_run` | Run examples | All succeed |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0 with `-Wall -Werror -Wextra`
- [ ] `ctest --output-on-failure` exits 0 (all tests)
- [ ] `./build/bin/lightwatch generate --prompt "The" | wc -w` >= 5
- [ ] `./build/bin/lightwatch benchmark --json` shows >= 50 tok/s
- [ ] `valgrind --leak-check=full` (if available) shows no leaks
- [ ] Memory usage < 2GB
- [ ] `.lightwatch_state.json` shows 40 completed phases
- [ ] `test -f docs/API.md && test -f docs/BUILDING.md`

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 5 |
| New test files | 2 |
| Complexity | MEDIUM |

## Notes
- This is CHECKPOINT 4 (FINAL) - project complete when all pass
- Run `scripts/verify_complete.sh` for comprehensive check
- If performance < 50 tok/s, document bottleneck in DECISIONS.md
- If performance < 25 tok/s, this is an ESCALATION TRIGGER
- Update README.md with final status and usage instructions
