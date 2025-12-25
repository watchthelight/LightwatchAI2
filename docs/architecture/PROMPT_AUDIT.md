# Master Prompt Audit

Audit performed: 2025-12-25
Document reviewed: `docs/Master Prompt.md` (2525 lines)

## Critical Issues (must fix before execution)

| ID | Location | Issue | Impact | Suggested Fix |
|----|----------|-------|--------|---------------|
| C1 | Line 697 | `/* hash */` placeholder in `BPETokenizer::merge_ranks_` type | Code won't compile; tokenizer unusable | Replace with actual `PairHash` struct implementation |
| C2 | STATE MANAGEMENT section | No partial failure recovery procedure | Session interruption mid-phase loses progress; unclear resume behavior | Add explicit partial failure handling with `PARTIAL_FAILURE` status and `pending_deliverables` recovery |
| C3 | verify_complete.sh | No memory budget enforcement | Could pass verification while using >2GB RAM (spec says ~500MB model + 75MB cache) | Add RSS memory check using `/usr/bin/time -v` or platform equivalent |

## High Priority (should fix)

| ID | Location | Issue | Impact | Suggested Fix |
|----|----------|-------|--------|---------------|
| H1 | docs/contracts/ | Missing `serialization.md` contract | Phase 37 implementer must guess HuggingFace weight format (safetensors vs pickle, tensor naming, endianness) | Create explicit serialization contract specifying exact format |
| H2 | docs/contracts/ | Missing `cli.md` contract | Phase 38 implementer has incomplete spec; JSON schema scattered across sections | Create unified CLI contract with all commands, flags, exit codes, JSON schemas |
| H3 | docs/architecture/ | Missing `PLATFORM_SUPPORT.md` | Unclear which platforms/architectures are supported; SIMD fallback strategy undefined | Create platform support document specifying x86-64/ARM64, SIMD requirements, OS differences |
| H4 | Phase 06 tests (line 1496-1506) | Tokenizer test table incomplete | Missing critical edge cases could cause production bugs | Add tests: empty string, unicode with emoji, whitespace preservation, long text (2000+ tokens) |
| H5 | PHASE EXECUTION ORDER | Phase 0.5 comes after 0.7 but numbering suggests otherwise | Confusing execution order; implementer might run 0.5 before 0.7 | Add explicit callout note about non-sequential numbering |
| H6 | Phase 38 CLI tests | Missing JSON schema validation test | JSON output might not match documented schema | Add test that validates JSON output against schema |
| H7 | Acceptance criteria | Some phases reference test binaries that don't exist yet | Acceptance criteria like `./build/bin/test_tensor_autograd` assume specific binary names | Document that test binary names are conventions, or specify exact CMake target names |

## Low Priority (nice to have)

| ID | Location | Issue | Impact | Suggested Fix |
|----|----------|-------|--------|---------------|
| L1 | TEST SPECIFICATIONS (lines 1450-1580) | ~130 lines of inline test tables | Increases context pressure on every session | Consider extracting to `docs/test_specs/` directory, reference by phase |
| L2 | BENCHMARK SPECIFICATION + Phase 38 | Redundant JSON output specifications | Two places define benchmark JSON format | Consolidate into cli.md contract, reference from both sections |
| L3 | validate_prompts.py | Hardcoded MIN_TESTS dict | Adding new complex phases requires script modification | Consider making MIN_TESTS configurable or derive from phase complexity field |
| L4 | Anti-patterns section | 14 items inline | Moderate context usage | Could extract to `docs/architecture/ANTI_PATTERNS.md` if context becomes issue |
| L5 | Contract files | HPP files in docs/contracts/ | Unusual location for header files; might confuse IDE tooling | Consider if these should be reference docs vs actual includable headers |

## Dependency Gap Analysis

| Consumer Phase | Expected Input | Source Phase | Status |
|----------------|----------------|--------------|--------|
| 37 (Serialization) | HuggingFace weight format spec | Contract | **MISSING** - no serialization.md |
| 38 (CLI) | Complete CLI interface spec | Contract | **MISSING** - no cli.md |
| 04 (SIMD) | Platform/architecture requirements | Architecture doc | **MISSING** - no PLATFORM_SUPPORT.md |
| All phases | Partial failure recovery | STATE MANAGEMENT | **INCOMPLETE** - procedure undefined |

## Edge Cases Not Covered

| Scenario | Current Handling | Recommendation |
|----------|------------------|----------------|
| macOS without valgrind | Skipped silently | Already handled (SKIP_VALGRIND flag) ✓ |
| Windows build | Undefined | Document in PLATFORM_SUPPORT.md |
| ARM64 Mac (Apple Silicon) | AVX2 not available | Document NEON fallback or ARM exclusion |
| Network failure during asset download | Escalation trigger exists | ✓ Adequate |
| Disk full during build | Not handled | Low priority - OS-level error |
| Git conflicts on merge | Not handled | Add conflict resolution procedure |

## Summary

- **3 Critical issues** requiring immediate fix before execution
- **7 High priority issues** that should be addressed for robustness
- **5 Low priority issues** for future improvement
- **4 Dependency gaps** where contracts or docs are missing
- **2 Edge cases** worth documenting (Windows, ARM64)

### Recommended Fix Order

1. C1: PairHash placeholder (blocks compilation)
2. H1-H3: Create missing contract/architecture docs
3. C2: Partial failure recovery procedure
4. H4-H5: Tokenizer tests and phase ordering callout
5. C3: Memory budget check in verify_complete.sh
6. Remaining high/low priority as time permits
