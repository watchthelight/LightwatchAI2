<!-- File: docs/references/examples/commit_messages.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > CLAUDE.md Content -->

# Commit Message Examples

All commits MUST follow this format:

```
[PHASE-XX] <type>: <subject>

<body>

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>
```

## Type Values

- `feat` — New feature or functionality
- `fix` — Bug fix
- `test` — Adding or updating tests
- `docs` — Documentation changes
- `refactor` — Code restructuring without behavior change
- `perf` — Performance improvement
- `chore` — Build system, dependencies, tooling

## Examples

### Feature Addition

```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-03] feat: Implement Tensor reshape and view operations

Add reshape() for copying data to new shape and view() for zero-copy
reshape when strides permit. Both validate that total elements match.

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"
```

### Bug Fix

```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-04] fix: Handle unaligned memory in SIMD dot product

AVX2 load instructions require 32-byte alignment. Added scalar fallback
for the first N elements until aligned, then SIMD for remainder.

Fixes: test_simd_alignment

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"
```

### Test Addition

```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" \
    commit -m "[PHASE-06] test: Add tokenizer edge case tests

- test_tokenizer_empty: empty string handling
- test_tokenizer_emoji: Unicode emoji roundtrip
- test_tokenizer_long: 2000 token stress test

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>"
```

## Rules

1. Subject line ≤ 72 characters
2. Body wrapped at 72 characters
3. Blank line between subject and body
4. Reference test names when fixing test failures
5. ALWAYS include Signed-off-by line
