<!-- File: docs/references/examples/decision_entry.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > PHASE 0: PROJECT BOOTSTRAP -->

# Example Decision Entry

Add entries like this to `docs/architecture/DECISIONS.md`:

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
