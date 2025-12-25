<!-- File: docs/references/test_specs/phase-05-autograd.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPEC FILE TEMPLATES -->

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
