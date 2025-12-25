# Phase 05: Autograd Engine - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 10

## Required Tests

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

## Implementation Notes

- Use numerical gradient checking with epsilon = 1e-5
- Tolerance for gradient comparison: 1e-4 (relative) or 1e-6 (absolute)
- Test gradient accumulation explicitly (don't zero_grad between backward calls)
