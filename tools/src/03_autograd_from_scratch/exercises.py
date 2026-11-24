# %% [markdown]
# # Module 03 — Exercises: Autograd
# Exercises 1–3 extend a small `Value` engine. Exercises 4–6 use PyTorch autograd.

# %%
#! SETUP

# %%
import math
import torch


class Value:
    """Minimal engine from the lesson (add, mul, pow only). You'll add more operations."""
    def __init__(self, data, _prev=()):
        self.data, self.grad, self._prev, self._backward = float(data), 0.0, set(_prev), lambda: None

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def _b():
            self.grad += out.grad; o.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def _b():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = _b
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _b():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _b
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * (o if isinstance(o, Value) else Value(o)) ** -1

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# %% [markdown]
# ### Exercise 1 — Add `exp` and `log`
# Implement `value_exp(v)` and `value_log(v)` returning new `Value`s with correct `_backward` closures.
# d/dx eˣ = eˣ, d/dx ln x = 1/x.

# %%
def value_exp(v: Value) -> Value:
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    out = Value(math.exp(v.data), (v,))
    def _b():
        v.grad += out.data * out.grad
    out._backward = _b
    return out
    ### END SOLUTION


def value_log(v: Value) -> Value:
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    out = Value(math.log(v.data), (v,))
    def _b():
        v.grad += (1 / v.data) * out.grad
    out._backward = _b
    return out
    ### END SOLUTION


# %%
x = Value(0.7)
y = value_log(value_exp(x) * x)        # = x + ln x  -> dy/dx = 1 + 1/x
y.backward()
assert math.isclose(y.data, 0.7 + math.log(0.7))
assert math.isclose(x.grad, 1 + 1 / 0.7)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — Sigmoid via composition (no new backward needed)
# Write `value_sigmoid(v)` using only operations that already have backward rules: `1 / (1 + exp(-v))`.
# Autograd then gives the gradient for free.

# %%
def value_sigmoid(v: Value) -> Value:
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return Value(1.0) / (value_exp(-v) + 1)
    ### END SOLUTION


# %%
x = Value(0.4)
s = value_sigmoid(x); s.backward()
sig = 1 / (1 + math.exp(-0.4))
assert math.isclose(s.data, sig) and math.isclose(x.grad, sig * (1 - sig))
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — Binary cross-entropy with your engine
# Logistic regression on two features: `p = sigmoid(w1*x1 + w2*x2 + b)`.
# Compute the BCE for **one** example, `-(y*log(p) + (1-y)*log(1-p))`, then call `backward()`.
# Check that `w1.grad == (p - y) * x1` (the formula from module 02).

# %%
w1, w2, b = Value(0.5), Value(-1.0), Value(0.1)
x1, x2, y_true = 2.0, 1.0, 1.0
### BEGIN SOLUTION
#> p = None     # TODO
#> loss = None  # TODO
p = value_sigmoid(w1 * x1 + w2 * x2 + b)
loss = -(value_log(p) * y_true + value_log(1 + (-p)) * (1 - y_true))
### END SOLUTION
loss.backward()

# %%
assert math.isclose(w1.grad, (p.data - y_true) * x1, rel_tol=1e-9)
assert math.isclose(b.grad, (p.data - y_true), rel_tol=1e-9)
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Gradients with PyTorch
# For $f(x) = \sum_i \sin(x_i)\,x_i^2$, use autograd to return ∂f/∂x for a given 1-D tensor (it should **not** need
# `requires_grad` set by the caller). Compare with the analytic answer `cos(x)·x² + 2x·sin(x)`.

# %%
def grad_f(x: torch.Tensor) -> torch.Tensor:
    ### BEGIN SOLUTION
    #> # TODO: clone x, enable grad, compute f, backward, return the gradient
    #> raise NotImplementedError
    x = x.detach().clone().requires_grad_(True)
    f = (torch.sin(x) * x ** 2).sum()
    f.backward()
    return x.grad
    ### END SOLUTION


# %%
x = torch.linspace(-2, 2, 7)
assert torch.allclose(grad_f(x), torch.cos(x) * x ** 2 + 2 * x * torch.sin(x), atol=1e-5)
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — Fit a quadratic with autograd
# Fit `y ≈ a·x² + b·x + c` by gradient descent with raw tensors (no `nn` and no `optim`). Remember to update inside
# `torch.no_grad()` and to reset the grads. Reach an MSE below 0.02.

# %%
torch.manual_seed(0)
xs = torch.linspace(-2, 2, 100)
ys = 1.5 * xs ** 2 - 0.8 * xs + 0.3 + 0.1 * torch.randn(100)
params = torch.zeros(3, requires_grad=True)     # [a, b, c]
### BEGIN SOLUTION
#> # TODO: training loop
for _ in range(2000):
    pred = params[0] * xs ** 2 + params[1] * xs + params[2]
    loss = ((pred - ys) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        params -= 0.05 * params.grad
    params.grad = None
### END SOLUTION

# %%
with torch.no_grad():
    mse = ((params[0] * xs ** 2 + params[1] * xs + params[2] - ys) ** 2).mean().item()
print("params:", params.detach().numpy().round(3), "mse:", round(mse, 4))
assert mse < 0.02
assert torch.allclose(params.detach(), torch.tensor([1.5, -0.8, 0.3]), atol=0.1)
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — A custom autograd Function: straight-through rounding
# `round()` has zero gradient almost everywhere, which would block learning. The *straight-through estimator* uses
# rounding in the forward pass but pretends it was the identity in the backward pass. (Quantized networks and the
# VQ-VAE use this trick.) Implement `forward` (round) and `backward` (pass the gradient through unchanged).

# %%
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        return torch.round(x)
        ### END SOLUTION

    @staticmethod
    def backward(ctx, grad_out):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        return grad_out
        ### END SOLUTION


# %%
x = torch.tensor([0.2, 1.7, -2.6], requires_grad=True)
y = RoundSTE.apply(x)
(y * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
assert torch.equal(y.detach(), torch.tensor([0.0, 2.0, -3.0]))
assert torch.equal(x.grad, torch.tensor([1.0, 2.0, 3.0]))
print("Exercise 6 passed ✅")
