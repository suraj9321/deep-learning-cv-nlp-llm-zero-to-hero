# %% [markdown]
# # Module 03 — Autograd: Build It, Then Use PyTorch's
#
# In module 02 you derived a backward pass by hand for one specific network. That doesn't scale: every new
# architecture would need new calculus. **Automatic differentiation** (autograd) records operations as they run and
# applies the chain rule for you. Each operation only has to know its own local derivative.
#
# We'll build a tiny scalar autograd engine (in the spirit of Andrej Karpathy's *micrograd*), train a network with
# it, and then see that `torch.autograd` does exactly the same thing on tensors.
#
# **Learning goals**
# - How a computation graph is recorded during the forward pass
# - Topological ordering, and why gradients are *accumulated* (`+=`)
# - `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`, `.detach()`
# - Why you must zero gradients every step
# - Writing a custom `torch.autograd.Function`

# %%
#! SETUP

# %%
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

random.seed(0)

# %% [markdown]
# ## 1. A `Value` that remembers how it was made
# > 🎥 **Watch alongside this lesson:** [Karpathy, 'The spelled-out intro to neural networks and backpropagation: building micrograd'](https://karpathy.ai/zero-to-hero.html). He builds this exact engine live. Pause the video and compare it with the code below.
#
# Each `Value` stores:
# - `data`: the number;
# - `grad`: ∂L/∂(this value), filled in by `backward()`;
# - `_prev`: the Values it was computed from;
# - `_backward`: a closure that pushes this node's gradient to its parents (upstream × local).

# %%
class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None     # leaf nodes have nothing to propagate

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # add passes the gradient through; += because a value may be used in several places
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad    # multiply "swaps" the inputs
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):                       # k is a plain number
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Order the graph so every node comes after everything it depends on (a topological sort)...
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        # ...then walk it in reverse, so a node's grad is complete before it is pushed to its parents.
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # conveniences so that expressions like 2 * x, x - y and x / y work
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return (-self) + other
    def __truediv__(self, other): return self * other ** -1

# %% [markdown]
# ### Check against module 02
# For $f = (x + y)\cdot z$ we found df/dx = df/dy = −4 and df/dz = 3.

# %%
x, y, z = Value(-2.0), Value(5.0), Value(-4.0)
f = (x + y) * z
f.backward()
print(f, x, y, z, sep="\n")

# %% [markdown]
# ### Why `+=`? A value used twice
# In `b = a * a`, `a` feeds the multiply **twice**. Its gradient is the sum of both paths: 2a. With `=` instead of
# `+=`, the second path would overwrite the first.

# %%
a = Value(3.0)
b = a * a
b.backward()
print("d(a²)/da at a=3:", a.grad)   # 6

# %% [markdown]
# ## 2. A neural network built from `Value`s
# The same layer and MLP idea as module 01, but every weight is a `Value`, so the gradients come for free.

# %%
class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i + 1], nonlin=i != len(n_outs) - 1) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# %% [markdown]
# Train on a tiny dataset with a hinge-style target of ±1 (tanh range) and MSE loss.
# The loop has the three steps that every PyTorch training loop also has:
# **zero the grads → forward + loss → backward → update**.

# %%
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
model = MLP(3, [4, 4, 1])
print("parameters:", len(model.parameters()))

losses = []
for step in range(40 if FAST_DEV_RUN else 100):
    preds = [model(x) for x in xs]
    loss = sum((p - t) ** 2 for p, t in zip(preds, ys))
    for p in model.parameters():
        p.grad = 0.0                 # zero the grads, otherwise they'd keep accumulating across steps
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad      # gradient descent
    losses.append(loss.data)

print("predictions:", [round(model(x).data, 3) for x in xs], "targets:", ys)
plt.plot(losses); plt.xlabel("step"); plt.ylabel("MSE"); plt.title("training with our autograd engine"); plt.show()

# %% [markdown]
# That's the whole idea of autograd. PyTorch does the same thing, but on **tensors** (so one node is a whole matrix
# multiply rather than a single scalar multiply) and in fast C++/GPU code.
#
# ## 3. PyTorch autograd
# - Set `requires_grad=True` on tensors you want gradients for (model parameters do this automatically).
# - Every operation on them records a node; you can see it in `.grad_fn`.
# - `loss.backward()` fills in `.grad` on the leaf tensors.

# %%
x = torch.tensor(-2.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
z = torch.tensor(-4.0, requires_grad=True)
f = (x + y) * z
print(f, f.grad_fn)               # MulBackward0: the node that will run backward
f.backward()
print(x.grad, y.grad, z.grad)     # same numbers as our Value engine

# %% [markdown]
# ### Our engine vs PyTorch on a bigger expression

# %%
def expr(lib_a, lib_b):
    c = lib_a * lib_b + lib_b ** 3
    d = (c * 2 + 1).tanh() if isinstance(c, Value) else torch.tanh(c * 2 + 1)
    return d * lib_a - lib_b / 4

va, vb = Value(0.3), Value(-0.7)
out_v = expr(va, vb); out_v.backward()
ta, tb = torch.tensor(0.3, dtype=torch.float64, requires_grad=True), torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
out_t = expr(ta, tb); out_t.backward()
print(f"ours : value={out_v.data:.6f} da={va.grad:.6f} db={vb.grad:.6f}")
print(f"torch: value={out_t.item():.6f} da={ta.grad.item():.6f} db={tb.grad.item():.6f}")

# %% [markdown]
# ## 4. Gradients accumulate in PyTorch too
# Calling `backward()` twice **adds** to `.grad`. That's why every training loop calls `optimizer.zero_grad()`
# (or `p.grad = None`). Forgetting it is one of the most common bugs.

# %%
w = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    loss = w ** 2
    loss.backward()
    print(f"call {i + 1}: w.grad = {w.grad.item()}")    # 4, 8, 12: accumulating
w.grad = None                                            # reset

# %% [markdown]
# ## 5. Turning tracking off: `no_grad` and `detach`
# - `with torch.no_grad():` → nothing inside is recorded. Use it for evaluation and for parameter updates.
# - `t.detach()` → the same data, cut out of the graph. Use it to log values, convert to NumPy, or stop gradients
#   flowing into part of a model (you'll do this for GANs in module 17).

# %%
w = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = w * 2
print("inside no_grad, requires_grad:", y.requires_grad)
print("detached:", (w * 2).detach().requires_grad)
print("numpy:", (w * 2).detach().cpu().numpy())

# %% [markdown]
# ## 6. Gradient descent with raw PyTorch tensors
# Before `nn.Module` and `optim` (module 04), here is linear regression with autograd and a manual update.

# %%
torch.manual_seed(0)
X = torch.randn(200, 3, device=device)
true_w = torch.tensor([2.0, -3.0, 0.5], device=device)
y = X @ true_w + 1.0 + 0.1 * torch.randn(200, device=device)

w = torch.zeros(3, device=device, requires_grad=True)
b = torch.zeros(1, device=device, requires_grad=True)
lr = 0.1
for step in range(200):
    loss = ((X @ w + b - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():          # the update itself must not be recorded in the graph
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad = None; b.grad = None   # zero for the next step
print("learned w:", w.detach().cpu().numpy().round(3), "b:", round(b.item(), 3), "loss:", round(loss.item(), 4))

# %% [markdown]
# ## 7. Custom autograd functions
# Occasionally you need to define your own backward rule (for numerical stability, or a non-differentiable trick
# such as the "straight-through estimator"). Subclass `torch.autograd.Function`:

# %%
class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(s)          # stash what backward needs (like our cache in module 02)
        return s

    @staticmethod
    def backward(ctx, grad_out):
        (s,) = ctx.saved_tensors
        return grad_out * s * (1 - s)     # upstream × local


x = torch.randn(5, dtype=torch.float64, requires_grad=True)
print("matches torch.sigmoid:", torch.allclose(MySigmoid.apply(x), torch.sigmoid(x)))
# gradcheck compares our backward with numerical gradients (it needs float64 for precision)
print("gradcheck:", torch.autograd.gradcheck(MySigmoid.apply, (x,)))

# %% [markdown]
# ## Common pitfalls
# - **Not zeroing gradients** → they accumulate across steps.
# - **Updating parameters outside `no_grad()`** → PyTorch either errors ("a leaf Variable that requires grad is
#   being used in an in-place operation") or records the update into the graph.
# - **Calling `.numpy()` on a tensor that requires grad** → `.detach()` first.
# - **`backward()` on a non-scalar** → it needs a scalar loss (or an explicit `gradient=` argument).
# - **Calling `backward()` twice on the same graph** → by default the graph is freed after the first call.
#
# ## Recap
# - Autograd records a graph during the forward pass, then runs each node's local backward in reverse topological order.
# - Gradients accumulate (`+=`), both inside the graph (shared values) and across `backward()` calls.
# - PyTorch: `requires_grad`, `.backward()`, `.grad`, `no_grad()`, `detach()`, custom `Function`s.
# - **Next (module 04):** `nn.Module`, `optim` and `DataLoader`, the standard PyTorch training loop.
