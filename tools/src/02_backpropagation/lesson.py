# %% [markdown]
# # Module 02 — Backpropagation
#
# Module 01 trained an MLP with numerical gradients: 2 forward passes **per parameter** per step.
# Backpropagation gets every gradient exactly with **one forward + one backward pass**, by applying the chain rule
# from the loss back towards the inputs and reusing intermediate results.
#
# **Learning goals**
# - Computational graphs and the chain rule (refresher from the ML course, module 02)
# - Local gradients for the building blocks: linear layer, ReLU, sigmoid, softmax + cross-entropy
# - Backprop through a whole MLP, written with matrices
# - Gradient checking against numerical gradients
# - Multi-class classification with softmax

# %%
#! SETUP

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
from common.plotting import plot_decision_boundary

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A computational graph by hand
# > 🎥 **Watch before this section:** [3Blue1Brown, Chapter 3: 'What is backpropagation really doing?'](https://www.3blue1brown.com/lessons/backpropagation). It gives the intuition, with no equations.
#
# Take $f(x, y, z) = (x + y)\cdot z$. Name the intermediate $q = x + y$, so $f = q z$.
#
# **Forward:** compute q, then f.
# **Backward:** start with $\partial f/\partial f = 1$ and walk backwards, multiplying by each node's *local* gradient:
# - $\partial f/\partial q = z$, $\partial f/\partial z = q$
# - $\partial f/\partial x = \partial f/\partial q \cdot \partial q/\partial x = z \cdot 1$, and the same for $y$.

# %%
x, y, z = -2.0, 5.0, -4.0
q = x + y          # forward
f = q * z
df_dq, df_dz = z, q                  # local gradients of the multiply node
df_dx, df_dy = df_dq * 1, df_dq * 1  # chain rule through the add node
print(f"f={f}, df/dx={df_dx}, df/dy={df_dy}, df/dz={df_dz}")

# %% [markdown]
# Two patterns to remember:
# - **add** passes the incoming gradient through unchanged to both inputs (a "gradient distributor");
# - **multiply** sends each input the incoming gradient times the **other** input (a "swap").
#
# ### Upstream × local
# Every node only needs to know its *local* derivative. During backward it receives the **upstream gradient**
# (∂L/∂output), multiplies by its local derivative, and passes the result to its inputs. If a value feeds into
# several places, the gradients arriving from each place are **summed**.

# %% [markdown]
# ## 2. Gradients of the building blocks (batched)
# > 🎥 **Watch:** [3Blue1Brown, Chapter 4: 'Backpropagation calculus'](https://www.3blue1brown.com/lessons/backpropagation-calculus). It covers the same chain-rule bookkeeping for a tiny network, one neuron per layer.
#
# Notation: `dX` means ∂L/∂X, always the **same shape as X**. That rule is the best way to sanity-check matrix
# gradients: if the shapes match, you've usually got the transposes right.
#
# **Linear layer** `Z = X @ W + b`, with X (N, D), W (D, H), b (H,), given upstream `dZ` (N, H):
# - `dW = X.T @ dZ` → (D, N) @ (N, H) = (D, H) ✓
# - `db = dZ.sum(axis=0)` → b was broadcast across the N rows, so its gradients from every row add up
# - `dX = dZ @ W.T` → (N, H) @ (H, D) = (N, D) ✓
#
# **ReLU** `A = max(0, Z)`: `dZ = dA * (Z > 0)`. The gradient flows only through units that were active.
#
# **Sigmoid + binary cross-entropy:** if `p = sigmoid(z)` and `L = BCE(p, y)` averaged over N, then
# `dz = (p - y) / N`. The messy terms cancel, which is why the two are always fused.
#
# **Softmax + cross-entropy** (multi-class, K classes): with `P = softmax(Z)` and one-hot labels `Y`,
# `dZ = (P - Y) / N`. It has the same form as above.

# %%
def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)    # stability trick from module 00
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(P, y, eps=1e-12):
    """P: (N, K) probabilities, y: (N,) integer labels."""
    return -np.mean(np.log(P[np.arange(len(y)), y] + eps))   # pick each row's true-class probability

# %% [markdown]
# ## 3. Backprop through an MLP
# > 🎥 **Optional, after this section:** [Karpathy, 'makemore Part 4: Becoming a Backprop Ninja'](https://karpathy.ai/zero-to-hero.html). It does manual backprop through a real network, including BatchNorm. Good practice once this lesson clicks.
#
# We cache what the forward pass computed (each layer's input `A_prev` and pre-activation `Z`), because the
# backward pass needs them. That cache is why training uses much more memory than inference.

# %%
def init_mlp(sizes, rng):
    return [(rng.normal(0, np.sqrt(2 / i), size=(i, o)), np.zeros(o)) for i, o in zip(sizes[:-1], sizes[1:])]


def forward(params, X):
    """Returns class probabilities and a cache for backward. Hidden: ReLU, output: softmax."""
    cache, A = [], X
    for i, (W, b) in enumerate(params):
        Z = A @ W + b
        cache.append((A, Z))
        A = softmax(Z) if i == len(params) - 1 else relu(Z)
    return A, cache


def backward(params, cache, P, y):
    """Gradients of mean cross-entropy w.r.t. every (W, b)."""
    N, K = P.shape
    Y = np.eye(K)[y]                    # one-hot labels, shape (N, K)
    dZ = (P - Y) / N                    # softmax + CE fused gradient
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        A_prev, Z = cache[i]
        W, _ = params[i]
        grads[i] = (A_prev.T @ dZ, dZ.sum(axis=0))   # dW, db
        if i > 0:
            dA_prev = dZ @ W.T
            _, Z_prev = cache[i - 1]
            dZ = dA_prev * (Z_prev > 0)              # through the previous layer's ReLU
    return grads

# %% [markdown]
# ## 4. Gradient checking
# **Always** verify a hand-written backward pass against numerical gradients on a tiny problem. The usual metric is
# the *relative error* `|a - b| / max(|a|, |b|)`. Below about 1e-6 is great; above 1e-3 means there's a bug.

# %%
def loss_fn(params, X, y):
    return cross_entropy(forward(params, X)[0], y)


def numerical_grads(params, X, y, eps=1e-5):
    grads = []
    for W, b in params:
        gs = []
        for arr in (W, b):
            g = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + eps; lp = loss_fn(params, X, y)
                arr[idx] = old - eps; lm = loss_fn(params, X, y)
                arr[idx] = old
                g[idx] = (lp - lm) / (2 * eps)
            gs.append(g)
        grads.append(tuple(gs))
    return grads


def rel_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


Xc, yc = rng.normal(size=(6, 4)), rng.integers(0, 3, size=6)
pc = init_mlp([4, 5, 3], rng)
P, cache = forward(pc, Xc)
analytic = backward(pc, cache, P, yc)
numeric = numerical_grads(pc, Xc, yc)
for i, ((dW, db), (nW, nb)) in enumerate(zip(analytic, numeric)):
    print(f"layer {i}: rel error W={rel_error(dW, nW):.2e}, b={rel_error(db, nb):.2e}")

# %% [markdown]
# ## 5. Train on a 3-class spiral
# Spirals can't be separated by straight lines, and three classes means we need softmax.

# %%
def make_spiral(n_per_class, k, rng, noise=0.2):
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.05, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


X_sp, y_sp = make_spiral(100, 3, rng)
plt.figure(figsize=(4, 4)); plt.scatter(*X_sp.T, c=y_sp, cmap="viridis", s=15, edgecolors="k"); plt.title("spiral"); plt.show()

# %%
params = init_mlp([2, 64, 64, 3], rng)
steps = 300 if FAST_DEV_RUN else 5000
lr = 1.0
losses = []
t0 = time.perf_counter()
for step in range(steps):
    P, cache = forward(params, X_sp)
    losses.append(cross_entropy(P, y_sp))
    grads = backward(params, cache, P, y_sp)
    params = [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]
elapsed = time.perf_counter() - t0
n_params = sum(W.size + b.size for W, b in params)
acc = np.mean(forward(params, X_sp)[0].argmax(axis=1) == y_sp)
print(f"{n_params} params, {steps} steps in {elapsed:.2f}s  -> loss {losses[-1]:.3f}, accuracy {acc:.1%}")

# %%
plt.plot(losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.show()
plot_decision_boundary(lambda G: forward(params, G)[0].argmax(axis=1), X_sp, y_sp, title="MLP on the spiral")

# %% [markdown]
# Compare with module 01: there, 337 parameters × 300 steps took seconds using numerical gradients. Here, about
# 4,500 parameters × thousands of steps take about the same time, because the backward pass costs roughly as much
# as one forward pass, however many parameters there are.
#
# ## 6. Vanishing gradients: a first look
# Each sigmoid layer multiplies the backward signal by σ'(z) ≤ 0.25. Stack many sigmoid layers and the
# gradients reaching the early layers shrink exponentially. ReLU's derivative is exactly 1 for active units, which
# is a big reason it replaced sigmoid in hidden layers. (Module 05 covers initialisation, which also matters.)

# %%
def grad_norms_per_layer(act, act_grad, depth=10, width=64):
    X = rng.normal(size=(128, width))
    Ws = [rng.normal(0, np.sqrt(1 / width), size=(width, width)) for _ in range(depth)]
    As, Zs = [X], []
    for W in Ws:
        Zs.append(As[-1] @ W); As.append(act(Zs[-1]))
    dA = rng.normal(size=As[-1].shape)                    # pretend upstream gradient
    norms = []
    for W, Z, A_prev in zip(reversed(Ws), reversed(Zs), reversed(As[:-1])):
        dZ = dA * act_grad(Z)
        norms.append(np.linalg.norm(A_prev.T @ dZ))
        dA = dZ @ W.T
    return norms[::-1]                                    # layer 1 first


sig_norms = grad_norms_per_layer(sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)))
relu_norms = grad_norms_per_layer(relu, lambda z: (z > 0).astype(float))
plt.semilogy(range(1, 11), sig_norms, "o-", label="sigmoid")
plt.semilogy(range(1, 11), relu_norms, "o-", label="ReLU")
plt.xlabel("layer (1 = closest to input)"); plt.ylabel("‖dW‖"); plt.legend(); plt.title("gradient size per layer")
plt.show()

# %% [markdown]
# ## Common pitfalls
# - **Forgetting `/ N`**: the loss is a mean, so the gradient must be too. Without it, your learning rate
#   silently depends on the batch size.
# - **Wrong transpose**: check that every gradient has the same shape as its parameter.
# - **Not summing `db` over the batch**: the bias was broadcast, so its gradient is a sum over rows.
# - **Using the ReLU mask from the wrong layer**: the mask comes from that layer's own `Z`.
# - **Skipping gradient checks**: a buggy backward pass often still lowers the loss, just badly.
#
# ## Recap
# - Backprop = the chain rule applied from the loss backwards, reusing cached forward values.
# - Linear layer: `dW = Xᵀ dZ`, `db = Σ dZ`, `dX = dZ Wᵀ`. ReLU: mask. Softmax + CE: `(P − Y)/N`.
# - Gradient checking catches bugs; relative error below 1e-6 is good.
# - **Next (module 03):** instead of deriving every backward by hand, build an engine that does it
#   automatically, which is exactly what PyTorch's autograd does.
