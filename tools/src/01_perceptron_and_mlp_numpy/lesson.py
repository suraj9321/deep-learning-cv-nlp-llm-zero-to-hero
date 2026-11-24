# %% [markdown]
# # Module 01 — From Perceptron to Multi-Layer Perceptron (NumPy)
#
# In the classical ML course, logistic regression was `sigmoid(w·x + b)`. That *is* a neural network with one
# neuron. This module stacks neurons into **layers** and shows why the stacking (plus a non-linearity) lets a
# network learn shapes that linear models can't.
#
# **Learning goals**
# - A neuron = weighted sum + bias + activation
# - Common activations (sigmoid, tanh, ReLU) and why we need them
# - The perceptron learning rule, and why it fails on XOR
# - Forward pass of an MLP written with matrices (batched)
# - Training by numerical gradients, and why that's too slow (motivation for backprop in module 02)

# %%
#! SETUP

# %%
import numpy as np
import matplotlib.pyplot as plt
from common.plotting import plot_decision_boundary

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. A single neuron
# > 🎥 **Watch before this section:** [3Blue1Brown, Chapter 1: 'But what is a neural network?'](https://www.3blue1brown.com/lessons/gradient-descent) (Chapter 1 is linked from that page). It's the visual version of sections 1–4: neurons, layers and weights.
#
# $$ z = \mathbf{w}\cdot\mathbf{x} + b, \qquad a = \sigma(z) $$
# `z` is the "pre-activation"; `σ` is an **activation function**. With a batch of inputs `X` (N, D), we compute all
# neurons at once: `Z = X @ W + b`, with `W` shaped (D, units).

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

z = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, (name, f) in zip(axes, [("sigmoid", sigmoid), ("tanh", tanh), ("ReLU", relu)]):
    ax.plot(z, f(z)); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5); ax.set_title(name)
plt.tight_layout(); plt.show()

# %% [markdown]
# | Activation | Range | Notes |
# |---|---|---|
# | sigmoid | (0, 1) | outputs a probability; saturates (flat) for large \|z\| → tiny gradients |
# | tanh | (-1, 1) | zero-centred sigmoid; still saturates |
# | ReLU | [0, ∞) | cheap, doesn't saturate for z>0; **the default for hidden layers** |

# %% [markdown]
# ## 2. The perceptron (1958)
# Output `1 if w·x + b > 0 else 0`. Learning rule: for every misclassified point, nudge `w` toward it
# (if it should be 1) or away from it (if it should be 0):
# $$ \mathbf{w} \leftarrow \mathbf{w} + \eta\,(y - \hat y)\,\mathbf{x}, \qquad b \leftarrow b + \eta\,(y-\hat y) $$
# It is guaranteed to converge **if** the classes can be separated by a straight line.

# %%
def train_perceptron(X, y, lr=0.1, epochs=50):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            y_hat = 1 if xi @ w + b > 0 else 0
            update = lr * (yi - y_hat)   # 0 when correct, ±lr when wrong
            w += update * xi
            b += update
            errors += int(update != 0)
        if errors == 0:
            break
    return w, b

# AND gate: linearly separable
X_logic = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1])
y_xor = np.array([0, 1, 1, 0])

for name, y in [("AND", y_and), ("XOR", y_xor)]:
    w, b = train_perceptron(X_logic, y)
    preds = (X_logic @ w + b > 0).astype(int)
    print(f"{name}: predictions={preds}, target={y}, accuracy={np.mean(preds == y):.2f}")

# %% [markdown]
# The perceptron learns AND but **can't** learn XOR: no single straight line separates {(0,1),(1,0)} from
# {(0,0),(1,1)}. This result (Minsky & Papert, 1969) stalled neural-network research for years.
# The fix is a **hidden layer**.

# %% [markdown]
# ## 3. A hidden layer solves XOR
# XOR = (x1 OR x2) AND NOT (x1 AND x2). Two hidden neurons compute OR and AND; the output neuron combines them.
# Here are hand-picked weights (with a steep sigmoid to act like a step):

# %%
W1 = np.array([[20.0, 20.0],     # column 0 -> "OR" neuron, column 1 -> "AND" neuron
               [20.0, 20.0]])
b1 = np.array([-10.0, -30.0])
W2 = np.array([[20.0], [-20.0]])  # OR and not AND
b2 = np.array([-10.0])

H = sigmoid(X_logic @ W1 + b1)    # (4, 2) hidden activations
out = sigmoid(H @ W2 + b2)        # (4, 1)
print(np.round(H, 2)); print("XOR output:", np.round(out.ravel(), 3))

# %% [markdown]
# The hidden layer **re-represents** the input so that the problem becomes linearly separable for the next layer.
# That is the core idea of deep learning: layers learn useful representations.
#
# ### Why the non-linearity matters
# Without an activation, two layers collapse into one: `(X @ W1) @ W2 = X @ (W1 @ W2)`, still linear.

# %%
W1r, W2r = rng.normal(size=(2, 8)), rng.normal(size=(8, 1))
print(np.allclose((X_logic @ W1r) @ W2r, X_logic @ (W1r @ W2r)))  # True: stacking linear layers = one linear layer

# %% [markdown]
# ## 4. An MLP forward pass (general, batched)
# For layer sizes `[D, h1, h2, ..., out]` we store a list of `(W, b)`. Hidden layers use ReLU, the output
# layer uses sigmoid (binary classification).
#
# **Weight init:** start with small random weights scaled by `sqrt(2 / fan_in)` ("He init", good for ReLU).
# All-zero weights would make every neuron in a layer identical forever (module 05 covers this).

# %%
def init_mlp(sizes, rng):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0, np.sqrt(2 / fan_in), size=(fan_in, fan_out))
        b = np.zeros(fan_out)
        params.append((W, b))
    return params


def mlp_forward(params, X):
    a = X
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        a = sigmoid(z) if i == len(params) - 1 else relu(z)  # last layer: sigmoid; others: ReLU
    return a.ravel()   # (N,) probabilities


params = init_mlp([2, 16, 16, 1], rng)
print([(W.shape, b.shape) for W, b in params])
print("param count:", sum(W.size + b.size for W, b in params))

# %% [markdown]
# ## 5. A harder dataset: two moons
# We'll make the classic "moons" data by hand (sklearn's `make_moons` does the same thing).

# %%
def make_moons(n, noise, rng):
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, n1)
    t2 = rng.uniform(0, np.pi, n - n1)
    upper = np.c_[np.cos(t1), np.sin(t1)]
    lower = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([upper, lower]) + rng.normal(0, noise, size=(n, 2))
    y = np.r_[np.zeros(n1), np.ones(n - n1)]
    return X, y

X_moons, y_moons = make_moons(200, 0.15, rng)
plt.figure(figsize=(5, 4)); plt.scatter(*X_moons.T, c=y_moons, cmap="RdBu", edgecolors="k", s=20)
plt.title("two moons"); plt.show()

# %% [markdown]
# ## 6. Loss: binary cross-entropy
# Same loss as logistic regression:
# $$ L = -\frac{1}{N}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] $$
# We clip `p` away from 0 and 1 so `log` never sees 0.

# %%
def bce(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print("loss of untrained net:", bce(mlp_forward(params, X_moons), y_moons))

# %% [markdown]
# ## 7. Training with *numerical* gradients
# > 🎥 **Watch:** [3Blue1Brown, Chapter 2: 'Gradient descent, how neural networks learn'](https://www.3blue1brown.com/lessons/gradient-descent). It shows what 'walking downhill on the loss' means for a network with thousands of weights.
#
# Gradient descent needs ∂L/∂θ for every parameter θ. The simplest way: wiggle each parameter by a tiny ε and
# measure how the loss changes (central difference):
# $$ \frac{\partial L}{\partial \theta} \approx \frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon} $$
# That works, but needs **2 forward passes per parameter, per step**.

# %%
def numerical_grads(params, X, y, eps=1e-5):
    grads = []
    for W, b in params:
        gW, gb = np.zeros_like(W), np.zeros_like(b)
        for arr, g in [(W, gW), (b, gb)]:
            for idx in np.ndindex(arr.shape):   # every single entry
                old = arr[idx]
                arr[idx] = old + eps; lp = bce(mlp_forward(params, X), y)
                arr[idx] = old - eps; lm = bce(mlp_forward(params, X), y)
                arr[idx] = old                  # restore
                g[idx] = (lp - lm) / (2 * eps)
        grads.append((gW, gb))
    return grads


import time
params = init_mlp([2, 16, 16, 1], rng)
steps = 30 if FAST_DEV_RUN else 300
lr = 0.5
losses = []
t0 = time.perf_counter()
for step in range(steps):
    grads = numerical_grads(params, X_moons, y_moons)
    for (W, b), (gW, gb) in zip(params, grads):
        W -= lr * gW      # in-place update so `params` sees the change
        b -= lr * gb
    losses.append(bce(mlp_forward(params, X_moons), y_moons))
elapsed = time.perf_counter() - t0
n_params = sum(W.size + b.size for W, b in params)
print(f"{steps} steps, {n_params} params, {elapsed:.1f}s  -> {2 * n_params * steps:,} forward passes")

acc = np.mean((mlp_forward(params, X_moons) > 0.5) == y_moons)
print(f"final loss {losses[-1]:.3f}, accuracy {acc:.2%}")

# %%
plt.plot(losses); plt.xlabel("step"); plt.ylabel("BCE loss"); plt.title("training with numerical gradients"); plt.show()
plot_decision_boundary(lambda G: mlp_forward(params, G), X_moons, y_moons, title="MLP [2,16,16,1] on moons")

# %% [markdown]
# It learns a curved boundary that a linear model never could. But the cost: a network with **N** parameters needs
# **2N** forward passes per step. A small image model has millions of parameters, so this is hopeless.
#
# **Backpropagation** (module 02) computes *all* gradients exactly with one forward and one backward pass.
# We'll keep `numerical_grads` around as a way to **check** backprop ("gradient checking").

# %% [markdown]
# ## Common pitfalls
# - **No activation between layers** → the whole network is linear.
# - **Zero initialisation** → all neurons in a layer stay identical (symmetry never breaks).
# - **`log(0)` in the loss** → clip probabilities or (in PyTorch) use the logits-based losses.
# - **Shape confusion** → with `X @ W`, `W` is (in_features, out_features). (PyTorch's `nn.Linear` stores the
#   transpose, (out, in); you'll see that in module 04.)
#
# ## Recap
# - Neuron = linear function + non-linearity. Logistic regression is a 1-neuron network.
# - One layer can only draw straight boundaries; hidden layers build new features and bend the boundary.
# - The forward pass is a chain of `a = act(a @ W + b)`.
# - We can train with numerical gradients, but it scales terribly → backprop is next.
