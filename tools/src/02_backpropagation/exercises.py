# %% [markdown]
# # Module 02 — Exercises: Backpropagation
# Each backward function takes the upstream gradient and returns gradients with the **same shapes** as the inputs.
# The checks compare against numerical gradients.

# %%
#! SETUP

# %%
import numpy as np

rng = np.random.default_rng(2)


def num_grad(f, x, eps=1e-6):
    """Numerical gradient of scalar f() w.r.t. array x (modified in place, then restored)."""
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps; fp = f()
        x[idx] = old - eps; fm = f()
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g

# %% [markdown]
# ### Exercise 1 — Scalar graph by hand
# For $f(a, b, c) = (a \cdot b + c)^2$ at a=2, b=-3, c=4, compute the forward value and all three partial
# derivatives by hand with the chain rule (write the intermediate values as code).

# %%
a, b, c = 2.0, -3.0, 4.0
### BEGIN SOLUTION
#> f = None      # TODO
#> df_da = None  # TODO
#> df_db = None  # TODO
#> df_dc = None  # TODO
m = a * b            # -6
s = m + c            # -2
f = s ** 2           # 4
df_ds = 2 * s        # -4
df_dc = df_ds * 1    # add node passes it through
df_dm = df_ds * 1
df_da = df_dm * b    # multiply node: swap
df_db = df_dm * a
### END SOLUTION

# %%
assert f == 4.0
assert (df_da, df_db, df_dc) == (12.0, -8.0, -4.0)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — Linear layer backward
# Forward: `Z = X @ W + b`. Given upstream `dZ`, return `(dX, dW, db)`.

# %%
def linear_backward(dZ, X, W):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return dZ @ W.T, X.T @ dZ, dZ.sum(axis=0)
    ### END SOLUTION


# %%
X = rng.normal(size=(5, 3)); W = rng.normal(size=(3, 4)); b = rng.normal(size=4)
R = rng.normal(size=(5, 4))                   # random projection -> scalar loss L = sum(Z * R), so dZ = R
L = lambda: np.sum((X @ W + b) * R)
dX, dW, db = linear_backward(R, X, W)
assert np.allclose(dX, num_grad(L, X), atol=1e-5)
assert np.allclose(dW, num_grad(L, W), atol=1e-5)
assert np.allclose(db, num_grad(L, b), atol=1e-5)
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — Activation backward passes
# Given the pre-activation `Z` and upstream `dA`, return `dZ` for ReLU, sigmoid and tanh.
# Derivatives: σ'(z) = σ(z)(1−σ(z)), tanh'(z) = 1 − tanh²(z).

# %%
def relu_backward(dA, Z):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return dA * (Z > 0)
    ### END SOLUTION


def sigmoid_backward(dA, Z):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    s = 1 / (1 + np.exp(-Z))
    return dA * s * (1 - s)
    ### END SOLUTION


def tanh_backward(dA, Z):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return dA * (1 - np.tanh(Z) ** 2)
    ### END SOLUTION


# %%
Z = rng.normal(size=(4, 6)); R = rng.normal(size=(4, 6))
for fwd, bwd in [(lambda z: np.maximum(0, z), relu_backward),
                 (lambda z: 1 / (1 + np.exp(-z)), sigmoid_backward),
                 (np.tanh, tanh_backward)]:
    assert np.allclose(bwd(R, Z), num_grad(lambda: np.sum(fwd(Z) * R), Z), atol=1e-5), bwd.__name__
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Softmax + cross-entropy
# Implement `softmax_ce(Z, y)` that returns `(loss, dZ)`, where loss is the **mean** cross-entropy over the batch and
# `dZ` its gradient. Use the stable softmax.

# %%
def softmax_ce(Z, y):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    N = len(y)
    Zs = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Zs); P /= P.sum(axis=1, keepdims=True)
    loss = -np.mean(np.log(P[np.arange(N), y]))
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1          # P - one_hot(y), without building the one-hot matrix
    return loss, dZ / N
    ### END SOLUTION


# %%
Z = rng.normal(size=(7, 4)); y = rng.integers(0, 4, size=7)
loss, dZ = softmax_ce(Z, y)
import torch
ref = torch.nn.functional.cross_entropy(torch.tensor(Z), torch.tensor(y))
assert np.isclose(loss, ref.item())
assert np.allclose(dZ, num_grad(lambda: softmax_ce(Z, y)[0], Z), atol=1e-6)
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — A full 2-layer network (tanh hidden layer)
# Network: `H = tanh(X @ W1 + b1)`, `Z = H @ W2 + b2`, loss = softmax-CE(Z, y).
# Return the loss and a dict of gradients with keys `W1, b1, W2, b2`. Reuse your functions above.

# %%
def two_layer_loss_and_grads(p, X, y):
    ### BEGIN SOLUTION
    #> # TODO: forward (keep Z1 and H), then backward in reverse order
    #> raise NotImplementedError
    Z1 = X @ p["W1"] + p["b1"]
    H = np.tanh(Z1)
    Z2 = H @ p["W2"] + p["b2"]
    loss, dZ2 = softmax_ce(Z2, y)
    dH, dW2, db2 = linear_backward(dZ2, H, p["W2"])
    dZ1 = tanh_backward(dH, Z1)
    _, dW1, db1 = linear_backward(dZ1, X, p["W1"])
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    ### END SOLUTION


# %%
p = {"W1": rng.normal(size=(3, 5)), "b1": rng.normal(size=5), "W2": rng.normal(size=(5, 4)), "b2": rng.normal(size=4)}
X = rng.normal(size=(8, 3)); y = rng.integers(0, 4, size=8)
loss, g = two_layer_loss_and_grads(p, X, y)
for k in p:
    num = num_grad(lambda: two_layer_loss_and_grads(p, X, y)[0], p[k])
    assert g[k].shape == p[k].shape, k
    assert np.allclose(g[k], num, atol=1e-5), k
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Train it
# Use your function to train on the XOR-like "checkerboard" data below with plain gradient descent.
# Reach **≥ 95% training accuracy**.

# %%
Xq = rng.uniform(-1, 1, size=(400, 2))
yq = ((Xq[:, 0] > 0) ^ (Xq[:, 1] > 0)).astype(int)   # XOR of the signs: label by quadrant

p = {"W1": rng.normal(0, 1, size=(2, 32)), "b1": np.zeros(32),
     "W2": rng.normal(0, 0.1, size=(32, 2)), "b2": np.zeros(2)}
### BEGIN SOLUTION
#> # TODO: loop: compute loss & grads, update every p[k] -= lr * g[k]
lr = 0.5
for step in range(3000):
    loss, g = two_layer_loss_and_grads(p, Xq, yq)
    for k in p:
        p[k] -= lr * g[k]
### END SOLUTION

# %%
pred = (np.tanh(Xq @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]).argmax(axis=1)
acc = np.mean(pred == yq)
print(f"accuracy {acc:.1%}")
assert acc >= 0.95
print("Exercise 6 passed ✅")
