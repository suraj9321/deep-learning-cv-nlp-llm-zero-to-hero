# %% [markdown]
# # Module 01 — Exercises: Perceptrons and MLPs
# Fill in each `# TODO`, then run the check cell below it.

# %%
#! SETUP

# %%
import numpy as np

rng = np.random.default_rng(1)

# %% [markdown]
# ### Exercise 1 — Activations
# Implement `leaky_relu(z, alpha=0.01)`: `z` where z > 0, else `alpha * z`.
# And `softplus(z) = log(1 + exp(z))`, a smooth version of ReLU. Make it numerically safe for large z:
# use `np.logaddexp(0, z)`, which computes `log(e^0 + e^z)` without overflow.

# %%
def leaky_relu(z, alpha=0.01):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return np.where(z > 0, z, alpha * z)
    ### END SOLUTION


def softplus(z):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return np.logaddexp(0, z)
    ### END SOLUTION


# %%
z = np.array([-2.0, -0.5, 0.0, 3.0])
assert np.allclose(leaky_relu(z), [-0.02, -0.005, 0.0, 3.0])
assert np.allclose(leaky_relu(z, 0.2), [-0.4, -0.1, 0.0, 3.0])
assert np.allclose(softplus(np.array([0.0])), np.log(2))
assert np.isfinite(softplus(np.array([1000.0]))).all() and np.isclose(softplus(np.array([1000.0]))[0], 1000.0)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — A dense layer
# `dense(X, W, b, activation)` should return `activation(X @ W + b)`. `activation` can be `None` (no activation).
# X: (N, in), W: (in, out), b: (out,).

# %%
def dense(X, W, b, activation=None):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    z = X @ W + b
    return z if activation is None else activation(z)
    ### END SOLUTION


# %%
X = rng.normal(size=(5, 3)); W = rng.normal(size=(3, 4)); b = rng.normal(size=4)
assert dense(X, W, b).shape == (5, 4)
assert np.allclose(dense(X, W, b), X @ W + b)
assert np.allclose(dense(X, W, b, np.tanh), np.tanh(X @ W + b))
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — Count parameters
# For an MLP with layer sizes like `[784, 128, 64, 10]`, return the total number of weights + biases.
# (Do it with a formula; don't build the arrays.)

# %%
def count_params(sizes):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))
    ### END SOLUTION


# %%
assert count_params([2, 1]) == 3
assert count_params([784, 128, 64, 10]) == 784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10
print("[784,128,64,10] has", count_params([784, 128, 64, 10]), "parameters")
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Solve XOR by hand with ReLU
# Choose `W1` (2, 2), `b1` (2,), `W2` (2,), `b2` (scalar) so that
# `relu(X @ W1 + b1) @ W2 + b2` gives exactly **[0, 1, 1, 0]** for the four XOR inputs.
#
# Hint: hidden unit 1 = `relu(x1 + x2)`, hidden unit 2 = `relu(x1 + x2 - 1)`. Look at their values on the four
# inputs and find a combination that equals XOR.

# %%
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

### BEGIN SOLUTION
#> W1 = None  # TODO
#> b1 = None  # TODO
#> W2 = None  # TODO
#> b2 = None  # TODO
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
W2 = np.array([1.0, -2.0])   # h1 - 2*h2: (0,1,1,2) - 2*(0,0,0,1) = (0,1,1,0)
b2 = 0.0
### END SOLUTION

# %%
out = np.maximum(0, X_xor @ W1 + b1) @ W2 + b2
assert np.allclose(out, [0, 1, 1, 0]), out
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — Generic MLP forward pass
# `params` is a list of `(W, b)`. Apply `hidden_act` after every layer **except** the last, and `out_act`
# after the last (either can be `None`). Return the final output.

# %%
def mlp_forward(params, X, hidden_act=lambda z: np.maximum(0, z), out_act=None):
    ### BEGIN SOLUTION
    #> # TODO: loop over layers; reuse `dense` from exercise 2
    #> raise NotImplementedError
    a = X
    for i, (W, b) in enumerate(params):
        act = out_act if i == len(params) - 1 else hidden_act
        a = dense(a, W, b, act)
    return a
    ### END SOLUTION


# %%
p = [(rng.normal(size=(3, 5)), rng.normal(size=5)), (rng.normal(size=(5, 2)), rng.normal(size=2))]
Xt = rng.normal(size=(4, 3))
expected = np.maximum(0, Xt @ p[0][0] + p[0][1]) @ p[1][0] + p[1][1]
assert np.allclose(mlp_forward(p, Xt), expected)
assert np.allclose(mlp_forward(p, Xt, out_act=np.tanh), np.tanh(expected))
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Numerical gradient of a simple function
# Write `numerical_grad(f, x, eps=1e-5)` for a function `f` that takes a 1-D array and returns a scalar.
# Return an array of the same shape as `x` using central differences. Don't modify `x` permanently.

# %%
def numerical_grad(f, x, eps=1e-5):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    x = x.astype(float).copy()
    g = np.zeros_like(x)
    for i in range(x.size):
        old = x[i]
        x[i] = old + eps; fp = f(x)
        x[i] = old - eps; fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g
    ### END SOLUTION


# %%
f = lambda v: np.sum(v ** 2) + 3 * v[0]          # gradient: 2v + [3, 0, 0]
x0 = np.array([1.0, -2.0, 0.5])
assert np.allclose(numerical_grad(f, x0), 2 * x0 + np.array([3.0, 0, 0]), atol=1e-6)
assert np.allclose(x0, [1.0, -2.0, 0.5]), "x was modified!"
print("Exercise 6 passed ✅")
