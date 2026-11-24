# %% [markdown]
# # Module 00 — Exercises: Tensors and Devices
# Fill in each `# TODO`, then run the check cell below it. If the asserts pass, you're done with that exercise.
# Solutions are in `solutions.ipynb` — try first!

# %%
#! SETUP

# %%
import numpy as np
import torch

# %% [markdown]
# ### Exercise 1 — NumPy to a float32 tensor on the device
# Write `to_device_tensor(arr)` that converts a NumPy array (any float dtype) into a **float32** tensor on `device`.

# %%
def to_device_tensor(arr: np.ndarray) -> torch.Tensor:
    ### BEGIN SOLUTION
    #> # TODO: convert to a float32 tensor and move it to `device`
    #> raise NotImplementedError
    return torch.from_numpy(arr).float().to(device)
    ### END SOLUTION


# %%
t = to_device_tensor(np.random.rand(3, 4))
assert t.dtype == torch.float32, "dtype should be float32"
assert t.device.type == device.type, "tensor should be on the selected device"
assert t.shape == (3, 4)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — Standardize features (column-wise)
# Given `X` of shape (N, D), return `(X - mean) / std` computed **per column** (use `unbiased=False` / `correction=0`
# for the std). Use broadcasting, no loops.

# %%
def standardize(X: torch.Tensor) -> torch.Tensor:
    ### BEGIN SOLUTION
    #> # TODO: column-wise standardization with broadcasting
    #> raise NotImplementedError
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, correction=0, keepdim=True)
    return (X - mean) / std
    ### END SOLUTION


# %%
X = torch.randn(100, 5) * torch.tensor([1.0, 2, 3, 4, 5]) + 10
Z = standardize(X)
assert torch.allclose(Z.mean(dim=0), torch.zeros(5), atol=1e-5)
assert torch.allclose(Z.std(dim=0, correction=0), torch.ones(5), atol=1e-5)
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — Image batch shapes
# `imgs` has shape (N, C, H, W). Return a tuple:
# 1. the batch in (N, H, W, C) layout (for plotting),
# 2. the batch flattened to (N, C*H*W) (for an MLP),
# 3. the **first** image with its batch dimension kept, shape (1, C, H, W).

# %%
def reshape_batch(imgs: torch.Tensor):
    ### BEGIN SOLUTION
    #> # TODO: return (nhwc, flat, first_with_batch_dim)
    #> raise NotImplementedError
    nhwc = imgs.permute(0, 2, 3, 1)
    flat = imgs.flatten(start_dim=1)
    first = imgs[0].unsqueeze(0)      # or imgs[0:1]
    return nhwc, flat, first
    ### END SOLUTION


# %%
imgs = torch.rand(8, 3, 28, 28)
nhwc, flat, first = reshape_batch(imgs)
assert nhwc.shape == (8, 28, 28, 3)
assert torch.equal(nhwc[2, 5, 7], imgs[2, :, 5, 7])
assert flat.shape == (8, 3 * 28 * 28)
assert first.shape == (1, 3, 28, 28) and torch.equal(first[0], imgs[0])
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Pairwise squared distances (no loops)
# For `A` (n, d) and `B` (m, d), return the (n, m) matrix `D[i, j] = ||A[i] - B[j]||²`.
# Hint: insert dimensions so that `A` becomes (n, 1, d) and `B` becomes (1, m, d), then broadcast.
# (You used this idea for KNN in the classical ML course.)

# %%
def pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ### BEGIN SOLUTION
    #> # TODO: broadcast (n,1,d) - (1,m,d) -> (n,m,d), square, sum over d
    #> raise NotImplementedError
    diff = A.unsqueeze(1) - B.unsqueeze(0)
    return (diff ** 2).sum(dim=-1)
    ### END SOLUTION


# %%
A, B = torch.randn(6, 4), torch.randn(9, 4)
D = pairwise_sq_dists(A, B)
assert D.shape == (6, 9)
assert torch.allclose(D, torch.cdist(A, B) ** 2, atol=1e-4)
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — einsum
# Use `torch.einsum` (one call each) to compute:
# 1. `batched_matmul(X, Y)`: X (B, n, k), Y (B, k, m) → (B, n, m)
# 2. `row_dots(X, Y)`: X, Y both (n, d) → (n,) where out[i] = X[i] · Y[i]

# %%
def batched_matmul(X, Y):
    ### BEGIN SOLUTION
    #> # TODO: one einsum call
    #> raise NotImplementedError
    return torch.einsum("bnk,bkm->bnm", X, Y)
    ### END SOLUTION


def row_dots(X, Y):
    ### BEGIN SOLUTION
    #> # TODO: one einsum call
    #> raise NotImplementedError
    return torch.einsum("nd,nd->n", X, Y)
    ### END SOLUTION


# %%
X, Y = torch.randn(4, 3, 5), torch.randn(4, 5, 2)
assert torch.allclose(batched_matmul(X, Y), X @ Y, atol=1e-5)
P, Q = torch.randn(7, 3), torch.randn(7, 3)
assert torch.allclose(row_dots(P, Q), (P * Q).sum(dim=1), atol=1e-5)
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Softmax, numerically stable
# Implement `softmax(logits)` over the **last** dimension. Subtract the per-row max before `exp` so that huge
# logits don't overflow to `inf` (the result is mathematically identical). You'll meet softmax constantly from
# module 04 on.

# %%
def softmax(logits: torch.Tensor) -> torch.Tensor:
    ### BEGIN SOLUTION
    #> # TODO: stable softmax over dim=-1
    #> raise NotImplementedError
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    e = shifted.exp()
    return e / e.sum(dim=-1, keepdim=True)
    ### END SOLUTION


# %%
L = torch.randn(5, 10) * 3
assert torch.allclose(softmax(L), torch.softmax(L, dim=-1), atol=1e-6)
big = torch.tensor([[1000.0, 1001.0, 1002.0]])
assert torch.isfinite(softmax(big)).all(), "overflow! did you subtract the max?"
print("Exercise 6 passed ✅")
