# %% [markdown]
# # Module 00 — PyTorch Tensors and Devices
#
# **Bridge from classical ML.** You already know NumPy arrays. A PyTorch *tensor* is the same idea plus two
# superpowers that deep learning needs:
# 1. it can live on a **GPU** (CUDA on Colab, **MPS** on your Mac) for fast math, and
# 2. it can **record the operations** applied to it so gradients can be computed automatically (module 03).
#
# **Learning goals**
# - Create tensors and convert to/from NumPy
# - Understand dtypes, shapes, and broadcasting in PyTorch
# - Reshape with `view` / `reshape` / `permute` / `unsqueeze` / `squeeze`
# - Move tensors between CPU and GPU/MPS, and know the common device errors
# - Read and write `torch.einsum` for batched matrix math

# %%
#! SETUP

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt

print(torch.__version__)

# %% [markdown]
# ## 1. Creating tensors
# Almost every NumPy constructor has a PyTorch twin.

# %%
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # from Python data
z = torch.zeros(2, 3)                         # note: sizes as separate args OR a tuple both work
o = torch.ones(2, 3)
r = torch.rand(2, 3)                          # uniform [0, 1)
n = torch.randn(2, 3)                         # standard normal — the usual way to initialise weights
ar = torch.arange(0, 10, 2)
ls = torch.linspace(0, 1, 5)
print(a, a.shape, a.dtype, sep="\n")

# %% [markdown]
# ### NumPy ↔ PyTorch
# `torch.from_numpy` **shares memory** with the NumPy array (no copy). Changing one changes the other.
# `torch.tensor(np_array)` always copies.

# %%
arr = np.arange(4.0)
shared = torch.from_numpy(arr)   # shares memory
copied = torch.tensor(arr)       # independent copy
arr[0] = 100
print("shared:", shared, "\ncopied:", copied)

back = shared.numpy()            # tensor -> NumPy (only works for CPU tensors without grad tracking)
print(type(back))

# %% [markdown]
# ## 2. dtypes
# Deep learning almost always uses **float32**. NumPy defaults to float64, so arrays converted from NumPy
# are often float64 — and mixing them with float32 model weights raises errors.
#
# | Use | dtype |
# |---|---|
# | inputs, weights, activations | `torch.float32` |
# | class labels for `CrossEntropyLoss` | `torch.long` (int64) |
# | masks | `torch.bool` |
# | mixed precision (later) | `torch.float16` / `torch.bfloat16` |

# %%
x64 = torch.from_numpy(np.random.rand(3))
print(x64.dtype)                  # float64 — inherited from NumPy
x32 = x64.float()                 # same as .to(torch.float32)
labels = torch.tensor([0, 2, 1])  # Python ints -> int64 (long) by default
print(x32.dtype, labels.dtype)

# %% [markdown]
# ## 3. Indexing, broadcasting, reductions
# These follow NumPy rules exactly. The only rename: PyTorch often says `dim=` where NumPy says `axis=`
# (both are accepted for most functions).

# %%
X = torch.arange(12.0).reshape(3, 4)
print(X)
print("row 1:", X[1])
print("col 2:", X[:, 2])
print("mask >5:", X[X > 5])

# broadcasting: (3, 4) - (4,) -> the (4,) row is "stretched" across all 3 rows
col_means = X.mean(dim=0)        # dim=0 -> collapse the rows, one result per column
X_centered = X - col_means
print(X_centered)

# keepdim=True keeps the reduced dim as size 1, which makes broadcasting back easy
row_sums = X.sum(dim=1, keepdim=True)   # shape (3, 1) instead of (3,)
print(row_sums.shape, (X / row_sums).sum(dim=1))  # each row now sums to 1

# %% [markdown]
# ## 4. Shapes: the #1 source of bugs in deep learning
# Image batches in PyTorch use **(N, C, H, W)**: batch size, channels, height, width.
# matplotlib wants **(H, W, C)**. You will constantly reshape between them.
#
# | Operation | What it does |
# |---|---|
# | `reshape(...)` / `view(...)` | same data, new shape (`view` needs contiguous memory; `reshape` copies if needed) |
# | `permute(...)` | reorder dimensions (e.g. NCHW → NHWC) |
# | `unsqueeze(d)` | insert a size-1 dim at position `d` |
# | `squeeze(d)` | remove a size-1 dim |
# | `flatten(start_dim)` | merge dims from `start_dim` onward |
# | `-1` in a shape | "infer this size" |

# %%
batch = torch.rand(16, 3, 32, 32)          # 16 RGB images of 32x32
print("NCHW:", batch.shape)
print("NHWC:", batch.permute(0, 2, 3, 1).shape)
print("flat per image:", batch.flatten(start_dim=1).shape)   # (16, 3072) — what an MLP expects
print("same with view:", batch.view(16, -1).shape)

single = torch.rand(3, 32, 32)
print("add batch dim:", single.unsqueeze(0).shape)           # models always expect a batch dimension

# %% [markdown]
# `permute` returns a tensor whose memory is no longer laid out row-by-row ("non-contiguous").
# `view` refuses to work on it; `reshape` (or `.contiguous().view(...)`) works.

# %%
p = batch.permute(0, 2, 3, 1)
print(p.is_contiguous())
try:
    p.view(16, -1)
except RuntimeError as e:
    print("view failed:", str(e)[:80], "...")
print(p.reshape(16, -1).shape)

# %% [markdown]
# ## 5. Devices: CPU, CUDA, MPS
# The setup cell chose `device` for you. Rules:
# - Every tensor in an operation must be on the **same device**.
# - Move with `.to(device)`. It returns a **new** tensor for tensors (but moves a model in place).
# - To plot or convert to NumPy, bring data back with `.cpu()`.
# - MPS does **not** support float64. Keep data float32.

# %%
x = torch.randn(1000, 1000)
x_dev = x.to(device)
print(x.device, "->", x_dev.device)

y_dev = x_dev @ x_dev          # runs on the accelerator
y_cpu = y_dev.cpu()            # back to CPU for NumPy / matplotlib
print(y_cpu.device)

if device.type != "cpu":
    try:
        x + x_dev              # x is on the CPU, x_dev is on the GPU
    except RuntimeError as e:
        print("Mixing devices fails:", str(e)[:90])

# %% [markdown]
# ### Is the GPU actually faster?
# GPUs win on **large** operations. For tiny tensors, the overhead of launching work on the GPU dominates.
# GPU work is also *asynchronous*, so we must `synchronize` before stopping the timer.

# %%
import time


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_matmul(size, dev, reps=5):
    a = torch.randn(size, size, device=dev)
    a @ a; sync()                      # warm-up (first call includes setup cost)
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ a
    sync()
    return (time.perf_counter() - t0) / reps


sizes = [64, 256, 1024] if FAST_DEV_RUN else [64, 256, 1024, 2048, 4096]
cpu_t = [time_matmul(s, "cpu") for s in sizes]
dev_t = [time_matmul(s, device) for s in sizes]
for s, c, d in zip(sizes, cpu_t, dev_t):
    print(f"{s:>5}x{s:<5} cpu={c * 1e3:8.2f} ms   {device.type}={d * 1e3:8.2f} ms")

plt.figure(figsize=(5, 3.5))
plt.loglog(sizes, cpu_t, "o-", label="cpu")
plt.loglog(sizes, dev_t, "o-", label=device.type)
plt.xlabel("matrix size"); plt.ylabel("seconds per matmul"); plt.legend(); plt.title("CPU vs accelerator")
plt.show()

# %% [markdown]
# ## 6. Matrix math and `einsum`
# `@` is matrix multiply. With 3-D tensors, `@` does a **batched** matmul: (B, n, k) @ (B, k, m) → (B, n, m).
#
# `torch.einsum` describes the operation with letters: each letter is a dimension; letters that appear in the
# inputs but **not** in the output are summed over. It reads like the maths and is used heavily in attention
# (module 14).

# %%
A = torch.randn(4, 3)
B = torch.randn(3, 5)
print(torch.allclose(A @ B, torch.einsum("ik,kj->ij", A, B)))   # matrix multiply: sum over k

v = torch.randn(3)
w = torch.randn(3)
print(torch.allclose(v @ w, torch.einsum("i,i->", v, w)))         # dot product

Q = torch.randn(8, 10, 16)   # (batch, sequence, features)
K = torch.randn(8, 10, 16)
scores = torch.einsum("bqd,bkd->bqk", Q, K)                       # every query vs every key, per batch
print(scores.shape, torch.allclose(scores, Q @ K.transpose(1, 2), atol=1e-5))

# %% [markdown]
# ## 7. In-place operations
# Methods ending in `_` modify the tensor in place: `x.add_(1)`, `x.zero_()`, `x.clamp_(0, 1)`.
# They save memory but can break gradient computation (module 03). Use them only when you mean it.

# %%
t = torch.ones(3)
t.mul_(5)
print(t)

# %% [markdown]
# ## Common pitfalls
# - **float64 sneaking in from NumPy** → `RuntimeError: expected scalar type Float but found Double`. Fix with `.float()`.
# - **Device mismatch** → `Expected all tensors to be on the same device`. Move both model and data with `.to(device)`.
# - **Forgetting the batch dimension** → models expect (N, ...). Use `unsqueeze(0)` for a single example.
# - **`view` on non-contiguous tensors** → use `reshape`.
# - **Timing GPU code without synchronizing** → times look impossibly fast.
# - **`.numpy()` on a GPU tensor** → call `.cpu()` first (and `.detach()` if it tracks gradients).
#
# ## Recap
# - Tensors = NumPy arrays + devices + autograd.
# - Images are (N, C, H, W); `permute` for plotting, `flatten(1)` for MLPs.
# - Keep everything float32 and on the same device.
# - `einsum` writes tensor contractions the way you'd write the maths.
#
# **Next:** module 01 — building a neural network from scratch in NumPy.
