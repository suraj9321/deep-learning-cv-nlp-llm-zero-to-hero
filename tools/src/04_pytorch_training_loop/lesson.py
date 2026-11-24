# %% [markdown]
# # Module 04 — The PyTorch Training Loop
#
# Modules 01–03 built everything by hand. PyTorch packages the same pieces:
#
# | Hand-made (modules 01–03) | PyTorch |
# |---|---|
# | list of `(W, b)` arrays | `nn.Module` with `nn.Linear` layers |
# | softmax + CE gradient `(P−Y)/N` | `nn.CrossEntropyLoss` + autograd |
# | `W -= lr * dW` | `torch.optim.SGD` / `Adam` |
# | slicing arrays into batches | `Dataset` + `DataLoader` |
#
# The goal of this module is to write the **training loop** yourself until it's second nature. Every model
# in this course, up to diffusion, is trained with a variant of it.
#
# **Learning goals**
# - `nn.Module`, `nn.Linear`, `nn.Sequential`, `parameters()`, `state_dict()`
# - `Dataset` / `DataLoader`, torchvision datasets and transforms
# - The training and evaluation loops, `model.train()` / `model.eval()`
# - Reaching about 98% accuracy on MNIST with an MLP
# - Saving and loading models

# %%
#! SETUP

# %%
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from common.plotting import show_image_grid, plot_loss_curves

# %% [markdown]
# ## 1. `nn.Linear` and `nn.Module`
# `nn.Linear(in, out)` holds a weight of shape **(out, in)** and a bias of shape (out,), and computes
# `x @ W.T + b`. (In NumPy we stored W as (in, out). PyTorch stores the transpose.)

# %%
layer = nn.Linear(4, 3)
print(layer.weight.shape, layer.bias.shape, layer.weight.requires_grad)
x = torch.randn(2, 4)
print(torch.allclose(layer(x), x @ layer.weight.T + layer.bias))

# %% [markdown]
# A model is a subclass of `nn.Module`. Any layer assigned as an attribute in `__init__` is **registered**:
# its parameters show up in `model.parameters()`, move with `model.to(device)`, and are saved in `state_dict()`.
# You only write `forward`; autograd handles backward.

# %%
class MLP(nn.Module):
    def __init__(self, in_features=784, hidden=(256, 128), n_classes=10):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))   # no softmax: the loss takes raw "logits"
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)                  # (N, 1, 28, 28) -> (N, 784)
        return self.net(x)


model = MLP()
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters")

# %% [markdown]
# ### Logits and `CrossEntropyLoss`
# `nn.CrossEntropyLoss` expects **raw scores (logits)** of shape (N, K) and **integer labels** (N,). Internally it
# applies log-softmax and picks out the true class. This is the fused softmax + CE from module 02, done in a
# numerically stable way. **Don't** put a softmax layer before it.

# %%
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])
labels = torch.tensor([0, 2])
loss_fn = nn.CrossEntropyLoss()
manual = -torch.log_softmax(logits, dim=1)[torch.arange(2), labels].mean()
print(loss_fn(logits, labels).item(), manual.item())

# %% [markdown]
# ## 2. Datasets and DataLoaders
# A `Dataset` only needs `__len__` and `__getitem__(i) -> (x, y)`. A `DataLoader` wraps it to give shuffled
# **mini-batches**, so each step uses a batch (typically 32–256 examples) instead of the whole dataset.
# This is *stochastic* gradient descent: noisier steps, but many more of them per second.

# %%
class TensorPairs(Dataset):
    """The simplest possible Dataset: wraps two tensors (torch.utils.data.TensorDataset does the same)."""
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


toy = TensorPairs(torch.arange(10).float().unsqueeze(1), torch.arange(10))
for xb, yb in DataLoader(toy, batch_size=4, shuffle=True):
    print(xb.squeeze(1).tolist(), yb.tolist())   # the last batch is smaller unless drop_last=True

# %% [markdown]
# ### MNIST with torchvision
# `transforms.ToTensor()` turns a PIL image into a float tensor of shape (1, 28, 28) with values in [0, 1].
# `Normalize(mean, std)` then standardizes it using the training set's statistics (as with the scaler in
# the ML course).

# %%
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST's well-known mean and std
])
train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

# hold out 5k training images for validation; the test set is used only once, at the very end
train_ds, val_ds = random_split(train_full, [55_000, 5_000], generator=torch.Generator().manual_seed(0))

if FAST_DEV_RUN:   # tiny subsets so automated checks finish quickly
    train_ds, val_ds, test_ds = Subset(train_ds, range(2000)), Subset(val_ds, range(500)), Subset(test_ds, range(500))

print(len(train_ds), len(val_ds), len(test_ds))
x0, y0 = train_full[0]
print(x0.shape, x0.dtype, y0)

# %%
# num_workers=0 is the safe choice inside notebooks on macOS; MNIST is small enough that it doesn't matter
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)

xb, yb = next(iter(train_loader))
print("batch:", xb.shape, yb.shape)
show_image_grid(xb[:16] * 0.3081 + 0.1307, labels=yb[:16].tolist())   # undo Normalize for display

# %% [markdown]
# ## 3. The training loop
# > 🎥 **Watch after this module:** [Karpathy, 'makemore Part 2: MLP'](https://karpathy.ai/zero-to-hero.html). It's a full PyTorch training loop with mini-batches, train/val/test splits and learning-rate tuning, on a language task.
#
# Memorize this shape:
# ```
# for epoch in range(epochs):
#     model.train()
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()            # 1. clear old gradients
#         loss = loss_fn(model(xb), yb)    # 2. forward + loss
#         loss.backward()                  # 3. backward
#         optimizer.step()                 # 4. update
#     evaluate on validation data with model.eval() and torch.no_grad()
# ```
# `model.train()` / `model.eval()` don't train or evaluate anything. They switch the behaviour of layers such as
# dropout and batch norm (module 06). Get into the habit now.

# %%
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)       # .item() copies one number to the CPU; fine once per batch
        n += len(xb)
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(xb)
    return total / n, correct / n

# %%
model = MLP().to(device)                    # move the parameters to the GPU/MPS *before* making the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # Adam: a robust default (details in module 05)

epochs = 1 if FAST_DEV_RUN else 10
history = {"train_loss": [], "val_loss": [], "val_acc": []}
t0 = time.perf_counter()
for epoch in range(epochs):
    tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
    va_loss, va_acc = evaluate(model, val_loader, loss_fn)
    history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
    print(f"epoch {epoch + 1:>2}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}")
print(f"training took {time.perf_counter() - t0:.1f}s on {device}")

# %%
plot_loss_curves(history, "MLP on MNIST")

# %% [markdown]
# Notice that the validation loss stops improving (and may start rising) while the training loss keeps falling.
# That's **overfitting**, the bias–variance story from the ML course. Module 06 is about fighting it.
#
# ### Final test evaluation (only once, after all decisions are made)

# %%
test_loss, test_acc = evaluate(model, test_loader, loss_fn)
print(f"test accuracy: {test_acc:.4f}")

# %% [markdown]
# ### Looking at mistakes
# The predicted class is `argmax(logits)`. `softmax(logits)` gives probabilities if you want a confidence score.

# %%
@torch.no_grad()
def collect_mistakes(model, loader, max_n=16):
    model.eval()
    imgs, info = [], []
    for xb, yb in loader:
        probs = torch.softmax(model(xb.to(device)), dim=1).cpu()
        pred = probs.argmax(dim=1)
        wrong = (pred != yb).nonzero().flatten()
        for i in wrong:
            imgs.append(xb[i]); info.append(f"{yb[i].item()}→{pred[i].item()} ({probs[i, pred[i]]:.2f})")
            if len(imgs) == max_n:
                return torch.stack(imgs), info
    return (torch.stack(imgs), info) if imgs else (None, [])


mist, info = collect_mistakes(model, test_loader)
if mist is not None:
    show_image_grid(mist * 0.3081 + 0.1307, labels=info, title="true→pred (confidence)")

# %% [markdown]
# ## 4. Saving and loading
# Save the **`state_dict`** (a dict of parameter tensors), not the whole model object. To load, rebuild the same
# architecture and then load the weights into it.

# %%
ckpt_path = ROOT / "checkpoints" / "mnist_mlp.pt"
torch.save(model.state_dict(), ckpt_path)

restored = MLP().to(device)
restored.load_state_dict(torch.load(ckpt_path, map_location=device))
print("restored test acc:", evaluate(restored, test_loader, loss_fn)[1])
print(list(model.state_dict().keys())[:4])

# %% [markdown]
# ## 5. The reusable helpers
# From module 05 on, notebooks import `train_one_epoch`, `evaluate` and `fit` from `common/training.py`. They are the
# same functions you just wrote, with a `max_batches` option for quick runs. Open the file and read it; there's
# nothing new in it.

# %%
from common.training import fit
m2 = MLP().to(device)
h2 = fit(m2, train_loader, val_loader, loss_fn, torch.optim.Adam(m2.parameters(), 1e-3), device, epochs=1)

# %% [markdown]
# ## Common pitfalls
# - **Softmax before `CrossEntropyLoss`** → the loss applies log-softmax itself. Feed it logits.
# - **Labels as float or one-hot** → `CrossEntropyLoss` wants `long` class indices of shape (N,).
# - **Creating the optimizer before `model.to(device)`** → it usually still works, but get used to moving first.
# - **Forgetting `model.eval()` / `torch.no_grad()` in evaluation** → wrong results with dropout or batch norm, and
#   wasted memory.
# - **Tuning on the test set** → use a validation split; touch the test set once.
# - **`num_workers > 0` in a notebook on macOS** → can hang or crash. Keep it at 0 unless you run a script.
#
# ## Recap
# - `nn.Module` registers parameters; you write `forward`.
# - `DataLoader` gives shuffled mini-batches.
# - The loop: `zero_grad → forward → loss → backward → step`, with train and eval modes.
# - An MLP reaches about 98% on MNIST. It ignores the fact that the input is an image; CNNs (module 08) don't.
# - **Next (module 05):** what the optimizer is actually doing, and how to choose learning rates.
