# %% [markdown]
# # Module 04 — Exercises: The Training Loop (FashionMNIST)
# FashionMNIST has the same format as MNIST (28×28 grayscale images, 10 classes) but shows clothing, and it is
# harder. Write every piece of the pipeline yourself.

# %%
#! SETUP

# %%
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

# %% [markdown]
# ### Exercise 1 — A custom Dataset
# Write `NoisyRegression`, a `Dataset` that generates `n` points with `x ~ U(-3, 3)` and `y = sin(x) + noise`
# (noise std 0.1), **once** in `__init__`. Each item should be `(x, y)` where both are float tensors of shape (1,).

# %%
class NoisyRegression(Dataset):
    def __init__(self, n, seed=0):
        g = torch.Generator().manual_seed(seed)
        ### BEGIN SOLUTION
        #> # TODO: create self.x and self.y
        #> raise NotImplementedError
        self.x = torch.rand(n, 1, generator=g) * 6 - 3
        self.y = torch.sin(self.x) + 0.1 * torch.randn(n, 1, generator=g)
        ### END SOLUTION

    def __len__(self):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        return len(self.x)
        ### END SOLUTION

    def __getitem__(self, i):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        return self.x[i], self.y[i]
        ### END SOLUTION


# %%
ds = NoisyRegression(300)
xb, yb = next(iter(DataLoader(ds, batch_size=32)))
assert len(ds) == 300 and xb.shape == (32, 1) and yb.shape == (32, 1) and xb.dtype == torch.float32
assert xb.min() >= -3 and xb.max() <= 3
assert torch.equal(NoisyRegression(300)[5][0], ds[5][0]), "data should be generated once, deterministically"
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — Fit it with an MLP (a regression loop)
# Build an `nn.Sequential` with one hidden layer of 64 units and `Tanh`, and train it with `nn.MSELoss` and
# `torch.optim.Adam(lr=1e-2)` for 200 epochs over a `DataLoader(ds, batch_size=32, shuffle=True)`.
# Reach a final MSE on the whole dataset below 0.02.

# %%
torch.manual_seed(0)
### BEGIN SOLUTION
#> reg_model = None  # TODO
#> # TODO: training loop
reg_model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(reg_model.parameters(), lr=1e-2)
mse = nn.MSELoss()
loader = DataLoader(ds, batch_size=32, shuffle=True)
for epoch in range(200):
    for xb, yb in loader:
        opt.zero_grad()
        loss = mse(reg_model(xb), yb)
        loss.backward()
        opt.step()
### END SOLUTION

# %%
with torch.no_grad():
    final = nn.functional.mse_loss(reg_model(ds.x), ds.y).item()
print("final MSE:", round(final, 4))
assert final < 0.02
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — FashionMNIST loaders
# Create `train_loader`, `val_loader`, `test_loader`:
# - transform: `ToTensor()` then `Normalize((0.2860,), (0.3530,))`;
# - split the 60k training images into 54k train / 6k validation with a seeded generator;
# - batch size 128 with shuffling for train; 512 without shuffling for val/test.
# (`FAST_DEV_RUN` shrinks the datasets afterwards; that code is already written.)

# %%
### BEGIN SOLUTION
#> # TODO: train_ds, val_ds, test_ds
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
test_ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
train_ds, val_ds = random_split(full, [54_000, 6_000], generator=torch.Generator().manual_seed(0))
### END SOLUTION
if FAST_DEV_RUN:
    train_ds, val_ds, test_ds = Subset(train_ds, range(3000)), Subset(val_ds, range(500)), Subset(test_ds, range(500))
### BEGIN SOLUTION
#> # TODO: the three DataLoaders
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)
test_loader = DataLoader(test_ds, batch_size=512)
### END SOLUTION

# %%
xb, yb = next(iter(train_loader))
assert xb.shape == (128, 1, 28, 28) and yb.dtype == torch.long
assert abs(xb.mean().item()) < 0.2, "did you normalize?"
if not FAST_DEV_RUN:
    assert len(train_loader.dataset) == 54_000 and len(val_loader.dataset) == 6_000
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — The model
# Write `FashionMLP(nn.Module)`: flatten → Linear(784, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 10).

# %%
class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        return self.net(x)
        ### END SOLUTION


# %%
m = FashionMLP()
assert m(torch.randn(3, 1, 28, 28)).shape == (3, 10)
assert sum(p.numel() for p in m.parameters()) == 784 * 512 + 512 + 512 * 256 + 256 + 256 * 10 + 10
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — Train and evaluate loops
# Write `train_epoch(model, loader, opt)` (returns the mean training loss) and `accuracy(model, loader)` (returns
# the fraction correct). Use `nn.functional.cross_entropy`. Remember `train()`/`eval()`, `no_grad`, and the device.

# %%
def train_epoch(model, loader, opt):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    return total / n
    ### END SOLUTION


@torch.no_grad()
def accuracy(model, loader):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(dim=1).cpu()
        correct += (pred == yb).sum().item(); n += len(yb)
    return correct / n
    ### END SOLUTION


# %% [markdown]
# Now train `FashionMLP` with Adam (lr 1e-3) for 8 epochs, printing the validation accuracy each epoch.
# Target: **≥ 87% test accuracy** (88–89% is typical for an MLP after 8 epochs).

# %%
torch.manual_seed(0)
### BEGIN SOLUTION
#> model = None  # TODO: create, move to device, train
model = FashionMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1 if FAST_DEV_RUN else 8):
    tl = train_epoch(model, train_loader, opt)
    print(f"epoch {epoch + 1}: train_loss={tl:.4f} val_acc={accuracy(model, val_loader):.4f}")
### END SOLUTION

# %%
test_acc = accuracy(model, test_loader)
print("test accuracy:", round(test_acc, 4))
assert test_acc >= (0.6 if FAST_DEV_RUN else 0.87)
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Save, load and predict
# Save `model`'s state_dict to `checkpoints/fashion_mlp.pt`, load it into a **new** `FashionMLP`, and write
# `predict(model, imgs)` that returns a list of class **names** for a batch of images.

# %%
def predict(model, imgs):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    model.eval()
    with torch.no_grad():
        idx = model(imgs.to(device)).argmax(dim=1).cpu()
    return [CLASSES[i] for i in idx]
    ### END SOLUTION


path = ROOT / "checkpoints" / "fashion_mlp.pt"
### BEGIN SOLUTION
#> loaded = None  # TODO
torch.save(model.state_dict(), path)
loaded = FashionMLP().to(device)
loaded.load_state_dict(torch.load(path, map_location=device))
### END SOLUTION

# %%
imgs, labels = next(iter(test_loader))
names = predict(loaded, imgs[:8])
assert names == predict(model, imgs[:8]) and all(n in CLASSES for n in names)
assert loaded is not model
print(list(zip(names, [CLASSES[i] for i in labels[:8]])))
print("Exercise 6 passed ✅")
