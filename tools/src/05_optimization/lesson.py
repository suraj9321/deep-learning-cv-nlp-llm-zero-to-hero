# %% [markdown]
# # Module 05 — Optimization
#
# `optimizer.step()` has been a black box so far. This module opens it, and covers the other settings that decide
# whether training works at all: the learning rate, its schedule, and how weights are initialised.
#
# **Learning goals**
# - SGD, momentum, RMSProp and Adam: the update rules, built from scratch and checked against `torch.optim`
# - Picking a learning rate (too small, too big, the LR range test)
# - Learning-rate schedules: step decay, cosine, warmup
# - Weight initialisation (Xavier/He) and its effect on activations and gradients
# - Gradient clipping

# %%
#! SETUP

# %%
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from common.training import train_one_epoch, evaluate

# %% [markdown]
# ## 1. A hard landscape: a long, narrow valley
# Loss surfaces of real networks have directions that are very steep and others that are almost flat. A stretched
# quadratic $f(x, y) = 0.5x^2 + 10y^2$ captures the problem: a learning rate that is safe for the steep `y`
# direction is painfully slow for the flat `x` direction.

# %%
def f(p):          # p: tensor of shape (2,)
    return 0.5 * p[0] ** 2 + 10 * p[1] ** 2


def run(opt_fn, steps=60, start=(-8.0, 2.0)):
    """Run a from-scratch optimizer step function and record the path."""
    p = torch.tensor(start, requires_grad=True)
    state, path = {}, [p.detach().clone()]
    for t in range(1, steps + 1):
        loss = f(p)
        loss.backward()
        with torch.no_grad():
            opt_fn(p, p.grad, state, t)
        p.grad = None
        path.append(p.detach().clone())
    return torch.stack(path)


def plot_paths(paths: dict, title=""):
    xs, ys = np.meshgrid(np.linspace(-9, 9, 200), np.linspace(-3, 3, 200))
    zs = 0.5 * xs ** 2 + 10 * ys ** 2
    plt.figure(figsize=(9, 3.5))
    plt.contour(xs, ys, zs, levels=30, cmap="Greys", linewidths=0.5)
    for name, path in paths.items():
        plt.plot(path[:, 0], path[:, 1], "o-", ms=2, label=name)
    plt.plot(0, 0, "r*", ms=12); plt.legend(); plt.title(title); plt.show()

# %% [markdown]
# ## 2. SGD
# $$ \theta \leftarrow \theta - \eta\, g $$

# %%
def sgd(lr):
    def step(p, g, state, t):
        p -= lr * g
    return step


plot_paths({f"SGD lr={lr}": run(sgd(lr)) for lr in [0.02, 0.09, 0.1]}, "plain SGD")

# %% [markdown]
# With lr=0.02 progress along `x` is slow. With lr=0.09 the path zig-zags across the valley. At lr=0.1 it can no
# longer settle in `y`, because the steep direction's curvature (20) times the lr reaches 2. Above that it diverges.
# **The steepest direction caps the learning rate for all directions.**
#
# ## 3. Momentum
# > 🎥 **Watch:** [Stanford CS231n (2017), Lecture 7: 'Training Neural Networks II'](https://cs231n.stanford.edu/2017/syllabus.html), the optimizer part. It animates SGD, momentum, RMSProp and Adam on loss surfaces like the one above.
#
# Keep a running *velocity* that accumulates past gradients. Zig-zags cancel out; consistent directions build speed.
# $$ v \leftarrow \beta v + g, \qquad \theta \leftarrow \theta - \eta v \qquad (\beta \approx 0.9) $$
# (This is PyTorch's formulation; some texts scale `g` by `(1−β)`.)

# %%
def momentum(lr, beta=0.9):
    def step(p, g, state, t):
        v = state.get("v", torch.zeros_like(p))
        v = beta * v + g
        state["v"] = v
        p -= lr * v
    return step


plot_paths({"SGD lr=0.02": run(sgd(0.02)), "momentum lr=0.02": run(momentum(0.02))}, "momentum")

# %% [markdown]
# ## 4. RMSProp and Adam: a step size per parameter
# **RMSProp** divides each parameter's step by a running average of its squared gradients. Steep directions
# (big gradients) take smaller steps; flat directions take bigger ones.
# $$ s \leftarrow \rho s + (1-\rho) g^2, \qquad \theta \leftarrow \theta - \eta\, \frac{g}{\sqrt{s}+\epsilon} $$
#
# **Adam** = momentum (a running mean of g, "first moment" m) + RMSProp (a running mean of g², "second moment" v),
# with a **bias correction**, because m and v start at 0 and would be too small in the first steps:
# $$ m \leftarrow \beta_1 m + (1-\beta_1) g, \quad v \leftarrow \beta_2 v + (1-\beta_2) g^2 $$
# $$ \hat m = \frac{m}{1-\beta_1^t}, \quad \hat v = \frac{v}{1-\beta_2^t}, \quad \theta \leftarrow \theta - \eta \frac{\hat m}{\sqrt{\hat v}+\epsilon} $$

# %%
def rmsprop(lr, rho=0.99, eps=1e-8):
    def step(p, g, state, t):
        s = state.get("s", torch.zeros_like(p))
        s = rho * s + (1 - rho) * g ** 2
        state["s"] = s
        p -= lr * g / (s.sqrt() + eps)
    return step


def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def step(p, g, state, t):
        m = b1 * state.get("m", torch.zeros_like(p)) + (1 - b1) * g
        v = b2 * state.get("v", torch.zeros_like(p)) + (1 - b2) * g ** 2
        state["m"], state["v"] = m, v
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        p -= lr * m_hat / (v_hat.sqrt() + eps)
    return step


plot_paths({"momentum lr=0.02": run(momentum(0.02)), "RMSProp lr=0.3": run(rmsprop(0.3)),
            "Adam lr=0.5": run(adam(0.5))}, "adaptive methods")

# %% [markdown]
# ### Do our versions match `torch.optim`?
# Run each from-scratch rule and its PyTorch twin from the same start, and compare the final parameters.

# %%
def run_torch(opt_cls, steps=60, **kw):
    p = torch.tensor([-8.0, 2.0], requires_grad=True)
    opt = opt_cls([p], **kw)
    for _ in range(steps):
        opt.zero_grad(); f(p).backward(); opt.step()
    return p.detach()


checks = [
    ("SGD", sgd(0.05), torch.optim.SGD, dict(lr=0.05)),
    ("momentum", momentum(0.02), torch.optim.SGD, dict(lr=0.02, momentum=0.9)),
    ("RMSProp", rmsprop(0.3), torch.optim.RMSprop, dict(lr=0.3, alpha=0.99, eps=1e-8)),
    ("Adam", adam(0.5), torch.optim.Adam, dict(lr=0.5)),
]
for name, ours, cls, kw in checks:
    print(f"{name:9s} match: {torch.allclose(run(ours)[-1], run_torch(cls, **kw), atol=1e-4)}")

# %% [markdown]
# ## 5. Comparing optimizers on a real network
# FashionMNIST MLP, same initial weights, three optimizers. Adam is the usual default because it's forgiving about
# the learning rate. Well-tuned SGD + momentum is still common for CNNs (it often generalises slightly better).

# %%
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
train_ds = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
val_ds = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
if FAST_DEV_RUN:
    train_ds, val_ds = Subset(train_ds, range(2000)), Subset(val_ds, range(500))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)


def make_mlp():
    torch.manual_seed(0)    # identical initial weights for a fair comparison
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)


loss_fn = nn.CrossEntropyLoss()
epochs = 1 if FAST_DEV_RUN else 5
setups = {
    "SGD lr=0.05": lambda ps: torch.optim.SGD(ps, lr=0.05),
    "SGD+momentum lr=0.05": lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9),
    "Adam lr=1e-3": lambda ps: torch.optim.Adam(ps, lr=1e-3),
}
results = {}
for name, make_opt in setups.items():
    model = make_mlp()
    opt = make_opt(model.parameters())
    accs = []
    for _ in range(epochs):
        train_one_epoch(model, train_loader, loss_fn, opt, device)
        accs.append(evaluate(model, val_loader, loss_fn, device)[1])
    results[name] = accs
    print(f"{name:22s} val acc per epoch: {[round(a, 3) for a in accs]}")

# %% [markdown]
# ## 6. Choosing the learning rate: the LR range test
# The single most important hyperparameter. A quick heuristic (Leslie Smith's "LR finder"): train for a few hundred
# steps while increasing the LR exponentially from 1e-6 to 1, and record the loss. Pick an LR about 10× below the
# point where the loss is lowest (the curve's steepest descent region).

# %%
def lr_range_test(make_model, loader, lr_min=1e-6, lr_max=1.0, steps=150):
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    gamma = (lr_max / lr_min) ** (1 / steps)      # multiply the LR by this every step
    lrs, losses, it = [], [], iter(loader)
    best, smooth = float("inf"), None
    for i in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        smooth = loss.item() if smooth is None else 0.9 * smooth + 0.1 * loss.item()  # exponential smoothing
        lrs.append(opt.param_groups[0]["lr"]); losses.append(smooth)
        best = min(best, smooth)
        if smooth > 4 * best:          # stop once the loss explodes
            break
        for g in opt.param_groups:
            g["lr"] *= gamma
    return lrs, losses


lrs, losses = lr_range_test(make_mlp, train_loader, steps=60 if FAST_DEV_RUN else 150)
plt.semilogx(lrs, losses); plt.xlabel("learning rate"); plt.ylabel("smoothed loss"); plt.title("LR range test (SGD+momentum)")
plt.show()
print("LR with the lowest loss:", f"{lrs[int(np.argmin(losses))]:.3g}")

# %% [markdown]
# ## 7. Learning-rate schedules
# A large LR early makes fast progress; a small LR late settles into a good minimum. Schedulers change
# `optimizer.param_groups[i]["lr"]` for you. Call `scheduler.step()` once per epoch (or once per batch for
# per-step schedules such as OneCycle).
#
# **Warmup:** starting at full LR can blow up the first steps (especially with Adam and large models), so
# ramp it up linearly over the first few hundred steps. Transformers and diffusion models (modules 14, 19) rely on it.

# %%
def schedule_curve(make_sched, steps=100):
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.1)
    sched = make_sched(opt)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sched.step()
    return lrs


warmup_cosine = lambda opt: torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, 100) / 100)))
curves = {
    "StepLR (÷10 every 30)": lambda o: torch.optim.lr_scheduler.StepLR(o, step_size=30, gamma=0.1),
    "Cosine": lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=100),
    "Warmup + cosine": warmup_cosine,
    "OneCycle": lambda o: torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=100),
}
plt.figure(figsize=(7, 3.5))
for name, mk in curves.items():
    plt.plot(schedule_curve(mk), label=name)
plt.xlabel("step"); plt.ylabel("lr"); plt.legend(); plt.title("LR schedules"); plt.show()

# %% [markdown]
# ## 8. Weight initialisation
# > 🎥 **Watch:** [Karpathy, 'makemore Part 3: Activations & Gradients, BatchNorm'](https://karpathy.ai/zero-to-hero.html), the first half. He diagnoses a badly initialised network by plotting its activation statistics, the same experiment as below.
#
# A forward pass multiplies by a weight matrix at every layer. If the weights are a bit too big, the activations
# grow exponentially with depth; a bit too small, and they shrink to 0. Both kill learning, and the gradients
# behave the same way on the way back.
#
# - **Xavier/Glorot** (for tanh/sigmoid): `std = sqrt(2 / (fan_in + fan_out))`
# - **He/Kaiming** (for ReLU): `std = sqrt(2 / fan_in)`. ReLU zeroes half the inputs, so the variance needs
#   doubling to compensate.
#
# PyTorch's `nn.Linear` default is a uniform Kaiming variant that's reasonable for most cases. Let's see what goes
# wrong without care:

# %%
def activation_stds(init_std_fn, act=torch.relu, depth=20, width=512):
    x = torch.randn(1000, width)
    stds = []
    for _ in range(depth):
        W = torch.randn(width, width) * init_std_fn(width)
        x = act(x @ W)
        stds.append(x.std().item())
    return stds


inits = {
    "std=0.01": lambda n: 0.01,
    "std=0.1": lambda n: 0.1,
    "He sqrt(2/n)": lambda n: math.sqrt(2 / n),
}
plt.figure(figsize=(6, 3.5))
for name, fn in inits.items():
    plt.semilogy(activation_stds(fn), "o-", ms=3, label=name)
plt.xlabel("layer"); plt.ylabel("activation std (log)"); plt.legend(); plt.title("20 ReLU layers, width 512")
plt.show()

# %% [markdown]
# Only He init keeps the signal at a stable scale through 20 layers. In PyTorch:
# ```python
# nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
# nn.init.xavier_uniform_(layer.weight)
# nn.init.zeros_(layer.bias)
# ```
# **Zero init** fails differently: every neuron in a layer computes the same thing and receives the same gradient,
# so they stay identical forever ("symmetry"). Biases can safely start at zero; weights can't.
#
# BatchNorm (module 06) and residual connections (module 09) make deep networks far less sensitive to initialisation.

# %% [markdown]
# ## 9. Gradient clipping
# An occasional huge gradient (from a bad batch, or an unstable model such as a GAN) can wreck the weights in
# one step. Clip the **total gradient norm** before `optimizer.step()`:

# %%
model = make_mlp()
xb, yb = next(iter(train_loader))
loss = loss_fn(model(xb.to(device)), yb.to(device)) * 1000          # artificially huge loss
loss.backward()
total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # returns the norm *before* clipping
after = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
print(f"grad norm before: {total_norm:.1f}, after: {after:.3f}")

# %% [markdown]
# ## Common pitfalls
# - **The learning rate is wrong.** It's the first thing to check when training misbehaves: loss that is NaN or
#   rising → too high; loss that barely moves → too low.
# - **Calling `scheduler.step()` before `optimizer.step()`**, or once per batch for an epoch-based scheduler.
# - **Reusing an optimizer after replacing the model's parameters** → it still points at the old tensors.
# - **Adam with a high LR (≥1e-2) on a big network** → unstable. Start at 1e-3 or 3e-4.
# - **Comparing optimizers with different random initialisations** → seed before building each model.
#
# ## Recap
# - SGD's step size is limited by the steepest direction. Momentum smooths the zig-zags; RMSProp and Adam scale
#   each parameter's step.
# - Adam (lr ≈ 1e-3) is the robust default; SGD + momentum with a schedule is a strong alternative for CNNs.
# - Find the LR with a range test; decay it with cosine or step schedules; warm up big models.
# - He init for ReLU, Xavier for tanh; never zero weights. Clip gradients when training is unstable.
# - **Next (module 06):** regularization and debugging.
