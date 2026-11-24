# %% [markdown]
# # Module 06 — Regularization, Normalization and Debugging
#
# Neural networks have enough capacity to memorise their training set. Regularization makes them generalise
# instead. Normalization layers make deep networks trainable. A short debugging checklist saves hours.
#
# **Learning goals**
# - See overfitting clearly on a small dataset
# - Dropout (from scratch → `nn.Dropout`), weight decay (L2 vs AdamW), early stopping
# - BatchNorm and LayerNorm from scratch, and why `train()`/`eval()` matter
# - A debugging checklist: initial loss, overfitting one batch, NaN hunting, TensorBoard

# %%
#! SETUP

# %%
import copy
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from common.training import train_one_epoch, evaluate
from common.plotting import plot_loss_curves

# %% [markdown]
# ## 1. An overfitting setup
# A big MLP trained on only **2,000** FashionMNIST images will memorise them. We validate on 10,000 test images.

# %%
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=tfm)
full_test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=tfm)
small_train = Subset(full_train, range(2000))
val = Subset(full_test, range(2000 if FAST_DEV_RUN else 10_000))
train_loader = DataLoader(small_train, batch_size=64, shuffle=True)
val_loader = DataLoader(val, batch_size=1000)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 3 if FAST_DEV_RUN else 40


def big_mlp(p_drop=0.0, norm=None):
    layers, prev = [nn.Flatten()], 784
    for h in (1024, 1024, 512):
        layers.append(nn.Linear(prev, h))
        if norm == "bn":
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if p_drop:
            layers.append(nn.Dropout(p_drop))
        prev = h
    layers.append(nn.Linear(prev, 10))
    return nn.Sequential(*layers)


def run(model, opt, epochs=EPOCHS):
    torch.manual_seed(0)
    model.to(device)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for _ in range(epochs):
        train_one_epoch(model, train_loader, loss_fn, opt, device)
        tl, ta = evaluate(model, train_loader, loss_fn, device)   # measured in eval mode, comparable to val
        vl, va = evaluate(model, val_loader, loss_fn, device)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
    return hist

# %%
torch.manual_seed(0)
m = big_mlp()
h_base = run(m, torch.optim.Adam(m.parameters(), lr=1e-3))
plot_loss_curves(h_base, "Baseline: big MLP on 2k images")
print(f"final train acc {h_base['train_acc'][-1]:.3f}  val acc {h_base['val_acc'][-1]:.3f}  "
      f"best val acc {max(h_base['val_acc']):.3f}")

# %% [markdown]
# The training accuracy reaches ~100% while the validation loss rises after a few epochs: textbook overfitting.
#
# ## 2. Dropout
# > 🎥 **Watch:** [Stanford CS231n (2017), Lecture 7: 'Training Neural Networks II'](https://cs231n.stanford.edu/2017/syllabus.html), the regularization part, which covers dropout and why it acts like an ensemble.
#
# During training, zero each activation with probability `p`, a different random mask on every step. The
# network can't rely on any single unit, which acts like training an ensemble of thinned networks.
#
# **Inverted dropout:** divide the kept activations by `1 − p` during training so that their expected value is
# unchanged. At evaluation time dropout then does **nothing**. That's why `model.eval()` matters.

# %%
def dropout(x, p, training):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) > p).float()   # 1 = keep, with probability 1-p
    return x * mask / (1 - p)


x = torch.ones(1, 10_000)
y = dropout(x, 0.3, training=True)
print("fraction zeroed:", (y == 0).float().mean().item(), " mean stays ≈1:", y.mean().item())

d = nn.Dropout(0.3)
d.train(); print("nn.Dropout train mean:", d(x).mean().item())
d.eval(); print("nn.Dropout eval is identity:", torch.equal(d(x), x))

# %% [markdown]
# ## 3. Weight decay
# Add $\frac{\lambda}{2}\|\theta\|^2$ to the loss (L2 / Ridge, from the ML course). Its gradient adds $\lambda\theta$,
# which pulls every weight toward 0 at each step: hence "decay".
#
# Subtlety: with **Adam**, adding `λθ` to the gradient gets rescaled by Adam's per-parameter step sizes, so it
# no longer behaves like real decay. **AdamW** decouples it: `θ ← θ − lr·(adam_step + λθ)`. Use `AdamW` with
# weight decay; it's the standard for modern models.

# %%
torch.manual_seed(0)
m = big_mlp(p_drop=0.5)
h_drop = run(m, torch.optim.Adam(m.parameters(), lr=1e-3))
torch.manual_seed(0)
m = big_mlp()
h_wd = run(m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1))
torch.manual_seed(0)
m = big_mlp(p_drop=0.5)
h_both = run(m, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, h in [("baseline", h_base), ("dropout 0.5", h_drop), ("AdamW wd=0.1", h_wd), ("dropout + wd", h_both)]:
    axes[0].plot(h["val_loss"], label=name); axes[1].plot(h["val_acc"], label=name)
    print(f"{name:14s} final val acc {h['val_acc'][-1]:.3f}   train-val gap {h['train_acc'][-1] - h['val_acc'][-1]:.3f}")
axes[0].set_title("validation loss"); axes[1].set_title("validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch"); ax.legend()
plt.show()

# %% [markdown]
# Read the numbers rather than expecting magic. In a typical run, dropout narrows the train–val gap and gives a
# small accuracy gain. Weight decay alone barely changes anything here, and can even cost a little. Regularizers
# aren't free wins: their strength is a hyperparameter you tune on the validation set. Look at the validation
# **loss** too: the baseline's loss climbs much more steeply than the regularized runs', which means its
# predictions are becoming over-confident.
#
# With only 2,000 images, the **best** fix for overfitting is still more data, or data augmentation (module 08).
#
# ## 4. Early stopping
# Keep the weights from the epoch with the best validation loss, and stop once it hasn't improved for
# `patience` epochs.

# %%
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs, self.best_state = math.inf, 0, None

    def step(self, val_loss, model) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best, self.bad_epochs = val_loss, 0
            self.best_state = copy.deepcopy(model.state_dict())   # deepcopy: state_dict tensors are views of live weights
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


torch.manual_seed(0)
m = big_mlp().to(device)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
stopper = EarlyStopping(patience=3)
for epoch in range(EPOCHS):
    train_one_epoch(m, train_loader, loss_fn, opt, device)
    vl, va = evaluate(m, val_loader, loss_fn, device)
    if stopper.step(vl, m):
        print(f"stopped at epoch {epoch + 1}, best val loss {stopper.best:.4f}")
        break
m.load_state_dict(stopper.best_state)
print("restored best-epoch weights, val acc:", round(evaluate(m, val_loader, loss_fn, device)[1], 3))

# %% [markdown]
# ## 5. Batch normalization
# > 🎥 **Watch:** [Karpathy, 'makemore Part 3: Activations & Gradients, BatchNorm'](https://karpathy.ai/zero-to-hero.html), the second half. He builds BatchNorm from scratch and explains the running statistics.
#
# For each feature, BatchNorm standardizes the activations **across the batch**, then applies a learnable scale
# γ and shift β:
# $$ \hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat x + \beta $$
# It keeps every layer's inputs well scaled, allows higher learning rates, and reduces sensitivity to initialisation.
#
# **Train vs eval:** in training it uses the current batch's mean and variance, and updates **running** averages.
# In eval it uses the running averages, so a prediction doesn't depend on the other examples in the batch.

# %%
class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.gamma = nn.Parameter(torch.ones(num_features))    # nn.Parameter -> learnable, shows in parameters()
        self.beta = nn.Parameter(torch.zeros(num_features))
        # buffers: saved in state_dict and moved by .to(device), but not trained
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():   # PyTorch stores the *unbiased* variance in the running estimate
                n = x.shape[0]
                self.running_mean.lerp_(mean, self.momentum)          # running = (1-m)*running + m*new
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


ours, ref = MyBatchNorm1d(6), nn.BatchNorm1d(6)
for _ in range(5):
    xb = torch.randn(32, 6) * 3 + 2
    assert torch.allclose(ours(xb), ref(xb), atol=1e-5)          # training mode
ours.eval(); ref.eval()
xt = torch.randn(4, 6)
print("train & eval outputs match nn.BatchNorm1d:", torch.allclose(ours(xt), ref(xt), atol=1e-5))

# %%
torch.manual_seed(0)
m = big_mlp(norm="bn")
h_bn = run(m, torch.optim.Adam(m.parameters(), lr=1e-3))
print(f"BatchNorm: final val acc {h_bn['val_acc'][-1]:.3f} (baseline {h_base['val_acc'][-1]:.3f})")

# %% [markdown]
# **LayerNorm** standardizes across the **features** of each example rather than across the batch. It needs no
# running statistics and works with any batch size, which is why Transformers (module 14) use it.
# The only difference in code: `x.mean(dim=-1, keepdim=True)` instead of `dim=0`.

# %%
x = torch.randn(4, 8)
ln = nn.LayerNorm(8, elementwise_affine=False)
manual = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
print("LayerNorm matches:", torch.allclose(ln(x), manual, atol=1e-5))

# %% [markdown]
# ## 6. The debugging checklist
# > 🎥 **Watch:** [Stanford CS231n (2017), Lecture 6: 'Training Neural Networks I'](https://cs231n.stanford.edu/2017/syllabus.html), the 'babysitting the learning process' part. It's the same checklist, in lecture form.
#
# When a model won't train, go through these **in order**.
#
# **(a) Check the initial loss.** With K balanced classes and a fresh network, predictions are close to uniform, so
# the cross-entropy should be about ln K (2.30 for 10 classes). A much larger value means a bad init or a bug in the
# loss or labels.

# %%
torch.manual_seed(0)
fresh = big_mlp().to(device)
print(f"initial val loss {evaluate(fresh, val_loader, loss_fn, device)[0]:.3f}  vs ln(10) = {math.log(10):.3f}")

# %% [markdown]
# **(b) Overfit one batch.** A correct model + loop must drive the loss on a single small batch to almost 0. If it
# can't, the bug is in the model or the loop, not the data or the hyperparameters.

# %%
xb, yb = next(iter(train_loader))
xb, yb = xb[:16].to(device), yb[:16].to(device)
model = big_mlp().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for i in range(100):
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward(); opt.step()
print("loss on one batch after 100 steps:", round(loss.item(), 5))

# %% [markdown]
# **(c) Hunt NaNs.** NaN loss usually means: LR too high, `log(0)` or division by 0 in a hand-written loss, or
# exploding gradients. `torch.autograd.set_detect_anomaly(True)` reports the operation that produced it (it's slow,
# so only use it while debugging). Checking `torch.isfinite(loss)` every step is cheap.

# %%
x = torch.tensor([0.0, 1.0], requires_grad=True)
with torch.autograd.set_detect_anomaly(True):
    try:
        (torch.log(x) * 0).sum().backward()      # 0 * -inf = nan in the backward pass
        print("x.grad:", x.grad)
    except RuntimeError as e:
        print("anomaly detected:", str(e).split("\n")[0][:110])

# %% [markdown]
# **(d) Watch the curves and weight/gradient statistics with TensorBoard.** Log scalars and histograms during training,
# then run `tensorboard --logdir runs` in a terminal (inside the venv) and open http://localhost:6006.

# %%
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=str(ROOT / "runs" / "module06_dropout"))
torch.manual_seed(0)
m = big_mlp(p_drop=0.5).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
for epoch in range(2 if FAST_DEV_RUN else 10):
    tl = train_one_epoch(m, train_loader, loss_fn, opt, device)
    vl, va = evaluate(m, val_loader, loss_fn, device)
    writer.add_scalars("loss", {"train": tl, "val": vl}, epoch)
    writer.add_scalar("val_acc", va, epoch)
    for name, p in m.named_parameters():
        if "weight" in name:
            writer.add_histogram(name, p.detach().cpu(), epoch)
writer.close()
print("logged to runs/module06_dropout — run: tensorboard --logdir runs")

# %% [markdown]
# **(e) Other usual suspects:** data and labels misaligned after shuffling; the wrong normalization constants; the
# eval transform differs from the train transform when it shouldn't; `model.eval()` forgotten; a learning rate that's
# off by 10×.
#
# ## Common pitfalls
# - **Evaluating in train mode** → dropout stays on, BatchNorm uses batch stats: noisy and worse results.
# - **BatchNorm with tiny batches** (≤ 4) → the batch statistics are too noisy. Use LayerNorm or GroupNorm.
# - **Weight decay with `Adam` instead of `AdamW`** → weaker, mis-scaled regularization.
# - **Early stopping without restoring the best weights** → you keep the last (worse) epoch.
# - **`state_dict()` saved without `deepcopy`** → it's a view of the live weights and keeps changing.
#
# ## Recap
# - Regularizers: dropout, weight decay (AdamW), early stopping, and above all more data or augmentation.
# - BatchNorm normalizes over the batch (running stats in eval); LayerNorm normalizes over features.
# - Debug in order: initial loss ≈ ln K → overfit one batch → NaN checks → curves and TensorBoard.
#
# **Phase 1 complete.** Next is Phase 2, computer vision, starting with convolutions from scratch.
