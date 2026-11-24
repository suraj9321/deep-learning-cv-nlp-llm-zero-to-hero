# %% [markdown]
# # Module 06 — Exercises: Regularization and Debugging

# %%
#! SETUP

# %%
import math
import torch
from torch import nn

# %% [markdown]
# ### Exercise 1 — Inverted dropout as an `nn.Module`
# Write `MyDropout(p)`. In training mode, zero elements with probability `p` and scale the survivors by `1/(1-p)`.
# In eval mode (`self.training == False`), return the input unchanged.

# %%
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).to(x.dtype)
        return x * mask / (1 - self.p)
        ### END SOLUTION


# %%
d = MyDropout(0.25)
x = torch.ones(200, 500)
y = d(x)
assert abs((y == 0).float().mean().item() - 0.25) < 0.01
assert abs(y.mean().item() - 1.0) < 0.02
assert torch.allclose(y[y != 0], torch.tensor(1 / 0.75))
d.eval()
assert torch.equal(d(x), x)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — LayerNorm from scratch
# `MyLayerNorm(dim)`: normalize over the **last** dimension (biased variance, `eps=1e-5`), then apply learnable
# `weight` (init 1) and `bias` (init 0), both `nn.Parameter` of shape (dim,). It must match `nn.LayerNorm(dim)`.

# %%
class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        #> # TODO
        #> raise NotImplementedError
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        ### END SOLUTION


# %%
ours, ref = MyLayerNorm(12), nn.LayerNorm(12)
with torch.no_grad():
    for m in (ours, ref):
        m.weight.copy_(torch.linspace(0.5, 2, 12)); m.bias.copy_(torch.linspace(-1, 1, 12))
x = torch.randn(3, 5, 12) * 4 + 1
assert torch.allclose(ours(x), ref(x), atol=1e-5)
assert len(list(ours.parameters())) == 2
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — L2 penalty by hand
# Write `loss_with_l2(model, base_loss, lam)` returning `base_loss + lam/2 * Σ w²` over all **weight** matrices only
# (parameter names ending in `"weight"`, not biases). Then confirm that one SGD step with this loss equals one SGD step
# with `weight_decay=lam` applied **only to the weights** (use two param groups).

# %%
def loss_with_l2(model, base_loss, lam):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    l2 = sum((p ** 2).sum() for n, p in model.named_parameters() if n.endswith("weight"))
    return base_loss + lam / 2 * l2
    ### END SOLUTION


# %%
torch.manual_seed(0)
X, y = torch.randn(16, 4), torch.randn(16, 1)
make = lambda: (torch.manual_seed(1), nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1)))[1]
m1, m2 = make(), make()

opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
opt1.zero_grad(); loss_with_l2(m1, nn.functional.mse_loss(m1(X), y), 0.5).backward(); opt1.step()

decay = [p for n, p in m2.named_parameters() if n.endswith("weight")]
no_decay = [p for n, p in m2.named_parameters() if not n.endswith("weight")]
opt2 = torch.optim.SGD([{"params": decay, "weight_decay": 0.5}, {"params": no_decay, "weight_decay": 0.0}], lr=0.1)
opt2.zero_grad(); nn.functional.mse_loss(m2(X), y).backward(); opt2.step()

for p1, p2 in zip(m1.parameters(), m2.parameters()):
    assert torch.allclose(p1, p2, atol=1e-6)
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Expected initial loss
# For a fresh classifier on K balanced classes, what loss do you expect? Write `expected_initial_ce(K)`.
# Then check whether `suspicious_model` below has a reasonable initial loss on 10 classes, and set `is_suspicious`.

# %%
def expected_initial_ce(K):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    return math.log(K)
    ### END SOLUTION


torch.manual_seed(0)
suspicious_model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
with torch.no_grad():
    suspicious_model[2].weight.mul_(50)             # someone "initialised" the last layer badly
Xs, ys = torch.randn(512, 20), torch.randint(0, 10, (512,))
init_loss = nn.functional.cross_entropy(suspicious_model(Xs), ys).item()
### BEGIN SOLUTION
#> is_suspicious = None  # TODO: compare init_loss with expected_initial_ce(10)
is_suspicious = init_loss > 1.5 * expected_initial_ce(10)
### END SOLUTION

# %%
assert math.isclose(expected_initial_ce(10), 2.302585, rel_tol=1e-5)
print(f"initial loss {init_loss:.2f} vs expected {expected_initial_ce(10):.2f}")
assert is_suspicious is True or is_suspicious == True
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — Find the bugs
# `buggy_train` has **three** bugs that stop it from overfitting a single batch. Fix them in `fixed_train` so that the
# loss on the batch falls below 0.05.

# %%
def buggy_train(model, xb, yb, steps=200):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.eval()
    for _ in range(steps):
        logits = torch.softmax(model(xb), dim=1)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
    return loss.item()


def fixed_train(model, xb, yb, steps=200):
    ### BEGIN SOLUTION
    #> # TODO: copy buggy_train and fix it
    #> raise NotImplementedError
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()                                   # bug 1: eval mode disables dropout during training
    for _ in range(steps):
        opt.zero_grad()                             # bug 2: gradients were accumulating
        logits = model(xb)                          # bug 3: no softmax before cross_entropy
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
    return loss.item()
    ### END SOLUTION


# %%
torch.manual_seed(0)
xb, yb = torch.randn(32, 20), torch.randint(0, 5, (32,))
make = lambda: (torch.manual_seed(0), nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 5)))[1]
print("buggy final loss:", round(buggy_train(make(), xb, yb), 4))
final = fixed_train(make(), xb, yb)
print("fixed final loss:", round(final, 4))
assert final < 0.05
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Early stopping helper with patience and restore
# Implement `train_with_early_stopping(step_fn, val_fn, model, max_epochs, patience)`:
# - each epoch: call `step_fn()` and then `val_loss = val_fn()`;
# - track the best val loss and a **deep copy** of the model's `state_dict` at that epoch;
# - stop after `patience` epochs without improvement;
# - restore the best weights and return `(best_epoch, epochs_run)` (both 1-based).

# %%
import copy


def train_with_early_stopping(step_fn, val_fn, model, max_epochs, patience):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    best, best_epoch, best_state, bad = math.inf, 0, None, 0
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        step_fn()
        vl = val_fn()
        if vl < best:
            best, best_epoch, bad = vl, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_epoch, epoch
    ### END SOLUTION


# %%
# A scripted validation curve: improves until epoch 4, then gets worse
curve = iter([1.0, 0.8, 0.7, 0.65, 0.66, 0.7, 0.75, 0.8, 0.9, 1.0])
m = nn.Linear(2, 1)


def fake_step():
    with torch.no_grad():
        m.weight += 1.0          # weights change every epoch


snapshot = {}


def fake_val():
    v = next(curve)
    snapshot[v] = m.weight.detach().clone()
    return v


best_epoch, ran = train_with_early_stopping(fake_step, fake_val, m, max_epochs=10, patience=3)
assert (best_epoch, ran) == (4, 7), (best_epoch, ran)
assert torch.equal(m.weight, snapshot[0.65]), "best weights were not restored"
print("Exercise 6 passed ✅")
