# %% [markdown]
# # Module 05 — Exercises: Optimization

# %%
#! SETUP

# %%
import math
import torch
from torch import nn

# %% [markdown]
# ### Exercise 1 — Your own `torch.optim.Optimizer`: SGD with momentum and weight decay
# Subclass `torch.optim.Optimizer`. For each parameter `p` with gradient `g`:
# 1. `g = g + weight_decay * p` (L2 penalty, applied to the gradient)
# 2. `v = momentum * v + g` (store `v` in `self.state[p]["v"]`, starting at zeros)
# 3. `p = p - lr * v`
#
# The result must match `torch.optim.SGD(lr, momentum, weight_decay)` exactly.

# %%
class MySGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:          # param groups allow different settings per layer
            for p in group["params"]:
                if p.grad is None:
                    continue
                ### BEGIN SOLUTION
                #> # TODO
                #> raise NotImplementedError
                g = p.grad + group["weight_decay"] * p
                state = self.state[p]
                v = state.get("v")
                v = g.clone() if v is None else group["momentum"] * v + g
                state["v"] = v
                p -= group["lr"] * v
                ### END SOLUTION


# %%
def train_copy(opt_factory, steps=25):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 1))
    X, y = torch.randn(64, 5), torch.randn(64, 1)
    opt = opt_factory(m.parameters())
    for _ in range(steps):
        opt.zero_grad(); nn.functional.mse_loss(m(X), y).backward(); opt.step()
    return torch.cat([p.flatten() for p in m.parameters()]).detach()


ours = train_copy(lambda ps: MySGD(ps, lr=0.05, momentum=0.9, weight_decay=1e-2))
ref = train_copy(lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9, weight_decay=1e-2))
assert torch.allclose(ours, ref, atol=1e-6)
print("Exercise 1 passed ✅")

# %% [markdown]
# ### Exercise 2 — Your own Adam
# Same idea for Adam (lr, betas=(b1, b2), eps). Keep `m`, `v` and the step count `t` per parameter.
# It must match `torch.optim.Adam`.

# %%
class MyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                ### BEGIN SOLUTION
                #> # TODO
                #> raise NotImplementedError
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                g = p.grad
                st["m"] = b1 * st["m"] + (1 - b1) * g
                st["v"] = b2 * st["v"] + (1 - b2) * g * g
                m_hat = st["m"] / (1 - b1 ** st["t"])
                v_hat = st["v"] / (1 - b2 ** st["t"])
                p -= group["lr"] * m_hat / (v_hat.sqrt() + group["eps"])
                ### END SOLUTION


# %%
ours = train_copy(lambda ps: MyAdam(ps, lr=1e-2))
ref = train_copy(lambda ps: torch.optim.Adam(ps, lr=1e-2))
assert torch.allclose(ours, ref, atol=1e-5)
print("Exercise 2 passed ✅")

# %% [markdown]
# ### Exercise 3 — Warmup + cosine schedule as a function
# Write `lr_at(step, base_lr, warmup_steps, total_steps, min_lr=0.0)`:
# - during warmup (`step < warmup_steps`): rise linearly from `base_lr / warmup_steps` (step 0) up to `base_lr`;
# - afterwards: cosine decay from `base_lr` (at `warmup_steps`) to `min_lr` (at `total_steps`), then stay at `min_lr`.

# %%
def lr_at(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    ### END SOLUTION


# %%
assert math.isclose(lr_at(0, 1.0, 10, 100), 0.1)
assert math.isclose(lr_at(9, 1.0, 10, 100), 1.0)
assert math.isclose(lr_at(10, 1.0, 10, 100), 1.0)
assert math.isclose(lr_at(55, 1.0, 10, 100), 0.5, abs_tol=1e-9)          # halfway through the decay
assert math.isclose(lr_at(100, 1.0, 10, 100, min_lr=0.01), 0.01)
assert math.isclose(lr_at(500, 1.0, 10, 100, min_lr=0.01), 0.01)
print("Exercise 3 passed ✅")

# %% [markdown]
# ### Exercise 4 — Apply He initialisation to a model
# Write `he_init_(model)`: for every `nn.Linear` in `model` (use `model.modules()`), set the weights with
# `nn.init.kaiming_normal_(..., nonlinearity="relu")` and the biases to zero. Then check the effect: in a deep ReLU MLP
# the activation std at the last hidden layer should stay within the same order of magnitude as the input's.

# %%
def he_init_(model: nn.Module):
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    ### END SOLUTION


# %%
torch.manual_seed(0)
deep = nn.Sequential(*[layer for _ in range(15) for layer in (nn.Linear(256, 256), nn.ReLU())])
with torch.no_grad():
    for m in deep.modules():
        if isinstance(m, nn.Linear):
            m.weight.normal_(0, 0.01)                  # a deliberately terrible init
    tiny = deep(torch.randn(512, 256)).std().item()
    he_init_(deep)
    good = deep(torch.randn(512, 256)).std().item()
print(f"std with N(0, 0.01): {tiny:.2e}   with He: {good:.3f}")
assert all((m.bias == 0).all() for m in deep.modules() if isinstance(m, nn.Linear))
assert 0.2 < good < 5
print("Exercise 4 passed ✅")

# %% [markdown]
# ### Exercise 5 — Gradient clipping by global norm
# Implement `clip_grad_norm(params, max_norm)`: compute the L2 norm over **all** gradients together; if it exceeds
# `max_norm`, scale every gradient by `max_norm / (norm + 1e-6)`. Return the norm before clipping.
# Compare with `nn.utils.clip_grad_norm_`.

# %%
def clip_grad_norm(params, max_norm):
    params = [p for p in params if p.grad is not None]
    ### BEGIN SOLUTION
    #> # TODO
    #> raise NotImplementedError
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in params))
    if total > max_norm:
        for p in params:
            p.grad.mul_(max_norm / (total + 1e-6))
    return total
    ### END SOLUTION


# %%
def grads_after(clip_fn):
    torch.manual_seed(0)
    m = nn.Linear(10, 3)
    (m(torch.randn(4, 10)).sum() * 100).backward()
    n = clip_fn(m.parameters(), 1.0)
    return n, torch.cat([p.grad.flatten() for p in m.parameters()])


n1, g1 = grads_after(clip_grad_norm)
n2, g2 = grads_after(nn.utils.clip_grad_norm_)
assert torch.isclose(n1, n2) and torch.allclose(g1, g2, atol=1e-6)
assert abs(g1.norm().item() - 1.0) < 1e-3
print("Exercise 5 passed ✅")

# %% [markdown]
# ### Exercise 6 — Which learning rate?
# For the stretched quadratic from the lesson, $f(x, y) = 0.5x^2 + 10y^2$, plain gradient descent diverges when
# `lr * curvature > 2` in any direction (the curvatures are 1 and 20).
# 1. Set `max_stable_lr` to the largest LR that does **not** diverge (the boundary value).
# 2. Run 100 steps of GD from (-8, 2) at `0.9 * max_stable_lr` and at `1.1 * max_stable_lr`, and store the final
#    distances to the origin in `d_ok` and `d_bad`.

# %%
def gd_final_dist(lr, steps=100):
    p = torch.tensor([-8.0, 2.0])
    for _ in range(steps):
        g = torch.tensor([1.0, 20.0]) * p          # the gradient of f
        p = p - lr * g
    return p.norm().item()


### BEGIN SOLUTION
#> max_stable_lr = None  # TODO
#> d_ok, d_bad = None, None  # TODO
max_stable_lr = 2 / 20
d_ok, d_bad = gd_final_dist(0.9 * max_stable_lr), gd_final_dist(1.1 * max_stable_lr)
### END SOLUTION

# %%
assert math.isclose(max_stable_lr, 0.1)
assert d_ok < 1e-2 and d_bad > 1e3
print(f"lr={0.9 * max_stable_lr:.3f}: dist {d_ok:.2e}   lr={1.1 * max_stable_lr:.3f}: dist {d_bad:.2e}")
print("Exercise 6 passed ✅")
