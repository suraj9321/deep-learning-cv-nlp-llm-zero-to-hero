"""Generic training helpers.

You write these loops by hand in module 04; from module 05 onward notebooks import them to save space.
"""
import torch
from torch import nn


def train_one_epoch(model: nn.Module, loader, loss_fn, optimizer, device, max_batches=None) -> float:
    """One pass over the training data. Returns the average loss per sample."""
    model.train()  # enables dropout / batch-norm updates
    total_loss, n = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()          # gradients accumulate by default, so clear them every step
        loss = loss_fn(model(xb), yb)
        loss.backward()                # compute d(loss)/d(param) for every parameter
        optimizer.step()               # update the parameters using those gradients
        total_loss += loss.item() * len(xb)  # .item() -> Python float; loss is a per-batch mean
        n += len(xb)
    return total_loss / max(n, 1)


@torch.no_grad()  # no gradient bookkeeping during evaluation: faster and uses less memory
def evaluate(model: nn.Module, loader, loss_fn, device, max_batches=None) -> tuple[float, float]:
    """Returns (average loss, accuracy) for a classification model."""
    model.eval()  # disables dropout, uses running batch-norm statistics
    total_loss, correct, n = 0.0, 0, 0
    for i, (xb, yb) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(xb)
    return total_loss / max(n, 1), correct / max(n, 1)


def fit(model, train_loader, val_loader, loss_fn, optimizer, device, epochs: int,
        scheduler=None, max_batches=None, verbose: bool = True) -> dict:
    """Train for several epochs and return a history dict for plot_loss_curves."""
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(epochs):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device, max_batches)
        vl, va = evaluate(model, val_loader, loss_fn, device, max_batches)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(tr); history["val_loss"].append(vl); history["val_acc"].append(va)
        if verbose:
            print(f"epoch {epoch + 1:>2}/{epochs}  train_loss={tr:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")
    return history
