"""
Train the BHR surrogate measurement model.

Usage:
    python train.py --npz data/synthetic.npz
"""

import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import HetRegModel, gaussian_nll_loss, _enable_dropout
from dataset import make_loaders


def _expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    P = len(theta)
    if len(prior_prec) == 1:
        return torch.ones(P, device=theta.device) * prior_prec
    elif len(prior_prec) == P:
        return prior_prec.to(theta.device)
    else:
        return torch.cat([
            delta * torch.ones_like(p).flatten()
            for delta, p in zip(prior_prec, model.parameters())
        ])


def _valid_perf(model, loader, device, n_mc=10):
    model.eval()
    N, mse, nll = len(loader.dataset), 0.0, 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            _enable_dropout(model)
            samples  = torch.stack([model(X_b) for _ in range(n_mc)], dim=1)
            mu_mean  = samples[:, :, :15].mean(dim=1)
            lv_mean  = samples[:, :, 15:].mean(dim=1)
            f_avg    = torch.cat([mu_mean, lv_mean], dim=1)
            mse     += (mu_mean - y_b).pow(2).sum().item() / N
            nll     += gaussian_nll_loss(f_avg, y_b).item()
    return mse, nll / len(loader)


def train(model, train_loader, valid_loader=None,
          n_epochs=200, lr=1e-3, lr_min=1e-5,
          beta=0.5, prior_prec_init=1.0, temperature=1.0, device=None,
          eval_every=10, patience=5):
    """
    patience : int
        Number of consecutive evaluations (each `eval_every` epochs) with no
        improvement in val_nll before stopping early. Ignored if valid_loader
        is None. The best-val_nll weights are restored before returning.
    """

    if device is None:
        device = next(model.parameters()).device

    X_train = train_loader.dataset.dataset.X[train_loader.dataset.indices]
    model.set_input_stats(X_train.mean(dim=0), X_train.std(dim=0))

    N  = len(train_loader.dataset)
    H  = len(list(model.parameters()))
    log_prior_prec = np.log(prior_prec_init) * torch.ones(H, device=device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, n_epochs * len(train_loader), eta_min=lr_min)

    train_losses, val_mses, val_nlls = [], [], []
    best_val_nll, best_state, bad_evals = float('inf'), None, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            delta = _expand_prior_precision(torch.exp(log_prior_prec).detach(), model)
            theta = parameters_to_vector(model.parameters())
            f     = model(X_b)
            loss  = gaussian_nll_loss(f, y_b, beta=beta) \
                    + (0.5 * (delta * theta) @ theta) / N / temperature

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() / len(train_loader)

        train_losses.append(epoch_loss)

        if epoch % eval_every == 0:
            if valid_loader is not None:
                val_mse, val_nll = _valid_perf(model, valid_loader, device)
                val_mses.append(val_mse)
                val_nlls.append(val_nll)
                print(f"Epoch {epoch:4d}/{n_epochs} | loss={epoch_loss:.4f} | "
                      f"val_mse={val_mse:.4f} | val_nll={val_nll:.4f}")

                if val_nll < best_val_nll:
                    best_val_nll, bad_evals = val_nll, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no val_nll improvement in {patience * eval_every} epochs, "
                              f"best={best_val_nll:.4f})")
                        break
            else:
                print(f"Epoch {epoch:4d}/{n_epochs} | loss={epoch_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_mses, val_nlls


if __name__ == '__main__':
    import os, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='data/synthetic.npz')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader = make_loaders(args.npz, batch_size=512)
    model = HetRegModel(in_dim=16, trunk_dims=(128, 128, 64), dropout_p=0.1).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model, *_ = train(model, train_loader, val_loader, n_epochs=200, lr=1e-3)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/hetreg_model.pt')
    print("Saved -> checkpoints/hetreg_model.pt")
