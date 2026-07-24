"""g_theta: predict a target feature's activation from other features' activations.

Ridge is the default because the minimality search refits it thousands of times and the
closed form makes that free. Feature activations are non-negative, so predictions are
clamped at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Ridge:
    w: torch.Tensor
    b: float
    mu: torch.Tensor
    sigma: torch.Tensor

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        Xs = (X.to(torch.float32) - self.mu) / self.sigma
        return (Xs @ self.w + self.b).clamp_min(0.0)


def fit_ridge(X: torch.Tensor, y: torch.Tensor, lam: float = 1.0) -> Ridge:
    X = X.to(torch.float64)
    y = y.to(torch.float64)
    mu = X.mean(0)
    sigma = X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sigma
    b = y.mean()
    yc = y - b

    k = Xs.shape[1]
    A = Xs.T @ Xs + lam * torch.eye(k, dtype=torch.float64)
    w = torch.linalg.solve(A, Xs.T @ yc)
    return Ridge(w.float(), float(b), mu.float(), sigma.float())


class MLP(torch.nn.Module):
    def __init__(self, k: int, hidden: int = 64):
        super().__init__()
        self.mu: torch.Tensor
        self.sigma: torch.Tensor
        self.net = torch.nn.Sequential(
            torch.nn.Linear(k, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        Xs = (X.to(torch.float32) - self.mu) / self.sigma
        with torch.no_grad():
            return self.net(Xs).squeeze(-1).clamp_min(0.0)


def fit_mlp(X: torch.Tensor, y: torch.Tensor, hidden: int = 64, epochs: int = 200) -> MLP:
    X = X.to(torch.float32)
    y = y.to(torch.float32)
    model = MLP(X.shape[1], hidden)
    model.mu = X.mean(0)
    model.sigma = X.std(0).clamp_min(1e-6)
    Xs = (X - model.mu) / model.sigma

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model.net(Xs).squeeze(-1), y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def fit(kind: str, X: torch.Tensor, y: torch.Tensor, lam: float = 1.0):
    if kind == "ridge":
        return fit_ridge(X, y, lam)
    if kind == "mlp":
        return fit_mlp(X, y)
    raise KeyError(f"unknown reconstructor {kind!r}")


def r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
