"""Reduced-rank linear map: predicted_error = U @ V.T @ hidden_state."""

from __future__ import annotations

import math

import torch


def fit_low_rank(hidden: torch.Tensor, error: torch.Tensor, rank: int, ridge: float = 1e-3):
    """Least-squares reduced-rank regression, no bias.

    `hidden` and `error` are row-major [N, D]. The returned factors satisfy
    `error_col ≈ U @ V.T @ hidden_col` with U, V shaped [D, rank].
    """
    if hidden.ndim != 2 or error.shape != hidden.shape:
        raise ValueError("hidden and error must be [N, D] and the same shape")
    n, d = hidden.shape
    if not 1 <= rank <= d:
        raise ValueError(f"rank {rank} is outside 1..{d}")
    if n < d:
        raise RuntimeError(f"need at least D={d} calibration rows, got {n}")

    hth = hidden.T @ hidden
    diag_mean = hth.diagonal().mean().clamp(min=1e-8)
    hth = hth + (ridge * diag_mean) * torch.eye(d, device=hidden.device, dtype=hidden.dtype)
    hte = hidden.T @ error
    weight = torch.linalg.solve(hth, hte)
    gram = 0.5 * (hte.T @ weight + (hte.T @ weight).T)
    _evals, evecs = torch.linalg.eigh(gram)
    top = evecs[:, -rank:]
    v = weight @ top
    u = top
    return u.contiguous(), v.contiguous()


def predict_error(hidden: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (hidden @ v) @ u.T


def prediction_scores(hidden: torch.Tensor, error: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> dict:
    pred = predict_error(hidden, u, v)
    diff = pred - error
    mse = float(diff.float().pow(2).mean().item())
    pf = pred.float().reshape(-1, pred.shape[-1])
    tf = error.float().reshape(-1, error.shape[-1])
    cos = (pf * tf).sum(-1) / (pf.norm(dim=-1).clamp(min=1e-8) * tf.norm(dim=-1).clamp(min=1e-8))
    denom = math.sqrt(float(pf.pow(2).sum().item()) * float(tf.pow(2).sum().item()))
    global_cos = float((pf * tf).sum().item() / denom) if denom > 0 else float("nan")
    return {"mse": mse, "cosine": float(cos.mean().item()), "global_cosine": global_cos}
