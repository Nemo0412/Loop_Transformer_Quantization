"""Inference-time quantization-error compensators for looped transformers.

Three methods, all post-training and all weight-frozen:

  A. loop-wise mean           h~_t = h^_t - lambda_t * mu_t,  mu_t = E_calib[e_t]
  B. loop-aware low-rank      h~_t = h^_t - U_t V_t^T h^_t
  C. error dynamics           e^_{t+1} = A_t e^_t + b_t,  h~_t = h^_t - e^_t

e_t = h^_t - h_t is the state error (quantized minus full precision).

Hard rule: at inference a compensator may read only the live quantized state
and its own previous estimate. Full-precision states are used to fit the
compensator on calibration data and to log prediction quality, never fed back
into the trajectory. The driver enforces this by never passing FP states to
`correct`.

Correctors act on a hidden state `h` shaped [B, T, D] and are applied after
every non-final loop. `reset` clears any per-sequence recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fit import fit_low_rank


class Compensator:
    name: str = "base"

    def reset(self) -> None:  # per-forward recurrent state
        pass

    def correct(self, loop_index: int, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predicted_error(self, loop_index: int, h: torch.Tensor) -> torch.Tensor:
        """The error estimate subtracted at this loop; for logging only."""
        raise NotImplementedError

    def extra_params(self) -> int:
        return 0


class NoComp(Compensator):
    name = "none"

    def correct(self, loop_index, h):
        return h

    def predicted_error(self, loop_index, h):
        return torch.zeros_like(h)


@dataclass
class MeanComp(Compensator):
    mus: list[torch.Tensor]            # mu_t, one [D] vector per correctable loop
    lams: list[float]                  # lambda_t
    name: str = "mean"

    def predicted_error(self, loop_index, h):
        if loop_index >= len(self.mus):   # not yet fitted (sequential calibration)
            return torch.zeros_like(h)
        return self.lams[loop_index] * self.mus[loop_index].to(h.dtype)

    def correct(self, loop_index, h):
        return h - self.predicted_error(loop_index, h)

    def extra_params(self) -> int:
        return sum(m.numel() for m in self.mus)


@dataclass
class LowRankComp(Compensator):
    factors: list[tuple[torch.Tensor, torch.Tensor]]   # (U_t, V_t), each [D, r]
    shared: bool = False
    name: str = "lowrank"

    def predicted_error(self, loop_index, h):
        if loop_index >= len(self.factors):   # not yet fitted (sequential calibration)
            return torch.zeros_like(h)
        u, v = self.factors[loop_index]
        return (h @ v.to(h.dtype)) @ u.to(h.dtype).t()

    def correct(self, loop_index, h):
        return h - self.predicted_error(loop_index, h)

    def extra_params(self) -> int:
        if self.shared and self.factors:
            u, v = self.factors[0]
            return u.numel() + v.numel()
        return sum(u.numel() + v.numel() for u, v in self.factors)


@dataclass
class DynamicsComp(Compensator):
    """Per-sample error dynamics seeded from the observable loop-0 state.

        e^_0 = (h^_0 @ V0) @ U0^T + c0
        e^_t = (e^_{t-1} @ V_{t-1}) @ U_{t-1}^T + b_t      for t >= 1

    Only the live quantized state and the compensator's own previous estimate
    are read, so no full-precision state is needed at inference.
    """

    init: tuple[torch.Tensor, torch.Tensor]            # (U0, V0), each [D, r0]
    c0: torch.Tensor                                   # [D]
    A: list[tuple[torch.Tensor, torch.Tensor]]         # (U_k, V_k) for k = 0..n-3
    b: list[torch.Tensor]                              # b_t for t = 1..n-2 (index 0 unused)
    rank: int = 0
    gamma: float = 1.0                                 # damping on the A term (C2)
    reset_period: int = 0                              # re-anchor every K loops (C3); 0 = never
    reseed_maps: list | None = None                    # per-loop observable maps for re-anchor
    name: str = "dynamics"
    _ehat: torch.Tensor | None = field(default=None, repr=False)

    def reset(self):
        self._ehat = None

    def _seed(self, loop_index, h):
        """Observable-only error estimate at this loop (no FP state)."""
        if self.reseed_maps is not None and loop_index < len(self.reseed_maps):
            u, v = self.reseed_maps[loop_index]
            return (h @ v.to(h.dtype)) @ u.to(h.dtype).t()
        u0, v0 = self.init
        return (h @ v0.to(h.dtype)) @ u0.to(h.dtype).t() + self.c0.to(h.dtype)

    def predicted_error(self, loop_index, h):
        reanchor = self.reset_period and loop_index > 0 and loop_index % self.reset_period == 0
        if loop_index == 0 or reanchor or self._ehat is None:
            e = self._seed(loop_index, h)
        else:
            u, v = self.A[loop_index - 1]
            prev = self._ehat
            e = self.gamma * ((prev @ v.to(h.dtype)) @ u.to(h.dtype).t()) + self.b[loop_index].to(h.dtype)
        return e

    def correct(self, loop_index, h):
        e = self.predicted_error(loop_index, h)
        self._ehat = e
        return h - e

    def extra_params(self) -> int:
        u0, v0 = self.init
        total = self.c0.numel() + u0.numel() + v0.numel()
        total += sum(u.numel() + v.numel() for u, v in self.A)
        total += sum(bt.numel() for bt in self.b)
        if self.reseed_maps is not None:
            total += sum(u.numel() + v.numel() for u, v in self.reseed_maps)
        return total

    def spectral_stats(self) -> list[dict]:
        """Spectral norm and radius of each transition matrix M_k = gamma * U_k V_k^T."""
        stats = []
        for k, (u, v) in enumerate(self.A):
            m = self.gamma * (u @ v.t())
            sv = torch.linalg.svdvals(m.float())
            ev = torch.linalg.eigvals(m.float())
            stats.append({
                "k": k,
                "spectral_norm": float(sv.max().item()),
                "spectral_radius": float(ev.abs().max().item()),
            })
        return stats


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def fit_mean(h_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
    """mu_t = mean error over calibration rows. h_t unused, kept for symmetry."""
    return e_t.mean(dim=0)


def best_lambda(mu: torch.Tensor, e_t: torch.Tensor) -> float:
    """Scalar that minimizes ||e_t - lambda * mu|| over calibration rows."""
    denom = float((mu * mu).sum().item()) * e_t.shape[0]
    if denom < 1e-12:
        return 0.0
    num = float((e_t * mu).sum().item())
    return num / denom


def fit_lowrank(h_t: torch.Tensor, e_t: torch.Tensor, rank: int):
    """(U, V) with e_t ≈ (h_t @ V) @ U^T."""
    return fit_low_rank(h_t, e_t, rank)


def constrain_spectral(u: torch.Tensor, v: torch.Tensor, cap: float):
    """Scale (U, V) so the spectral norm of U V^T is at most `cap` (C1)."""
    if cap is None or cap <= 0:
        return u, v
    with torch.no_grad():
        smax = float(torch.linalg.svdvals((u @ v.t()).float()).max().item())
    if smax > cap:
        u = u * (cap / smax)
    return u, v


def fit_dynamics(errors: list[torch.Tensor], h0: torch.Tensor, rank: int, device,
                 refine: int = 1, spectral_cap: float | None = None,
                 gamma: float = 1.0, reset_period: int = 0, reseed_maps=None):
    """Fit the error dynamics from true per-loop errors and observable loop-0 state.

    `errors[t]` is [N, D], the true error at loop t under the uncompensated
    quantized rollout. `h0` is [N, D], the observable quantized state at loop 0.
    A teacher-forced fit uses the true previous error as the regressor; `refine`
    DAgger passes then refit the propagation against the compensator's own
    free-running estimate so it stays stable on its own trajectory.
    """
    n = len(errors)
    d = errors[0].shape[1]
    r = min(rank, d)

    def _fit_step(prev, cur):
        u, v = fit_low_rank(prev, cur, r)
        u, v = constrain_spectral(u, v, spectral_cap)
        pred = gamma * ((prev @ v) @ u.t())
        bias = (cur - pred).mean(dim=0)
        return (u.to(device), v.to(device)), bias.to(device)

    # observable seed for loop 0: map h0 -> e_0
    u0, v0 = fit_low_rank(h0.to(device), errors[0].to(device), r)
    c0 = (errors[0].to(device) - (h0.to(device) @ v0) @ u0.t()).mean(dim=0)
    init = (u0.to(device), v0.to(device))

    def _make(A, b):
        return DynamicsComp(init=init, c0=c0, A=A, b=b, rank=r, gamma=gamma,
                            reset_period=reset_period, reseed_maps=reseed_maps)

    A: list = []
    b: list = [torch.zeros(d, device=device)]  # index 0 unused
    for t in range(1, n):
        (u, v), bias = _fit_step(errors[t - 1].to(device), errors[t].to(device))
        A.append((u, v))
        b.append(bias)

    comp = _make(A, b)

    for _ in range(refine):
        # free-run the estimate on calibration, then refit each step against it
        ehat = [(h0.to(device) @ v0) @ u0.t() + c0]
        for t in range(1, n):
            u, v = comp.A[t - 1]
            ehat.append(gamma * ((ehat[t - 1] @ v) @ u.t()) + comp.b[t])
        new_A, new_b = [], [torch.zeros(d, device=device)]
        for t in range(1, n):
            (u, v), bias = _fit_step(ehat[t - 1].detach(), errors[t].to(device))
            new_A.append((u, v))
            new_b.append(bias)
        comp = _make(new_A, new_b)
    return comp


# ---------------------------------------------------------------------------
# synthetic self-test
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    d, n, N = 16, 5, 4096
    device = torch.device("cpu")

    # synthetic looped error with linear dynamics plus a bias, to exercise fits.
    # e_0 is a linear function of an observable state h0 (so the seed map is learnable).
    A_true = 0.6 * torch.randn(d, d) / (d ** 0.5)
    b_true = 0.1 * torch.randn(d)
    h0 = torch.randn(N, d)
    seed = (torch.randn(d, d) / d)
    e = [h0 @ seed + 0.05 * torch.randn(N, d)]
    for _ in range(1, n):
        e.append(e[-1] @ A_true.t() + b_true + 0.01 * torch.randn(N, d))

    # mean
    mu = [fit_mean(None, e[t]) for t in range(n - 1)]
    assert torch.allclose(mu[0], e[0].mean(0))

    # low-rank maps h -> e; use an exactly rank-8 target so the fit can hit it
    h0 = torch.randn(N, d)
    M = (torch.randn(d, 8) @ torch.randn(8, d)) / d
    target = h0 @ M
    u, v = fit_lowrank(h0, target, rank=8)
    pred = (h0 @ v) @ u.t()
    rel = (pred - target).norm() / target.norm()
    assert rel < 0.1, rel

    # dynamics should recover the per-sample recursion well in free-running mode,
    # seeded only from the observable loop-0 state h0 (no FP error fed in)
    comp = fit_dynamics(e, h0, rank=d, device=device, refine=2)
    comp.reset()
    ehat = []
    for t in range(n - 1):
        pred = comp.predicted_error(t, h0 if t == 0 else torch.zeros(N, d))
        comp._ehat = pred
        ehat.append(pred)
    err = (ehat[-1] - e[n - 2]).norm() / e[n - 2].norm()
    assert err < 0.45, err

    # no-FP guarantee: correct() signature takes only (loop_index, h)
    for comp in (NoComp(), MeanComp(mu, [1.0] * len(mu)),
                 LowRankComp([(u, v)] * (n - 1))):
        comp.reset()
        h = torch.randn(2, 3, d)
        out = comp.correct(0, h)
        assert out.shape == h.shape
    print("compensate self-test ok")


if __name__ == "__main__":
    _self_test()
