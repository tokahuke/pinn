"""
Monte Carlo shoot-out: the trained PINN policy vs explore-then-commit (50/50
until T = 1/(2 rho), then bang-bang). Dimensionless units (rho = sigma = 1),
posterior-space simulation from the ridge (m = 0, tau = TAU0).
"""

from __future__ import annotations

import math
import torch

from pathlib import Path
from torch import Tensor

from pinn.problems.two_arm import ExplorationPremium, ValueFunction

TAU0 = 1.0
COMMIT_TIME = 0.5
DT = 2e-3
HORIZON = 12.0
PATHS = 50_000

state = torch.load(Path("data") / "value.pt")
hidden = [weight.shape[0] for key, weight in state.items() if key.endswith(".weight")][
    :-1
]

value = ValueFunction(ExplorationPremium(hidden))
value.load_state_dict(state)

# Policy table on muhat >= 0; bilinear lookup + arm-swap symmetry at run time.
MU_MAX = 6.0
TAU_MAX = TAU0 + HORIZON / 4 + 0.1
mu_axis = torch.linspace(0.0, MU_MAX, 481)
tau_axis = torch.linspace(TAU0, TAU_MAX, 481)


def policy_table() -> Tensor:
    mu_grid, tau_grid = torch.meshgrid(mu_axis, tau_axis, indexing="ij")
    muhat = mu_grid.flatten().requires_grad_(True)
    tauhat = tau_grid.flatten().requires_grad_(True)

    v = value(muhat, tauhat)
    v_muhat, v_tauhat = torch.autograd.grad(v.sum(), [muhat, tauhat], create_graph=True)
    (v_muhat_muhat,) = torch.autograd.grad(v_muhat.sum(), muhat)

    lhat = v_tauhat.detach() + v_muhat_muhat / (2 * tauhat.detach() ** 2)
    concave = lhat > 0
    safe_lhat = lhat.masked_fill(~concave, 1.0)
    alpha = torch.where(
        concave,
        ((muhat.detach() + lhat) / (2 * safe_lhat)).clamp(0.0, 1.0),
        (muhat.detach() > 0).float(),
    )

    return alpha.detach().reshape(len(mu_axis), len(tau_axis))


TABLE = policy_table()


def pinn_policy(m: Tensor, tau: Tensor, t: float) -> Tensor:
    i = ((m.abs() / MU_MAX) * (len(mu_axis) - 1)).clamp(0, len(mu_axis) - 2)
    j = (((tau - TAU0) / (TAU_MAX - TAU0)) * (len(tau_axis) - 1)).clamp(
        0, len(tau_axis) - 2
    )
    i0, j0 = i.long(), j.long()
    di, dj = i - i0, j - j0

    a = (
        TABLE[i0, j0] * (1 - di) * (1 - dj)
        + TABLE[i0 + 1, j0] * di * (1 - dj)
        + TABLE[i0, j0 + 1] * (1 - di) * dj
        + TABLE[i0 + 1, j0 + 1] * di * dj
    )

    return torch.where(m >= 0, a, 1.0 - a)


def baseline_policy(m: Tensor, tau: Tensor, t: float) -> Tensor:
    if t < COMMIT_TIME:
        return torch.full_like(m, 0.5)

    return (m > 0).float()


def simulate(policy) -> Tensor:
    """
    Discounted value per path. Saturated paths freeze, so they close in exact
    arithmetic: e^-t * m for commit-up, 0 for commit-down.
    """
    torch.manual_seed(0)
    m = torch.zeros(PATHS)
    tau = torch.full((PATHS,), TAU0)
    payoff = torch.zeros(PATHS)
    active = torch.ones(PATHS, dtype=torch.bool)
    t = 0.0

    while t < HORIZON and bool(active.any()):
        alpha = policy(m, tau, t)
        discount = math.exp(-t)

        committed_up = active & (alpha >= 1.0 - 1e-6)
        committed_down = active & (alpha <= 1e-6)
        payoff = payoff + committed_up * discount * m
        active = active & ~(committed_up | committed_down)

        rate = alpha * (1 - alpha)
        payoff = payoff + active * discount * alpha * m * DT
        noise = torch.randn(PATHS)
        m = m + active * (rate.sqrt() / tau) * math.sqrt(DT) * noise
        tau = tau + active * rate * DT
        t += DT

    return payoff + active * math.exp(-HORIZON) * m.relu()


with torch.no_grad():
    predicted = float(value(torch.zeros(1), torch.tensor([TAU0])))

pinn_values = simulate(pinn_policy)
baseline_values = simulate(baseline_policy)

variance = 1.0 / TAU0 - 1.0 / (TAU0 + COMMIT_TIME / 4)
analytic_baseline = math.exp(-COMMIT_TIME) * math.sqrt(variance / (2 * math.pi))
stderr = 2.0 / math.sqrt(PATHS)

print(f"net's own claim v(0, {TAU0})    = {predicted:.4f}")
print(
    f"PINN policy (simulated)      = {pinn_values.mean():.4f}"
    f" +/- {stderr * pinn_values.std():.4f}"
)
print(
    f"explore-then-commit          = {baseline_values.mean():.4f}"
    f" +/- {stderr * baseline_values.std():.4f}"
)
print(f"explore-then-commit (exact)  = {analytic_baseline:.4f}")
print(
    f"PINN / baseline              = {pinn_values.mean() / baseline_values.mean():.3f}"
)
