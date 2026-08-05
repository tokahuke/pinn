"""
Monte Carlo shoot-out: the trained PINN policy vs optimally-timed
explore-then-commit (50/50 until the commit time maximizing the analytic
ETC value at TAU0, then bang-bang). Dimensionless units (rho = sigma = 1),
posterior-space simulation from the ridge (m = 0, tau = TAU0).
"""

from __future__ import annotations

import math
import torch

from pathlib import Path
from torch import Tensor

from pinn.problems.two_arm import ExplorationPremium, ValueFunction

TAU0 = 1.0
DT = 1e-3
HORIZON = 12.0
PATHS = 50_000

# Optimally-timed explore-then-commit at THIS tau0: maximize the analytic
# commit value e^-T sqrt(variance(T) / 2 pi) over the commit time.
_times = torch.linspace(0.01, 2.0, 2000)
_variances = 1.0 / TAU0 - 1.0 / (TAU0 + _times / 4)
COMMIT_TIME = float(_times[(torch.exp(-2.0 * _times) * _variances).argmax()])

state = torch.load(Path("data") / "two_arm.pt")
hidden = [weight.shape[0] for key, weight in state.items() if key.endswith(".weight")][
    :-1
]

value = ValueFunction(ExplorationPremium(hidden))
value.load_state_dict(state)

# Policy table on the similarity chart (z, s) = (muhat sqrt(tauhat), log tauhat):
# the policy is smooth there at every information level, and the similarity
# operator M grades alpha without the 1/tauhat^2 error amplification of the raw
# form. Bilinear lookup + arm-swap symmetry at run time.
Z_MAX = 8.0
TAU_MAX = TAU0 + HORIZON / 4 + 0.1
z_axis = torch.linspace(0.0, Z_MAX, 481)
s_axis = torch.linspace(math.log(TAU0), math.log(TAU_MAX), 481)


def policy_table() -> Tensor:
    from pinn.problems.two_arm.simplex import maximize_quadratic

    z_grid, s_grid = torch.meshgrid(z_axis, s_axis, indexing="ij")
    z = z_grid.flatten().requires_grad_(True)
    s = s_grid.flatten().requires_grad_(True)

    g = (s / 2).exp() * value.premium(z * (-s / 2).exp(), s.exp())
    g_z, g_s = torch.autograd.grad(g.sum(), [z, s], create_graph=True)
    (g_zz,) = torch.autograd.grad(g_z.sum(), z)

    m_of_g = (g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g).detach()
    best = maximize_quadratic(-m_of_g, s.detach().exp() * z.detach() + m_of_g)

    return best.x.reshape(len(z_axis), len(s_axis))


TABLE = policy_table()


def pinn_policy(m: Tensor, tau: Tensor, t: float) -> Tensor:
    i = ((m.abs() * tau.sqrt() / Z_MAX) * (len(z_axis) - 1)).clamp(0, len(z_axis) - 2)
    j = ((tau.log() - s_axis[0]) / (s_axis[-1] - s_axis[0]) * (len(s_axis) - 1)).clamp(
        0, len(s_axis) - 2
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


def ztest_policy(m: Tensor, tau: Tensor, t: float) -> Tensor:
    """
    The impatient experimenter: 50/50 until the continuously-peeked z-test
    rejects at 5%, then commit to the winner. The statistic is data-only, so
    the prior must be subtracted back out of the posterior state: data
    precision tau - TAU0, data mean m tau/(tau - TAU0), hence
    z = m tau / sqrt(tau - TAU0). Commitment is absorbing via simulate().
    """
    z_data = m * tau / (tau - TAU0).clamp_min(1e-12).sqrt()
    significant = z_data.abs() > 1.96

    return torch.where(significant, (m > 0).float(), torch.full_like(m, 0.5))


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
ztest_values = simulate(ztest_policy)

variance = 1.0 / TAU0 - 1.0 / (TAU0 + COMMIT_TIME / 4)
analytic_baseline = math.exp(-COMMIT_TIME) * math.sqrt(variance / (2 * math.pi))
stderr = 2.0 / math.sqrt(PATHS)

print(f"ETC commit time (optimized)  = {COMMIT_TIME:.3f}")
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
    f"z-test at 5% (peeking)       = {ztest_values.mean():.4f}"
    f" +/- {stderr * ztest_values.std():.4f}"
)
# Differences headline: real money is sigma/sqrt(rho) times these, and the
# ratio divides out exactly the prior-width scaling that pays the bills.
print(
    f"PINN - baseline              = {pinn_values.mean() - baseline_values.mean():.4f}"
    f"   (ratio {pinn_values.mean() / baseline_values.mean():.3f})"
)
print(
    f"PINN - z-test                = {pinn_values.mean() - ztest_values.mean():.4f}"
    f"   (ratio {pinn_values.mean() / ztest_values.mean():.3f})"
)
