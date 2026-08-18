"""
Inverse-cdf laws for the quasi-random samplers.

Every problem draws scrambled Sobol points in the unit cube and pushes them
through these. Keeping the laws here rather than per-problem means one
definition, one self-check, and no chance of two samplers disagreeing about
what "an exponential tail" or "a decade spread" is.

All of them take u in (0, 1) and are pure elementwise maps, so they compose
with any Sobol dimension without caring which.
"""

from __future__ import annotations

import torch

from torch import Tensor


def exponential(u: Tensor) -> Tensor:
    """
    Unit exponential. The tail two_arm uses for muhat and its tauhat spread,
    three_arm for its wall means, two_arm_drift for etahat.
    """
    return -(1.0 - u).log()


def laplace(u: Tensor) -> Tensor:
    """
    Unit Laplace: two-sided, exponential in each direction. three_arm's means,
    which must reach both signs before the wedge fold sorts them.
    """
    centered = u - 0.5

    return -centered.sign() * (1.0 - 2.0 * centered.abs()).log()


def chi_squared_1(u: Tensor) -> Tensor:
    """
    Chi-squared with 1 degree of freedom, the square of a standard normal. Its density
    **diverges** at 0, so `P(X < eps) ~ sqrt(eps)` and tiny values are favoured rather
    than coincidental: three_arm draws pairwise precisions this way to give the startup
    corner real mass, where exponential tails leave a ~1e-5 triple coincidence.
    """
    return torch.special.ndtri(0.5 + 0.5 * u) ** 2


def decade_scale(u: Tensor, decades: float) -> Tensor:
    """
    A multiplicative scale spread over `decades` powers of ten, densest at 1 and
    reaching down: `10**(-decades * u**2)`. The squared exponent is what makes it dense
    near 1 rather than uniform per decade; every problem multiplies its precision law
    by one so the low-information decades get mass an exponential tail would miss.
    """
    return torch.pow(10.0, -decades * u**2)


def truncated_pareto(u: Tensor, low: float, high: float) -> Tensor:
    """
    Log-uniform on [low, high]: a truncated Pareto at shape alpha = 0, equal mass per
    decade, reaching its bounds only in the limit so neither carries point mass.
    **Monotone** in u, which beats a product of two laws because a Sobol coordinate is
    near-uniform within a batch (gradient cosine 0.22 two-coordinate, 0.94 here).
    """
    return low * (high / low) ** u


if __name__ == "__main__":
    torch.manual_seed(0)
    u = torch.rand(400_000, dtype=torch.float64)

    # Each law pinned at the points that fix its shape, so a reparameterisation
    # fails here rather than silently reshaping three samplers at once.
    assert abs(exponential(torch.tensor(0.5)).item() - 0.6931472) < 1e-6
    assert abs(exponential(u).mean().item() - 1.0) < 0.01

    assert abs(laplace(torch.tensor(0.5)).item()) < 1e-12
    assert abs(laplace(u).mean().item()) < 0.02
    assert abs(laplace(u).abs().mean().item() - 1.0) < 0.02

    # chi-squared-1: mean 1, median 0.4549 (the square of the normal's 0.6745),
    # and the divergent density shows up as heavy mass near 0.
    assert abs(chi_squared_1(u).mean().item() - 1.0) < 0.02
    assert abs(chi_squared_1(u).median().item() - 0.4549) < 0.01
    assert chi_squared_1(u).lt(0.01).float().mean().item() > 0.07

    # decade_scale spans exactly the decades asked for, densest at the top.
    scale = decade_scale(u, 3.0)

    assert scale.max().item() <= 1.0 and scale.min().item() >= 1e-3
    assert scale.gt(0.1).float().mean().item() > 0.5

    # truncated_pareto: bounded, log-uniform, and equal mass per decade, so the
    # three decades of [1e-3, 1] must each carry a third of the draw.
    pareto = truncated_pareto(u, 1.0e-3, 1.0)

    assert pareto.min().item() >= 1.0e-3 and pareto.max().item() <= 1.0
    assert abs(truncated_pareto(torch.tensor(0.5), 1.0e-4, 1.0).item() - 1.0e-2) < 1e-9

    for low, high in [(1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 1.0)]:
        share = ((pareto >= low) & (pareto < high)).float().mean().item()

        assert abs(share - 1.0 / 3.0) < 0.01, (low, high, share)

    # Monotone, which is the property the samplers rely on.
    rising = truncated_pareto(torch.linspace(0.0, 1.0, 1000), 1e-3, 50.0)

    assert (rising.diff() > 0).all()
    print("ok")
