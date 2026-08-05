"""
Small shared math utilities, problem-agnostic.
"""

from __future__ import annotations

import math
import torch

from torch import Tensor


def nu(mean: Tensor, stddev: Tensor) -> Tensor:
    """
    Expected positive part of a Gaussian: E[max(0, X)] for X ~ N(mean,
    stddev**2),

        nu = mean * Phi(mean/stddev) + stddev * phi(mean/stddev)

    with Phi, phi the standard normal cdf and density. The building block of
    the free-information premium envelopes (docs/three_arm.md section 13).
    Limits: 0 as mean -> -inf, stddev/sqrt(2 pi) at mean = 0, mean as
    mean -> +inf.
    """
    z = mean / stddev
    expectation = mean * torch.special.ndtr(z) + stddev * torch.exp(
        -0.5 * z**2
    ) / math.sqrt(2.0 * math.pi)

    # nu >= 0 exactly; float32 cancellation in the deep negative tail can
    # return ~ -1e-8, which downstream positivity guarantees inherit.
    return expectation.clamp_min(0.0)


if __name__ == "__main__":
    stddev = torch.rand(1000) + 0.5
    mean = torch.randn(1000) * 2.0

    # Exact at the three limits.
    assert torch.allclose(
        nu(torch.zeros(5), stddev[:5]),
        stddev[:5] / math.sqrt(2.0 * math.pi),
        atol=1e-6,
    )
    assert torch.allclose(nu(20.0 * stddev, stddev), 20.0 * stddev, atol=1e-4)
    assert (nu(-20.0 * stddev, stddev).abs() < 1e-6).all()
    assert (nu(torch.randn(10000) * 50.0, torch.rand(10000) + 0.1) >= 0).all()

    # Against Monte Carlo, and always at least the positive part of the mean.
    torch.manual_seed(0)
    draws = mean[:, None] + stddev[:, None] * torch.randn(1000, 20000)
    monte_carlo = draws.relu().mean(dim=1)
    errors = (nu(mean, stddev) - monte_carlo).abs()

    assert errors.mean() < 0.01 and errors.max() < 0.06, (errors.mean(), errors.max())
    assert (nu(mean, stddev) >= mean.relu() - 1e-6).all()
    print("ok")
