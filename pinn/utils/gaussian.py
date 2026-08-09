"""
Gaussian expectations: the free-information envelopes every problem is
built on.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from torch import Tensor


SQRT_2PI = math.sqrt(2.0 * math.pi)

# Genz quadrature rule for the bivariate cdf: Gauss-Legendre on [0, 1], nodes fixed
# at import. Fixed nodes are what make Phi2 an elementary smooth composition
# (exp/sin/cos), so it survives repeated differentiation with create_graph=True.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)
_NODES = 0.5 * (_NODES + 1.0)
_WEIGHTS = 0.5 * _WEIGHTS

# Correlations are clamped off +-1, where asin'(rho) and 1/sqrt(1 - rho**2) both
# blow up. 1e-6 keeps float64 gradients finite; float32 callers should keep
# |rho| <= 1 - 1e-3, below which 1 - rho has lost all but a digit of precision.
_RHO_MAX = 1.0 - 1e-6


def _normal_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x**2) / SQRT_2PI


def _bivariate_ndtr(h: Tensor, k: Tensor, rho: Tensor) -> Tensor:
    """
    Standard bivariate normal cdf P(X <= h, Y <= k) with corr(X, Y) = rho, in the
    Genz/Sheppard single-integral form

        Phi2(h, k; rho) = Phi(h) Phi(k) + 1/(2 pi)
            int_0^{asin rho} exp(-(h**2 + k**2 - 2 h k sin t) / (2 cos(t)**2)) dt

    on 24 fixed Gauss-Legendre nodes. The exponent is never positive (the
    quadratic is at least (h - k)**2), so the tails underflow to 0 instead of
    overflowing, and cos(t)**2 stays bounded away from 0 by the rho clamp.
    """
    h, k, rho = torch.broadcast_tensors(h, k, rho)
    nodes = torch.as_tensor(_NODES, dtype=h.dtype, device=h.device)
    weights = torch.as_tensor(_WEIGHTS, dtype=h.dtype, device=h.device)

    top = torch.asin(rho.clamp(-_RHO_MAX, _RHO_MAX))
    theta = top[..., None] * nodes
    quadratic = (
        h[..., None] ** 2
        + k[..., None] ** 2
        - 2.0 * h[..., None] * k[..., None] * torch.sin(theta)
    )
    integral = (torch.exp(-quadratic / (2.0 * torch.cos(theta) ** 2)) * weights).sum(-1)

    return torch.special.ndtr(h) * torch.special.ndtr(k) + top * integral / (
        2.0 * math.pi
    )


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


def nu2(
    mean_b: Tensor, mean_c: Tensor, stddev_b: Tensor, stddev_c: Tensor, rho: Tensor
) -> Tensor:
    """
    Two-arm version of nu: E[max(0, X, Y)] for (X, Y) bivariate normal with means
    (mean_b, mean_c), standard deviations (stddev_b, stddev_c) and correlation
    rho. Writing h_b = mean_b/stddev_b, h_c = mean_c/stddev_c, and using the
    difference X - Y, which has standard deviation

        a = sqrt((stddev_b - stddev_c)**2 + 2 stddev_b stddev_c (1 - rho)),
        d = (mean_b - mean_c) / a,   root = sqrt(1 - rho**2),

        nu2 = mean_b Phi2(h_b, d; r_b) + mean_c Phi2(h_c, -d; r_c)
              + stddev_b phi(h_b) Phi(g_b) + stddev_c phi(h_c) Phi(g_c)
              + a phi(d) Phi(A)

    with r_b = (stddev_b - rho stddev_c) / a, r_c = (stddev_c - rho stddev_b) / a,
    g_b = (rho h_b - h_c) / root, g_c = (rho h_c - h_b) / root, and

        A = (stddev_c mean_b (stddev_c - rho stddev_b)
             + stddev_b mean_c (stddev_b - rho stddev_c))
            / (stddev_b stddev_c a root).

    The three-arm free-information envelope (docs/three_arm.md section 13). Limits:
    nu(mean_b, stddev_b) as mean_c -> -inf, and max(nu_b, nu_c) <= nu2 <=
    nu_b + nu_c. rho is clamped to +-(1 - 1e-6) inside; see _RHO_MAX.
    """
    rho = rho.clamp(-_RHO_MAX, _RHO_MAX)
    a = torch.sqrt((stddev_b - stddev_c) ** 2 + 2.0 * stddev_b * stddev_c * (1.0 - rho))
    root = torch.sqrt((1.0 - rho) * (1.0 + rho))
    d = (mean_b - mean_c) / a
    h_b, h_c = mean_b / stddev_b, mean_c / stddev_c
    tilt = (
        stddev_c * mean_b * (stddev_c - rho * stddev_b)
        + stddev_b * mean_c * (stddev_b - rho * stddev_c)
    ) / (stddev_b * stddev_c * a * root)
    expectation = (
        mean_b * _bivariate_ndtr(h_b, d, (stddev_b - rho * stddev_c) / a)
        + mean_c * _bivariate_ndtr(h_c, -d, (stddev_c - rho * stddev_b) / a)
        + stddev_b * _normal_pdf(h_b) * torch.special.ndtr((rho * h_b - h_c) / root)
        + stddev_c * _normal_pdf(h_c) * torch.special.ndtr((rho * h_c - h_b) / root)
        + a * _normal_pdf(d) * torch.special.ndtr(tilt)
    )

    # Same float32 cancellation guard as nu: the deep negative tail is exactly 0.
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

    # nu2 at the triple point: three symmetric arms, ratio to two independent
    # one-arm premia is (2 + sqrt(2)) / 4.
    zero, one = torch.zeros(1), torch.ones(1)
    ratio = nu2(zero, zero, one, one, zero) / (2.0 * nu(zero, one))

    assert (ratio - (2.0 + math.sqrt(2.0)) / 4.0).abs().item() < 1e-6, ratio

    # Against Monte Carlo on 200 correlated states.
    n_states = 200
    stddev_b = torch.rand(n_states) + 0.5
    stddev_c = torch.rand(n_states) + 0.5
    mean_b = torch.randn(n_states) * 2.0
    mean_c = torch.randn(n_states) * 2.0
    rho = torch.rand(n_states) * 1.8 - 0.9
    z_b = torch.randn(n_states, 100000)
    z_c = rho[:, None] * z_b + (1.0 - rho[:, None] ** 2).sqrt() * torch.randn(
        n_states, 100000
    )
    monte_carlo = (
        torch.maximum(
            mean_b[:, None] + stddev_b[:, None] * z_b,
            mean_c[:, None] + stddev_c[:, None] * z_c,
        )
        .relu()
        .mean(dim=1)
    )
    errors = (nu2(mean_b, mean_c, stddev_b, stddev_c, rho) - monte_carlo).abs()

    assert errors.mean() < 0.01 and errors.max() < 0.06, (errors.mean(), errors.max())

    # One arm sent to -20 sd recovers the one-arm nu (float64, quadrature-limited).
    stddev_b = torch.rand(200, dtype=torch.float64) + 0.5
    stddev_c = torch.rand(200, dtype=torch.float64) + 0.5
    mean_b = torch.randn(200, dtype=torch.float64) * 2.0
    rho = torch.rand(200, dtype=torch.float64) * 1.8 - 0.9
    dead = nu2(mean_b, -20.0 * stddev_c, stddev_b, stddev_c, rho) - nu(mean_b, stddev_b)

    assert dead.abs().max() < 1e-5, dead.abs().max()

    # Sandwiched between the best single arm and the sum, on 10k float32 states
    # (the tolerance is a couple of float32 ulps at these magnitudes).
    stddev_b = torch.rand(10000) + 0.5
    stddev_c = torch.rand(10000) + 0.5
    mean_b = torch.randn(10000) * 2.0
    mean_c = torch.randn(10000) * 2.0
    rho = torch.rand(10000) * 1.98 - 0.99
    both = nu2(mean_b, mean_c, stddev_b, stddev_c, rho)
    nu_b, nu_c = nu(mean_b, stddev_b), nu(mean_c, stddev_c)

    assert (both >= torch.maximum(nu_b, nu_c) - 2e-6).all()
    assert (both <= nu_b + nu_c + 2e-6).all()
    assert (both >= 0.0).all()

    # Degenerate rho = +-1 stays finite and positive, deep tails are exactly 0.
    edge = nu2(mean_b, mean_c, stddev_b, stddev_c, rho.sign())

    assert torch.isfinite(edge).all() and (edge >= 0.0).all()
    assert (nu2(-50.0 * stddev_b, -50.0 * stddev_c, stddev_b, stddev_c, rho) == 0).all()

    # Twice differentiable with create_graph, then backpropagated, rho at +-1.
    mean_b = (torch.randn(64) * 2.0).requires_grad_(True)
    rho = torch.linspace(-1.0, 1.0, 64)
    premium = nu2(
        mean_b, torch.randn(64), torch.rand(64) + 0.5, torch.rand(64) + 0.5, rho
    )
    first = torch.autograd.grad(premium.sum(), mean_b, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), mean_b, create_graph=True)[0]
    second.sum().backward()

    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert torch.isfinite(mean_b.grad).all()
    print("ok")
