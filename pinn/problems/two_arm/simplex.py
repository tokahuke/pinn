"""
Maximize a quadratic function of one variable over the interval [0, 1] (the
1-simplex in coordinates, as three_arm's triangle is the 2-simplex). Plain
calculus; this module knows nothing about arms, beliefs, or networks.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import Tensor


@dataclass
class Maximum:
    """
    The max value of f over [0, 1] and the point x attaining it.
    """

    value: Tensor
    x: Tensor


def maximize_quadratic(c_xx: Tensor, c_x: Tensor) -> Maximum:
    """
    Maximize the quadratic

        f(x) = c_xx x^2 + c_x x

    over the interval [0, 1], elementwise over batched coefficients.

    The max can only sit at the interior stationary point or at an endpoint.
    Evaluate all three candidates and take the biggest: no case analysis on
    curvature signs, convex vertices simply lose. Every candidate is clamped
    into the interval before evaluation, so a garbage vertex is a valid lower
    bound that loses the argmax; it can never inflate the answer. Selection is
    by gather, whose backward zeroes every unselected row.
    """

    def f(x: Tensor) -> Tensor:
        return c_xx * x**2 + c_x * x

    def finite(denominator: Tensor) -> Tensor:
        # Only prevents literal division by zero; near-degenerate denominators
        # still pass and produce a garbage vertex, which the clamp-and-lose
        # argument above absorbs.
        return denominator.masked_fill(denominator.abs() < 1e-12, 1.0)

    x_vertex = (-c_x / (2.0 * finite(c_xx))).clamp(0.0, 1.0)
    x_candidates = torch.stack([x_vertex, torch.zeros_like(c_x), torch.ones_like(c_x)])
    values = f(x_candidates)

    best = values.argmax(dim=0, keepdim=True)

    return Maximum(
        value=values.gather(0, best).squeeze(0),
        x=x_candidates.gather(0, best).squeeze(0),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    c_xx, c_x = 2.0 * torch.randn(300), 2.0 * torch.randn(300)
    best = maximize_quadratic(c_xx, c_x)

    assert (best.x >= 0).all() and (best.x <= 1).all()

    # Brute force sandwich: the grid is a subset of the interval, so the closed
    # form must beat every grid point and stay within gridding error of the
    # grid max.
    axis = torch.linspace(0.0, 1.0, 401)
    grid_values = c_xx[:, None] * axis[None, :] ** 2 + c_x[:, None] * axis[None, :]
    brute = grid_values.max(dim=1).values

    assert (best.value >= brute - 1e-6).all()
    assert (best.value <= brute + 0.01).all()
    print("ok")
