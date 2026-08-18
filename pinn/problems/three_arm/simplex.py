"""
Maximize a quadratic function of two variables over a triangle. Plain
calculus; this module knows nothing about arms, beliefs, or networks.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import Tensor


@dataclass
class Maximum:
    """A quadratic's maximum over the triangle, and the point where it sits."""

    value: Tensor
    """The largest value f takes on the triangle."""

    x: Tensor
    """First coordinate of the point attaining it."""

    y: Tensor
    """Second coordinate of the point attaining it."""


def maximize_quadratic(
    c_xx: Tensor, c_yy: Tensor, c_xy: Tensor, c_x: Tensor, c_y: Tensor
) -> Maximum:
    """
    Maximize f(x, y) = c_xx x^2 + c_yy y^2 + c_xy x y + c_x x + c_y y over the
    triangle {x >= 0, y >= 0, x + y <= 1}, elementwise over batched coefficients.
    The max sits at the interior stationary point, at an edge's, or at a corner
    (doc section 10): evaluate all seven and take the biggest, so saddles lose.
    """

    def f(x: Tensor, y: Tensor) -> Tensor:
        """The quadratic itself, elementwise over the batch."""
        return c_xx * x**2 + c_yy * y**2 + c_xy * x * y + c_x * x + c_y * y

    def finite(denominator: Tensor) -> Tensor:
        """
        The denominator with its exact zeros replaced by 1, which only prevents
        literal division by zero. Near-degenerate denominators still pass and
        produce garbage candidates; correctness does not rest here, but on the
        candidate stack below.
        """
        return denominator.masked_fill(denominator.abs() < 1e-12, 1.0)

    zeros = torch.zeros_like(c_x)
    ones = torch.ones_like(c_x)

    # Interior stationary point: grad f = 0, a 2x2 linear solve. Clamped so
    # dead-branch evaluation stays finite; degenerate or outside points are
    # masked below.
    det = 4.0 * c_xx * c_yy - c_xy**2
    x_interior = (c_xy * c_y - 2.0 * c_yy * c_x) / finite(det)
    y_interior = (c_xy * c_x - 2.0 * c_xx * c_y) / finite(det)
    feasible = (
        (x_interior >= 0)
        & (y_interior >= 0)
        & (x_interior + y_interior <= 1)
        & (det.abs() >= 1e-12)
    )
    x_interior = x_interior.clamp(0.0, 1.0)
    y_interior = y_interior.clamp(0.0, 1.0)

    # Edge stationary points: two legs are 1-D parabolas in one variable; the
    # hypotenuse x + y = 1 is a 1-D parabola in x after substitution.
    x_leg = (-c_x / (2.0 * finite(c_xx))).clamp(0.0, 1.0)
    y_leg = (-c_y / (2.0 * finite(c_yy))).clamp(0.0, 1.0)
    x_hyp = (
        (2.0 * c_yy - c_xy - c_x + c_y) / (2.0 * finite(c_xx + c_yy - c_xy))
    ).clamp(0.0, 1.0)

    # Candidates: interior, 3 edges, 3 corners, all through f. A degraded
    # candidate loses the argmax rather than inflating it, and gather's backward
    # keeps its gradients off the parameters (kb section 19.6).
    x_candidates = torch.stack([x_interior, x_leg, zeros, x_hyp, zeros, ones, zeros])
    y_candidates = torch.stack(
        [y_interior, zeros, y_leg, 1.0 - x_hyp, zeros, zeros, ones]
    )
    values = f(x_candidates, y_candidates)
    values[0] = torch.where(feasible, values[0], torch.full_like(c_x, -torch.inf))

    best = values.argmax(dim=0, keepdim=True)

    return Maximum(
        value=values.gather(0, best).squeeze(0),
        x=x_candidates.gather(0, best).squeeze(0),
        y=y_candidates.gather(0, best).squeeze(0),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    c_xx, c_yy, c_xy, c_x, c_y = (2.0 * torch.randn(300) for _ in range(5))
    best = maximize_quadratic(c_xx, c_yy, c_xy, c_x, c_y)

    assert (best.x >= 0).all() and (best.y >= 0).all()
    assert (best.x + best.y <= 1 + 1e-6).all()

    # Brute force sandwich: the grid is a subset of the triangle, so the
    # closed form must beat every grid point and stay within gridding error
    # of the grid max.
    axis = torch.linspace(0.0, 1.0, 201)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    on_triangle = (grid_x + grid_y) <= 1.0
    grid_x, grid_y = grid_x[on_triangle], grid_y[on_triangle]
    grid_values = (
        c_xx[:, None] * grid_x[None, :] ** 2
        + c_yy[:, None] * grid_y[None, :] ** 2
        + c_xy[:, None] * grid_x[None, :] * grid_y[None, :]
        + c_x[:, None] * grid_x[None, :]
        + c_y[:, None] * grid_y[None, :]
    )
    brute = grid_values.max(dim=1).values

    assert (best.value >= brute - 1e-5).all()
    assert (best.value <= brute + 0.05).all()
    print("ok")
