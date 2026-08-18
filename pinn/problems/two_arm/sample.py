"""
Collocation samplers for the two-arm problem: long-tailed interior draws and
the ridge family.
"""

from __future__ import annotations

import torch

from torch import Tensor
from torch.quasirandom import SobolEngine

from ...utils import decade_scale, exponential

PRIOR_FLOOR = 1e-3
"""
Keeps `tauhat` off the singular corner: numerical stability *only*, with no prior
baked in, since the net trains general down to priors ~30 sd wide. No real experiment
starts more agnostic than that, and each decade below costs another 100x in PDE
stiffness (`1/tauhat**2` in the residual) for territory nobody visits.
"""

SCALE_DECADES = 3.0
"""
The log10 range of the scale that spreads `tauhat` mass across decades. An Exp tail
alone would leave the low decades unsampled.
"""

_SOBOL = SobolEngine(dimension=3, scramble=True)
"""The one scrambled sequence every interior draw continues."""


def _tauhat(u_scale: Tensor, u_tail: Tensor) -> Tensor:
    """
    The tauhat law: floor + decade-spread scale times an Exp(mean 2) tail,
    densest around 1 and reaching the floor with real per-decade mass.
    """
    scale = decade_scale(u_scale, SCALE_DECADES)

    return PRIOR_FLOOR + 2.0 * scale * exponential(u_tail)


def sample_sobol(n: int) -> tuple[Tensor, Tensor]:
    """
    Scrambled Sobol through the long-tail law: `tauhat` from `_tauhat`, then
    `muhat ~ Exp(mean 2 / sqrt(tauhat))`, so the cloud tracks the corridor at every
    information level. Successive calls continue one sequence, refining coverage.
    """
    t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1e-7)
    tauhat = _tauhat(t[:, 0], t[:, 1])
    muhat = (2.0 / tauhat.sqrt()) * exponential(t[:, 2])

    return muhat, tauhat


def sample_ridge(n: int) -> Tensor:
    """
    Draw n ridge points (muhat = 0 implied), tauhat from the same law as the
    interior draw.
    """
    return _tauhat(torch.rand(n), torch.rand(n).clamp(max=1.0 - 1e-7))


if __name__ == "__main__":
    muhat, tauhat = sample_sobol(100_000)

    assert muhat.shape == tauhat.shape == (100_000,)
    assert (muhat > 0).all() and (tauhat >= PRIOR_FLOOR).all()
    assert (sample_ridge(100) >= PRIOR_FLOOR).all()

    # The decades are actually covered: real mass near 1, near the floor, and in the
    # far tail, where the log-spread scale only shrinks and leaves ~1e-3 beyond 10.
    for low, high, at_least in [(1e-3, 1e-2, 0.03), (0.5, 2.0, 0.1), (10.0, 1e9, 5e-4)]:
        fraction = ((tauhat > low) & (tauhat < high)).float().mean().item()

        assert fraction > at_least, (low, high, fraction)
    print("ok")
