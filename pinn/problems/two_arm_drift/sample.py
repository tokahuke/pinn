"""
Collocation samplers for the drift problem: two_arm's interior and ridge laws
plus the drift coordinate.
"""

from __future__ import annotations

import torch

from torch import Tensor
from torch.quasirandom import SobolEngine

from ...utils import decade_scale, exponential

# tauhat law: two_arm's, scaled by the ceiling 1/(2 etahat) and clipped there.
# dtauhat/dt = rho[design - etahat^2 tauhat^2], design <= 1/4, so the ceiling
# is one-way: never crossed from below. Floor is numerics, see two_arm.
PRIOR_FLOOR = 1e-3
SCALE_DECADES = 3.0

# etahat law: tauhat's shape, reaching 0 (the two_arm anchor). etahat/2 is the
# truth's wander over one discount time divided by the best measurement of it,
# so deployment sits near 10, not 0.1; decades because nothing pins it tighter.
ETAHAT_SCALE = 20.0
ETAHAT_DECADES = 4.0

# Past this the ceiling 1/(2 etahat) nears PRIOR_FLOOR and tauhat degenerates.
ETAHAT_MAX = 50.0

# The ceiling diverges as etahat -> 0, so the coupled law needs its own bound.
TAUHAT_MAX = 50.0

_SOBOL = SobolEngine(dimension=5, scramble=True)


def _tauhat(u_scale: Tensor, u_tail: Tensor, etahat: Tensor) -> Tensor:
    """
    two_arm's tauhat law scaled by the ceiling 1/(2 etahat) and clipped there:
    the clipped mass lands ON the ceiling, which is the attractor. Below
    etahat = 1/(2 TAUHAT_MAX) the ceiling stops binding and the cap takes over.
    """
    scale = decade_scale(u_scale, SCALE_DECADES)
    ceiling = 0.5 / etahat.clamp_min(0.5 / TAUHAT_MAX)
    drawn = (scale * exponential(u_tail)).clamp(max=1.0) * ceiling

    # clamp_min, not PRIOR_FLOOR + ...: an addend would carry tauhat over the
    # ceiling once etahat is large. Keeps 2 etahat tauhat <= 1 exact.
    return drawn.clamp_min(PRIOR_FLOOR)


def _etahat(u_scale: Tensor, u_tail: Tensor) -> Tensor:
    """
    Decade-spread scale times an Exp tail, tauhat's shape at ETAHAT_SCALE.
    """
    drawn = ETAHAT_SCALE * decade_scale(u_scale, ETAHAT_DECADES) * exponential(u_tail)

    return drawn.clamp(max=ETAHAT_MAX)


def sample_sobol(n: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Scrambled Sobol points pushed through the laws: etahat from _etahat first,
    then tauhat from _tauhat as a fraction of the ceiling it implies, then
    muhat ~ Exp(mean 2 / sqrt(tauhat)) given tauhat (the cloud tracks the
    corridor at every information level). Successive calls continue one
    sequence, so coverage keeps refining across iterations.
    """
    t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1e-7)
    etahat = _etahat(t[:, 3], t[:, 4])
    tauhat = _tauhat(t[:, 0], t[:, 1], etahat)
    muhat = (2.0 / tauhat.sqrt()) * exponential(t[:, 2])

    return muhat, tauhat, etahat


def sample_ridge(n: int) -> tuple[Tensor, Tensor]:
    """
    Ridge points (muhat = 0 implied): tauhat and etahat from the same laws.
    BC1 holds at every etahat, so the drift coordinate is sampled here too.
    """
    etahat = _etahat(torch.rand(n), torch.rand(n))

    return _tauhat(torch.rand(n), torch.rand(n), etahat), etahat


if __name__ == "__main__":
    muhat, tauhat, etahat = sample_sobol(100_000)

    assert muhat.shape == tauhat.shape == etahat.shape == (100_000,)
    assert (muhat > 0).all() and (tauhat >= PRIOR_FLOOR).all()
    assert (etahat >= 0.0).all() and etahat.isfinite().all()

    ridge_tauhat, ridge_etahat = sample_ridge(1000)

    assert (ridge_tauhat >= PRIOR_FLOOR).all()
    assert (ridge_etahat >= 0.0).all()

    # Every tauhat decade the ceiling allows; the top ones thin as etahat grows.
    for low, high, at_least in [(1e-3, 1e-2, 0.1), (1e-2, 0.5, 0.2), (0.5, 1e9, 0.1)]:
        fraction = ((tauhat > low) & (tauhat < high)).float().mean().item()

        assert fraction > at_least, (low, high, fraction)

    # The coupling, on the PHYSICAL ratio: no offset subtracted off the left
    # side, which would only test that clamp clamps. 1e-6 is float32 slack.
    for name, tau, eta in [
        ("interior", tauhat, etahat),
        ("ridge", ridge_tauhat, ridge_etahat),
    ]:
        ratio = 2.0 * eta * tau

        assert (tau <= TAUHAT_MAX).all(), (name, tau.max().item())
        assert (eta <= ETAHAT_MAX).all(), (name, eta.max().item())
        assert (ratio <= 1.0 + 1e-6).all(), (name, ratio.max().item())
        assert (ratio > 0.99).float().mean().item() > 0.06, name
        assert ((ratio > 0.3) & (ratio < 0.99)).float().mean().item() > 0.12, name

    # Deployment (etahat ~ 10.7) and the two_arm anchor both carry real mass.
    assert (etahat < 0.01).float().mean().item() > 0.08
    assert ((etahat > 3.0) & (etahat < 30.0)).float().mean().item() > 0.2
    assert (etahat > 10.67).float().mean().item() > 0.1
    assert etahat.min().item() < 1e-4

    print("ok")
