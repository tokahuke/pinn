"""The drift problem's free-information envelope (kb/two_arm_drift.md section 7)."""

from __future__ import annotations

import math
import torch

from torch import Tensor

from ...utils import nu

SQRT2 = math.sqrt(2.0)

FLOOR = 1e-20
"""
Floor on etahat inside the reciprocal only (see `correction`). Small enough that the
correction it gates is O(etahat**2) = 1e-40, large enough that root/FLOOR stays finite
in float32.
"""


def correction(muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
    """
    Everything drift adds to the zero-drift bound, on its own, so three_arm_drift can
    call it once per pair of arms without subtracting `nu` back out and rounding
    twice. The closed form, why `erfcx`, the `a < b` branch and why etahat is floored
    inside `a` alone: kb/two_arm_drift.md section 7.
    """
    root = tauhat.rsqrt()
    a, b = root / etahat.clamp_min(FLOOR), muhat / (root * SQRT2)
    excess = a - b
    scale = torch.exp(-(b**2))
    # Both branches are clamped: `torch.where` differentiates both, so the
    # unselected one must not go nan (CLAUDE.md's traps). a < b is that branch, and
    # the product a(a - 2b) beats the equal a^2 - b^2 by four digits.
    tail = torch.where(
        excess >= 0.0,
        scale * torch.special.erfcx(excess.clamp_min(0.0)),
        torch.exp((a * (a - 2.0 * b)).clamp_max(0.0))
        * torch.erfc(excess.clamp_max(0.0)),
    )

    return (etahat / (4.0 * SQRT2)) * (scale * torch.special.erfcx(a + b) + tail)


def envelope(muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
    """
    The discounted perfect-information premium in closed form, `nu` plus `correction`
    (derivation, accuracy and the inherited nu limit: kb/two_arm_drift.md section 7).
    At etahat = 0 the correction is exactly +0.0 and nu is clamped at 0, so this
    returns nu *bitwise*, which is what makes the two_arm champion an exact bootstrap.
    """
    return nu(-muhat, tauhat.rsqrt()) + correction(muhat, tauhat, etahat)


if __name__ == "__main__":
    muhat = torch.linspace(0.0, 4.0, 64)
    tauhat = torch.logspace(-2.0, 1.0, 64)

    # The anchor is an identity, not a tolerance, in both dtypes.
    exact = nu(-muhat, tauhat.rsqrt())

    assert torch.equal(envelope(muhat, tauhat, torch.zeros_like(muhat)), exact)

    wide = torch.linspace(0.0, 4.0, 64, dtype=torch.float64)
    wide_tauhat = torch.logspace(-2.0, 1.0, 64, dtype=torch.float64)

    assert torch.equal(
        envelope(wide, wide_tauhat, torch.zeros_like(wide)),
        nu(-wide, wide_tauhat.rsqrt()),
    )

    # Drift can only add premium: nonnegative, and increasing in etahat from the
    # etahat = 0 bound upward.
    previous = exact
    for etahat in [0.01, 0.1, 0.5, 2.0]:
        current = envelope(muhat, tauhat, torch.full_like(muhat, etahat))

        assert (current >= previous - 1e-7).all(), etahat
        assert (current >= 0.0).all() and current.isfinite().all()
        previous = current

    # The float32 training path: two create_graph derivatives, then backward.
    muhat = torch.rand(256, dtype=torch.float32).requires_grad_(True)
    tauhat = torch.rand(256, dtype=torch.float32) + 0.05
    etahat = torch.rand(256, dtype=torch.float32) * 0.5
    bound = envelope(muhat, tauhat, etahat)
    first = torch.autograd.grad(bound.sum(), muhat, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), muhat, create_graph=True)[0]
    second.sum().backward()

    assert bound.isfinite().all() and (bound >= 0).all()
    assert first.isfinite().all() and second.isfinite().all()
    assert muhat.grad.isfinite().all()
    print("ok")
