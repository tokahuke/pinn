"""
The drift problem's free-information envelope (docs/two_arm_drift.md section 7).
"""

from __future__ import annotations

import math
import torch

from torch import Tensor

from ...utils import nu

SQRT2 = math.sqrt(2.0)

# Floor on etahat inside the reciprocal only; see forward. Small enough that
# the correction it gates is O(etahat**2) = 1e-40, large enough that
# root/FLOOR stays finite in float32.
FLOOR = 1e-20


def correction(muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
    """
    Everything drift adds to the zero-drift bound, on its own:

        (etahat/(4 sqrt 2)) exp(-b^2) [ erfcx(a + b) + erfcx(a - b) ]

    Exposed separately because three_arm_drift needs it three times, once per
    pair of arms (docs/three_arm_drift.md section 7), and subtracting nu back
    out of `envelope` would round twice and lose the exact zero at etahat = 0.

    Three things the code cannot be read safely without:

    erfcx, not erfc: the raw form carries exp(1/(tauhat etahat^2)), exponent
    1e8 at the corner of the domain.

    The a < b branch: there erfcx of a negative argument overflows while
    exp(-b^2) underflows, giving 0 * inf = nan. The algebraically identical
    exp(a(a - 2b)) erfc(a - b) has a negative exponent exactly on that branch.
    Both sides are clamped so the unselected one cannot poison the backward
    pass (torch.where differentiates both -- CLAUDE.md traps), and the product
    a(a - 2b) beats the equal a^2 - b^2 by four digits.

    etahat is floored ONLY inside a, never in the prefactor: at etahat = 0 the
    prefactor is exactly 0 so the whole term is exactly 0, while a stays finite
    so the BACKWARD pass does too (a = root/0 is inf, whose forward survives
    but whose derivative is inf * 0 = nan -- found by policy() returning
    allocations outside [0, 1]).
    """
    root = tauhat.rsqrt()
    a, b = root / etahat.clamp_min(FLOOR), muhat / (root * SQRT2)
    excess = a - b
    scale = torch.exp(-(b**2))
    tail = torch.where(
        excess >= 0.0,
        scale * torch.special.erfcx(excess.clamp_min(0.0)),
        torch.exp((a * (a - 2.0 * b)).clamp_max(0.0))
        * torch.erfc(excess.clamp_max(0.0)),
    )

    return (etahat / (4.0 * SQRT2)) * (scale * torch.special.erfcx(a + b) + tail)


def envelope(muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
    """
    The discounted perfect-information premium

        u_env = integral_0^inf e^(-t) nu( -muhat, sqrt(1/tauhat + etahat^2 t) ) dt

    in closed form (derivation, citation, accuracy and the inherited nu limit:
    docs/two_arm_drift.md section 7). With

        a = 1/(etahat sqrt(tauhat)),   b = muhat sqrt(tauhat/2)

        u_env = nu(-muhat, tauhat**(-1/2))
              + (etahat/(4 sqrt 2)) exp(-b^2) [ erfcx(a + b) + erfcx(a - b) ]

    The second term is `correction` above. At etahat = 0 it is exactly +0.0 and
    nu is clamped at 0, so this returns nu BITWISE, not to a tolerance.
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

    # Drift can only add premium: nonnegative, and increasing in etahat from
    # the etahat = 0 bound upward.
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
