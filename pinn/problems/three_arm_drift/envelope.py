"""
The three-arm drift problem's premium cap (docs/three_arm_drift.md section 7).
"""

from __future__ import annotations

import math
import torch

from torch import Tensor

from ...utils import nu2
from ..two_arm_drift.envelope import correction

SQRT2 = math.sqrt(2.0)


def envelope(
    m_b: Tensor,
    m_c: Tensor,
    tau_bb: Tensor,
    tau_bc: Tensor,
    tau_cc: Tensor,
    etahat: Tensor,
) -> Tensor:
    """
    A cap on the premium: the value of being told the right answer for free,
    over and over as it goes stale, discounted.

        u_env = integral_0^inf e^(-t) nu2( m, Sigma_0 + E t ) dt

    That integral has no closed form, but its rate of change splits into three
    pieces, one per pair of arms, and each is (how likely that pair is tied)
    times (how likely that pair is the one that decides it). The first factor
    integrates to two_arm_drift's formula exactly; the second has no formula
    and is bounded by 1, which is what we use. Full derivation, the measured
    looseness, and why the tighter version is not built: section 7 of the doc.

        u_env <= nu2(m, Sigma_0)  +  sum over pairs of correction(...)

    Three things worth knowing before editing:

    The pair precisions are ALREADY three_arm's feature quantities -- 1/Sigma_bb
    is precision_b, 1/Sigma_cc is precision_c, and 1/Var(theta_b - theta_c) is
    precision_bc. Reusing those expressions verbatim rather than inverting T is
    what keeps the etahat = 0 anchor bitwise against the three_arm champion.

    sqrt2 etahat, not etahat: two_arm's eta is a CONTRAST volatility, ours is
    per arm, and two arms wandering independently make their contrast wander
    sqrt2 faster (doc section 0).

    abs on the means: `correction` is even in its mean mathematically, but its
    overflow guard is written for the muhat >= 0 half that two_arm trains on,
    and the unguarded branch overflows in float32. The tie density depends on
    |mean| anyway, so this is the maths, not a workaround.
    """
    det = tau_bb * tau_cc - tau_bc**2
    precision_b = tau_bb - tau_bc**2 / tau_cc
    precision_c = tau_cc - tau_bc**2 / tau_bb
    precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
    correlation = -tau_bc / (tau_bb * tau_cc).sqrt()
    pair_etahat = SQRT2 * etahat

    return (
        nu2(m_b, m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation)
        + correction(m_b.abs(), precision_b, pair_etahat)
        + correction(m_c.abs(), precision_c, pair_etahat)
        + correction((m_b - m_c).abs(), precision_bc, pair_etahat)
    )


if __name__ == "__main__":
    from ..three_arm.sample import Sample

    def bare(draw: Sample) -> Tensor:
        """three_arm's envelope, the etahat = 0 target."""
        det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2
        precision_b = draw.tau_bb - draw.tau_bc**2 / draw.tau_cc
        precision_c = draw.tau_cc - draw.tau_bc**2 / draw.tau_bb

        return nu2(
            draw.m_b,
            draw.m_c,
            precision_b.rsqrt(),
            precision_c.rsqrt(),
            -draw.tau_bc / (draw.tau_bb * draw.tau_cc).sqrt(),
        )

    draw = Sample.draw(4096).fold()
    zero = torch.zeros_like(draw.m_b)
    fields = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)

    # The anchor is an identity, not a tolerance: at etahat = 0 every
    # correction carries an exactly-zero prefactor, so this IS three_arm's
    # envelope and the champion graft is exact.
    assert torch.equal(envelope(*fields, zero), bare(draw))

    wide = Sample(*(field.double() for field in vars(draw).values()))
    wide_fields = (wide.m_b, wide.m_c, wide.tau_bb, wide.tau_bc, wide.tau_cc)

    assert torch.equal(envelope(*wide_fields, torch.zeros_like(wide.m_b)), bare(wide))

    # Drift can only add: nonnegative, finite, above the zero-drift cap, and
    # increasing in etahat.
    previous = bare(draw)
    for etahat in [0.01, 0.1, 1.0, 10.0, 50.0]:
        current = envelope(*fields, torch.full_like(draw.m_b, etahat))

        assert current.isfinite().all(), etahat
        assert (current >= previous - 1e-6).all(), etahat
        previous = current

    # Shuffling the arm labels leaves it alone -- the b <-> c swap, which is
    # what both wall conditions are statements about.
    drift = torch.rand_like(draw.m_b) * 20.0
    swapped = envelope(draw.m_c, draw.m_b, draw.tau_cc, draw.tau_bc, draw.tau_bb, drift)

    assert torch.allclose(envelope(*fields, drift), swapped, rtol=1e-6)

    # The float32 training path: two create_graph derivatives, then backward.
    m_b = draw.m_b.clone().requires_grad_(True)
    bound = envelope(m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, drift)
    first = torch.autograd.grad(bound.sum(), m_b, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), m_b, create_graph=True)[0]
    second.sum().backward()

    assert bound.isfinite().all() and (bound >= 0.0).all()
    assert first.isfinite().all() and second.isfinite().all()
    assert m_b.grad.isfinite().all()
    print("ok")
