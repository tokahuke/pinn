"""The three-arm drift problem's premium cap (kb/three_arm_drift.md section 7)."""

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
    A cap on the premium: the value of being told the right answer for free, over
    and over as it goes stale, discounted. `nu2` at the current belief plus one
    closed-form correction per pair. Derivation, the measured looseness, and why the
    tight version is not built: doc section 7.
    """
    # Spelled as three_arm's feature quantities rather than by inverting T: reusing
    # those expressions verbatim is what keeps the etahat = 0 anchor bitwise against
    # the three_arm champion.
    det = tau_bb * tau_cc - tau_bc**2
    precision_b = tau_bb - tau_bc**2 / tau_cc
    precision_c = tau_cc - tau_bc**2 / tau_bb
    precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
    correlation = -tau_bc / (tau_bb * tau_cc).sqrt()

    # sqrt2, because two_arm's eta is a *contrast* volatility and ours is per arm:
    # two arms wandering independently make their contrast wander that much faster
    # (doc section 0).
    pair_etahat = SQRT2 * etahat

    # abs on the means: `correction` is even in its mean mathematically, but its
    # overflow guard is written for the muhat >= 0 half two_arm trains on, and the
    # unguarded branch overflows in float32. The tie density depends on |mean| anyway.
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

    # The anchor is an identity, not a tolerance: at etahat = 0 every correction
    # carries an exactly-zero prefactor, so this *is* three_arm's envelope and the
    # champion graft is exact.
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

    # Shuffling the arm labels leaves it alone: the b <-> c swap, which is what both
    # wall conditions are statements about.
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
