"""
Models for the three-arm problem: the premium net and the value wrapper that
carries the commit envelope's kinks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from ...net import GainedTanh
from ...utils import nu

# Width of the feature stack in ExplorationPremium.forward.
FEATURE_COUNT = 15


class ExplorationPremium(nn.Module):
    """
    The premium net over the fundamental wedge.

    Plain function on the wedge, no symmetry machinery (fold + wall losses
    carry S3); the feature list in forward; one GainedTanh MLP over the
    stacked features, Xavier-tanh init, shallow profile; the free-information
    envelope of doc section 13 multiplying the relu-squared rational response
    (relu(r)**2 / (1 + relu(r)**2)).
    Warm-starting from the two_arm champion is deferred to v2 by the
    one-variable-at-a-time rule.
    """

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            # Xavier with the tanh gain (the PyTorch default is relu-flavored
            # and too small for tanh); plain gain for the linear head.
            head = i == len(sizes) - 2
            nn.init.xavier_uniform_(
                linear.weight, gain=1.0 if head else nn.init.calculate_gain("tanh")
            )

            # Head bias 1: start alive everywhere. An all-dead relu**2 net
            # has zero loss gradient and never recovers (CLAUDE.md traps).
            if head is True:
                nn.init.ones_(linear.bias)
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        # Linear head (final GainedTanh sliced off): the response is mapped
        # through relu(r)**2 / (1 + relu(r)**2) in forward, which both
        # confines u inside the proven bound and builds in the free-boundary
        # regularity.
        self.net = nn.Sequential(*layers[:-1])

        # Envelope scale, init 0: at scale exactly 1 the envelope is a proven
        # upper bound on the true premium (doc section 13), so the whole
        # solution starts inside the tanh range.
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        # Marginal precision per contrast, via Schur complements of T. NOT the
        # raw diagonals: tau_bb is the precision GIVEN the other contrast --
        # conditioning you do not have -- and overstates certainty exactly in
        # the correlated states where the shared control matters. The b-c
        # denominator is the sum of the a-b and a-c pair coordinates, positive
        # on reachable states (harmonic combination when tau_bc = 0: verified
        # 1/(1/2 + 1/3) = 6/5 in-session).
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
        m_bc = m_b - m_c

        features = torch.stack(
            [
                # Raw anchors, as in two_arm.
                m_b,
                m_c,
                tau_bb,
                tau_bc,
                tau_cc,
                # Logs of every positive scale: the two_arm log-tauhat win,
                # once per contrast.
                precision_b.log(),
                precision_c.log(),
                precision_bc.log(),
                # z-scores: each mean in units of its own marginal sd -- the
                # two_arm corridor-alignment win, once per contrast.
                m_b * precision_b.sqrt(),
                m_c * precision_c.sqrt(),
                m_bc * precision_bc.sqrt(),
                # Tail coordinates: far-field boundary level sets (two_arm's
                # G ~ 1/(2 tauhat) law), once per contrast.
                m_b * precision_b,
                m_c * precision_c,
                m_bc * precision_bc,
                # The one cross-pair coordinate: Pearson correlation of the
                # two contrasts' beliefs, in [0, 1) on reachable states;
                # r = 0 is the decoupled two_arm limit.
                -tau_bc / (tau_bb * tau_cc).sqrt(),
            ],
            dim=-1,
        )

        response = self.net(features).squeeze(-1)

        # The free-information envelope (doc section 13): what learning could
        # possibly be worth, per challenger contrast. Decays in every far
        # field. NOT tight at the triple point (corrected 2026-08-05): the
        # max-splitting slack there is 1.17x to 1.87x with correlation, so
        # the response must learn ~0.85 down to ~0.53 at startup, not 1.
        envelope = self.log_scale.exp() * (
            nu(m_b, precision_b.rsqrt()) + nu(m_c, precision_c.rsqrt())
        )

        # y/(1+y) with y = relu(r)**2 maps the response into [0, 1): both
        # proven properties hold by construction, 0 <= u < envelope. The
        # squared relu is the free-boundary trick (see two_arm's
        # ExplorationPremium): committing all traffic to the leader observes
        # no contrast, so v = commit solves the PDE exactly in the deep wedge
        # and the true premium is exactly 0 there, pasting with u = u_m = 0
        # and a jump only in the second derivative. relu(r)**2 has exactly
        # that regularity on the learned zero set of r; a plain clip
        # (first-derivative kink) would break the smooth pasting. Rational
        # saturation, NOT tanh(y): tanh is float32-exactly 1 beyond r ~ 2.5
        # and its gradient underflows, a cliff with no way back (the first
        # three_arm start died there, r ~ 14 everywhere, only log_scale left
        # trainable). y/(1+y) saturates with a polynomial tail, so gradients
        # survive any overshoot.
        response_squared = torch.relu(response) ** 2

        return envelope * response_squared / (1.0 + response_squared)


class ValueFunction(nn.Module):
    """
    Value on top of the premium: v = max(0, m_b, m_c) + u. The commit envelope
    carries all three kinks (hand-written relu-of-max); the premium net stays
    smooth, exactly the two_arm division of labor. Units rho = sigma = 1,
    which is the dimensionless form (doc section 11).
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        commit = torch.relu(torch.maximum(m_b, m_c))

        return commit + self.premium(m_b, m_c, tau_bb, tau_bc, tau_cc)


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return torch.zeros_like(m_b)

    m_b, m_c = torch.randn(100), torch.randn(100)
    taus = (torch.rand(100) + 0.5, -torch.rand(100) * 0.3, torch.rand(100) + 0.5)
    v = ValueFunction(_ZeroPremium())(m_b, m_c, *taus)

    assert torch.allclose(v, torch.relu(torch.maximum(m_b, m_c)))

    # The premium runs end to end on wedge states, stays finite, and at
    # log_scale = 0 it obeys both proven properties: 0 <= u <= envelope.
    from .sample import Sample

    premium = ExplorationPremium([32, 16])
    draw = Sample.draw(1000).fold()
    state = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    u = premium(*state)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == draw.m_b.shape and u.isfinite().all()

    precision_b = draw.tau_bb - draw.tau_bc**2 / draw.tau_cc
    precision_c = draw.tau_cc - draw.tau_bc**2 / draw.tau_bb
    bound = nu(draw.m_b, precision_b.rsqrt()) + nu(draw.m_c, precision_c.rsqrt())

    assert (u >= 0).all() and (u <= bound + 1e-6).all()
    print("ok")
