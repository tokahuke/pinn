"""
Models for the three-arm problem: the premium net and the value wrapper that
carries the commit envelope's kinks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import DeclaresTopology, DimensionlessValue, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ...utils import nu2
from .sample import Sample
from .simplex import Maximum, maximize_quadratic

FEATURE_COUNT = 15
"""Width of the feature stack that `ExplorationPremium.forward` builds."""


class ExplorationPremium(DeclaresTopology):
    """
    The premium net over the fundamental wedge.

    Plain function on the wedge, no symmetry machinery (fold + wall losses carry
    S3); the feature list in forward; one GainedTanh MLP over the stacked
    features, Xavier-tanh init, shallow profile; the free-information envelope of
    doc section 13 multiplying the rational response of kb section 19.1.
    """

    def __init__(self, hidden: list[int], kinks: int = 0) -> None:
        super().__init__(FEATURE_COUNT, hidden, kinks)

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

        # Linear head, the final GainedTanh sliced off: forward maps the
        # response through the rational squared relu (kb section 19.1).
        self.net = nn.Sequential(*layers[:-1])

        # Movable curvature-jump primitives for the free-boundary junction, each
        # bounded so it cannot colonize the smooth bulk, and zero-init so a graft
        # is bit-exact at step 0 (kb section 19.2).
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)

            # Alive start: a relu**2 unit that is never active never recovers,
            # and default init left 3 of 8 dead (kb section 19.2).
            nn.init.constant_(self.kink_in.bias, 0.5)

        # Envelope scale, init 0: at scale exactly 1 the envelope is a proven
        # upper bound (doc section 13), so the solution starts inside the tanh
        # range.
        self.log_scale = nn.Parameter(torch.zeros(()))

        # A fixed per-feature scale, calibrated from the sampling law once at
        # init and kept as a buffer so checkpoints carry it (kb section 19.3).
        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        draw = Sample.draw(4096).fold()
        self.feature_scale = (
            self._features(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)[0]
            .std(dim=0)
            .clamp_min(1e-3)
        )

    def stitch(self, source: dict) -> None:
        """
        Adopt a smooth premium's parameters into this (possibly kinked) net.

        A source without a kink branch keeps this net's zero-init one, so the
        graft is bit-exact at step 0; anything else missing fails loudly.
        """
        state = dict(source)
        mine = self.state_dict()

        if self.kinks > 0:
            for name in (
                "kink_in.weight",
                "kink_in.bias",
                "kink_out.weight",
                "kink_out.bias",
            ):
                state.setdefault(name, mine[name])
        self.load_state_dict(state)

    def _features(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        The raw (uncalibrated) feature stack, plus the marginal precisions and
        correlation the envelope reuses.
        """
        # Marginal precision per contrast, via Schur complements and **not** the
        # raw diagonals, which are precisions *given* the other contrast and so
        # overstate certainty exactly where the shared control matters.
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
        m_bc = m_b - m_c
        correlation = -tau_bc / (tau_bb * tau_cc).sqrt()

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
                # z-scores: each mean in units of its own marginal sd, which is
                # the two_arm corridor-alignment win, once per contrast.
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
                correlation,
            ],
            dim=-1,
        )

        return features, precision_b, precision_c, correlation

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        features, precision_b, precision_c, correlation = self._features(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        scaled = features / self.feature_scale
        response = self.net(scaled).squeeze(-1)

        if self.kinks > 0:
            bumps = torch.relu(self.kink_in(scaled)) ** 2
            response = response + self.kink_out(bumps / (1.0 + bumps)).squeeze(-1)

        # The **exact** free-information value E[max(0, theta_b, theta_c)]: a
        # proven upper bound and the exact startup solution, so the floor decades
        # ask nothing of the net (doc section 13, kb section 19.1).
        envelope = self.log_scale.exp() * nu2(
            m_b, m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
        )

        # y/(1+y) with y = relu(r)**2 maps the response into [0, 1), so
        # 0 <= u < envelope holds by construction and the free boundary keeps its
        # curvature-jump regularity. Not tanh: kb section 19.1.
        response_squared = torch.relu(response) ** 2

        return envelope * response_squared / (1.0 + response_squared)


class DimensionlessValueFunction(DimensionlessValue):
    """
    The thing training grades: v = max(0, m_b, m_c) + u. The commit envelope
    carries all three kinks (hand-written relu-of-max) while the premium net
    stays smooth, exactly the two_arm division of labor. Units rho = sigma = 1,
    which is the dimensionless form (doc section 11), and the premium is only
    trained on the fundamental wedge. Deployment goes through `ValueFunction`,
    which papers over both cuts.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path, kinks: int = 0) -> Self:
        """
        A trained checkpoint as a model, at the architecture the checkpoint
        DECLARES (`read_topology`, never inferred from weight shapes). `kinks`
        only sizes a first-time stitch onto a checkpoint that has none.
        """
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        value = cls(ExplorationPremium(hidden, kinks=kinks))
        value.load_state_dict(state)

        return value

    def bind(self, rho: float, sigma: float) -> ValueFunction:
        """
        This net tied to one experiment, which is what makes it usable: means
        and precisions go in and come back in your own units.
        """
        return ValueFunction(self, rho, sigma)

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        commit = torch.relu(torch.maximum(m_b, m_c))

        return commit + self.premium(m_b, m_c, tau_bb, tau_bc, tau_cc)

    def hamiltonian(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Tensor, Maximum, tuple[Tensor, Tensor, Tensor]]:
        """
        The HJB's two sides on wedge states: the value v and the maximized
        Hamiltonian (doc section 10), plus the pairwise learning numbers, which
        carry the Hamiltonian's Hessian and so its concavity. One derivative
        chain, graph-connected to the premium, serves loss and policy alike.
        """
        m_b = m_b.detach().requires_grad_(True)
        m_c = m_c.detach().requires_grad_(True)
        tau_bb = tau_bb.detach().requires_grad_(True)
        tau_bc = tau_bc.detach().requires_grad_(True)
        tau_cc = tau_cc.detach().requires_grad_(True)

        v = self(m_b, m_c, tau_bb, tau_bc, tau_cc)
        v_mb, v_mc, v_tbb, v_tbc, v_tcc = torch.autograd.grad(
            v.sum(),
            [m_b, m_c, tau_bb, tau_bc, tau_cc],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        v_mbmb, v_mbmc = torch.autograd.grad(
            v_mb.sum(),
            [m_b, m_c],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        (v_mcmc,) = torch.autograd.grad(
            v_mc.sum(),
            [m_c],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

        det = tau_bb * tau_cc - tau_bc**2

        def mean_diffusion(d_b: Tensor, d_c: Tensor) -> Tensor:
            """The Ito term (1/2) d' D2m(v) d along one pair direction."""
            return 0.5 * (d_b**2 * v_mbmb + 2.0 * d_b * d_c * v_mbmc + d_c**2 * v_mcmc)

        l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + v_tbb
        l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + v_tcc
        l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
            v_tbb + v_tcc - v_tbc
        )

        return (
            v,
            maximize_quadratic(
                -l_ab, -l_ac, l_bc - l_ab - l_ac, m_b + l_ab, m_c + l_ac
            ),
            (l_ab, l_ac, l_bc),
        )

    def policy(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        """
        The argmax allocation in wedge roles, shape (n, 3) rows
        (alpha_a, alpha_b, alpha_c). Wedge states only, like forward;
        ValueFunction.policy handles fold and physical arm labels.
        """
        _, best, _ = self.hamiltonian(m_b, m_c, tau_bb, tau_bc, tau_cc)

        return torch.stack([1.0 - best.x - best.y, best.x, best.y], dim=-1).detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, any reachable state.

    Wraps a trained `DimensionlessValueFunction` with the doc section 11 readout
    dictionary and folds arbitrary states into the fundamental wedge for the
    premium (S3-invariant, doc section 6), keeping the commit term in physical
    labels. States must be reachable (tau_bc <= 0, pair coordinates nonnegative),
    and rho = sigma = 1 on wedge states recovers the dimensionless form.
    """

    def __init__(
        self, dimensionless: DimensionlessValueFunction, rho: float, sigma: float
    ) -> None:
        super().__init__()

        self.dimensionless = dimensionless
        self.rho = rho
        self.sigma = sigma

    def _fold(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Sample, Tensor]:
        """
        Nondimensionalize and roll into the fundamental wedge; the returned
        order un-permutes wedge roles back to physical arms.
        """
        mean_scale = self.sigma * self.rho**0.5
        precision_scale = self.rho * self.sigma**2

        return Sample(
            m_b / mean_scale,
            m_c / mean_scale,
            precision_scale * tau_bb,
            precision_scale * tau_bc,
            precision_scale * tau_cc,
        ).fold_ordered()

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        folded, _ = self._fold(m_b, m_c, tau_bb, tau_bc, tau_cc)
        premium = self.dimensionless.premium(
            folded.m_b, folded.m_c, folded.tau_bb, folded.tau_bc, folded.tau_cc
        )
        commit = torch.relu(torch.maximum(m_b, m_c)) / (self.sigma * self.rho**0.5)

        return (self.sigma / self.rho**0.5) * (commit + premium)

    def policy(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        """
        The argmax allocation over physical arms, rows (alpha_a, alpha_b,
        alpha_c): the wedge policy, folded in and un-permuted back through
        fold_ordered's relabel. Allocations are fractions, so only inputs
        rescale.
        """
        folded, order = self._fold(m_b, m_c, tau_bb, tau_bc, tau_cc)
        roles = self.dimensionless.policy(
            folded.m_b, folded.m_c, folded.tau_bb, folded.tau_bc, folded.tau_cc
        )

        return torch.zeros_like(roles).scatter_(1, order, roles)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from: fresh at `topology`, or adapted from an
    existing checkpoint's `state`. Exactly one of the two. The CLI reads the
    file and passes the dict, so `pinn init --from` stays the only place that
    decides what a source file means.
    """
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))

        # Both given: topology is the **target** shape and state the source
        # adapted into it, which grafts a kink branch onto a trained smooth net,
        # bit-exact at step 0 (kb section 19.2).
        if state is not None:
            value.premium.stitch(
                {k.removeprefix("premium."): v for k, v in state.items()}
            )

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))
    value.load_state_dict(state)

    return value


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        """A premium that is identically zero, so v is the bare commit envelope."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return torch.zeros_like(m_b)

    m_b, m_c = torch.randn(100), torch.randn(100)
    taus = (torch.rand(100) + 0.5, -torch.rand(100) * 0.3, torch.rand(100) + 0.5)
    v = DimensionlessValueFunction(_ZeroPremium())(m_b, m_c, *taus)

    assert torch.allclose(v, torch.relu(torch.maximum(m_b, m_c)))

    # The premium runs end to end on wedge states, stays finite, and at
    # log_scale = 0 it obeys both proven properties: 0 <= u <= envelope.
    premium = ExplorationPremium([32, 16])
    draw = Sample.draw(1000).fold()
    state = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    u = premium(*state)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == draw.m_b.shape and u.isfinite().all()

    precision_b = draw.tau_bb - draw.tau_bc**2 / draw.tau_cc
    precision_c = draw.tau_cc - draw.tau_bc**2 / draw.tau_bb
    correlation = -draw.tau_bc / (draw.tau_bb * draw.tau_cc).sqrt()
    bound = nu2(
        draw.m_b, draw.m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
    )

    assert (u >= 0).all() and (u <= bound + 1e-6).all()

    # Stitch identity: a state without kink keys loaded into a kinked net is the
    # same function, because the zero-init output layer keeps the branch silent,
    # and the branch's parameters are still trainable.
    stitched = ExplorationPremium([32, 16], kinks=8)
    stitched.stitch(premium.state_dict())

    assert torch.allclose(stitched(*state), u)
    assert stitched.kink_out.weight.requires_grad

    # The graft must keep declaring **its own** shape. The smooth source says
    # kink_count = 0, and letting that win would save a net whose declaration its
    # own kink weights disprove, so the next load would fail on them.
    assert int(stitched.kink_count) == 8, int(stitched.kink_count)
    assert read_topology(DimensionlessValueFunction(stitched).state_dict()) == (
        [32, 16],
        8,
    )

    # The deployment wrapper: at rho = sigma = 1 on wedge states it equals
    # the dimensionless form; it is b<->c relabel invariant on any state; and
    # units scale exactly by the section 11 dictionary.
    dimensionless = DimensionlessValueFunction(premium)
    wrapper = ValueFunction(dimensionless, rho=1.0, sigma=1.0)

    assert torch.allclose(wrapper(*state), dimensionless(*state), atol=1e-6)

    anywhere = (m_b, m_c, *taus)
    swapped = (m_c, m_b, taus[2], taus[1], taus[0])

    assert torch.allclose(wrapper(*anywhere), wrapper(*swapped), atol=1e-5)

    rho, sigma = 0.04, 2.5
    real = ValueFunction(dimensionless, rho=rho, sigma=sigma)
    mean_scale, precision_scale = sigma * rho**0.5, rho * sigma**2
    dimensional = (
        state[0] * mean_scale,
        state[1] * mean_scale,
        state[2] / precision_scale,
        state[3] / precision_scale,
        state[4] / precision_scale,
    )

    assert torch.allclose(
        real(*dimensional),
        (sigma / rho**0.5) * dimensionless(*state),
        rtol=1e-5,
    )

    # The policy: valid simplex rows on any state, and the b<->c relabel
    # permutes the allocation columns with it.
    alpha = wrapper.policy(*anywhere)

    assert (alpha >= -1e-6).all() and (alpha <= 1 + 1e-6).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(len(m_b)), atol=1e-5)

    swapped_alpha = wrapper.policy(*swapped)

    assert torch.allclose(alpha[:, [0, 2, 1]], swapped_alpha, atol=1e-5)
    print("ok")
