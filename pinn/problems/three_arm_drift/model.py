"""
Models for the three-arm drift problem: three_arm's, with the drift each state
is drawn for as one more input. Read three_arm/model.py first; this only
records what changes.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import DeclaresTopology, DimensionlessValue, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ..three_arm.simplex import Maximum, maximize_quadratic
from .envelope import envelope as premium_cap
from .sample import Sample

FEATURE_COUNT = 16
"""three_arm's fifteen, plus the drift coordinate."""

SQRT3 = math.sqrt(3.0)


class ExplorationPremium(DeclaresTopology):
    """
    The premium net over the fundamental wedge.

    Plain function on the wedge, no symmetry machinery (fold + wall losses carry S3);
    the feature list in forward; one GainedTanh MLP over the stacked features,
    Xavier-tanh init, shallow profile; the free-information envelope of doc section 7
    multiplying the saturated softplus response (y / (1 + y) with
    y = (softplus(k r) / k)**2).
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

        # Linear head (final GainedTanh sliced off): the response is mapped through
        # the saturated softplus gate in forward, which both confines u inside the
        # proven bound and builds in the free-boundary regularity.
        self.net = nn.Sequential(*layers[:-1])

        # Movable curvature jumps for the free-boundary junction, saturated so the
        # branch cannot colonize the smooth bulk, and zero-init so a graft is silent
        # at step 0. What that buys and what it cost to learn: doc section 9.
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)

            # Alive start: a relu**2 unit that is never active gets no gradient and
            # never recovers (doc section 9).
            nn.init.constant_(self.kink_in.bias, 0.5)

        # Envelope scale, init 0: at scale exactly 1 the envelope is a proven
        # upper bound on the true premium (doc section 7), so the whole
        # solution starts inside the tanh range.
        self.log_scale = nn.Parameter(torch.zeros(()))

        # Gate sharpness k for the softplus response (see forward). Init 10, near the
        # relu gate it replaces, so grafting a relu-gate net moves the function only by
        # O(1/k^2) at the seam. Trainable: the net picks how hard its boundary is.
        self.log_gate = nn.Parameter(torch.tensor(math.log(10.0)))

        # Xavier assumes unit-variance inputs and the raw stack breaks that by
        # 10-100x, railing the first tanh layer (2026-08-05: 49% of units saturated).
        # Calibrated once at init into a buffer, so a saved net carries it.
        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        draw = Sample.draw(4096).fold()
        self.feature_scale = (
            self._features(
                draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat
            )
            .std(dim=0)
            .clamp_min(1e-3)
        )

    def stitch(self, source: dict) -> None:
        """
        Adopt another premium's parameters: a three_arm checkpoint (one feature
        narrower), or a drift one whose kink branch does not match this net's.
        Anything else missing is a real mismatch and fails loudly. What it pads, what
        it defaults, and how close the three_arm anchor lands: doc section 9.
        """
        state = dict(source)

        # Both layers that read the feature stack get the pad, not just the
        # trunk: the kink branch reads it too, and forgetting it is a shape
        # error the moment anyone grafts a kinked checkpoint.
        if read_features(state, prefix="") == FEATURE_COUNT - 1:
            for name in ("net.0.weight", "kink_in.weight"):
                weight = state.get(name)

                if weight is not None:
                    state[name] = torch.cat(
                        [weight, torch.zeros_like(weight[:, :1])], dim=1
                    )

            state["feature_scale"] = torch.cat(
                [state["feature_scale"], self.feature_scale[-1:]]
            )

        # The branch by name, never by a "kink_" prefix, which would also match the
        # kink_count buffer and leave the graft undeclared. DeclaresTopology restores
        # both shape buffers to this net's own, so a stale declaration cannot survive.
        mine = self.state_dict()
        branch = ("kink_in.weight", "kink_in.bias", "kink_out.weight", "kink_out.bias")

        for name in branch:
            if name in mine:
                state.setdefault(name, mine[name])

        # Sources predating the softplus gate (2026-08-14) have no sharpness;
        # this net's init (k = 10, near-relu) is the graft-faithful default.
        state.setdefault("log_gate", mine["log_gate"])

        if self.kinks == 0:
            for name in branch:
                state.pop(name, None)
        self.load_state_dict(state)

    def _features(
        self,
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        etahat: Tensor,
    ) -> Tensor:
        """
        The raw (uncalibrated) feature stack. Unlike three_arm this returns no
        marginal precisions: the drift envelope recomputes them from the state, which
        is what keeps the etahat = 0 anchor bitwise.
        """
        # Marginal precision per contrast, via Schur complements of T. Not the raw
        # diagonals: tau_bb is the precision *given* the other contrast, conditioning
        # you do not have, so it overstates certainty where the control matters.
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
                # z-scores: each mean in units of its own marginal sd, the two_arm
                # corridor-alignment win, once per contrast.
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
                # The drift coordinate, *last* because the graft pads on the right:
                # the state's share of the ceiling drift imposes, bounded, relabel-
                # invariant, and exactly 0 at etahat = 0 (doc sections 6 and 8).
                torch.log1p(2.0 * SQRT3 * etahat**2 * det),
            ],
            dim=-1,
        )

        return features

    def forward(
        self,
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        etahat: Tensor,
    ) -> Tensor:
        scaled = (
            self._features(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)
            / self.feature_scale
        )
        response = self.net(scaled).squeeze(-1)

        if self.kinks > 0:
            bumps = torch.relu(self.kink_in(scaled)) ** 2
            response = response + self.kink_out(bumps / (1.0 + bumps)).squeeze(-1)

        # The free-information envelope of doc section 7, a proven upper bound that
        # collapses *bitwise* onto three_arm's nu2 at etahat = 0. No correlation clamp,
        # which would kink d(envelope)/d(tau_bc) at the corner that needs it.
        envelope = self.log_scale.exp() * premium_cap(
            m_b, m_c, tau_bb, tau_bc, tau_cc, etahat
        )

        # y/(1+y) with y = (softplus(k r)/k)**2: 0 < u < envelope architecturally, and
        # u strictly positive, which drift needs since there is no contact set here.
        # Why softplus and not relu, with the measurement: doc section 9.
        gate = self.log_gate.exp()
        y = (torch.nn.functional.softplus(gate * response) / gate) ** 2

        return envelope * y / (1.0 + y)


class DimensionlessValueFunction(DimensionlessValue):
    """
    The thing training grades: v = max(0, m_b, m_c) + u. The commit envelope carries
    all three kinks (hand-written relu-of-max); the premium net stays smooth, exactly
    the two_arm division of labor. Units rho = sigma = 1, which is the dimensionless
    form (doc section 4), and the premium is only trained on the fundamental wedge,
    since deployment goes through ValueFunction, which papers over both cuts.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """A trained checkpoint as a model, architecture inferred from the state dict."""
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        value = cls(ExplorationPremium(hidden, kinks=kinks))
        # Checkpoints predating the softplus gate carry no sharpness; the
        # init (k = 10) reads them as near-relu, which is what they were.
        state.setdefault("premium.log_gate", value.premium.log_gate.detach().clone())
        value.load_state_dict(state)

        return value

    def bind(self, rho: float, sigma: float, eta: float) -> ValueFunction:
        """
        This net tied to one experiment, which is what makes it usable: means
        and precisions go in and come back in your own units. `eta` is the
        drift rate of the effect itself.
        """
        return ValueFunction(self, rho, sigma, eta)

    def forward(
        self,
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        etahat: Tensor,
    ) -> Tensor:
        commit = torch.relu(torch.maximum(m_b, m_c))

        return commit + self.premium(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)

    def hamiltonian(
        self,
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        etahat: Tensor,
    ) -> tuple[Tensor, Maximum, tuple[Tensor, Tensor, Tensor]]:
        """
        The HJB's two sides on wedge states: v, the maximized Hamiltonian, and the
        learning numbers (l_ab, l_ac, l_bc) for the concavity grading. Drift adds the
        control-free erosion T E T on the *left*, so the maximization is three_arm's
        byte for byte (doc section 2; three_arm.md section 10).
        """
        m_b = m_b.detach().requires_grad_(True)
        m_c = m_c.detach().requires_grad_(True)
        tau_bb = tau_bb.detach().requires_grad_(True)
        tau_bc = tau_bc.detach().requires_grad_(True)
        tau_cc = tau_cc.detach().requires_grad_(True)

        v = self(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)
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
            return 0.5 * (d_b**2 * v_mbmb + 2.0 * d_b * d_c * v_mbmc + d_c**2 * v_mcmc)

        l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + v_tbb
        l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + v_tcc
        l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
            v_tbb + v_tcc - v_tbc
        )

        # The erosion, T E T = etahat^2 (T^2 + v v^T) with v = T 1, whose two entries
        # are the a-b and a-c pair coordinates. Coefficient one on each independent
        # entry, the convention the learning numbers above already use.
        pair_ab = tau_bb + tau_bc
        pair_ac = tau_cc + tau_bc
        erosion = etahat**2
        erosion_bb = erosion * (tau_bb**2 + tau_bc**2 + pair_ab**2)
        erosion_bc = erosion * (tau_bc * (tau_bb + tau_cc) + pair_ab * pair_ac)
        erosion_cc = erosion * (tau_bc**2 + tau_cc**2 + pair_ac**2)

        left = v + erosion_bb * v_tbb + erosion_bc * v_tbc + erosion_cc * v_tcc

        return (
            left,
            maximize_quadratic(
                -l_ab, -l_ac, l_bc - l_ab - l_ac, m_b + l_ab, m_c + l_ac
            ),
            (l_ab, l_ac, l_bc),
        )

    def policy(
        self,
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        etahat: Tensor,
    ) -> Tensor:
        """
        The argmax allocation in wedge roles, shape (n, 3) rows
        (alpha_a, alpha_b, alpha_c). Wedge states only, like forward;
        ValueFunction.policy handles fold and physical arm labels.
        """
        _, best, _ = self.hamiltonian(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)

        return torch.stack([1.0 - best.x - best.y, best.x, best.y], dim=-1).detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, any reachable state.

    Wraps a trained DimensionlessValueFunction with the three_arm.md section 11
    readout dictionary (mhat = m / (sigma sqrt(rho)), tauhat = rho sigma^2 tau,
    etahat = eta / (rho sigma), V = (sigma / sqrt(rho)) vhat) and folds arbitrary
    states into the fundamental wedge for the premium, keeping the commit term in
    physical labels. States must still be reachable (tau_bc <= 0, pair coordinates
    nonnegative).
    """

    def __init__(
        self,
        dimensionless: DimensionlessValueFunction,
        rho: float,
        sigma: float,
        eta: float,
    ) -> None:
        super().__init__()

        self.dimensionless = dimensionless
        self.rho = rho
        self.sigma = sigma
        self.eta = eta

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
            torch.full_like(m_b, self.eta / (self.rho * self.sigma)),
        ).fold_ordered()

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        folded, _ = self._fold(m_b, m_c, tau_bb, tau_bc, tau_cc)
        premium = self.dimensionless.premium(
            folded.m_b,
            folded.m_c,
            folded.tau_bb,
            folded.tau_bc,
            folded.tau_cc,
            folded.etahat,
        )
        commit = torch.relu(torch.maximum(m_b, m_c)) / (self.sigma * self.rho**0.5)

        return (self.sigma / self.rho**0.5) * (commit + premium)

    def policy(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        """
        The argmax allocation over physical arms, shape (n, 3) rows
        (alpha_a, alpha_b, alpha_c): the wedge policy folded in and un-permuted back
        through fold_ordered's relabel. Allocations are fractions, so only the inputs
        rescale.
        """
        folded, order = self._fold(m_b, m_c, tau_bb, tau_bc, tau_cc)
        roles = self.dimensionless.policy(
            folded.m_b,
            folded.m_c,
            folded.tau_bb,
            folded.tau_bc,
            folded.tau_cc,
            folded.etahat,
        )

        return torch.zeros_like(roles).scatter_(1, order, roles)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from: fresh at `topology`, adapted from an existing
    `state`, or both, which adapts the source into the target shape. Taking the state
    dict rather than a path keeps `pinn init --from` the only place that decides what
    a source file means.
    """
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))

        # Both: topology is the *target* shape, state the source to adapt into it.
        # That is how a three_arm checkpoint becomes a drift one, and how its kink
        # branch is kept or dropped.
        if state is not None:
            value.premium.stitch(
                {k.removeprefix("premium."): v for k, v in state.items()}
            )

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))
    features = read_features(state)

    # A three_arm checkpoint is one feature narrower: stitch it as the
    # etahat = 0 slice.
    if features == FEATURE_COUNT - 1:
        value.premium.stitch({k.removeprefix("premium."): v for k, v in state.items()})

        return value
    value.load_state_dict(state)

    return value


if __name__ == "__main__":
    from ..three_arm.model import ExplorationPremium as ThreeArm

    class _ZeroPremium(nn.Module):
        """No premium at all, so the value is the bare commit envelope."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return torch.zeros_like(m_b)

    m_b, m_c = torch.randn(100), torch.randn(100)
    taus = (torch.rand(100) + 0.5, -torch.rand(100) * 0.3, torch.rand(100) + 0.5)
    drift = torch.rand(100) * 10.0
    v = DimensionlessValueFunction(_ZeroPremium())(m_b, m_c, *taus, drift)

    assert torch.allclose(v, torch.relu(torch.maximum(m_b, m_c)))

    # The premium runs end to end on wedge states, stays finite, and at
    # log_scale = 0 it obeys both proven properties: 0 <= u <= the cap.
    premium = ExplorationPremium([32, 16])
    draw = Sample.draw(1000).fold()
    state = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat)
    u = premium(*state)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == draw.m_b.shape and u.isfinite().all()
    assert (u >= 0).all() and (u <= premium_cap(*state) + 1e-6).all()

    # Kinks both ways: a source without a branch keeps this net's zero-init
    # one (silent at step 0), and a source with one loaded into a net without
    # is dropped rather than refused.
    stitched = ExplorationPremium([32, 16], kinks=8)
    stitched.stitch(premium.state_dict())

    assert torch.allclose(stitched(*state), u)
    assert stitched.kink_out.weight.requires_grad

    smooth = ExplorationPremium([32, 16])
    smooth.stitch(stitched.state_dict())

    assert smooth.kinks == 0 and torch.allclose(smooth(*state), u)

    # *The* graft: a three_arm source is one feature narrower, and the anchor is
    # close, *not* bitwise, because the two gates differ near the seam. The 1.0e-2
    # tolerance is measured (true max 7.17e-3), not derived: doc section 9.
    three_arm = ThreeArm([32, 16])
    grafted = ExplorationPremium([32, 16])
    grafted.stitch(three_arm.state_dict())
    zero = torch.zeros_like(draw.m_b)

    graft_cap = premium_cap(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, zero
    )
    graft_gap = (
        grafted(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, zero)
        - three_arm(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    ).abs()

    assert (graft_gap <= 1.0e-2 * graft_cap + 1e-7).all(), (
        (graft_gap / graft_cap.clamp_min(1e-9)).max().item()
    )
    assert torch.equal(grafted.feature_scale[:-1], three_arm.feature_scale)

    # The deployment wrapper: at rho = sigma = 1, eta = 0 on wedge states it
    # equals the dimensionless form; it is b<->c relabel invariant on any
    # state; and units scale exactly by the three_arm.md section 11 dictionary.
    dimensionless = DimensionlessValueFunction(premium)
    wrapper = ValueFunction(dimensionless, rho=1.0, sigma=1.0, eta=0.0)
    flat = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, zero)

    assert torch.allclose(wrapper(*state[:-1]), dimensionless(*flat), atol=1e-6)

    anywhere = (m_b, m_c, *taus)
    swapped = (m_c, m_b, taus[2], taus[1], taus[0])
    drifting = ValueFunction(dimensionless, rho=1.0, sigma=1.0, eta=3.0)

    assert torch.allclose(drifting(*anywhere), drifting(*swapped), atol=1e-5)

    rho, sigma, eta = 0.04, 2.5, 0.3
    real = ValueFunction(dimensionless, rho=rho, sigma=sigma, eta=eta)
    unit = ValueFunction(dimensionless, rho=1.0, sigma=1.0, eta=eta / (rho * sigma))
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
        (sigma / rho**0.5) * unit(*state[:-1]),
        rtol=1e-5,
    )

    # The policy: valid simplex rows on any state, and the b<->c relabel
    # permutes the allocation columns with it.
    alpha = drifting.policy(*anywhere)

    assert (alpha >= -1e-6).all() and (alpha <= 1 + 1e-6).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(len(m_b)), atol=1e-5)

    swapped_alpha = drifting.policy(*swapped)

    assert torch.allclose(alpha[:, [0, 2, 1]], swapped_alpha, atol=1e-5)
    print("ok")
