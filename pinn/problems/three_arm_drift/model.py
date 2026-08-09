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

from ...net import GainedTanh, hidden_widths, parse_topology
from ..three_arm.simplex import Maximum, maximize_quadratic
from .envelope import envelope as premium_cap
from .sample import Sample

# three_arm's fourteen, plus the drift coordinate.
FEATURE_COUNT = 16

SQRT3 = math.sqrt(3.0)


def _kinks(state: dict) -> int:
    """
    A checkpoint's kink count, 0 if it has no branch.
    """
    weight = state.get("premium.kink_in.weight")

    return 0 if weight is None else weight.shape[0]


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

    def __init__(self, hidden: list[int], kinks: int = 0) -> None:
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

        # Kink units: parallel saturated relu(.)**2 primitives added to the
        # response -- movable curvature jumps for the free-boundary junction
        # (the blob), which tanh ridges cannot synthesize cheaply. Each unit
        # is y/(1+y), y = relu(.)**2: same curvature-jump regularity at the
        # crease, but bounded output, so the branch is architecturally bad at
        # painting the smooth bulk and cannot colonize it in from-scratch
        # co-training (observed 2026-08-06 with bare relu**2: the branch
        # outgrew the tanh stack early and took half the field, 2 orders
        # worse). Zero-init output layer: stitched onto a trained checkpoint,
        # the branch contributes exactly 0 at step 0, so training resumes
        # from the checkpoint's function.
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)

            # Alive start (the head-bias-1 lesson, one level down): a relu**2
            # unit that is never active gets no gradient and never recovers;
            # a positive bias opens every unit on a healthy slice of the
            # cloud (observed 2026-08-06: default init left 3 of 8 dead).
            nn.init.constant_(self.kink_in.bias, 0.5)

        # Envelope scale, init 0: at scale exactly 1 the envelope is a proven
        # upper bound on the true premium (doc section 13), so the whole
        # solution starts inside the tanh range.
        self.log_scale = nn.Parameter(torch.zeros(()))

        # Xavier assumes unit-variance inputs; the raw feature stack breaks
        # that by 10-100x under the general sampling law, railing the first
        # tanh layer (measured 2026-08-05: 49% of units saturated, fatal for
        # deep profiles whose later layers see only those units). Calibrate a
        # fixed per-feature scale from the law once at init; it is a buffer,
        # so checkpoints carry it.
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

        Explicit, where three_arm does this implicitly in _load_from_state_dict.
        It has to be: padding the first layer with a zero column for the drift
        feature is a shape change, and a setdefault cannot express one. At
        etahat = 0 that feature is exactly 0 and the envelope collapses onto
        three_arm's, so the stitched net IS the source, bitwise.

        Kinks go both ways. A source without a branch keeps this net's
        zero-init one, so the graft is a no-op at step 0; a source WITH one
        being loaded into a net without is dropped, which is the smooth-first
        path. Anything else missing is a real mismatch and should fail loudly.
        """
        state = dict(source)

        # Both layers that read the feature stack get the pad, not just the
        # trunk: the kink branch reads it too, and forgetting it is a shape
        # error the moment anyone grafts a kinked checkpoint.
        for name in ("net.0.weight", "kink_in.weight"):
            weight = state.get(name)

            if weight is not None and weight.shape[1] == FEATURE_COUNT - 1:
                state[name] = torch.cat(
                    [weight, torch.zeros_like(weight[:, :1])], dim=1
                )

        if state["feature_scale"].shape[0] == FEATURE_COUNT - 1:
            state["feature_scale"] = torch.cat(
                [state["feature_scale"], self.feature_scale[-1:]]
            )

        for name, tensor in self.state_dict().items():
            if name.startswith("kink_"):
                state.setdefault(name, tensor)

        for name in list(state):
            if name.startswith("kink_") and self.kinks == 0:
                del state[name]
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
        The raw (uncalibrated) feature stack. three_arm returns the marginal
        precisions alongside for its envelope to reuse; the drift envelope
        recomputes them from the state instead, which is what keeps the
        etahat = 0 anchor bitwise, so they are not returned here.
        """
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
                correlation,
                # The drift coordinate, LAST because the graft pads on the
                # right. It is the state's share of the ceiling drift imposes,
                # which is exactly the erosion term's own coefficient
                # (docs/three_arm_drift.md sections 6 and 8): bounded on the
                # reachable set, unchanged by shuffling arm labels since det T
                # is the invariant, and exactly 0 at etahat = 0 -- which is
                # what makes the three_arm graft bitwise rather than close.
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

        # The EXACT free-information value E[max(0, theta_b, theta_c)] under
        # the posterior (doc sections 13-14): a proven upper bound that is
        # also the exact startup solution, correlation dependence included --
        # the response tends to 1 as information goes to zero, so the floor
        # decades ask nothing of the net (the nu-sum predecessor was 1.17x
        # to 1.87x loose there and bred a never-explore attractor). No
        # correlation clamp: k < 1 strictly on reachable states, and a clamp
        # would kink d(envelope)/d(tau_bc) exactly where the corner needs it.
        envelope = self.log_scale.exp() * premium_cap(
            m_b, m_c, tau_bb, tau_bc, tau_cc, etahat
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


class DimensionlessValueFunction(nn.Module):
    """
    The thing training grades: v = max(0, m_b, m_c) + u. The commit envelope
    carries all three kinks (hand-written relu-of-max); the premium net stays
    smooth, exactly the two_arm division of labor. Units rho = sigma = 1,
    which is the dimensionless form (doc section 11), and the premium is
    only trained on the fundamental wedge -- deployment goes through
    ValueFunction, which papers over both cuts.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        A trained checkpoint as a model, architecture inferred from the state
        dict.
        """
        state = torch.load(path)
        value = cls(ExplorationPremium(hidden_widths(state), kinks=_kinks(state)))
        value.load_state_dict(state)

        return value

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
    ) -> tuple[Tensor, Maximum]:
        """
        The HJB's two sides on wedge states: the value v and the maximized
        Hamiltonian (doc section 10). Pairwise learning numbers: for pair
        direction dir, w = T^{-1} dir, and L = mean-diffusion Ito term
        (1/2) w' D2m(v) w plus the precision-drift term (dir-form on dv/dT);
        H(b, c) = b (m_b + l_ab) + c (m_c + l_ac) - l_ab b^2 - l_ac c^2
        + (l_bc - l_ab - l_ac) b c, handed to plain calculus in
        triangle-quadratic coefficients. Everything is graph-connected to the
        premium's parameters, so pde_loss grades the gap and policy reads the
        argmax off the same derivation.

        Drift adds one thing, on the LEFT: what the wandering erodes from the
        precision each instant, T E T, contracted with the value's precision
        derivatives. It does not depend on the allocation, so the maximization
        below is byte-identical to three_arm's and the simplex code is that
        module's, imported unchanged (doc section 2).
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

        # The erosion, T E T = etahat^2 (T^2 + v v^T) with v = T 1, whose two
        # entries are the a-b and a-c pair coordinates. Coefficient one on each
        # of the three independent entries, matching the chain-rule convention
        # the learning numbers above already use.
        pair_ab = tau_bb + tau_bc
        pair_ac = tau_cc + tau_bc
        erosion = etahat**2
        erosion_bb = erosion * (tau_bb**2 + tau_bc**2 + pair_ab**2)
        erosion_bc = erosion * (tau_bc * (tau_bb + tau_cc) + pair_ab * pair_ac)
        erosion_cc = erosion * (tau_bc**2 + tau_cc**2 + pair_ac**2)

        left = v + erosion_bb * v_tbb + erosion_bc * v_tbc + erosion_cc * v_tcc

        return left, maximize_quadratic(
            -l_ab, -l_ac, l_bc - l_ab - l_ac, m_b + l_ab, m_c + l_ac
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
        _, best = self.hamiltonian(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)

        return torch.stack([1.0 - best.x - best.y, best.x, best.y], dim=-1).detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, any reachable state.

    Wraps a trained DimensionlessValueFunction with the doc section 11
    readout dictionary -- mhat = m / (sigma sqrt(rho)), tauhat = rho sigma^2
    tau, etahat = eta / (rho sigma), V = (sigma / sqrt(rho)) vhat -- and folds
    arbitrary states into the
    fundamental wedge for the premium (S3-invariant, doc section 6), keeping
    the commit term in physical labels. States must still be reachable
    (tau_bc <= 0, pair coordinates nonnegative); rho = sigma = 1 on wedge
    states recovers the dimensionless form exactly.
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
        (alpha_a, alpha_b, alpha_c): the dimensionless wedge policy, folded
        in and un-permuted back through fold_ordered's relabel. Allocations
        are dimensionless fractions, so only the inputs rescale.
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
    A model to start training from: fresh at `topology`, or adapted from an
    existing checkpoint's `state`, or both -- which means adapt the source into
    the target shape.

    The CLI reads the file; this takes the state dict. A problem module has no
    business knowing where checkpoints live, and passing the dict keeps
    `pinn init --from` the only place that decides what a source file means.
    """
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))

        # Both: topology is the TARGET shape, state the source to adapt into
        # it. That is how a three_arm checkpoint becomes a drift one, and how
        # its kink branch is kept or dropped.
        if state is not None:
            value.premium.stitch(
                {k.removeprefix("premium."): v for k, v in state.items()}
            )

        return value

    value = DimensionlessValueFunction(
        ExplorationPremium(hidden_widths(state), kinks=_kinks(state))
    )
    features = state["premium.net.0.weight"].shape[1]

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

    # THE graft: a three_arm checkpoint is one feature narrower, and at
    # etahat = 0 the extra feature is exactly 0 and the cap collapses onto
    # three_arm's, so the stitched net IS the source. Bitwise, not close --
    # this is the check that says the whole feature/cap/stitch chain is right.
    three_arm = ThreeArm([32, 16])
    grafted = ExplorationPremium([32, 16])
    grafted.stitch(three_arm.state_dict())
    zero = torch.zeros_like(draw.m_b)

    assert torch.equal(
        grafted(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, zero),
        three_arm(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc),
    )
    assert torch.equal(grafted.feature_scale[:-1], three_arm.feature_scale)

    # The deployment wrapper: at rho = sigma = 1, eta = 0 on wedge states it
    # equals the dimensionless form; it is b<->c relabel invariant on any
    # state; and units scale exactly by the section 11 dictionary.
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
