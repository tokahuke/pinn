"""
Models for the three-arm v3 problem: the drop-one subsolution base under a
learned interpolation toward the free-information envelope (kb/three_arm.md
section 17).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import DeclaresTopology, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ...utils import nu2
from ..two_arm.model import ExplorationPremium as TwoArmPremium
from .sample import Sample
from .simplex import Maximum, maximize_quadratic

# Width of the feature stack in ExplorationPremium.forward.
FEATURE_COUNT = 16

# What a two_arm checkpoint declares -- how init_model tells a fresh base
# bootstrap from a v3 resume.
_TWO_ARM_FEATURES = 4


class ExplorationPremium(DeclaresTopology):
    """
    The v3 premium over the fundamental wedge (kb/three_arm.md section 17):

        u = B * (1 - G-) + exp(log_scale) * (nu2 - B) * G+

    with G+ and G- the saturated relu(.)**2 gate on each sign of the response
    (see forward). B = max(p_ab, p_ac) is the drop-one subsolution -- the
    frozen two_arm premium at each surviving pair's marginal state, exact in
    the far fields and scoring 8x below the trained three_arm champion at step
    0. HARD max: the Boltzmann softmax was refuted by measurement (section 17
    issue 2, 8 orders -- the mixing band's weight derivatives are physics the
    HJB rejects); the seam kinks are null sets no loss samples.

    0 <= u < nu2 is architectural (given B <= nu2: a theorem at base scale
    <= 1, measured max ratio 0.978 over 65k states). u >= B is NOT, and must
    not be: the base is a trained net rather than the true u2, so it
    overclaims on 27% of states, and a one-sided gate deadlocked a run there
    (forward).

    The base is FROZEN and embedded: its parameters ride in this net's state
    dict, so a checkpoint is self-contained -- the champion symlink ages in
    place, and a referenced base would silently change under a trained
    correction. A co-trained base is the v2 failure mode (section 16 verdict).

    The correction net is three_arm's response machinery unchanged (feature
    stack, GainedTanh MLP, optional saturated kink branch, relu(r)**2
    response) plus ONE new feature: the ballasted drop-a candidate
    m_b + u2(m_bc), demoted from base candidate per the section 17 rulings.
    """

    def __init__(self, base: TwoArmPremium, hidden: list[int], kinks: int = 0) -> None:
        super().__init__(FEATURE_COUNT, hidden, kinks)

        self.base = base.requires_grad_(False)

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            # Head: ZERO weights, small bias -- the kink_out trick one level
            # up. The residual reads DERIVATIVES, so a random Xavier head is
            # not a small start however small the bias: its response
            # derivatives are O(1) and, times (nu2 - B), scored pde 1.4e+1
            # against the bare base's 1.6e-4. The bias cannot be 0 (relu**2
            # has no gradient there, so the correction would freeze at the
            # base forever) and each step up costs starting residual --
            # measured 2026-08-15 on the fenced law, pde at bias
            # 0 / 0.01 / 0.03 / 0.1 / 0.3 = 8.5e-6 / 8.7e-6 / 1.3e-5 /
            # 4.3e-4 / 2.8e-2. 0.03 is 1.5x the frozen-base floor and keeps
            # the response an order livelier than 0.01.
            head = i == len(sizes) - 2

            if head is True:
                nn.init.zeros_(linear.weight)
                nn.init.constant_(linear.bias, 0.03)
            else:
                nn.init.xavier_uniform_(
                    linear.weight, gain=nn.init.calculate_gain("tanh")
                )
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])

        # Kink units, as three_arm: saturated relu(.)**2 curvature-jump
        # primitives, zero-init output so a stitch is bit-exact at step 0,
        # alive-start bias. NOTE these cannot cancel the base's seam creases
        # (gradient jumps, one class rougher); that is accepted for v1 --
        # section 17 issue 2's verdict.
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)
            nn.init.constant_(self.kink_in.bias, 0.5)

        # Scale on the (nu2 - B) gap, init 0: at exactly 1 both bounds are
        # proven, so the whole true premium starts inside the net's range.
        self.log_scale = nn.Parameter(torch.zeros(()))

        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        draw = Sample.draw(4096).fold()
        self.feature_scale = (
            self._features(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)[0]
            .std(dim=0)
            .clamp_min(1e-3)
        )

    def stitch(self, source: dict) -> None:
        """
        Adopt a smooth v3 premium's parameters into this (possibly kinked)
        net: a source without a kink branch keeps this net's zero-init one,
        so the graft is bit-exact at step 0. The source carries its own base
        under base.*, so the graft keeps the source's base too.
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        The raw (uncalibrated) feature stack, plus the two pair premia the
        base is built from and the marginals/correlation the envelope reuses.
        """
        # Schur marginals, as three_arm (see that model.py for the why).
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
        m_bc = m_b - m_c
        correlation = -tau_bc / (tau_bb * tau_cc).sqrt()

        # The drop values: two_arm at each pair's marginal state (section 17;
        # on the wedge a leads every pair, so the mean argument is the lead).
        p_ab = self.base(-m_b, precision_b)
        p_ac = self.base(-m_c, precision_c)

        features = torch.stack(
            [
                m_b,
                m_c,
                tau_bb,
                tau_bc,
                tau_cc,
                precision_b.log(),
                precision_c.log(),
                precision_bc.log(),
                m_b * precision_b.sqrt(),
                m_c * precision_c.sqrt(),
                m_bc * precision_bc.sqrt(),
                m_b * precision_b,
                m_c * precision_c,
                m_bc * precision_bc,
                correlation,
                # The ballasted drop-a candidate, a FEATURE not a base
                # candidate (section 17 rulings): it hard-argmaxes on 0.1% of
                # the wedge, and bare p_bc is unbounded relative to nu2
                # (section 16) -- the m_b ballast is what makes it legal
                # information.
                m_b + self.base(m_bc, precision_bc),
            ],
            dim=-1,
        )

        return features, p_ab, p_ac, precision_b, precision_c, correlation

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        features, p_ab, p_ac, precision_b, precision_c, correlation = self._features(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        scaled = features / self.feature_scale
        response = self.net(scaled).squeeze(-1)

        if self.kinks > 0:
            bumps = torch.relu(self.kink_in(scaled)) ** 2
            response = response + self.kink_out(bumps / (1.0 + bumps)).squeeze(-1)

        envelope = nu2(m_b, m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation)
        base = torch.maximum(p_ab, p_ac)

        # TWO-SIDED gate (2026-08-15): the sibling relu(r)**2 primitive on
        # each sign, so r = 0 is exactly B, r > 0 climbs toward nu2 and
        # r < 0 DESCENDS toward 0. The one-sided version deadlocked a run
        # after 10k iterations: the base overclaims on 27% of states (it is
        # a trained net, not the true u2, so B <= u is not a theorem), the
        # correction needed to go negative there, could not, and the
        # optimizer drove the response through zero everywhere -- relu(r)**2
        # then zeroed both the correction AND every gradient through it, and
        # the net froze at u = B with a loss that looked healthy (pde 8.6e-6,
        # ties and concavity EXACTLY 0). Shrinking B does not help: scaling
        # it also scales the learning numbers that build max H, and the
        # overclaim measured WORSE (27% -> 47%).
        #
        # Both gates have zero value and zero SLOPE at r = 0, so the branches
        # meet C1 -- no crease at u = B -- and the downward branch saturates
        # rationally, so it has no absorbing state: a negative response keeps
        # its gradient. Where B = 0 (the commit region B inherits from its two
        # dead pair premia) the formula reduces to three_arm's own
        # nu2 * G+, so the free boundary stays learnable and exactly zero
        # inside. clamp_min on the gap guards the sign if B ever exceeds nu2
        # (measured max ratio 0.978, so it never binds).
        gain_squared = torch.relu(response) ** 2
        gain = gain_squared / (1.0 + gain_squared)
        deficit_squared = torch.relu(-response) ** 2
        deficit = deficit_squared / (1.0 + deficit_squared)

        return (
            base * (1.0 - deficit)
            + self.log_scale.exp() * (envelope - base).clamp_min(0.0) * gain
        )


class DimensionlessValueFunction(nn.Module):
    """
    The thing training grades: v = max(0, m_b, m_c) + u, exactly three_arm's
    wrapper (see that model.py); only the premium inside differs.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        A trained v3 checkpoint as a model: the correction's architecture
        from its declaration, the embedded base's from the nested one.
        """
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        base_hidden, _ = read_topology(state, "premium.base.")
        value = cls(ExplorationPremium(TwoArmPremium(base_hidden), hidden, kinks=kinks))
        value.load_state_dict(state)

        return value

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        commit = torch.relu(torch.maximum(m_b, m_c))

        return commit + self.premium(m_b, m_c, tau_bb, tau_bc, tau_cc)

    def hamiltonian(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Tensor, Maximum, tuple[Tensor, Tensor, Tensor]]:
        """
        The HJB's two sides on wedge states; identical derivation to
        three_arm (one chain serves pde_loss and policy readout). The chain
        runs through the frozen base: its INPUT derivatives are part of v's,
        only its parameters are off the optimization.
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
        (alpha_a, alpha_b, alpha_c); ValueFunction.policy handles fold and
        physical arm labels.
        """
        _, best, _ = self.hamiltonian(m_b, m_c, tau_bb, tau_bc, tau_cc)

        return torch.stack([1.0 - best.x - best.y, best.x, best.y], dim=-1).detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, any reachable state.
    Identical to three_arm's wrapper (see that model.py for the fold and
    readout dictionary).
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
        folded, order = self._fold(m_b, m_c, tau_bb, tau_bc, tau_cc)
        roles = self.dimensionless.policy(
            folded.m_b, folded.m_c, folded.tau_bb, folded.tau_bc, folded.tau_cc
        )

        return torch.zeros_like(roles).scatter_(1, order, roles)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from. v3 always needs a base, so `state` is
    mandatory and read_features decides what it means:

    - a two_arm checkpoint (4 features) is the BASE: with --topology, a fresh
      v3 net is built around it (`pinn init --problem three_arm_v3
      --topology 64:64:64 --from data/two_arm.pt`);
    - a v3 checkpoint (16 features) resumes as-is, or grafts a kink branch
      when --topology is also given, keeping its embedded base either way.
    """
    if state is None:
        raise ValueError(
            "three_arm_v3 always needs a base: pass --from "
            "(a two_arm checkpoint for a fresh net, or a v3 checkpoint)"
        )

    if read_features(state) == _TWO_ARM_FEATURES:
        if topology is None:
            raise ValueError("a fresh v3 net from a two_arm base needs --topology")
        hidden, kinks = parse_topology(topology)
        base = TwoArmPremium(read_topology(state)[0])
        base.load_state_dict({k.removeprefix("premium."): v for k, v in state.items()})

        return DimensionlessValueFunction(ExplorationPremium(base, hidden, kinks=kinks))

    base = TwoArmPremium(read_topology(state, "premium.base.")[0])

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(
            ExplorationPremium(base, hidden, kinks=kinks)
        )
        value.premium.stitch({k.removeprefix("premium."): v for k, v in state.items()})

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(base, hidden, kinks=kinks))
    value.load_state_dict(state)

    return value


if __name__ == "__main__":
    from ..two_arm.model import DimensionlessValueFunction as TwoArmValue

    # Untrained base: log_scale = 0, so every bound below is the proven one.
    base = TwoArmPremium([8, 8])
    premium = ExplorationPremium(base, [32, 16])
    draw = Sample.draw(1000).fold()
    state = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    u = premium(*state)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == draw.m_b.shape and u.isfinite().all()

    # The base is frozen; the correction (and log_scale) train.
    assert all(p.requires_grad is False for p in premium.base.parameters())
    assert any(p.requires_grad for p in premium.net.parameters())
    assert premium.log_scale.requires_grad

    # Architectural bounds at log_scale = 0: B <= u < nu2 (B <= nu2 is a
    # theorem at base scale <= 1), and the 0.1-bias start is alive.
    precision_b = draw.tau_bb - draw.tau_bc**2 / draw.tau_cc
    precision_c = draw.tau_cc - draw.tau_bc**2 / draw.tau_bb
    correlation = -draw.tau_bc / (draw.tau_bb * draw.tau_cc).sqrt()
    b = torch.maximum(base(-draw.m_b, precision_b), base(-draw.m_c, precision_c))
    bound = nu2(
        draw.m_b, draw.m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
    )

    # Both proven bounds hold architecturally; u >= B does NOT, by design --
    # the two-sided gate exists so the correction can go below an
    # overclaiming base.
    assert (u >= -1e-6).all() and (u <= bound + 1e-6).all()
    assert (u > b).float().mean() > 0.5

    # The two-sided gate: r = 0 is exactly B, and the negative branch keeps a
    # live gradient where the one-sided version had an absorbing wall.
    dead_head = [m for m in premium.net if isinstance(m, nn.Linear)][-1]

    with torch.no_grad():
        nn.init.constant_(dead_head.bias, 0.0)

    assert torch.allclose(premium(*state), b, atol=1e-6)

    with torch.no_grad():
        nn.init.constant_(dead_head.bias, -3.0)

    deep = premium(*state)

    assert (deep < b + 1e-9).all() and (deep >= 0).all()
    assert deep.sum() > 0, "the descending branch must not be an absorbing zero"

    gradient = torch.autograd.grad(premium(*state).sum(), dead_head.bias)[0]

    assert gradient.abs().item() > 0, "a negative response must keep its gradient"

    with torch.no_grad():
        nn.init.constant_(dead_head.bias, 0.03)

    # Stitch identity: a smooth v3 source into a kinked v3 net is the same
    # function (zero-init branch), base included, and declarations hold --
    # the net's own at premium., the embedded base's at premium.base.
    stitched = ExplorationPremium(TwoArmPremium([8, 8]), [32, 16], kinks=8)
    stitched.stitch(premium.state_dict())

    assert torch.allclose(stitched(*state), u)
    assert stitched.kink_out.weight.requires_grad
    assert torch.allclose(stitched.base.log_scale, base.log_scale)

    wrapped = DimensionlessValueFunction(stitched).state_dict()

    assert read_topology(wrapped) == ([32, 16], 8)
    assert read_features(wrapped) == FEATURE_COUNT
    assert read_topology(wrapped, "premium.base.") == ([8, 8], 0)

    # init_model, all three paths. Base bootstrap from a two_arm state:
    source = TwoArmValue(TwoArmPremium([8, 8])).state_dict()
    fresh = init_model(state=source, topology="16:16")

    assert torch.allclose(
        fresh.premium.base.net[0].weight, source["premium.net.0.weight"]
    )

    # Roundtrip through the state dict is the same function; a kink graft on
    # the saved state keeps base and smooth weights.
    saved = fresh.state_dict()
    again = init_model(state=saved)

    assert torch.allclose(again(*state), fresh(*state))

    grafted = init_model(state=saved, topology="16:16k8")

    assert torch.allclose(grafted(*state), fresh(*state))
    assert read_topology(grafted.state_dict()) == ([16, 16], 8)

    for bad in [dict(state=None, topology="8:8"), dict(state=source)]:
        try:
            init_model(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    # The derivative chain runs end to end through the frozen base.
    value = DimensionlessValueFunction(premium)
    v, best, learning = value.hamiltonian(*(t[:100] for t in state))

    assert v.isfinite().all() and best.value.isfinite().all()
    assert all(l.isfinite().all() for l in learning)

    # The deployment wrapper: dimensionless identity, relabel invariance,
    # unit scaling, policy validity -- as three_arm's checks.
    wrapper = ValueFunction(value, rho=1.0, sigma=1.0)

    assert torch.allclose(wrapper(*state), value(*state), atol=1e-6)

    m_b, m_c = torch.randn(100), torch.randn(100)
    taus = (torch.rand(100) + 0.5, -torch.rand(100) * 0.3, torch.rand(100) + 0.5)
    anywhere = (m_b, m_c, *taus)
    swapped = (m_c, m_b, taus[2], taus[1], taus[0])

    assert torch.allclose(wrapper(*anywhere), wrapper(*swapped), atol=1e-5)

    rho, sigma = 0.04, 2.5
    real = ValueFunction(value, rho=rho, sigma=sigma)
    mean_scale, precision_scale = sigma * rho**0.5, rho * sigma**2
    dimensional = (
        state[0] * mean_scale,
        state[1] * mean_scale,
        state[2] / precision_scale,
        state[3] / precision_scale,
        state[4] / precision_scale,
    )

    assert torch.allclose(
        real(*dimensional), (sigma / rho**0.5) * value(*state), rtol=1e-5
    )

    alpha = wrapper.policy(*anywhere)

    assert (alpha >= -1e-6).all() and (alpha <= 1 + 1e-6).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(len(m_b)), atol=1e-5)
    assert torch.allclose(alpha[:, [0, 2, 1]], wrapper.policy(*swapped), atol=1e-5)
    print("ok")
