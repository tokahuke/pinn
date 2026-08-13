"""
The pairwise ansatz: three_arm's premium with a two_arm net as basis.

Same envelope and same saturated response as three_arm -- the proven bound
`0 <= u < nu2` stays architectural -- but the response also sees the two_arm
premium of each PAIR of the wedge, computed by a frozen basis net. The basis
is a submodule, so a checkpoint is self-contained and reproducible.

Why a basis at all (measured 2026-08-12): three two_arm premia explain 98.0%
of three_arm's premium by linear regression, their leading pair's dead set
matches three_arm's commit region at Jaccard 0.93, and inside the feature
stack they carry the largest d/dtau of anything there -- which is what the
learning numbers are built from. The basis is ONE net whatever N is; only the
call count grows, which is the whole point (CLAUDE.md, three_arm_v2).

The combiner is EXPLICIT: a linear term in the three pair premia plus a
correction net that sees everything (all fifteen three_arm features and the
three premia). Both are summed in RESPONSE space, not premium space, which is
what keeps `0 <= u < nu2` architectural -- the same three coefficients added
in premium space overcount by half (they fit to 1.501, measured 2026-08-13)
and the result is not a value function.

The split is earned rather than assumed. Regressing the three_arm champion on
the pair premia, 20k wedge points, basis data/two_arm.pt (2026-08-13):

    premium value   R2 0.977    coefficients +0.923, +0.476, +0.102
    d/d tau_bb      R2 0.761    coefficients +0.724, +0.703, +0.697
    d/d tau_cc      R2 0.872    coefficients +0.944, +0.359, +0.600

So the linear part is nearly the whole story for the VALUE and only ~three
quarters of it for the DERIVATIVES, which is what the HJB residual is built
from -- hence a correction net with real capacity, not a perturbation. Note
the coefficients differ between value and slope (the leading pair dominates
the value; all three enter the slope about equally), so the linear head is
LEARNED, not seeded from the value fit: those coefficients are the wrong ones
for the quantity that gets graded.
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
from ..three_arm.model import ExplorationPremium as ThreeArmPremium
from ..three_arm.model import DimensionlessValueFunction as ThreeArmValue
from ..three_arm.model import ValueFunction as ThreeArmDeployment
from ..three_arm.sample import Sample

# The canonical two_arm champion. Which two_arm net it is barely matters: a
# 406-parameter basis and the 9,154-parameter one give identical R2 (0.9797),
# identical Jaccard and identical regression coefficients, so the basis
# contributes SHAPE, not precision.
BASIS = Path("data") / "two_arm.pt"

# three_arm's fifteen, then one two_arm premium per pair of the wedge. The
# premia are the LAST PAIR_COUNT columns, which is what the linear combiner
# slices; keep them last if the stack ever grows.
PAIR_COUNT = 3
FEATURE_COUNT = 15 + PAIR_COUNT

# The basis is only trained on muhat > 0 and down to its sampler's floor;
# below these it extrapolates. Binds on ~1% of wedge draws, all of it in the
# low-precision corner.
GAP_FLOOR = 1.0e-6
TAU_FLOOR = 1.0e-3


def basis_widths(state: dict) -> list[int] | None:
    """
    The basis net's hidden widths as the checkpoint declares them, or None when
    it holds no basis at all (a fresh init, which reads BASIS off disk instead).

    A saved v2 carries its basis weights, so loading one must not consult
    `data/two_arm.pt` -- that path is the champion symlink and it moves. Reading
    it to recover a shape the checkpoint already states made a v2 unloadable
    whenever the link pointed at a differently shaped net.
    """
    if "premium.basis.topology" not in state:
        return None

    return read_topology(state, prefix="premium.basis.")[0]


class ExplorationPremium(DeclaresTopology):
    """
    Premium over the fundamental wedge, three_arm's architecture plus the
    pairwise basis features.
    """

    def __init__(
        self, hidden: list[int], kinks: int = 0, basis_hidden: list[int] | None = None
    ) -> None:
        super().__init__(FEATURE_COUNT, hidden, kinks)

        # Frozen, and a real submodule so the checkpoint carries it: a v2 net
        # whose basis lives in a separate file is not reproducible. `basis_hidden`
        # is that promise kept -- given the shape, this touches no file, and the
        # weights arrive with the rest of the state dict.
        from ..two_arm.model import DimensionlessValueFunction as TwoArm
        from ..two_arm.model import ExplorationPremium as TwoArmPremium

        self.basis = (
            TwoArm.load(BASIS).premium
            if basis_hidden is None
            else TwoArmPremium(basis_hidden)
        )
        self.basis.requires_grad_(False)

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])
            head = i == len(sizes) - 2
            nn.init.xavier_uniform_(
                linear.weight, gain=1.0 if head else nn.init.calculate_gain("tanh")
            )

            if head is True:
                nn.init.ones_(linear.bias)
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)
            nn.init.constant_(self.kink_in.bias, 0.5)

        # The combiner: four parameters against the correction net's thousands.
        # Zero-init so a fresh net starts as the correction alone and the
        # linear term has to earn every unit of what it takes over.
        self.pair_head = nn.Linear(PAIR_COUNT, 1)
        nn.init.zeros_(self.pair_head.weight)
        nn.init.zeros_(self.pair_head.bias)

        self.log_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        draw = Sample.draw(4096).fold()
        self.feature_scale = (
            self._features(draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)[0]
            .std(dim=0)
            .clamp_min(1e-3)
        )

    def pairs(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        """
        The basis premium of each pair, stacked (n, 3) in order ab, ac, bc.

        Each pair is a two_arm problem in that pair's own marginal (Schur)
        precision. On the fundamental wedge m_c <= m_b <= 0, so arm a leads
        every pair and each gap is nonnegative.
        """
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)

        return torch.stack(
            [
                self.basis(
                    (-m_b).clamp_min(GAP_FLOOR), precision_b.clamp_min(TAU_FLOOR)
                ),
                self.basis(
                    (-m_c).clamp_min(GAP_FLOOR), precision_c.clamp_min(TAU_FLOOR)
                ),
                self.basis(
                    (m_b - m_c).clamp_min(GAP_FLOOR), precision_bc.clamp_min(TAU_FLOOR)
                ),
            ],
            dim=-1,
        )

    def _features(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # three_arm's stack is a pure function of the state, so it is called
        # unbound rather than duplicated here.
        features, precision_b, precision_c, correlation = ThreeArmPremium._features(
            self, m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        pairs = self.pairs(m_b, m_c, tau_bb, tau_bc, tau_cc)

        return (
            torch.cat([features, pairs], dim=-1),
            precision_b,
            precision_c,
            correlation,
        )

    def forward(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        features, precision_b, precision_c, correlation = self._features(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        scaled = features / self.feature_scale
        # Linear combiner on the calibrated premia, plus the correction net on
        # everything. Calibrated, not raw: the premia span decades, and the
        # feature_scale buffer already divides that out for the net.
        response = self.net(scaled).squeeze(-1) + self.pair_head(
            scaled[..., -PAIR_COUNT:]
        ).squeeze(-1)

        if self.kinks > 0:
            bumps = torch.relu(self.kink_in(scaled)) ** 2
            response = response + self.kink_out(bumps / (1.0 + bumps)).squeeze(-1)

        envelope = self.log_scale.exp() * nu2(
            m_b, m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
        )
        response_squared = torch.relu(response) ** 2

        return envelope * response_squared / (1.0 + response_squared)


class DimensionlessValueFunction(ThreeArmValue):
    """three_arm's derivation chain over the v2 premium."""

    @classmethod
    def load(cls, path: Path, kinks: int = 0) -> Self:
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        value = cls(
            ExplorationPremium(hidden, kinks=kinks, basis_hidden=basis_widths(state))
        )
        value.load_state_dict(state)

        return value


class ValueFunction(ThreeArmDeployment):
    """Deployment adapter, unchanged from three_arm."""


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    # Whenever a state dict is present it dictates the basis shape, so no path
    # on disk is consulted; only a bare --topology init reads BASIS.
    basis_hidden = basis_widths(state) if state is not None else None

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(
            ExplorationPremium(hidden, kinks=kinks, basis_hidden=basis_hidden)
        )

        if state is not None:
            value.load_state_dict(state)

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(
        ExplorationPremium(hidden, kinks=kinks, basis_hidden=basis_hidden)
    )
    value.load_state_dict(state)

    return value


if __name__ == "__main__":
    premium = ExplorationPremium([32, 32])
    draw = Sample.draw(2000).fold()
    state = (draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    u = premium(*state)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == draw.m_b.shape and u.isfinite().all()

    # The proven bound is architectural, exactly as in three_arm.
    _, precision_b, precision_c, correlation = premium._features(*state)
    bound = nu2(
        draw.m_b, draw.m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
    )

    assert (u >= 0).all() and (u <= bound + 1e-6).all()

    # The basis is frozen and IN the checkpoint: a v2 net that needed a
    # separate file to be loadable would not be reproducible.
    assert all(p.requires_grad is False for p in premium.basis.parameters())
    assert any(k.startswith("basis.") for k in premium.state_dict())
    assert all(p.requires_grad for p in premium.net.parameters())

    # The pair features are alive and differentiable: a detached one would
    # train to something plausible and mean nothing, since the learning
    # numbers are built from d/dtau (learnings, and CLAUDE.md three_arm_v2).
    probe = draw.tau_bb.clone().requires_grad_(True)
    pairs = premium.pairs(draw.m_b, draw.m_c, probe, draw.tau_bc, draw.tau_cc)
    (gradient,) = torch.autograd.grad(pairs.sum(), probe)

    assert gradient.abs().mean() > 0
    assert pairs.shape == (len(draw.m_b), 3) and pairs.std(dim=0).min() > 1e-6

    # The linear combiner is four trainable parameters, it starts silent, and
    # it moves the response once it is not zero -- a head that could not change
    # the output would be four parameters of decoration.
    assert premium.pair_head.weight.numel() + premium.pair_head.bias.numel() == 4
    assert all(p.requires_grad for p in premium.pair_head.parameters())
    assert torch.allclose(premium.pair_head.weight, torch.zeros(1, PAIR_COUNT))

    with torch.no_grad():
        premium.pair_head.weight.fill_(0.5)

    assert (premium(*state) - u).abs().max() > 0

    with torch.no_grad():
        premium.pair_head.weight.zero_()

    # Round trip through the loader, basis included.
    value = DimensionlessValueFunction(premium)
    torch.save(value.state_dict(), "/tmp/_v2_roundtrip.pt")
    again = DimensionlessValueFunction.load(Path("/tmp/_v2_roundtrip.pt"))

    assert torch.allclose(again.premium(*state), u)

    # A checkpoint carries its basis, so loading one must not read BASIS at all.
    # Point it at a nonexistent path AND at a differently shaped net: both used
    # to break the load, the second silently depending on where the champion
    # symlink happened to point.
    saved, globals()["BASIS"] = BASIS, Path("data") / "does_not_exist.pt"

    try:
        blind = DimensionlessValueFunction.load(Path("/tmp/_v2_roundtrip.pt"))
        assert torch.allclose(blind.premium(*state), u)
        assert basis_widths(value.state_dict()) == [16, 16]
        assert basis_widths({"premium.net.0.weight": torch.zeros(3, 3)}) is None
    finally:
        globals()["BASIS"] = saved
    print("ok")
