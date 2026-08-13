"""
State-space sampling for the three-arm drift problem: three_arm's cloud with
the drift coordinate and the ceiling it implies.
"""

from __future__ import annotations

import math
import torch

from dataclasses import dataclass
from torch import Tensor
from torch.quasirandom import SobolEngine

from ...utils import chi_squared_1, decade_scale, exponential, laplace

# three_arm's constants, unchanged: the state law IS three_arm's, and drift
# only clips it from above (see _precision_from_uniforms). Keeping the numbers
# identical is what makes the low-etahat slice the static problem rather than
# something adjacent to it.
PRIOR_FLOOR = 1e-3
PRECISION_MEAN = 2.0
SCALE_DECADES = 3.0
MEAN_SCALE = 2.0

# Additive fence on the three SHAPE draws, so a triple-near-zero chi-squared
# cannot send the ceiling multiplier to infinity. Relative, not absolute: the
# state is scaled to the ceiling right after, so an absolute floor here would
# mean a different thing at every etahat.
SHAPE_FLOOR = 1e-3

# Absolute cap on the det ceiling as etahat -> 0 (two_arm_drift's TAUHAT_MAX,
# det edition): the ceiling diverges as 1/etahat^2, and uncapped it put 17% of
# the cloud beyond det ~ 1e2 (up to 3.8e12, measured 2026-08-09) -- decades
# the static law never reaches (its max det ~ 8e2) and whose taus miscalibrate
# feature_scale ~1000x on the raw tau features while railing grafted nets'
# first tanh layer. 1e3 sits just above the static law's reach.
DET_MAX = 1e3

# etahat law, two_arm_drift's shape at 1/sqrt2 of its scale: two_arm's eta is a
# contrast volatility and ours is per arm, so the same physical world sits
# lower here (doc section 0).
ETAHAT_SCALE = 14.0
ETAHAT_DECADES = 4.0
ETAHAT_MAX = 35.0

SQRT3 = math.sqrt(3.0)

_SOBOL = SobolEngine(dimension=9, scramble=True)
_SOBOL_WALL = SobolEngine(dimension=8, scramble=True)


@dataclass
class Sample:
    """
    A batch of collocation states, precision in matrix entries (tau_bc <= 0
    always: reachable states only), plus the drift each state is drawn for.
    """

    m_b: Tensor
    m_c: Tensor
    tau_bb: Tensor
    tau_bc: Tensor
    tau_cc: Tensor
    etahat: Tensor

    @classmethod
    def draw(cls, n: int) -> Sample:
        """
        Scrambled 9-D Sobol: the drift first, then the precision as a fraction
        of the ceiling that drift implies, then the two means conditionally.
        """
        t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        etahat = _etahat(t[:, 7], t[:, 8])
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 5], etahat
        )

        # Means last, conditionally on the precisions: each mean is drawn with
        # width proportional to its own posterior standard deviation, so the
        # cloud tracks the corridor at every information level.
        det = tau_bb * tau_cc - tau_bc**2
        m_b = MEAN_SCALE * (tau_cc / det).sqrt() * laplace(t[:, 3])
        m_c = MEAN_SCALE * (tau_bb / det).sqrt() * laplace(t[:, 4])

        return cls(m_b, m_c, tau_bb, tau_bc, tau_cc, etahat)

    def fold(self) -> Sample:
        """
        Roll the batch into the fundamental wedge; see fold_ordered.
        """
        return self.fold_ordered()[0]

    def fold_ordered(self) -> tuple[Sample, Tensor]:
        """
        three_arm's fold, unchanged, with etahat carried through: shuffling arm
        labels leaves the drift alone (doc section 1), so it is not folded, it
        just rides along. Read three_arm/sample.py for what the fold does.
        """
        levels = torch.stack([torch.zeros_like(self.m_b), self.m_b, self.m_c], dim=-1)
        order = levels.argsort(dim=-1, descending=True)
        sorted_levels = levels.gather(-1, order)
        m_b = sorted_levels[:, 1] - sorted_levels[:, 0]
        m_c = sorted_levels[:, 2] - sorted_levels[:, 0]

        pair_coordinates = torch.stack(
            [
                self.tau_bb + self.tau_bc,
                self.tau_cc + self.tau_bc,
                -self.tau_bc,
            ],
            dim=-1,
        )

        def pair_coordinate(arm_one: Tensor, arm_two: Tensor) -> Tensor:
            return pair_coordinates.gather(
                -1, (arm_one + arm_two - 1).unsqueeze(-1)
            ).squeeze(-1)

        c_ab = pair_coordinate(order[:, 0], order[:, 1])
        c_ac = pair_coordinate(order[:, 0], order[:, 2])
        c_bc = pair_coordinate(order[:, 1], order[:, 2])

        return (
            Sample(m_b, m_c, c_ab + c_bc, -c_bc, c_ac + c_bc, self.etahat),
            order,
        )


@dataclass
class RidgeSample:
    """
    A batch of wall states: three_arm's four coordinates plus the drift. Both
    wall conditions hold at every etahat (doc section 5), so it is sampled here
    too.
    """

    mean: Tensor
    tau_bb: Tensor
    tau_bc: Tensor
    tau_cc: Tensor
    etahat: Tensor

    @classmethod
    def control_tie(cls, n: int) -> RidgeSample:
        """
        Wall states on the control tie {m_b = 0}: the free mean is m_c <= 0.
        """
        t = _SOBOL_WALL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        etahat = _etahat(t[:, 6], t[:, 7])
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 4], etahat
        )

        det = tau_bb * tau_cc - tau_bc**2
        m_c = -MEAN_SCALE * (tau_bb / det).sqrt() * exponential(t[:, 3])

        return cls(m_c, tau_bb, tau_bc, tau_cc, etahat)

    @classmethod
    def treatment_tie(cls, n: int) -> RidgeSample:
        """
        Wall states on the treatment tie {m_b = m_c}: the free mean is the
        common value m <= 0.
        """
        t = _SOBOL_WALL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        etahat = _etahat(t[:, 6], t[:, 7])
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 4], etahat
        )

        det = tau_bb * tau_cc - tau_bc**2
        deviation = 0.5 * ((tau_cc / det).sqrt() + (tau_bb / det).sqrt())
        mean = -MEAN_SCALE * deviation * exponential(t[:, 3])

        return cls(mean, tau_bb, tau_bc, tau_cc, etahat)


def _etahat(u_scale: Tensor, u_tail: Tensor) -> Tensor:
    """
    Decade-spread scale times an Exp tail, reaching 0 (the three_arm anchor).
    """
    drawn = ETAHAT_SCALE * decade_scale(u_scale, ETAHAT_DECADES) * exponential(u_tail)

    return drawn.clamp(max=ETAHAT_MAX)


def _precision_from_uniforms(
    u_ab: Tensor,
    u_ac: Tensor,
    u_bc: Tensor,
    u_scale: Tensor,
    etahat: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Uniforms -> pairwise precisions -> precision entries: three_arm's law,
    clipped at the drift ceiling.

    Drift caps how much can ever be known: what you buy stops keeping up with
    what the wandering destroys, and the cap has a shape, not just a size (doc
    section 6). Its determinant is the part that is both provable and unchanged
    by shuffling arm labels, so that is what is imposed here:

        det T  <=  det T*  =  1 / (2 sqrt3 etahat^2)

    The three chi-squared draws set only the SHAPE of T; three_arm's own scale
    law sizes it, and the ceiling only CLIPS -- so where drift does not bind
    this sampler IS three_arm's, and the clipped mass lands on the ceiling
    where trajectories converge.

    Drawing a FRACTION of the ceiling instead (until 2026-08-13) made every
    state ceiling-relative, so the low-drift slice inflated with the ceiling
    rather than matching the static problem: at etahat < 0.01 its median det
    was 1,250x three_arm's and its p05 42,000x, which pushed the means 6x
    small (they are drawn conditionally on det). The net trained on
    high-information states and was graded on the stiff low-information ones.

    Placing det rather than flooring the pair coordinates afterwards is what
    makes both bounds EXACT. three_arm floors the taus additively, which under
    drift lifts a lopsided state back over the ceiling -- measured at 0.89% of
    the cloud, and it made this module's own check flaky 5 runs in 10. The
    floor's stated purpose in three_arm is det anyway ("keeps det T >=
    PRIOR_FLOOR**2"), so imposing it on det directly is the same intent without
    the failure mode. The shapes get a small additive fence of their own so a
    triple-near-zero draw cannot send the multiplier to infinity.
    """
    shape_ab = chi_squared_1(u_ab) + SHAPE_FLOOR
    shape_ac = chi_squared_1(u_ac) + SHAPE_FLOOR
    shape_bc = chi_squared_1(u_bc) + SHAPE_FLOOR

    # det of the unscaled shape, in pair coordinates.
    shape_det = shape_ab * shape_ac + shape_bc * (shape_ab + shape_ac)

    # three_arm's own scale law, ABSOLUTE, and its det.
    scale = PRECISION_MEAN * decade_scale(u_scale, SCALE_DECADES)
    static_det = scale**2 * shape_det

    # etahat floored where the drift ceiling meets the static cap DET_MAX, so
    # the etahat -> 0 anchor stays inside the decades training can see.
    ceiling_det = 1.0 / (
        2.0 * SQRT3 * etahat.clamp_min((2.0 * SQRT3 * DET_MAX) ** -0.5) ** 2
    )
    target_det = static_det.clamp(max=ceiling_det).clamp_min(PRIOR_FLOOR**2)

    # det scales as the square, so the linear multiplier is the square root.
    multiplier = (target_det / shape_det).sqrt()

    pair_bc = multiplier * shape_bc

    return (
        multiplier * shape_ab + pair_bc,
        -pair_bc,
        multiplier * shape_ac + pair_bc,
    )


if __name__ == "__main__":
    draw = Sample.draw(20000)

    assert draw.m_b.shape == draw.tau_bc.shape == draw.etahat.shape == (20000,)
    assert (draw.tau_bc <= 0).all()

    # Pair coordinates strictly inside the box. The floor moved onto det (see
    # _precision_from_uniforms), so these are only required positive -- an
    # absolute floor on them is what broke the ceiling.
    assert (draw.tau_bb + draw.tau_bc > 0).all()
    assert (draw.tau_cc + draw.tau_bc > 0).all()
    assert (draw.etahat >= 0.0).all() and (draw.etahat <= ETAHAT_MAX).all()

    det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2

    # Relative slack, same float32 cancellation as the ceiling check below.
    assert (det >= PRIOR_FLOOR**2 * 0.99).all(), det.min().item()
    assert (det <= DET_MAX * 1.001).all(), det.max().item()

    # The ceiling, on the PHYSICAL quantity and with nothing subtracted off the
    # left side -- subtracting the floor back out would only test that clamp
    # clamps.
    ratio = 2.0 * SQRT3 * draw.etahat**2 * det

    # Exact in exact arithmetic -- det is placed, not floored. The slack is
    # float32: det is a difference of similar numbers when the b-c pair
    # dominates, which costs about four digits (three_arm's own det check is
    # in float64 for the same reason). Measured worst excess 1.6e-4.
    assert (ratio <= 1.0 + 1e-3).all(), ratio.max().item()

    # The ceiling CLIPS, it does not SIZE. Where drift is negligible nothing
    # should sit on it; where drift is real most states should. Drawing a
    # fraction OF the ceiling (until 2026-08-13) inverted this and inflated
    # the whole low-drift slice.
    quiet, loud = draw.etahat < 0.01, draw.etahat > 10.0

    assert (ratio[quiet] > 0.9).float().mean() < 0.01, "clip binds at etahat ~ 0"
    assert (ratio[loud] > 0.9).float().mean() > 0.4, "ceiling unreached under drift"

    # And the quiet slice IS three_arm: same constants, ceiling inactive, so
    # its det distribution must match. This is the regression test for the bug
    # above, which put the median 1,250x out.
    from ..three_arm.sample import Sample as StaticSample

    static = StaticSample.draw(40000)
    static_det = static.tau_bb * static.tau_cc - static.tau_bc**2

    for level in (0.25, 0.5, 0.75):
        mine = det[quiet].quantile(level).item()
        theirs = static_det.quantile(level).item()

        assert 0.4 < mine / theirs < 2.5, (level, mine, theirs)

    # The law's own high decades carry real mass, and etahat still reaches
    # the three_arm anchor at 0.
    assert (draw.etahat < 0.01).float().mean() > 0.05, "no mass at the anchor"
    assert ((draw.etahat > 3.0) & (draw.etahat < 20.0)).float().mean() > 0.15
    assert draw.etahat.min().item() < 1e-4

    # Fold: lands in the wedge, is idempotent, preserves det T, and carries the
    # drift through untouched.
    folded, order = draw.fold_ordered()
    physical_levels = torch.stack([torch.zeros_like(draw.m_b), draw.m_b, draw.m_c], -1)
    sorted_levels = physical_levels.gather(-1, order)

    assert (sorted_levels.diff(dim=-1) <= 1e-6).all()
    assert (folded.m_b <= 1e-6).all() and (folded.m_c <= folded.m_b + 1e-6).all()
    assert (folded.tau_bc <= 1e-6).all()
    assert torch.equal(folded.etahat, draw.etahat), "the fold dropped etahat"

    refolded = folded.fold()

    for name in vars(folded):
        assert torch.allclose(getattr(refolded, name), getattr(folded, name), atol=1e-5)

    # det T preserved by the fold, in float64 for the reason three_arm records.
    wide = Sample(*(field.double() for field in vars(draw).values()))
    wide_folded, _ = wide.fold_ordered()

    det_before = wide.tau_bb * wide.tau_cc - wide.tau_bc**2
    det_after = wide_folded.tau_bb * wide_folded.tau_cc - wide_folded.tau_bc**2

    assert torch.allclose(det_after, det_before, rtol=1e-10)

    # Wall samplers: sign conventions, reachability, drift present.
    for wall in (RidgeSample.control_tie(2000), RidgeSample.treatment_tie(2000)):
        wall_det = wall.tau_bb * wall.tau_cc - wall.tau_bc**2
        assert (wall.mean <= 0).all() and (wall.tau_bc <= 0).all()
        assert (wall.tau_bb + wall.tau_bc > 0).all()
        assert (wall.tau_cc + wall.tau_bc > 0).all()
        assert (wall_det >= PRIOR_FLOOR**2 * 0.99).all(), wall_det.min().item()
        assert (wall_det <= DET_MAX * 1.001).all(), wall_det.max().item()
        assert (2.0 * SQRT3 * wall.etahat**2 * wall_det <= 1.0 + 1e-3).all()
        assert (wall.etahat >= 0.0).all() and (wall.etahat <= ETAHAT_MAX).all()
        assert (wall.etahat < 0.01).float().mean() > 0.03
    print("ok")
