"""
State-space sampling for the three-arm problem: the interior cloud, the two
wall families, and the fold into the fundamental wedge.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import Tensor
from torch.quasirandom import SobolEngine

from ...utils import chi_squared_1, decade_scale, exponential, laplace

PRIOR_FLOOR = 1e-3
"""
Keeps det T >= PRIOR_FLOOR**2, fencing the singular boundary of the information
box. Numerical stability **only**: it encodes no prior, and the net trains general
down to priors ~30 sd wide. Each decade below costs another 100x in PDE stiffness
(the learning numbers carry 1/det) for territory nobody visits.
"""

PRECISION_MEAN = 2.0
"""Scale of the chi-squared-1 law drawn for each pairwise precision."""

SCALE_DECADES = 3.0
"""
The log10 range of the common scale multiplying all three precisions. It moves
them together, so jointly-low states get whole-decade mass instead of needing a
triple coincidence.
"""

MEAN_SCALE = 2.0
"""The Laplace width, in units of each mean's own posterior standard deviation."""

FUNNEL_SHARE = 0.25
"""
Share of interior draws given a *negative* bc pair precision (tau_bc > 0):
the negatively-correlated funnel, unreachable from any experiment history but
open to hand-authored priors, and transient (learning and drift both drain it
into the reachable set). Sampled so the served function is trained there
instead of extrapolated; 2026-08-18's probe measured up to 24% underclaims
against the drop-one floor and 0.49 policy jumps at the funnel mouth on the
extrapolating champion. Walls stay reachable: the tie conditions are
statements about the reachable boundary.
"""

DET_KEEP = 1e-3
"""
The funnel's relative det floor: det >= DET_KEEP * AB, which keeps ~4 of
float32's ~7 digits through the cancellation and still reaches precision
correlations near sqrt(1 - DET_KEEP) ~ 0.9995.
"""

_SOBOL = SobolEngine(dimension=7, scramble=True)
"""
Seven uniforms per interior state: three precisions, two means, one common
scale, one funnel branch-and-depth.
"""

_SOBOL_WALL = SobolEngine(dimension=5, scramble=True)
"""Five uniforms per wall state: the interior draw with one mean left out."""


@dataclass
class Sample:
    """
    A batch of collocation states, with the precision in matrix entries. The
    a-pair coordinates are floored and det >= PRIOR_FLOOR**2 always; tau_bc is
    <= 0 on the reachable branch and > 0 on the FUNNEL_SHARE of draws that
    train the negatively-correlated funnel.
    """

    m_b: Tensor
    """Posterior mean of arm b, as a contrast against the control."""

    m_c: Tensor
    """Posterior mean of arm c, as a contrast against the control."""

    tau_bb: Tensor
    """Precision matrix entry for b against itself."""

    tau_bc: Tensor
    """Off-diagonal precision entry, never positive on a reachable state."""

    tau_cc: Tensor
    """Precision matrix entry for c against itself."""

    @classmethod
    def draw(cls, n: int) -> Sample:
        """
        Scrambled 6-D Sobol pushed through the sampling alchemy of doc
        sections 6-7: a common information scale, three pairwise precisions,
        two conditionally-scaled means.
        """
        t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 5]
        )

        # The funnel branch: FUNNEL_SHARE of draws swap their bc pair precision
        # for a negative one, depth uniform up to the ceiling that keeps
        # det = AB - q(A + B) above a **relative** floor: the funnel reaches the
        # det floor by cancellation of O(AB) products, and float32 carries ~7
        # digits, so an absolute 1e-6 target under O(1e2) products computes
        # negative and NaNs the mean draw (kb section 19.7's lesson; it killed
        # the first funnel run at iteration 3k). The swap leaves the a-pair
        # coordinates invariant (tau_bb + tau_bc is A on both branches).
        into_funnel = t[:, 6] < FUNNEL_SHARE
        a_pair = tau_bb + tau_bc
        b_pair = tau_cc + tau_bc
        det_floor = torch.maximum(
            torch.full_like(a_pair, PRIOR_FLOOR**2),
            DET_KEEP * a_pair * b_pair,
        )
        depth = (t[:, 6] / FUNNEL_SHARE).clamp(0.0, 1.0) * (
            (a_pair * b_pair - det_floor) / (a_pair + b_pair)
        ).clamp_min(0.0)
        tau_bb = torch.where(into_funnel, a_pair - depth, tau_bb)
        tau_cc = torch.where(into_funnel, b_pair - depth, tau_cc)
        tau_bc = torch.where(into_funnel, depth, tau_bc)

        # Means last, conditionally on the precisions: width proportional to
        # each posterior standard deviation, so the cloud tracks the corridor at
        # every information level (the two_arm 2/sqrt(tauhat) trick, in matrix).
        det = tau_bb * tau_cc - tau_bc**2
        m_b = MEAN_SCALE * (tau_cc / det).sqrt() * laplace(t[:, 3])
        m_c = MEAN_SCALE * (tau_bb / det).sqrt() * laplace(t[:, 4])

        return cls(m_b, m_c, tau_bb, tau_bc, tau_cc)

    def fold(self) -> Sample:
        """Roll the batch into the fundamental wedge; see `fold_ordered`."""
        return self.fold_ordered()[0]

    def fold_ordered(self) -> tuple[Sample, Tensor]:
        """
        Roll the batch into the fundamental wedge {m_c <= m_b <= 0} by the relabel
        sorting the arm levels (0, m_b, m_c) descending; in pair coordinates that
        is a permutation, and the premium is invariant under it (doc sections 6,
        7). The returned order maps wedge role k to physical arm order[:, k].
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
            """The pair coordinate of the two named arms, read at index i + j - 1."""
            return pair_coordinates.gather(
                -1, (arm_one + arm_two - 1).unsqueeze(-1)
            ).squeeze(-1)

        c_ab = pair_coordinate(order[:, 0], order[:, 1])
        c_ac = pair_coordinate(order[:, 0], order[:, 2])
        c_bc = pair_coordinate(order[:, 1], order[:, 2])

        return Sample(m_b, m_c, c_ab + c_bc, -c_bc, c_ac + c_bc), order


@dataclass
class RidgeSample:
    """
    A batch of wall states, four coordinates: one free mean plus the precision
    entries. The missing fifth coordinate is implied by which wall the batch is
    drawn for: on the control tie, mean is m_c and m_b = 0; on the treatment tie,
    mean is the common value m_b = m_c. Same reachability contract as `Sample`.
    """

    mean: Tensor
    """The wall's one free mean; which mean it is depends on the wall."""

    tau_bb: Tensor
    """Precision matrix entry for b against itself."""

    tau_bc: Tensor
    """Off-diagonal precision entry, never positive on a reachable state."""

    tau_cc: Tensor
    """Precision matrix entry for c against itself."""

    @classmethod
    def control_tie(cls, n: int) -> RidgeSample:
        """
        Wall states on the control tie {m_b = 0}: the free mean is m_c <= 0,
        drawn as a one-sided Laplace at MEAN_SCALE times its own posterior
        standard deviation.
        """
        t = _SOBOL_WALL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 4]
        )

        det = tau_bb * tau_cc - tau_bc**2
        m_c = -MEAN_SCALE * (tau_bb / det).sqrt() * exponential(t[:, 3])

        return cls(m_c, tau_bb, tau_bc, tau_cc)

    @classmethod
    def treatment_tie(cls, n: int) -> RidgeSample:
        """
        Wall states on the treatment tie {m_b = m_c}: the free mean is the
        common value m <= 0, scaled by the average of the two posterior
        standard deviations.
        """
        t = _SOBOL_WALL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(
            t[:, 0], t[:, 1], t[:, 2], t[:, 4]
        )

        det = tau_bb * tau_cc - tau_bc**2
        deviation = 0.5 * ((tau_cc / det).sqrt() + (tau_bb / det).sqrt())
        mean = -MEAN_SCALE * deviation * exponential(t[:, 3])

        return cls(mean, tau_bb, tau_bc, tau_cc)


def _precision_from_uniforms(
    u_ab: Tensor, u_ac: Tensor, u_bc: Tensor, u_scale: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Uniforms into pairwise precisions (chi-squared-1 shapes times one common
    log-spread scale, every draw reachable), then into precision entries via the
    doc section 7 affine map, the floor keeping T invertible on the box faces.
    One scale moves all three, so every magnitude of ignorance is first-class.
    """
    scale = PRECISION_MEAN * decade_scale(u_scale, SCALE_DECADES)
    precision_ab = scale * chi_squared_1(u_ab)
    precision_ac = scale * chi_squared_1(u_ac)
    precision_bc = scale * chi_squared_1(u_bc)

    tau_bb = PRIOR_FLOOR + precision_ab + precision_bc
    tau_cc = PRIOR_FLOOR + precision_ac + precision_bc
    tau_bc = -precision_bc

    return tau_bb, tau_bc, tau_cc


if __name__ == "__main__":
    draw = Sample.draw(1000)

    assert draw.m_b.shape == draw.tau_bc.shape == (1000,)
    assert (draw.tau_bb + draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()
    assert (draw.tau_cc + draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()

    # The funnel branch fills its share, and only its share, of the cloud,
    # and its det keeps the relative floor that makes float32 survivable.
    funnel_share = (draw.tau_bc > 0).float().mean().item()

    assert abs(funnel_share - FUNNEL_SHARE) < 0.05, funnel_share

    in_funnel = draw.tau_bc > 0
    products = (draw.tau_bb + draw.tau_bc) * (draw.tau_cc + draw.tau_bc)
    funnel_det = (draw.tau_bb * draw.tau_cc - draw.tau_bc**2)[in_funnel]

    assert (funnel_det >= 0.5 * DET_KEEP * products[in_funnel]).all()

    det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2

    assert (det >= PRIOR_FLOOR**2 - 1e-6).all()

    # Fold: lands in the wedge, is idempotent, and preserves det T (the
    # relabels are congruences by determinant -1/+1 matrices). The order
    # really sorts the physical levels descending.
    folded, order = draw.fold_ordered()
    physical_levels = torch.stack([torch.zeros_like(draw.m_b), draw.m_b, draw.m_c], -1)
    sorted_levels = physical_levels.gather(-1, order)

    assert (sorted_levels.diff(dim=-1) <= 1e-6).all()

    assert (folded.m_b <= 1e-6).all() and (folded.m_c <= folded.m_b + 1e-6).all()

    refolded = folded.fold()

    for name in vars(folded):
        assert torch.allclose(getattr(refolded, name), getattr(folded, name), atol=1e-5)

    # det T is preserved by the relabel congruences, which is exact algebra, so
    # test it in float64: in float32 the reassembly cancels and this fails on ~7%
    # of unseeded scrambles (kb section 19.7).
    wide = Sample(*(field.double() for field in vars(draw).values()))
    wide_folded, _ = wide.fold_ordered()

    det_before = wide.tau_bb * wide.tau_cc - wide.tau_bc**2
    det_after = wide_folded.tau_bb * wide_folded.tau_cc - wide_folded.tau_bc**2

    assert torch.allclose(det_after, det_before, rtol=1e-10)

    # Wall samplers: correct sign conventions and reachability.
    for wall in (RidgeSample.control_tie(500), RidgeSample.treatment_tie(500)):
        assert (wall.mean <= 0).all() and (wall.tau_bc <= 0).all()
        assert (wall.tau_bb + wall.tau_bc >= PRIOR_FLOOR - 1e-6).all()
        assert (wall.tau_cc + wall.tau_bc >= PRIOR_FLOOR - 1e-6).all()
    print("ok")
