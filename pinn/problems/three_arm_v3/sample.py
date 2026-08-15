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

# Sampling constants: the floor keeps det T >= PRIOR_FLOOR**2, fencing the
# singular boundary of the information box. Numerical stability ONLY -- it
# encodes no prior; the net is trained general down to priors ~30 sd wide.
# No real experiment starts more agnostic than that, and each decade below
# costs another 100x in PDE stiffness (the learning numbers carry 1/det) for
# territory nobody visits. PRECISION_MEAN scales the chi-squared-1 law
# per pairwise precision; SCALE_DECADES is the log10 range of the common
# scale multiplying all three (jointly-low states get whole-decade mass,
# never a triple coincidence); MEAN_SCALE is the Laplace width in units of
# each mean's own posterior standard deviation.
PRIOR_FLOOR = 1e-3
PRECISION_MEAN = 2.0
SCALE_DECADES = 3.0
MEAN_SCALE = 2.0

_SOBOL = SobolEngine(dimension=6, scramble=True)
_SOBOL_WALL = SobolEngine(dimension=5, scramble=True)


@dataclass
class Sample:
    """
    A batch of collocation states, precision in matrix entries (tau_bc <= 0
    always: reachable states only).
    """

    m_b: Tensor
    m_c: Tensor
    tau_bb: Tensor
    tau_bc: Tensor
    tau_cc: Tensor

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

        # Means last, conditionally on the precisions: each mean is drawn with
        # width proportional to its own posterior standard deviation, so the
        # cloud tracks the corridor at every information level (the two_arm
        # 2/sqrt(tauhat) trick, matrix edition).
        det = tau_bb * tau_cc - tau_bc**2
        m_b = MEAN_SCALE * (tau_cc / det).sqrt() * laplace(t[:, 3])
        m_c = MEAN_SCALE * (tau_bb / det).sqrt() * laplace(t[:, 4])

        return cls(m_b, m_c, tau_bb, tau_bc, tau_cc)

    def fold(self) -> Sample:
        """
        Roll the batch into the fundamental wedge; see fold_ordered.
        """
        return self.fold_ordered()[0]

    def fold_ordered(self) -> tuple[Sample, Tensor]:
        """
        Roll the batch into the fundamental wedge {m_c <= m_b <= 0} by the arm
        relabel that sorts the three arm levels (0, m_b, m_c) descending: best
        becomes control, runner-up becomes b. In the pair coordinates of the precision,
        (tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc) the relabel is a pure
        permutation of pair labels -- pair {i, j} lives at index i + j - 1 --
        so the taus fold by permuting three coordinates and reassembling. The
        premium is invariant under all of this (doc section 6), so training
        folds freely. Policy readout needs the applied relabel back: the
        returned order maps wedge role k to physical arm order[:, k]
        (0 = a, 1 = b, 2 = c), so wedge-role allocations un-permute by
        alpha_physical.scatter_(1, order, alpha_roles).
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

        return Sample(m_b, m_c, c_ab + c_bc, -c_bc, c_ac + c_bc), order


@dataclass
class RidgeSample:
    """
    A batch of wall states, four coordinates: one free mean plus the precision
    entries. The missing fifth coordinate is implied by which wall the batch
    is drawn for: on the control tie, mean is m_c and m_b = 0; on the
    treatment tie, mean is the common value m_b = m_c. Same reachability
    contract as Sample (tau_bc <= 0).
    """

    mean: Tensor
    tau_bb: Tensor
    tau_bc: Tensor
    tau_cc: Tensor

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
    Uniforms -> pairwise precisions (chi-squared-1 shapes times one common
    log-spread scale, every draw reachable) -> precision entries via the doc
    section 7 affine map. The common scale spans SCALE_DECADES decades
    downward, densest at 1: it moves all three precisions together, so
    low-information states of every magnitude are first-class events.

    The floor is on ALL THREE PAIR COORDINATES, which is what makes it
    S3-symmetric and therefore fold-invariant. three_arm floors the two
    diagonal entries instead, leaving the b-c coordinate bare; the fold then
    permutes that bare coordinate into any slot, so every Schur marginal
    reaches ~0.5x the floor and the two_arm base is called outside its
    support. Measured on this problem 2026-08-15: those states are 1.0% of
    the cloud and carried 99.996% of the pde loss (mean residual squared
    1.2e3 against 4.7e-4 fenced). Flooring the pair coordinates gives every
    marginal >= the floor by construction -- no rejection step.
    """
    scale = PRECISION_MEAN * decade_scale(u_scale, SCALE_DECADES)
    precision_ab = PRIOR_FLOOR + scale * chi_squared_1(u_ab)
    precision_ac = PRIOR_FLOOR + scale * chi_squared_1(u_ac)
    precision_bc = PRIOR_FLOOR + scale * chi_squared_1(u_bc)

    tau_bb = precision_ab + precision_bc
    tau_cc = precision_ac + precision_bc
    tau_bc = -precision_bc

    return tau_bb, tau_bc, tau_cc


if __name__ == "__main__":
    draw = Sample.draw(1000)

    assert draw.m_b.shape == draw.tau_bc.shape == (1000,)
    assert (draw.tau_bc <= 0).all()
    # All THREE pair coordinates floored, the b-c one included: that is what
    # survives the fold, and it is what keeps every Schur marginal (the
    # states the two_arm base is called at) inside the base's support.
    for folded_draw in (draw, draw.fold()):
        assert (folded_draw.tau_bb + folded_draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()
        assert (folded_draw.tau_cc + folded_draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()
        assert (-folded_draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()

        determinant = folded_draw.tau_bb * folded_draw.tau_cc - folded_draw.tau_bc**2
        marginals = torch.stack(
            [
                determinant / folded_draw.tau_cc,
                determinant / folded_draw.tau_bb,
                determinant
                / (folded_draw.tau_bb + folded_draw.tau_cc + 2.0 * folded_draw.tau_bc),
            ]
        )

        assert (marginals >= PRIOR_FLOOR - 1e-6).all(), marginals.min().item()

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
    assert (folded.tau_bc <= 1e-6).all()

    refolded = folded.fold()

    for name in vars(folded):
        assert torch.allclose(getattr(refolded, name), getattr(folded, name), atol=1e-5)

    # det T is preserved because the relabels are congruences by permutation
    # matrices. That is exact algebra, so test it in float64: in float32 the
    # fold's own pair-coordinate reassembly cancels (tau_bb + tau_bc is
    # precision_ab, a difference of similar numbers when precision_bc
    # dominates) and det then floors at PRIOR_FLOOR**2 = 1e-6 against O(1)
    # products -- about one significant digit. Asserted in float32 at
    # rtol=1e-4 this failed on ~7% of unseeded Sobol scrambles, silently,
    # because the scramble is drawn from the global rng at import.
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
