"""
State-space sampling for the three-arm problem: the interior cloud, the two
wall families, and the fold into the fundamental wedge.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import Tensor
from torch.quasirandom import SobolEngine

# Sampling constants: the diagonal prior floor keeps det T >= PRIOR_FLOOR**2
# (the generalization of two_arm's tauhat >= 0.1, fencing the singular
# boundary of the information box); PRECISION_MEAN is the Exp tail per pairwise
# precision; MEAN_SCALE is the Laplace width in units of each mean's own
# posterior standard deviation.
PRIOR_FLOOR = 0.1
PRECISION_MEAN = 2.0
MEAN_SCALE = 2.0

_SOBOL = SobolEngine(dimension=5, scramble=True)
_SOBOL_WALL = SobolEngine(dimension=4, scramble=True)


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
        Scrambled 5-D Sobol pushed through the sampling alchemy of doc
        sections 6-7: three pairwise precisions + two
        conditionally-scaled means.
        """
        t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(t[:, 0], t[:, 1], t[:, 2])

        # Means last, conditionally on the precisions: each mean is drawn with
        # width proportional to its own posterior standard deviation, so the
        # cloud tracks the corridor at every information level (the two_arm
        # 2/sqrt(tauhat) trick, matrix edition).
        det = tau_bb * tau_cc - tau_bc**2
        m_b = MEAN_SCALE * (tau_cc / det).sqrt() * _laplace(t[:, 3])
        m_c = MEAN_SCALE * (tau_bb / det).sqrt() * _laplace(t[:, 4])

        return cls(m_b, m_c, tau_bb, tau_bc, tau_cc)

    def fold(self) -> Sample:
        """
        Roll the batch into the fundamental wedge {m_c <= m_b <= 0} by the arm
        relabel that sorts the three arm levels (0, m_b, m_c) descending: best
        becomes control, runner-up becomes b. In the pair coordinates of the precision,
        (tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc) the relabel is a pure
        permutation of pair labels -- pair {i, j} lives at index i + j - 1 --
        so the taus fold by permuting three coordinates and reassembling. The
        premium is invariant under all of this (doc section 6), so training
        and readout may fold freely. TODO before policy extraction: the
        applied relabel is discarded here, but the readout of alpha* needs it
        to un-permute the argmax back to physical arms.
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

        return Sample(m_b, m_c, c_ab + c_bc, -c_bc, c_ac + c_bc)


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
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(t[:, 0], t[:, 1], t[:, 2])

        det = tau_bb * tau_cc - tau_bc**2
        m_c = -MEAN_SCALE * (tau_bb / det).sqrt() * _exponential(t[:, 3])

        return cls(m_c, tau_bb, tau_bc, tau_cc)

    @classmethod
    def treatment_tie(cls, n: int) -> RidgeSample:
        """
        Wall states on the treatment tie {m_b = m_c}: the free mean is the
        common value m <= 0, scaled by the average of the two posterior
        standard deviations.
        """
        t = _SOBOL_WALL.draw(n).clamp(1e-7, 1.0 - 1.0e-7)
        tau_bb, tau_bc, tau_cc = _precision_from_uniforms(t[:, 0], t[:, 1], t[:, 2])

        det = tau_bb * tau_cc - tau_bc**2
        deviation = 0.5 * ((tau_cc / det).sqrt() + (tau_bb / det).sqrt())
        mean = -MEAN_SCALE * deviation * _exponential(t[:, 3])

        return cls(mean, tau_bb, tau_bc, tau_cc)


def _exponential(u: Tensor) -> Tensor:
    """
    Unit exponential via inverse CDF, u in (0, 1).
    """
    return -(1.0 - u).log()


def _laplace(u: Tensor) -> Tensor:
    """
    Unit Laplace via inverse CDF, u in (0, 1) -> two-sided heavy-ish tails.
    """
    centered = u - 0.5

    return -centered.sign() * (1.0 - 2.0 * centered.abs()).log()


def _precision_from_uniforms(
    u_ab: Tensor, u_ac: Tensor, u_bc: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Uniforms -> pairwise precisions (Exp tails on the box corner,
    every draw reachable) -> precision entries via the doc section 7 affine
    map, with the prior floor keeping T invertible on the box faces.
    """
    precision_ab = PRECISION_MEAN * _exponential(u_ab)
    precision_ac = PRECISION_MEAN * _exponential(u_ac)
    precision_bc = PRECISION_MEAN * _exponential(u_bc)

    tau_bb = PRIOR_FLOOR + precision_ab + precision_bc
    tau_cc = PRIOR_FLOOR + precision_ac + precision_bc
    tau_bc = -precision_bc

    return tau_bb, tau_bc, tau_cc


if __name__ == "__main__":
    draw = Sample.draw(1000)

    assert draw.m_b.shape == draw.tau_bc.shape == (1000,)
    assert (draw.tau_bc <= 0).all()
    assert (draw.tau_bb + draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()
    assert (draw.tau_cc + draw.tau_bc >= PRIOR_FLOOR - 1e-6).all()

    det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2

    assert (det >= PRIOR_FLOOR**2 - 1e-6).all()

    # Fold: lands in the wedge, is idempotent, and preserves det T (the
    # relabels are congruences by determinant -1/+1 matrices).
    folded = draw.fold()

    assert (folded.m_b <= 1e-6).all() and (folded.m_c <= folded.m_b + 1e-6).all()
    assert (folded.tau_bc <= 1e-6).all()

    refolded = folded.fold()

    for name in vars(folded):
        assert torch.allclose(getattr(refolded, name), getattr(folded, name), atol=1e-5)

    det_after = folded.tau_bb * folded.tau_cc - folded.tau_bc**2

    assert torch.allclose(det_after, det, rtol=1e-4)

    # Wall samplers: correct sign conventions and reachability.
    for wall in (RidgeSample.control_tie(500), RidgeSample.treatment_tie(500)):
        assert (wall.mean <= 0).all() and (wall.tau_bc <= 0).all()
        assert (wall.tau_bb + wall.tau_bc >= PRIOR_FLOOR - 1e-6).all()
        assert (wall.tau_cc + wall.tau_bc >= PRIOR_FLOOR - 1e-6).all()
    print("ok")
