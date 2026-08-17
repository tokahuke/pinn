"""
The three-arm drift objective: maximize the premium subject to not
overclaiming. three_arm/loss.py is the same objective without drift and
kb/three_arm.md section 18 is its record; read that first, then this.

Everything three_arm's version says carries over -- the climb needs no units
factor (value-form grading), and concavity is not subsumed by feasibility
because at N = 3 the max can sit on a simplex EDGE. Two things are different
in degree rather than kind, and both come from this being the least converged
of the four problems:

- CONCAVITY IS LIVE HERE. three_arm's champion violates it on 0.02% of states,
  so the term is a guard; this problem's best net violates on 8.6%, so it is
  doing real work and its weight is load-bearing rather than nominal.
- SLACK_PRICE SHIPS AT 0.5 HERE AND AT 0 EVERYWHERE ELSE, which is not an
  oversight. On this problem the bound and the policy are INVERTED: annealing
  the price down to 0.005 took overshoot from 2.4e-2 to 1.4e-3 and made the
  ARENA WORSE (48,686 -> 60,442 regret), and the price-0.1 stage was
  catastrophic at 164,776. The champion promoted 2026-08-17 is the SYMMETRIC
  net, so the objective that produced it is the objective that ships.
  kb/three_arm_drift.md section 11 has the ladder and the arena table.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from torch import Tensor

from ...train import Objective
from ..three_arm.loss import directional_learning, on_device
from .model import DimensionlessValueFunction
from .sample import RidgeSample, Sample

# The ties PLACE the solution: the climb term kills the never-explore
# degeneracy on its own, but nothing else in the objective knows where the free
# boundary belongs. Carried over from the two-sided loss, where it was set
# against a pde median of 2.4; the anchor is now a violation of 3.6e-2, so this
# is provisional and wants re-deriving once the retrain settles.
TIE_WEIGHT = 2.4e2

# OFF: SLACK_PRICE is the upward pull now, and the ties are the degeneracy
# breaker they were for years under the two-sided loss. The 10x-dead-solution
# -floor rule that set this at 2.4e-1 is WRONG on a problem whose violation is
# two orders above three_arm's -- measured on the cold start, the climb term
# came to 1.08 against a violation of 0.18, so the objective was 85% "maximize
# u" and the premium duly ran to 3x the champion's. That rule only looked sane
# on three_arm because its violation was already tiny.
CLIMB_WEIGHT = 0.0

# What one unit of SLACK costs, as a share of the residual budget: the two
# sides are priced (1 - SLACK_PRICE) and SLACK_PRICE, so they sum to 1 and
# moving this reallocates between them WITHOUT rescaling the objective. The
# pinball loss at q = 1 - SLACK_PRICE; 0 is the pure subsolution objective,
# 0.5 the symmetric two-sided loss in L1.
#
# NONZERO HERE, unlike the three promoted problems, because this net is a COLD
# START and they were polished from converged two-sided nets. At 0 the climb is
# a global mean that cannot say WHERE to climb: measured on two_arm from
# scratch, the premium settled 37% below V* while the learning number inflated
# to 9x the champion's, and pricing slack at 0.02 fixed both (two_arm/loss.py
# carries the table). It is also the direct charge for the runaway this
# problem showed at every CLIMB_WEIGHT including 0 -- inflating max H drives
# the residual negative, which is exactly what slack now costs.
SLACK_PRICE = 0.5

# Concavity, three_arm's term verbatim -- the erosion is control-free, so the
# drift Hessian is the static one, and the SCALE is three_arm's too. It sat at
# 1e-1 until 2026-08-13 "because this net is far from converged", which made
# violation mean something different in the two problems and left the terms
# incomparable across the pair.
CONCAVITY_SCALE = 1.0e-3
# Calibrated on the MEDIAN pde over seven draws, not one: the pde varies
# several-fold between batches and a single-draw calibration put this at 730%
# of it. Target ~5%; re-derived 2026-08-13 against the pair-floor fence in
# sample.py, which cut the champion's median pde ~2x more and left the old
# 1.2 at 28% of the equation. TIE_WEIGHT remeasured at the same time: 4.8%
# and still 2.2x the 100x dead-solution floor, kept.
CONCAVITY_WEIGHT = 2.2e-1


def subsolution_loss(
    value: DimensionlessValueFunction, draw: Sample
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns THREE numbers: the violation, the climb and the concavity term.

    The violation is the POSITIVE part of the residual of the HJB in value form
    (doc section 2; the static pieces in three_arm.md sections 4, 7, 10),
    units rho = sigma = 1, i.e. the dimensionless form (doc section 4):
    v = max over the simplex of alpha.m + pairwise learning terms; graded in
    the equation's OWN units, never scaled (three_arm/loss.py holds the rule).

    KNOWN DEGENERATE on its own: the commit envelope v = max(0, m_b, m_c)
    zeroes this residual exactly (the never-explore solution), just as the
    two_arm residual was degenerate before BC1. The tie losses of doc
    section 5 break the degeneracy.

    The second is the sampled-direction concavity term, three_arm's verbatim
    (see three_arm/loss.py for the derivation and design record) --
    the erosion is control-free and never touches the Hamiltonian's Hessian.
    """
    # The derivation (learning numbers, Hamiltonian, simplex max) lives on
    # the model: one chain serves training and policy readout.
    v, best, (l_ab, l_ac, l_bc) = value.hamiltonian(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat
    )
    theta = torch.rand_like(l_ab) * math.pi
    non_concave = torch.relu(-directional_learning(l_ab, l_ac, l_bc, theta))
    # SATURATED, as two_arm_drift's positivity term: relu is linear in depth,
    # so one deep point dominates -- the worst carried 7x the whole term's
    # mean. y / (s + y) is bounded per point, linear below s, 1/y^2 above; NOT
    # tanh (float32 gradient underflows, CLAUDE.md traps). Zero set untouched.
    # s sits just above the median violation, so lower it as the net converges.
    concavity = (non_concave / (CONCAVITY_SCALE + non_concave)).mean()

    # NATURAL UNITS, NEVER SCALED -- the premium-units weight is gone, and the
    # standing rule against reintroducing one lives in three_arm/loss.py.
    # LINEAR, because an L1 penalty is EXACT at a finite weight where a
    # quadratic only approaches feasibility.
    residual = v - best.value
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()
    climb = value.premium(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat
    ).mean()

    return violation, climb, concavity


def control_tie_loss(value: DimensionlessValueFunction, draw: RidgeSample) -> Tensor:
    """
    Wall condition on the control tie {m_b = 0} (doc section 5), the
    degeneracy breaker. Pair-sampled: each wall point is scored together with
    its mirror under the a<->b relabel,

        dU/dm_b (s) + dU/dm_b (s') + dU/dm_c (s') = 1

    with s = (0, m_c, T), s' = (0, m_c, T'), T' the relabeled precision.
    Derivatives on the premium: it is smooth at the wall (the commit
    envelope's kink sits exactly at m_b = 0, so v must not be differentiated
    here).
    """
    m_b = torch.zeros_like(draw.mean).requires_grad_(True)
    u = value.premium(
        m_b, draw.mean, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat
    )
    (normal,) = torch.autograd.grad(
        u.sum(), m_b, create_graph=True, allow_unused=True, materialize_grads=True
    )

    # The mirror point: T' = (tau_bb + 2 tau_bc + tau_cc, -(tau_bc + tau_cc), tau_cc),
    # still reachable (its off-diagonal is -(floor + precision_ac) < 0).
    mirror_m_b = torch.zeros_like(draw.mean).requires_grad_(True)
    mirror_m_c = draw.mean.clone().requires_grad_(True)
    mirror_u = value.premium(
        mirror_m_b,
        mirror_m_c,
        draw.tau_bb + 2.0 * draw.tau_bc + draw.tau_cc,
        -(draw.tau_bc + draw.tau_cc),
        draw.tau_cc,
        draw.etahat,
    )
    mirror_normal, mirror_tangent = torch.autograd.grad(
        mirror_u.sum(),
        [mirror_m_b, mirror_m_c],
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )

    return (normal + mirror_normal + mirror_tangent - 1.0).pow(2).mean()


def treatment_tie_loss(value: DimensionlessValueFunction, draw: RidgeSample) -> Tensor:
    """
    Wall condition on the treatment tie {m_b = m_c} (doc section 5), the
    mirror condition. Pair-sampled under the b<->c swap,

        d_n U(s) + d_n U(s') = 0,    d_n = d/dm_b - d/dm_c

    with s = (m, m, T), s' = (m, m, PTP) (tau_bb and tau_cc swapped).
    Derivatives on the premium, as on the control tie.
    """

    def crossing_derivative(tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor) -> Tensor:
        m_b = draw.mean.clone().requires_grad_(True)
        m_c = draw.mean.clone().requires_grad_(True)
        u = value.premium(m_b, m_c, tau_bb, tau_bc, tau_cc, draw.etahat)
        along_b, along_c = torch.autograd.grad(
            u.sum(),
            [m_b, m_c],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

        return along_b - along_c

    crossing = crossing_derivative(draw.tau_bb, draw.tau_bc, draw.tau_cc)
    mirror_crossing = crossing_derivative(draw.tau_cc, draw.tau_bc, draw.tau_bb)

    return (crossing + mirror_crossing).pow(2).mean()


def loss(
    value: DimensionlessValueFunction,
    draw: Sample,
    control_draw: RidgeSample,
    treatment_draw: RidgeSample,
    iteration: int | None = None,
) -> Tensor:
    violation, climb, concavity = subsolution_loss(value, draw)
    control_residual = control_tie_loss(value, control_draw)
    treatment_residual = treatment_tie_loss(value, treatment_draw)

    if iteration is not None:
        print(
            f"iter {iteration}: violation {violation.item():.3e}"
            f"  climb {climb.item():.4e}"
            f"  control_tie {control_residual.item():.3e}"
            f"  treatment_tie {treatment_residual.item():.3e}"
            f"  concavity {concavity.item():.3e}"
        )

    return (
        violation
        + TIE_WEIGHT * (control_residual + treatment_residual)
        + CONCAVITY_WEIGHT * concavity
        - CLIMB_WEIGHT * climb
    )


def draw(batch: int, device: str = "cpu") -> tuple:
    """
    One step's collocation samples, in loss()'s argument order: every training
    state lives in the fundamental wedge.

    Split out of objective so the graphed trainer can hold them as fixed
    buffers: a captured cuda graph replays the same tensor addresses, so the
    sampling has to live outside it.
    """
    return (
        on_device(Sample.draw(batch).fold(), device),
        on_device(RidgeSample.control_tie(batch // 4), device),
        on_device(RidgeSample.treatment_tie(batch // 4), device),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol draws scored by
    loss.

    `device` is the ONE place the trainer's device enters the problem. Sobol
    draws on CPU (SobolEngine ignores the default device) and the batch moves
    once; everything downstream inherits from its inputs. Defaults to CPU,
    which is what the arena, probes and the module self-checks rely on.
    """

    def step(value: nn.Module, iteration: int | None) -> Tensor:
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            # 0 * m_b * everything, not zeros_like: keeps the output
            # graph-connected through EVERY input, including the precisions
            # the erosion term differentiates and etahat itself. Autograd
            # refuses to start from a constant (CLAUDE.md traps).
            product = m_b
            for field in rest:
                product = product * field

            return 0.0 * product

    # Zero premium solves the interior PDE exactly at every drift -- the
    # documented degeneracy, which doubles as the end-to-end test of the
    # derivative -> L -> erosion -> Hamiltonian pipeline. Run across the
    # etahat range: the erosion is the only new term and this is what says it
    # was wired up at all.
    zero = DimensionlessValueFunction(_ZeroPremium())

    for etahat in [0.0, 1.0, 10.0, 35.0]:
        batch = Sample.draw(2048)
        batch.etahat = torch.full_like(batch.etahat, etahat)
        envelope_violation, envelope_climb, envelope_concavity = subsolution_loss(
            zero, batch
        )

        # THE POINT OF THIS OBJECTIVE: the commit envelope is FEASIBLE at every
        # drift -- it solves the interior equation exactly -- but climbs
        # nothing, so the objective rejects it without help from the ties.
        assert envelope_violation.item() < 1e-8, (etahat, envelope_violation.item())
        assert envelope_climb.item() == 0.0, (etahat, envelope_climb.item())

        # Concavity is EXACTLY silent on the degenerate solution at every
        # drift: all learning numbers are 0, no clamp residue to tolerate.
        assert envelope_concavity.item() == 0.0, (etahat, envelope_concavity.item())

    # The erosion coefficients against the matrix form: with a premium linear
    # in ONE precision entry, left - v is exactly that entry's erosion
    # coefficient, which must equal T etahat^2 (I + 11') T -- off-diagonal
    # with coefficient one, the independent-entry chain-rule convention. The
    # only runnable pin on the erosion FORMULA: the S3 test below cannot tell
    # it from any other invariant (a dropped v v^T part, an etahat**4).
    class _PickTau(nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()

            self.index = index

        def forward(self, m_b: Tensor, m_c: Tensor, *taus: Tensor) -> Tensor:
            connected = m_b * m_c
            for field in taus:
                connected = connected * field

            return taus[self.index] + 0.0 * connected

    wide = Sample(*(field.double() for field in vars(Sample.draw(512)).values()))
    wide.etahat = torch.full_like(wide.etahat, 7.5)
    precision = torch.stack(
        [
            torch.stack([wide.tau_bb, wide.tau_bc], -1),
            torch.stack([wide.tau_bc, wide.tau_cc], -1),
        ],
        dim=-2,
    )
    spread = wide.etahat[:, None, None] ** 2 * (torch.eye(2, dtype=torch.float64) + 1.0)
    reference = precision @ spread @ precision

    for index, (row, col) in enumerate([(0, 0), (0, 1), (1, 1)]):
        picked = DimensionlessValueFunction(_PickTau(index))
        state = (wide.m_b, wide.m_c, wide.tau_bb, wide.tau_bc, wide.tau_cc, wide.etahat)
        left, _, _ = picked.hamiltonian(*state)

        assert torch.allclose(
            left - picked(*state), reference[:, row, col], rtol=1e-10
        ), index

    # The tie losses on the zero premium, both exact: the control tie reads
    # (0 + 0 + 0 - 1)^2 = 1 -- the degeneracy breaker doing its job -- and
    # the treatment tie is homogeneous, so it is satisfied.
    control_zero = control_tie_loss(zero, RidgeSample.control_tie(512))
    treatment_zero = treatment_tie_loss(zero, RidgeSample.treatment_tie(512))

    assert abs(control_zero.item() - 1.0) < 1e-6, control_zero.item()
    assert treatment_zero.item() < 1e-10, treatment_zero.item()

    # Analytic satisfiers, each for one wall only: u = m_b / 2 zeroes the
    # control tie (1/2 + 1/2 + 0 = 1) but not the treatment tie; u = m_b + m_c
    # zeroes the treatment tie (d_n = 0) but not the control tie.
    class _HalfB(nn.Module):
        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return 0.5 * m_b

    class _Symmetric(nn.Module):
        def forward(self, m_b: Tensor, m_c: Tensor, *rest: Tensor) -> Tensor:
            return m_b + m_c

    assert (
        control_tie_loss(
            DimensionlessValueFunction(_HalfB()), RidgeSample.control_tie(64)
        ).item()
        < 1e-10
    )
    assert (
        treatment_tie_loss(
            DimensionlessValueFunction(_HalfB()), RidgeSample.treatment_tie(64)
        ).item()
        > 0.1
    )
    assert (
        treatment_tie_loss(
            DimensionlessValueFunction(_Symmetric()), RidgeSample.treatment_tie(64)
        ).item()
        < 1e-10
    )
    assert (
        control_tie_loss(
            DimensionlessValueFunction(_Symmetric()), RidgeSample.control_tie(64)
        ).item()
        > 0.1
    )

    class _TinyPremium(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.head = nn.Linear(6, 1)

        def forward(self, *state: Tensor) -> Tensor:
            return self.head(torch.stack(state, dim=-1)).squeeze(-1).tanh()

    tiny = DimensionlessValueFunction(_TinyPremium())
    tiny_loss = loss(
        tiny,
        Sample.draw(512),
        RidgeSample.control_tie(128),
        RidgeSample.treatment_tie(128),
    )
    tiny_loss.backward()
    gradients = [p.grad for p in tiny.parameters()]

    assert all(g is not None and g.isfinite().all() for g in gradients)
    assert tiny_loss.item() > 0

    # Full-loss S3 test (three_arm.md section 6): with an invariant premium (det T is
    # preserved by the relabel congruences), the loss must be identical on a
    # relabeled batch. Exercises the derivative chain, the L assembly, the
    # Hamiltonian mapping, and the solver in one identity.
    class _InvariantPremium(nn.Module):
        def forward(
            self,
            m_b: Tensor,
            m_c: Tensor,
            tbb: Tensor,
            tbc: Tensor,
            tcc: Tensor,
            etahat: Tensor,
        ) -> Tensor:
            return ((tbb * tcc - tbc**2) * (1.0 + etahat)).tanh()

    # Run with the drift ON. E is unchanged by shuffling arm labels, so the
    # erosion has to be too -- this is the cheapest possible test that the
    # three erosion entries were assembled symmetrically, and it fails if any
    # of them is transposed or mis-paired.
    invariant = DimensionlessValueFunction(_InvariantPremium())
    batch = Sample.draw(2048)
    batch.etahat = torch.full_like(batch.etahat, 7.5)
    swapped = Sample(  # b <-> c relabel
        batch.m_c, batch.m_b, batch.tau_cc, batch.tau_bc, batch.tau_bb, batch.etahat
    )
    relabeled = Sample(  # a <-> b relabel: m -> Jm, T -> J^T T J
        -batch.m_b,
        batch.m_c - batch.m_b,
        batch.tau_bb + 2.0 * batch.tau_bc + batch.tau_cc,
        -(batch.tau_bc + batch.tau_cc),
        batch.tau_cc,
        batch.etahat,
    )

    def clone(state: Sample) -> Sample:
        return Sample(*(field.detach().clone() for field in vars(state).values()))

    # The graded residual only: the concavity term draws a fresh direction
    # per call (only its zero set is chart-invariant).
    original_loss, _, _ = subsolution_loss(invariant, clone(batch))
    swapped_loss, _, _ = subsolution_loss(invariant, clone(swapped))
    relabeled_loss, _, _ = subsolution_loss(invariant, clone(relabeled))

    # RELATIVE tolerance, not absolute: the natural-units loss is no longer a
    # fixed-size number (it moved 4 orders on 2026-08-10 and moves again with
    # every checkpoint), so an atol calibrated at one magnitude is meaningless
    # at another -- the old atol=1e-4 flaked one run in three. Relative is also
    # the honest float32 budget for a tail-dominated mean over a double-backward
    # chain: measured agreement is 1e-7 to 1e-5. Nothing is lost in detection --
    # a transposed or mis-paired erosion entry moves this by O(1) or more.
    assert torch.allclose(swapped_loss, original_loss, rtol=1e-3, atol=1e-9), (
        swapped_loss.item(),
        original_loss.item(),
    )
    assert torch.allclose(relabeled_loss, original_loss, rtol=1e-3, atol=1e-9), (
        relabeled_loss.item(),
        original_loss.item(),
    )
    from pathlib import Path

    champion = Path("data/three_arm_drift.pt")

    if champion.exists():
        trained = DimensionlessValueFunction.load(champion)
        fit = subsolution_loss(trained, Sample.draw(8192).fold())

        # THE TIES are the degeneracy breaker here, not the climb: the commit
        # envelope scores the priced residual 0 on BOTH sides at any
        # SLACK_PRICE, so nothing in the residual can reject it. The control
        # tie reads exactly 1.0 on it (asserted above), which at TIE_WEIGHT
        # buys a margin no live net comes close to.
        envelope_total = TIE_WEIGHT * 1.0
        fit_total = (
            fit[0]
            + TIE_WEIGHT
            * (
                control_tie_loss(trained, RidgeSample.control_tie(2048))
                + treatment_tie_loss(trained, RidgeSample.treatment_tie(2048))
            ).item()
            + CONCAVITY_WEIGHT * fit[2]
            - CLIMB_WEIGHT * fit[1]
        )

        assert fit_total < envelope_total, (fit_total, envelope_total)
    print("ok")
