"""
Losses for the three-arm problem: the interior HJB residual, the two tie
losses of doc section 12, and the trainer-facing objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import RidgeSample, Sample

TIE_WEIGHT = 10.0
POWER = 2.0


def pde_loss(value: DimensionlessValueFunction, draw: Sample) -> Tensor:
    """
    Interior HJB residual in value form (doc sections 4, 7, 10), units
    rho = sigma = 1, i.e. the dimensionless form (doc section 11):
    v = max over the simplex of alpha.m + pairwise learning terms; graded in
    similarity PREMIUM units (doc section 14 and its postscript).

    KNOWN DEGENERATE on its own: the commit envelope v = max(0, m_b, m_c)
    zeroes this residual exactly (the never-explore solution), just as the
    two_arm residual was degenerate before BC1. The tie losses of doc
    section 12 break the degeneracy.
    """
    # The derivation (learning numbers, Hamiltonian, simplex max) lives on
    # the model: one chain serves training and policy readout.
    v, best = value.hamiltonian(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2

    # Similarity grading in PREMIUM units (doc section 14 postscript): one
    # power of S below the equation's own units, so the never-explore mode
    # (residual ~ u ~ S^(-1/2), one power smaller than the equation scale)
    # stays loudly visible at every information level -- graded in equation
    # units it fades like S and a dead-Hamiltonian net scores well (observed
    # 2026-08-05). This weight is the analytic form of the old reactive
    # 1 + |H - commit| scale. The denominator is the sum of the three
    # pairwise precisions (the tau_bc terms telescope); S3-invariant.
    weight = (det**0.75 / (draw.tau_bb + draw.tau_cc + draw.tau_bc)).detach()

    # Power-mean attention: a plain mean starves the fat-tail subpopulation
    # (the b/c-flux blob, a flat shelf at 5-6 sd) of gradient once the bulk
    # converges. The p-mean's gradient weights each point by (g / M_p)^(P-1)
    # -- size relative to the batch's own population, annealing back to the
    # plain mean as the tail thins -- while staying a mean: same units, same
    # magnitude, P = 1 recovers mean-of-squares exactly. Normalized by the
    # detached batch mean so pow(P) sees O(1) numbers, not float32 dust.
    graded = (weight * (v - best.value)).pow(2)
    scale = graded.mean().detach().clamp_min(1e-30)

    return scale * (graded / scale).pow(POWER).mean().pow(1.0 / POWER)


def control_tie_loss(value: DimensionlessValueFunction, draw: RidgeSample) -> Tensor:
    """
    Wall condition on the control tie {m_b = 0} (doc section 12), the
    degeneracy breaker. Pair-sampled: each wall point is scored together with
    its mirror under the a<->b relabel,

        dU/dm_b (s) + dU/dm_b (s') + dU/dm_c (s') = 1

    with s = (0, m_c, T), s' = (0, m_c, T'), T' the relabeled precision.
    Derivatives on the premium: it is smooth at the wall (the commit
    envelope's kink sits exactly at m_b = 0, so v must not be differentiated
    here).
    """
    m_b = torch.zeros_like(draw.mean).requires_grad_(True)
    u = value.premium(m_b, draw.mean, draw.tau_bb, draw.tau_bc, draw.tau_cc)
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
    Wall condition on the treatment tie {m_b = m_c} (doc section 12), the
    mirror condition. Pair-sampled under the b<->c swap,

        d_n U(s) + d_n U(s') = 0,    d_n = d/dm_b - d/dm_c

    with s = (m, m, T), s' = (m, m, PTP) (tau_bb and tau_cc swapped).
    Derivatives on the premium, as on the control tie.
    """

    def crossing_derivative(tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor) -> Tensor:
        m_b = draw.mean.clone().requires_grad_(True)
        m_c = draw.mean.clone().requires_grad_(True)
        u = value.premium(m_b, m_c, tau_bb, tau_bc, tau_cc)
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
    pde_residual = pde_loss(value, draw)
    control_residual = control_tie_loss(value, control_draw)
    treatment_residual = treatment_tie_loss(value, treatment_draw)

    if iteration is not None:
        print(
            f"iter {iteration}: pde {pde_residual.item():.3e}"
            f"  control_tie {control_residual.item():.3e}"
            f"  treatment_tie {treatment_residual.item():.3e}"
        )

    return pde_residual + TIE_WEIGHT * (control_residual + treatment_residual)


def objective(batch: int = 1024) -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol draws scored by
    loss.
    """

    def step(value: nn.Module, iteration: int | None) -> Tensor:
        # The rollup: every training state lives in the fundamental wedge.
        return loss(
            value,
            Sample.draw(batch).fold(),
            RidgeSample.control_tie(batch // 4),
            RidgeSample.treatment_tie(batch // 4),
            iteration,
        )

    return step


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            # 0 * m_b, not zeros_like: keeps the output graph-connected so
            # the tie losses can differentiate it.
            return 0.0 * m_b

    # The commit envelope solves the interior PDE exactly (the documented
    # degeneracy), which doubles as an end-to-end test of the derivative ->
    # L -> Hamiltonian pipeline: zero premium must give ~zero pde loss.
    zero = DimensionlessValueFunction(_ZeroPremium())
    envelope_loss = pde_loss(zero, Sample.draw(2048))

    assert envelope_loss.item() < 1e-8, envelope_loss.item()

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

            self.head = nn.Linear(5, 1)

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

    # Full-loss S3 test (doc section 6): with an invariant premium (det T is
    # preserved by the relabel congruences), the loss must be identical on a
    # relabeled batch. Exercises the derivative chain, the L assembly, the
    # Hamiltonian mapping, and the solver in one identity.
    class _InvariantPremium(nn.Module):
        def forward(
            self, m_b: Tensor, m_c: Tensor, tbb: Tensor, tbc: Tensor, tcc: Tensor
        ) -> Tensor:
            return (tbb * tcc - tbc**2).tanh()

    invariant = DimensionlessValueFunction(_InvariantPremium())
    batch = Sample.draw(2048)
    swapped = Sample(  # b <-> c relabel
        batch.m_c, batch.m_b, batch.tau_cc, batch.tau_bc, batch.tau_bb
    )
    relabeled = Sample(  # a <-> b relabel: m -> Jm, T -> J^T T J
        -batch.m_b,
        batch.m_c - batch.m_b,
        batch.tau_bb + 2.0 * batch.tau_bc + batch.tau_cc,
        -(batch.tau_bc + batch.tau_cc),
        batch.tau_cc,
    )

    def clone(state: Sample) -> Sample:
        return Sample(*(field.detach().clone() for field in vars(state).values()))

    original_loss = pde_loss(invariant, clone(batch))
    swapped_loss = pde_loss(invariant, clone(swapped))
    relabeled_loss = pde_loss(invariant, clone(relabeled))

    assert torch.allclose(swapped_loss, original_loss, atol=1e-5)
    assert torch.allclose(relabeled_loss, original_loss, atol=1e-4)
    print("ok")
