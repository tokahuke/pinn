"""
The three-arm drift objective: maximize the premium subject to not overclaiming.
three_arm/loss.py is the same objective without drift and kb/three_arm.md section 18
is its record; read that first, then this.

Everything three_arm's version says carries over. Two things differ in degree, both
because this is the least converged of the four problems: concavity does real work
rather than guarding (this problem's best net violates it on 8.6% against
three_arm's 0.02%), and `SLACK_PRICE` ships at 0.5 here and at 0 everywhere else,
because on this problem the bound and the policy are *inverted*. Doc section 11 has
the anneal ladder and the arena table behind both.
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

TIE_WEIGHT = 2.4e2
"""
The ties *place* the solution: nothing else in the objective knows where the free
boundary belongs. Provisional, since it was set against a pde median the current
objective no longer produces; doc section 8 carries the calibration.
"""

CLIMB_WEIGHT = 0.0
"""
Off: SLACK_PRICE is the upward pull and the ties are the degeneracy breaker. The
10x-dead-solution-floor rule that would put this at 2.4e-1 is *wrong* on a problem
whose violation is two orders above three_arm's, measurably; doc section 11.
"""

SLACK_PRICE = 0.5
"""
What one unit of *slack* costs, as a share of the residual budget: the pinball loss
at q = 1 - SLACK_PRICE, the two sides summing to 1 so moving this reallocates between
them without rescaling the objective. Nonzero here alone, and symmetric rather than
merely nonzero, both measured: doc section 11.
"""

CONCAVITY_SCALE = 1.0e-3
"""
three_arm's, deliberately: the erosion is control-free, so the drift Hessian is the
static one, and sharing the scale keeps a violation meaning the same thing in both
problems and the two terms comparable across the pair.
"""

CONCAVITY_WEIGHT = 2.2e-1
"""
Target ~5% of the equation, calibrated on the *median* pde over seven draws rather
than one, which is what stops a heavy tail putting it two orders out; doc section 8.
"""


def subsolution_loss(
    value: DimensionlessValueFunction, draw: Sample
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns *three* numbers: the violation, the climb and the concavity term. The
    violation is the *positive* part of the value-form HJB residual in the equation's
    own units, which the commit envelope zeroes exactly, so the ties of doc section 5
    are what break that degeneracy (doc section 2; three_arm/loss.py holds the rest).
    """
    # The derivation (learning numbers, Hamiltonian, simplex max) lives on
    # the model: one chain serves training and policy readout.
    v, best, (l_ab, l_ac, l_bc) = value.hamiltonian(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc, draw.etahat
    )
    theta = torch.rand_like(l_ab) * math.pi
    non_concave = torch.relu(-directional_learning(l_ab, l_ac, l_bc, theta))
    # Saturated, so one deep point cannot dominate (the worst carried 7x the term's
    # mean): bounded per point, linear below s, 1/y^2 above, zero set untouched. s
    # sits just above the median violation, so lower it as the net converges.
    concavity = (non_concave / (CONCAVITY_SCALE + non_concave)).mean()

    # Natural units, never scaled (the standing rule lives in three_arm/loss.py), and
    # linear, because an L1 penalty is *exact* at a finite weight where a quadratic
    # only approaches feasibility.
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
    Wall condition on the control tie {m_b = 0} (doc section 5), the degeneracy
    breaker: `dU/dm_b(s) + dU/dm_b(s') + dU/dm_c(s') = 1`, each wall point scored with
    its mirror s' under the a<->b relabel. Derivatives on the premium, which is smooth
    at the wall where v is not (the commit envelope kinks exactly at m_b = 0).
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
    Wall condition on the treatment tie {m_b = m_c} (doc section 5), the mirror
    condition: `d_n U(s) + d_n U(s') = 0` with `d_n = d/dm_b - d/dm_c`, pair-sampled
    under the b<->c swap. Derivatives on the premium, as on the control tie.
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
    """The whole objective: stay feasible, hold both walls, stay concave."""
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
    One step's sample points, in loss()'s argument order: every training state lives
    in the fundamental wedge. Split out of objective so the graphed trainer can hold
    them as fixed buffers, a captured cuda graph replaying the same addresses.
    """
    return (
        on_device(Sample.draw(batch).fold(), device),
        on_device(RidgeSample.control_tie(batch // 4), device),
        on_device(RidgeSample.treatment_tie(batch // 4), device),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol draws scored by loss.

    `device` is the *one* place the trainer's device enters the problem. Sobol draws
    on CPU and the batch moves once, everything downstream inheriting from its inputs.
    """

    def step(value: nn.Module, iteration: int | None) -> Tensor:
        """One step's loss, on collocation drawn fresh for it."""
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        """The never-explore solution as a premium: identically zero."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            # Not zeros_like: keeps the output graph-connected through *every* input,
            # including the precisions the erosion differentiates. Autograd refuses to
            # start from a constant (CLAUDE.md traps).
            product = m_b
            for field in rest:
                product = product * field

            return 0.0 * product

    # Zero premium solves the interior PDE exactly at every drift, so this doubles as
    # the end-to-end test of the derivative -> L -> erosion -> Hamiltonian pipeline,
    # run across the etahat range: the erosion is the only new term.
    zero = DimensionlessValueFunction(_ZeroPremium())

    for etahat in [0.0, 1.0, 10.0, 35.0]:
        batch = Sample.draw(2048)
        batch.etahat = torch.full_like(batch.etahat, etahat)
        envelope_violation, envelope_climb, envelope_concavity = subsolution_loss(
            zero, batch
        )

        # The point of this objective: the commit envelope is *feasible* at every
        # drift (it solves the interior equation exactly) but climbs nothing, so the
        # objective rejects it without help from the ties.
        assert envelope_violation.item() < 1e-8, (etahat, envelope_violation.item())
        assert envelope_climb.item() == 0.0, (etahat, envelope_climb.item())

        # Concavity is *exactly* silent on the degenerate solution at every drift:
        # all learning numbers are 0, no clamp residue to tolerate.
        assert envelope_concavity.item() == 0.0, (etahat, envelope_concavity.item())

    # With a premium linear in *one* precision entry, left - v is that entry's erosion
    # coefficient, which must equal T etahat^2 (I + 11') T. The only runnable pin on
    # the *formula*: the S3 test below passes for any invariant.
    class _PickTau(nn.Module):
        """A premium that is one precision entry, so its erosion coefficient reads off."""

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
    # (0 + 0 + 0 - 1)^2 = 1, the degeneracy breaker doing its job, and the treatment
    # tie is homogeneous, so it is satisfied.
    control_zero = control_tie_loss(zero, RidgeSample.control_tie(512))
    treatment_zero = treatment_tie_loss(zero, RidgeSample.treatment_tie(512))

    assert abs(control_zero.item() - 1.0) < 1e-6, control_zero.item()
    assert treatment_zero.item() < 1e-10, treatment_zero.item()

    # Analytic satisfiers, each for one wall only: u = m_b / 2 zeroes the
    # control tie (1/2 + 1/2 + 0 = 1) but not the treatment tie; u = m_b + m_c
    # zeroes the treatment tie (d_n = 0) but not the control tie.
    class _HalfB(nn.Module):
        """u = m_b / 2, which satisfies the control tie and not the treatment tie."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return 0.5 * m_b

    class _Symmetric(nn.Module):
        """u = m_b + m_c, which satisfies the treatment tie and not the control tie."""

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
        """A trainable premium, small enough to check that the loss backpropagates."""

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

    # Full-loss S3 test (three_arm.md section 6): with a premium the relabel
    # congruences preserve, the loss must be identical on a relabeled batch, which
    # exercises derivatives, L assembly, Hamiltonian and solver in one identity.
    class _InvariantPremium(nn.Module):
        """A premium invariant under the relabel congruences, which preserve det T."""

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

    # Drift *on*: E is unchanged by shuffling arm labels, so the erosion has to be
    # too. The cheapest test that the three entries were assembled symmetrically, and
    # it fails if any is transposed or mis-paired.
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
        """A detached copy, so each scoring starts from its own leaves."""
        return Sample(*(field.detach().clone() for field in vars(state).values()))

    # The graded residual only: the concavity term draws a fresh direction
    # per call (only its zero set is chart-invariant).
    original_loss, _, _ = subsolution_loss(invariant, clone(batch))
    swapped_loss, _, _ = subsolution_loss(invariant, clone(swapped))
    relabeled_loss, _, _ = subsolution_loss(invariant, clone(relabeled))

    # *Relative*, because the natural-units loss moves with every net and an atol
    # calibrated at one magnitude is meaningless at another. Measured agreement is
    # 1e-7 to 1e-5; a mis-paired erosion entry moves this by O(1) or more.
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

        # *The ties* break the degeneracy here, not the climb: the commit envelope
        # scores the priced residual 0 on both sides at any SLACK_PRICE, and the
        # control tie reads exactly 1.0 on it, which at TIE_WEIGHT is the margin.
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
