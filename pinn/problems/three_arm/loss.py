"""
Losses for the three-arm problem: the interior HJB residual, the two tie
losses of doc section 12, and the trainer-facing objective.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import RidgeSample, Sample

# The ties are the ONLY terms breaking the never-explore degeneracy: on that
# solution pde is exactly 0 and the control tie exactly 1.0, so the weight must
# beat the live pde. At pde 1.3e-2 the 100x floor is 1.3; 10.0 clears it.
TIE_WEIGHT = 10.0

# Plain mean-of-squares. P = 2 compensated for the chart weight's suppressed
# tail; in natural units it over-corrects, dropping the effective sample size
# at batch 4096 to 2.1 points on three_arm (54 at P = 1).
POWER = 1.0

# Concavity: L[f] >= 0 for every contrast direction, provable at the answer and
# invisible to the residual, so it moves the path and not the fixed point. Zero
# on the dead solution, so the tie floor says nothing about it. Both constants
# are on the post-saturation, post-natural-units scale and share nothing with
# the pre-2026-08-10 record; the weight is calibrated on the MEDIAN pde over
# seven draws, since a single draw put it at 730% of the equation.
CONCAVITY_SCALE = 1.0e-3
CONCAVITY_WEIGHT = 2.6e0


def directional_learning(
    l_ab: Tensor, l_ac: Tensor, l_bc: Tensor, theta: Tensor
) -> Tensor:
    """
    The learning operator along the contrast direction at angle theta of the
    (b, c) tangent chart: the Hamiltonian's negated Hessian as a quadratic
    form, L[f] = f' M f with M = [[l_ab, h], [h, l_ac]],
    h = (l_ab + l_ac - l_bc)/2. The pair directions are theta = 0 (l_ab),
    pi/2 (l_ac), and 3pi/4 (l_bc / 2).

    Linear in the learning numbers, so it is smooth everywhere and its
    gradient scale does not depend on their magnitude -- the property the
    eigenvalue form loses to its sqrt (whose tie set is the entire contact
    set; the clamped version silently pays to un-flatten the dead region).
    """
    h = 0.5 * (l_ab + l_ac - l_bc)
    along_b, along_c = torch.cos(theta), torch.sin(theta)

    return l_ab * along_b**2 + l_ac * along_c**2 + 2.0 * h * along_b * along_c


def pde_loss(value: DimensionlessValueFunction, draw: Sample) -> tuple[Tensor, Tensor]:
    """
    Returns TWO numbers. The first is the interior HJB residual in value form
    (doc sections 4, 7, 10), units rho = sigma = 1, i.e. the dimensionless
    form (doc section 11):
    v = max over the simplex of alpha.m + pairwise learning terms; graded in
    the equation's OWN units, never scaled (see the standing rule below).

    KNOWN DEGENERATE on its own: the commit envelope v = max(0, m_b, m_c)
    zeroes this residual exactly (the never-explore solution), just as the
    two_arm residual was degenerate before BC1. The tie losses of doc
    section 12 break the degeneracy.

    The second is the concavity term: mean of relu(-L[f]) along ONE freshly
    sampled contrast direction per point. Concavity of the simplex quadratic
    is L[f] >= 0 for EVERY direction f (derived 2026-08-08), provable at the
    answer by the mean-preserving-spread argument, and pairwise positivity
    alone misses ~3x of the violations (champion: 93.1% pairwise-positive,
    78.5% concave). The direction is redrawn every step, so the population
    loss vanishes exactly on concavity -- the same trick the collocation
    cloud plays with space. LINEAR in the violation, not squared, for the
    pos_learning reason: the policy reads the SIGN. Sampled on the wedge's
    own chart; the zero set (concavity) is chart-invariant even though the
    angular weighting is not. Zero on the never-explore solution -- all
    three learning numbers vanish -- so it neither breaks nor deepens the
    degeneracy.
    """
    # The derivation (learning numbers, Hamiltonian, simplex max) lives on
    # the model: one chain serves training and policy readout.
    v, best, (l_ab, l_ac, l_bc) = value.hamiltonian(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    theta = torch.rand_like(l_ab) * math.pi
    violation = torch.relu(-directional_learning(l_ab, l_ac, l_bc, theta))
    # SATURATED, as two_arm_drift's positivity term: relu is linear in depth,
    # so one deep point dominates -- the worst carried 7x the whole term's
    # mean. y / (s + y) is bounded per point, linear below s, 1/y^2 above; NOT
    # tanh (float32 gradient underflows, CLAUDE.md traps). Zero set untouched.
    # s sits just above the median violation, so lower it as the net converges.
    concavity = (violation / (CONCAVITY_SCALE + violation)).mean()

    # NATURAL UNITS, NEVER SCALED. This once graded in similarity PREMIUM
    # units, det**0.75 / (tau_bb + tau_cc + tau_bc), to keep the never-explore
    # mode loud at low information. Gone: a chart-derived weight on the
    # residual is an undeclared reweighting of the domain (learnings section
    # 3). The degeneracy it defended is the tie losses' job; if a dead
    # Hamiltonian scores well again, strengthen the breaker, not the thumb.
    #
    # Power-mean attention (learnings section 7); at POWER = 1 this is plain
    # mean-of-squares. Normalized by the detached batch mean so pow(P) sees
    # O(1) numbers, not float32 dust.
    graded = (v - best.value).pow(2)
    scale = graded.mean().detach().clamp_min(1e-30)

    return (
        scale * (graded / scale).pow(POWER).mean().pow(1.0 / POWER),
        concavity,
    )


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
    pde_residual, concavity = pde_loss(value, draw)
    control_residual = control_tie_loss(value, control_draw)
    treatment_residual = treatment_tie_loss(value, treatment_draw)

    if iteration is not None:
        print(
            f"iter {iteration}: pde {pde_residual.item():.3e}"
            f"  control_tie {control_residual.item():.3e}"
            f"  treatment_tie {treatment_residual.item():.3e}"
            f"  concavity {concavity.item():.3e}"
        )

    return (
        pde_residual
        + TIE_WEIGHT * (control_residual + treatment_residual)
        + CONCAVITY_WEIGHT * concavity
    )


def on_device(state: Sample | RidgeSample, device: str):
    """
    A draw moved to the trainer's device, field by field. Dataclasses of
    tensors, so the same reconstruct-from-vars trick the self-checks use for
    cloning; field order is the constructor's.
    """
    return type(state)(*(field.to(device) for field in vars(state).values()))


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
            # 0 * m_b, not zeros_like: keeps the output graph-connected so
            # the tie losses can differentiate it.
            return 0.0 * m_b

    # The commit envelope solves the interior PDE exactly (the documented
    # degeneracy), which doubles as an end-to-end test of the derivative ->
    # L -> Hamiltonian pipeline: zero premium must give ~zero pde loss. The
    # concavity term must be EXACTLY silent on it (all learning numbers are
    # 0, relu(-0) contributes nothing): it cannot break or deepen the
    # degeneracy, and there is no clamp residue to tolerate.
    zero = DimensionlessValueFunction(_ZeroPremium())
    envelope_loss, envelope_concavity = pde_loss(zero, Sample.draw(2048))

    assert envelope_loss.item() < 1e-8, envelope_loss.item()
    assert envelope_concavity.item() == 0.0, envelope_concavity.item()

    # The directional form: pair directions are its anchors, and it fires
    # exactly on the negative cone.
    one = torch.ones(4)

    # l_bc = 4 != l_ab + l_ac, so h != 0 and the third identity pins the
    # cross term, not just the diagonal.
    assert torch.allclose(
        directional_learning(1.0 * one, 2.0 * one, 4.0 * one, torch.zeros(4)), one
    )
    assert torch.allclose(
        directional_learning(
            1.0 * one, 2.0 * one, 4.0 * one, torch.full((4,), math.pi / 2.0)
        ),
        2.0 * one,
    )
    assert torch.allclose(
        directional_learning(
            1.0 * one, 2.0 * one, 4.0 * one, torch.full((4,), 3.0 * math.pi / 4.0)
        ),
        2.0 * one,
    )
    assert (
        directional_learning(-1.0 * one, -1.0 * one, -2.0 * one, torch.zeros(4)) < 0
    ).all()

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

    # The graded residual only: the concavity term draws a fresh direction
    # per call, so per-draw values differ across relabels (only its zero set
    # is chart-invariant).
    original_loss, _ = pde_loss(invariant, clone(batch))
    swapped_loss, _ = pde_loss(invariant, clone(swapped))
    relabeled_loss, _ = pde_loss(invariant, clone(relabeled))

    # RELATIVE tolerance: the natural-units loss is no longer a fixed-size
    # number, so an atol calibrated at one magnitude means nothing at another
    # (see three_arm_drift/loss.py, where the old atol flaked one run in
    # three). A real S3 bug moves this by O(1), so nothing is lost.
    assert torch.allclose(swapped_loss, original_loss, rtol=1e-3, atol=1e-9), (
        swapped_loss.item(),
        original_loss.item(),
    )
    assert torch.allclose(relabeled_loss, original_loss, rtol=1e-3, atol=1e-9), (
        relabeled_loss.item(),
        original_loss.item(),
    )
    print("ok")
