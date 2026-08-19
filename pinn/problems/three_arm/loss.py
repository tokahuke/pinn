"""
The three-arm objective: maximize the premium subject to not overclaiming,
rather than driving a two-sided residual to zero. two_arm/loss.py holds the
record (kb/two_arm.md section 10) and two_arm_drift is the same move on a
second problem (kb/two_arm_drift.md section 10); read those first.

V* is the **maximal** subsolution of the HJB, so `maximize u subject to
v <= max H` has the true value function as its optimum and every feasible
point is a proven lower bound. What differs at N = 3, with the measurements
that settled each one: kb/three_arm.md section 18.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import RidgeSample, Sample

TIE_WEIGHT = 10.0
"""
The ties **place** the solution: the climb kills the never-explore degeneracy
on its own (u = 0 climbs nothing), but nothing else knows where the free
boundary belongs. Set to beat a live pde of 1.3e-2, so provisional against an
anchor three orders smaller; re-derive once the retrain settles.
"""

CLIMB_WEIGHT = 4.0e-2
"""
The single knob, since climb and violation share units here, and small is the
exact-penalty side. It sits 10x above the dead-solution floor, violation / climb
= 4.1e-3 on the 2026-08-16 champion. Judge over 100k+ on block medians, never a
short sweep: violation *rising* on a flat climb is too high, climb below 0.466 low.
"""

SLACK_PRICE = 0.0
"""
The pinball tilt: what one unit of slack costs as a share of the residual
budget, 0 being the pure subsolution objective and 0.5 the symmetric two-sided
loss in L1. 0 is the polish value; 0.02 is for **cold starts** and for legs
that open new territory, like the funnel extension (kb/three_arm.md
section 19.5).
"""

CONCAVITY_SCALE = 1.0e-3
"""
The saturation knee of the concavity term, sitting just above the median
violation, so lower it as the net converges.
"""

CONCAVITY_WEIGHT = 2.6e0
"""
Weight on L[f] >= 0 for every contrast direction (kb section 19.4): provable
at the answer and invisible to the residual, so it moves the path and not the
fixed point, and zero on the dead solution. Post-saturation natural units,
calibrated on the *median* pde over seven draws; one draw put it at 730%.
"""

LEARNING_TIE_WEIGHT = 1.0
"""
Weight on the learning ties, the value ties' second-order siblings: placers,
not auxiliaries, since the subsolution objective is blind to the l-ratios
that decide the policy (the max *value* is envelope-insensitive to the
argmax) and one-sidedness pays for inflating an l. Started at 0.1 (the climb
term's scale, from medians on the 2026-08-16 champion: 9.6e-2 control,
4.5e-2 treatment over 7 draws); raised 10x at 14k iterations when
treatment_learning sat flat at ~7e-3 for 7k iterations with the corner
policy stalled at 0.13 control share. Falling is fine, stuck means raise 10x.
"""


def _learning_numbers(
    value: DimensionlessValueFunction,
    m_b: Tensor,
    m_c: Tensor,
    tau_bb: Tensor,
    tau_bc: Tensor,
    tau_cc: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """The learning numbers alone, through the model's one derivative chain."""
    _, _, learning = value.hamiltonian(m_b, m_c, tau_bb, tau_bc, tau_cc)

    return learning


def relative_mismatch(left: Tensor, right: Tensor) -> Tensor:
    """
    A scale-free equality score in [0, 1]: zero exactly on left = right (both
    zero included, so the dead solution scores 0), and the rescale is a
    positive per-point factor on an equality constraint, the one sanctioned
    exception to never-scale-the-residual: it cannot move the zero set, and
    without it the floor decade's l ~ 1/det owns the gradient.
    """
    return ((left - right) / (left.abs() + right.abs() + 1e-8)).pow(2)


def control_learning_loss(
    value: DimensionlessValueFunction, draw: RidgeSample
) -> Tensor:
    """
    Learning ties on the control tie {m_b = 0}: the a<->b swap fixes the wall,
    maps pair ab to itself and swaps ac <-> bc, so at the mirror s' of
    control_tie_loss the true solution has l_ab(s) = l_ab(s'),
    l_ac(s) = l_bc(s'), l_bc(s) = l_ac(s'). The value ties hold first
    derivatives only; the policy reads these second-order numbers, and no
    other term grades them (kb section 19.8).
    """
    zero = torch.zeros_like(draw.mean)
    l_ab, l_ac, l_bc = _learning_numbers(
        value, zero, draw.mean, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    mirror_ab, mirror_ac, mirror_bc = _learning_numbers(
        value,
        zero,
        draw.mean,
        draw.tau_bb + 2.0 * draw.tau_bc + draw.tau_cc,
        -(draw.tau_bc + draw.tau_cc),
        draw.tau_cc,
    )

    return (
        relative_mismatch(l_ab, mirror_ab)
        + relative_mismatch(l_ac, mirror_bc)
        + relative_mismatch(l_bc, mirror_ac)
    ).mean()


def treatment_learning_loss(
    value: DimensionlessValueFunction, draw: RidgeSample
) -> Tensor:
    """
    Learning ties on the treatment tie {m_b = m_c}: the b<->c swap fixes the
    wall, swaps ab <-> ac and fixes bc, so at the swapped-precision mirror
    s' = (m, m, tau_cc, tau_bc, tau_bb) the true solution has
    l_ab(s) = l_ac(s'), l_ac(s) = l_ab(s'), l_bc(s) = l_bc(s'). At self-mirror
    states (tau_bb = tau_cc, the flat-prior corner included) this collapses to
    l_ab = l_ac on the state itself, which the 2026-08-17 corner probe found
    violated 2:1 on the champion (kb section 19.8).
    """
    l_ab, l_ac, l_bc = _learning_numbers(
        value, draw.mean, draw.mean, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    mirror_ab, mirror_ac, mirror_bc = _learning_numbers(
        value, draw.mean, draw.mean, draw.tau_cc, draw.tau_bc, draw.tau_bb
    )

    return (
        relative_mismatch(l_ab, mirror_ac)
        + relative_mismatch(l_ac, mirror_ab)
        + relative_mismatch(l_bc, mirror_bc)
    ).mean()


def directional_learning(
    l_ab: Tensor, l_ac: Tensor, l_bc: Tensor, theta: Tensor
) -> Tensor:
    """
    The learning operator along the contrast direction at angle theta of the
    (b, c) tangent chart: L[f] = f' M f, M = [[l_ab, h], [h, l_ac]] and
    h = (l_ab + l_ac - l_bc)/2, pair directions at 0, pi/2 and 3pi/4. Linear in
    the learning numbers; why not the eigenvalue form: kb section 19.4.
    """
    h = 0.5 * (l_ab + l_ac - l_bc)
    along_b, along_c = torch.cos(theta), torch.sin(theta)

    return l_ab * along_b**2 + l_ac * along_c**2 + 2.0 * h * along_b * along_c


def subsolution_loss(
    value: DimensionlessValueFunction, draw: Sample
) -> tuple[Tensor, Tensor, Tensor]:
    """
    The violation, the climb and the concavity term: the *positive* part of the
    interior HJB residual in value form (doc sections 4, 7, 10), the plain mean
    premium, and relu(-L[f]) along one sampled direction (kb section 19.4).
    Linear and never scaled; slack is free, the climb being what tightens it.
    """
    # The derivation (learning numbers, Hamiltonian, simplex max) lives on
    # the model: one chain serves training and policy readout.
    v, best, (l_ab, l_ac, l_bc) = value.hamiltonian(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    theta = torch.rand_like(l_ab) * math.pi
    non_concave = torch.relu(-directional_learning(l_ab, l_ac, l_bc, theta))
    # **Saturated** so one deep point cannot dominate, as two_arm_drift's
    # positivity term: bounded per point, zero set untouched, and not tanh
    # (kb section 19.4).
    concavity = (non_concave / (CONCAVITY_SCALE + non_concave)).mean()

    # Natural units, **never scaled**. A chart-derived weight on the residual is
    # an undeclared reweighting of the domain (learnings section 3).
    residual = v - best.value
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()
    climb = value.premium(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc
    ).mean()

    return violation, climb, concavity


def control_tie_loss(value: DimensionlessValueFunction, draw: RidgeSample) -> Tensor:
    """
    Wall condition on the control tie {m_b = 0} (doc section 12), the degeneracy
    breaker: each wall point is scored with its a<->b mirror,
    dU/dm_b (s) + dU/dm_b (s') + dU/dm_c (s') = 1. Derivatives on the premium,
    smooth where the commit envelope's kink sits.
    """
    m_b = torch.zeros_like(draw.mean).requires_grad_(True)
    u = value.premium(m_b, draw.mean, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    (normal,) = torch.autograd.grad(
        u.sum(), m_b, create_graph=True, allow_unused=True, materialize_grads=True
    )

    # The mirror point, T' = (tau_bb + 2 tau_bc + tau_cc, -(tau_bc + tau_cc),
    # tau_cc), still reachable since its off-diagonal is
    # -(floor + precision_ac) < 0.
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
    Wall condition on the treatment tie {m_b = m_c} (doc section 12), the mirror
    condition, pair-sampled under the b<->c swap: d_n U(s) + d_n U(s') = 0 with
    d_n = d/dm_b - d/dm_c and s' = (m, m, PTP). Derivatives on the premium, as
    on the control tie.
    """

    def crossing_derivative(tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor) -> Tensor:
        """The premium's derivative across the tie, d/dm_b minus d/dm_c."""
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
    """
    The whole objective: climb the premium, stay feasible, hold both walls
    concave. Pass an `iteration` to print the breakdown, or `None` to stay
    silent, which is what a captured cuda graph replay needs.
    """
    violation, climb, concavity = subsolution_loss(value, draw)
    control_residual = control_tie_loss(value, control_draw)
    treatment_residual = treatment_tie_loss(value, treatment_draw)
    control_learning = control_learning_loss(value, control_draw)
    treatment_learning = treatment_learning_loss(value, treatment_draw)

    if iteration is not None:
        print(
            f"iter {iteration}: violation {violation.item():.3e}"
            f"  climb {climb.item():.4e}"
            f"  control_tie {control_residual.item():.3e}"
            f"  treatment_tie {treatment_residual.item():.3e}"
            f"  control_learning {control_learning.item():.3e}"
            f"  treatment_learning {treatment_learning.item():.3e}"
            f"  concavity {concavity.item():.3e}"
        )

    return (
        violation
        + TIE_WEIGHT * (control_residual + treatment_residual)
        + LEARNING_TIE_WEIGHT * (control_learning + treatment_learning)
        + CONCAVITY_WEIGHT * concavity
        - CLIMB_WEIGHT * climb
    )


def on_device(state: Sample | RidgeSample, device: str) -> Sample | RidgeSample:
    """
    A draw moved to the trainer's device, field by field. These are dataclasses
    of tensors, so this is the same reconstruct-from-vars trick the self-checks
    use for cloning, and field order is the constructor's.
    """
    return type(state)(*(field.to(device) for field in vars(state).values()))


def draw(batch: int, device: str = "cpu") -> tuple:
    """
    One step's samples, in loss()'s argument order; every training state lives
    in the fundamental wedge. Split out of objective because a captured cuda
    graph replays the same tensor addresses, so the sampling lives outside it.
    """
    return (
        on_device(Sample.draw(batch).fold(), device),
        on_device(RidgeSample.control_tie(batch // 4), device),
        on_device(RidgeSample.treatment_tie(batch // 4), device),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol draws scored by
    loss. `device` is the **one** place the trainer's device enters, defaulting
    to CPU for the arena, probes and self-checks; Sobol draws on CPU regardless,
    since SobolEngine ignores the default device.
    """

    def step(value: nn.Module, iteration: int | None) -> Tensor:
        """One training step's loss, on a batch drawn fresh for it."""
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":

    class _ZeroPremium(nn.Module):
        """A premium that is identically zero, so v is the bare commit envelope."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            # 0 * m_b, not zeros_like: keeps the output graph-connected so
            # the tie losses can differentiate it.
            return 0.0 * m_b

    # The commit envelope is **feasible** and climbs nothing, so the objective
    # rejects it without the ties. Concavity must be **exactly** silent on it,
    # all learning numbers being 0, with no clamp residue to tolerate.
    zero = DimensionlessValueFunction(_ZeroPremium())
    envelope_violation, envelope_climb, envelope_concavity = subsolution_loss(
        zero, Sample.draw(2048)
    )

    assert envelope_violation.item() < 1e-8, envelope_violation.item()
    assert envelope_climb.item() == 0.0, envelope_climb.item()
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
    # (0 + 0 + 0 - 1)^2 = 1, which is the degeneracy breaker doing its job, and
    # the treatment tie is homogeneous, so it is satisfied.
    control_zero = control_tie_loss(zero, RidgeSample.control_tie(512))
    treatment_zero = treatment_tie_loss(zero, RidgeSample.treatment_tie(512))

    assert abs(control_zero.item() - 1.0) < 1e-6, control_zero.item()
    assert treatment_zero.item() < 1e-10, treatment_zero.item()

    # The learning ties on the zero premium: every learning number is 0, and
    # the relative form scores 0/eps = 0 exactly, so the dead solution gains
    # nothing from them (the degeneracy floor stays where it was).
    assert control_learning_loss(zero, RidgeSample.control_tie(256)).item() == 0.0
    assert treatment_learning_loss(zero, RidgeSample.treatment_tie(256)).item() == 0.0

    # A premium that breaks the swap symmetries (tau_bb alone is not invariant
    # under either transposition) must fire both learning ties.
    class _AsymmetricPremium(nn.Module):
        """u = tanh(tau_bb), invariant under neither wall's swap."""

        def forward(
            self, m_b: Tensor, m_c: Tensor, tbb: Tensor, tbc: Tensor, tcc: Tensor
        ) -> Tensor:
            return tbb.tanh()

    asymmetric = DimensionlessValueFunction(_AsymmetricPremium())

    assert control_learning_loss(asymmetric, RidgeSample.control_tie(256)).item() > 1e-3
    assert (
        treatment_learning_loss(asymmetric, RidgeSample.treatment_tie(256)).item()
        > 1e-3
    )

    # Analytic satisfiers, each for one wall only: u = m_b / 2 zeroes the
    # control tie (1/2 + 1/2 + 0 = 1) but not the treatment tie; u = m_b + m_c
    # zeroes the treatment tie (d_n = 0) but not the control tie.
    class _HalfB(nn.Module):
        """u = m_b / 2, which satisfies the control tie and nothing else."""

        def forward(self, m_b: Tensor, *rest: Tensor) -> Tensor:
            return 0.5 * m_b

    class _Symmetric(nn.Module):
        """u = m_b + m_c, which satisfies the treatment tie and nothing else."""

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
        """One trainable linear layer, enough to check gradients reach parameters."""

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

    # Full-loss S3 test (doc section 6): with a premium invariant under the
    # relabel congruences the loss must be identical on a relabeled batch,
    # exercising derivatives, L assembly, Hamiltonian and solver in one.
    class _InvariantPremium(nn.Module):
        """A premium in det T alone, so the relabel congruences leave it fixed."""

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
        """A detached copy, so each relabel is scored on its own graph."""
        return Sample(*(field.detach().clone() for field in vars(state).values()))

    # The violation only: concavity draws a fresh direction per call, so its
    # per-draw value differs across relabels (only its zero set is
    # chart-invariant), and the relabel maps the climb onto itself.
    original_loss, _, _ = subsolution_loss(invariant, clone(batch))
    swapped_loss, _, _ = subsolution_loss(invariant, clone(swapped))
    relabeled_loss, _, _ = subsolution_loss(invariant, clone(relabeled))

    # **Relative** tolerance: an atol calibrated at one loss magnitude means
    # nothing at another, and it flaked one run in three. A real S3 bug moves
    # this by O(1), so nothing is lost.
    assert torch.allclose(swapped_loss, original_loss, rtol=1e-3, atol=1e-9), (
        swapped_loss.item(),
        original_loss.item(),
    )
    assert torch.allclose(relabeled_loss, original_loss, rtol=1e-3, atol=1e-9), (
        relabeled_loss.item(),
        original_loss.item(),
    )

    # An S3-invariant premium satisfies the learning ties identically; float32
    # through two second-derivative chains leaves only roundoff, and the
    # relative form keeps that roundoff dimensionless.
    assert control_learning_loss(invariant, RidgeSample.control_tie(512)).item() < 1e-6
    assert (
        treatment_learning_loss(invariant, RidgeSample.treatment_tie(512)).item() < 1e-6
    )

    from pathlib import Path

    champion = Path("data/three_arm.pt")

    if champion.exists():
        trained = DimensionlessValueFunction.load(champion)
        # On the reachable slice: this is a calibration statement about the
        # core cloud, and a champion predating the funnel branch is expected
        # to fail out there until retrained.
        cloud = Sample.draw(8192)
        reachable = Sample(
            *(field[cloud.tau_bc <= 0] for field in vars(cloud).values())
        )
        fit = subsolution_loss(trained, reachable.fold())

        # The LP objective **alone**, without the ties, prefers the champion
        # to the commit envelope. This is what CLIMB_WEIGHT clears, by 10x:
        # the champion's violation / climb is 4.1e-3.
        assert (
            fit[0] - CLIMB_WEIGHT * fit[1]
            < envelope_violation - CLIMB_WEIGHT * envelope_climb
        ), "the climb term must reject the commit envelope without the ties"
    print("ok")
