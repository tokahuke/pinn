"""
The three-arm objective: maximize the premium subject to not overclaiming,
rather than driving a two-sided residual to zero. two_arm/loss.py holds the
record (kb/two_arm.md section 10) and two_arm_drift is the same move on a
second problem (kb/two_arm_drift.md section 10); read those first.

V* is the MAXIMAL subsolution of the HJB, so `maximize u subject to
v <= max H` has the true value function as its optimum and every feasible
point is a proven lower bound. Three notes specific to N = 3:

- The climb needs NO units factor. Both two-arm problems grade a similarity
  chart and have to divide `tauhat**1.5` back out of the residual, which the
  climb must then match or the floor decade is sacrificed. This problem is
  graded in value form already, so the residual and the premium are in the
  same units by construction and the climb is a plain mean.
- CONCAVITY STAYS, and it is not subsumed the way two_arm's pos_learning was.
  That argument needs a violated learning number to force the max onto a
  vertex; at N = 3 the max can sit on an EDGE of the simplex and stay
  feasible while a direction is non-concave. Measured on the 2026-08-16
  champion: 84% of live non-concave states were feasible.
- The tie losses keep their job. They break the never-explore degeneracy, and
  now the climb term does too -- u = 0 is feasible, so only the climb rejects
  it -- but the ties are what PLACE the solution, which the climb cannot see.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import RidgeSample, Sample

# The ties PLACE the solution: the climb term now kills the never-explore
# degeneracy on its own (u = 0 climbs nothing), but nothing else in the
# objective knows where the free boundary belongs. Carried over from the
# two-sided loss, where it was set to beat a live pde of 1.3e-2; the anchor is
# now a violation three orders smaller, so this is provisional and wants
# re-deriving once the retrain settles.
TIE_WEIGHT = 10.0

# Climb and violation are in the SAME units here without any factor (see the
# docstring), so this is the single knob and small is the exact-penalty side.
# The floor is the dead solution: below violation / climb = 4.1e-3, measured on
# the 2026-08-16 champion, the never-explore net scores better on the LP
# objective alone. 10x above that floor.
#
# NOT calibrated on a short sweep. two_arm_drift found five decades
# indistinguishable at 3k iterations and a monotone collapse of the bound by
# 50k at the cheapest of them (kb/two_arm_drift.md section 10). Judge this over
# 100k+ on block medians: violation RISING with a flat climb means too high,
# climb SAGGING below the champion's 0.466 means too low.
CLIMB_WEIGHT = 4.0e-2

# What one unit of SLACK costs, as a share of the residual budget: the two
# sides are priced (1 - SLACK_PRICE) and SLACK_PRICE, so they always sum to 1
# and moving this reallocates between them WITHOUT rescaling the objective,
# which keeps every weight calibrated against the residual valid across a
# sweep. The pinball loss at q = 1 - SLACK_PRICE: 0 is the pure subsolution
# objective, 0.5 the symmetric two-sided loss in L1, above 0.5 the wrong side
# for a lower bound.
#
# The climb is a global MEAN and cannot say WHERE to climb, so at 0 a
# from-scratch net sags below V* and inflates the learning number to buy free
# slack -- measured on two_arm, whose loss carries the table. 0 is the default
# because this problem's champion was polished at 0; 0.02 is for COLD STARTS.
SLACK_PRICE = 0.0

# Concavity: L[f] >= 0 for every contrast direction, provable at the answer and
# invisible to the residual, so it moves the path and not the fixed point. Zero
# on the dead solution, so the tie floor says nothing about it. Both constants
# are on the post-saturation, post-natural-units scale and share nothing with
# the pre-2026-08-10 record; the weight is calibrated on the MEDIAN pde over
# seven draws, since a single draw put it at 730% of the equation.
CONCAVITY_SCALE = 1.0e-3
CONCAVITY_WEIGHT = 2.6e0

# EXCHANGEABILITY. At a state fixed by the whole relabelling group the three
# learning numbers must be EQUAL -- the arms are interchangeable, so no pair
# can be worth more to learn about than another. The residual cannot see this
# (it reads the numbers only through the maximization), the tie losses only
# reach it as a limit from the walls, and the trained net misses it badly
# exactly where it matters: 58% spread at the precision floor against 6.5% at
# I ~ 1, and the arena boots EVERY run at a state of this family.
#
# What it costs is not the value but the POLICY, because the maximizer of a
# near-flat quadratic is badly conditioned: a 58% spread in the coefficients
# came out as 0.446/0.376/0.178 where exchangeability demands exact thirds,
# and on three_arm_drift a 240% spread flips a sign and starves an arm on zero
# evidence. Zero at the answer, so it moves the path and not the fixed point.
#
# RELATIVE, and the denominator is DETACHED. Relative because the policy reads
# the spread against the magnitude, which ranges over four decades here (L ~
# 400 at the floor, 0.6 at I ~ 1) and would otherwise be graded almost
# entirely at the bottom; detached because a live denominator is minimized by
# INFLATING the learning numbers, which is the failure two_arm already paid
# for once (kb/two_arm.md section 10).
# CALIBRATED ON GRADIENT SHARE, NOT VALUE SHARE, and the two disagree by three
# orders here: on the champion the term's VALUE is 1.09e-1 against a violation
# of 1.78e-5 (6,000x) while its GRADIENT is 5.79 against 8.0e-2 (72x), so the
# house 1-10%-of-the-anchor rule would set 1e-5 and a gradient rule sets 1e-3.
# Value share is a proxy for gradient share and the proxy breaks when a term
# starts far from satisfied rather than dormant: concavity reads exactly 0 on
# this net and is a guard, this one has real work to do. 1e-3 puts the
# symmetry gradient at 7% of the violation's.
SYMMETRY_WEIGHT = 1.0e-3


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


def subsolution_loss(
    value: DimensionlessValueFunction, draw: Sample
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns THREE numbers: the violation, the climb and the concavity term.

    The violation is the POSITIVE part of the residual of the interior HJB in
    value form (doc sections 4, 7, 10), units rho = sigma = 1:
    v = max over the simplex of alpha.m + pairwise learning terms, graded in
    the equation's OWN units, never scaled. v > max H is the overclaim that
    breaks the bound; v < max H is merely a slack subsolution and is left
    free, since the climb is what tightens it. LINEAR, because an L1 penalty is
    EXACT at a finite weight where a quadratic only approaches feasibility.

    The climb is the plain mean premium -- no units factor, unlike either
    two-arm problem (see the module docstring).

    The third is the concavity term: mean of relu(-L[f]) along ONE freshly
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
    non_concave = torch.relu(-directional_learning(l_ab, l_ac, l_bc, theta))
    # SATURATED, as two_arm_drift's positivity term: relu is linear in depth,
    # so one deep point dominates -- the worst carried 7x the whole term's
    # mean. y / (s + y) is bounded per point, linear below s, 1/y^2 above; NOT
    # tanh (float32 gradient underflows, CLAUDE.md traps). Zero set untouched.
    # s sits just above the median violation, so lower it as the net converges.
    concavity = (non_concave / (CONCAVITY_SCALE + non_concave)).mean()

    # NATURAL UNITS, NEVER SCALED. This once graded in similarity PREMIUM
    # units, det**0.75 / (tau_bb + tau_cc + tau_bc), to keep the never-explore
    # mode loud at low information. Gone: a chart-derived weight on the
    # residual is an undeclared reweighting of the domain (learnings section
    # 3).
    residual = v - best.value
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()
    climb = value.premium(
        draw.m_b, draw.m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc
    ).mean()

    return violation, climb, concavity


def symmetry_loss(value: DimensionlessValueFunction, draw: RidgeSample) -> Tensor:
    """
    Relative spread of the three learning numbers on exchangeable states,
    where the symmetry forces them equal.
    """
    zero = torch.zeros_like(draw.mean)
    _, _, learning = value.hamiltonian(
        zero, zero, draw.tau_bb, draw.tau_bc, draw.tau_cc
    )
    mean = (learning[0] + learning[1] + learning[2]) / 3.0
    scale = mean.detach().abs().clamp_min(1e-12)

    return sum(((l - mean.detach()) / scale).pow(2) for l in learning).mean()


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
    symmetry_draw: RidgeSample,
    iteration: int | None = None,
) -> Tensor:
    violation, climb, concavity = subsolution_loss(value, draw)
    control_residual = control_tie_loss(value, control_draw)
    treatment_residual = treatment_tie_loss(value, treatment_draw)
    symmetry = symmetry_loss(value, symmetry_draw)

    if iteration is not None:
        print(
            f"iter {iteration}: violation {violation.item():.3e}"
            f"  climb {climb.item():.4e}"
            f"  control_tie {control_residual.item():.3e}"
            f"  treatment_tie {treatment_residual.item():.3e}"
            f"  concavity {concavity.item():.3e}"
            f"  symmetry {symmetry.item():.3e}"
        )

    return (
        violation
        + TIE_WEIGHT * (control_residual + treatment_residual)
        + CONCAVITY_WEIGHT * concavity
        + SYMMETRY_WEIGHT * symmetry
        - CLIMB_WEIGHT * climb
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
        on_device(RidgeSample.exchangeable(batch // 4), device),
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

    # THE POINT OF THIS OBJECTIVE: the commit envelope v = max(0, m_b, m_c) is
    # FEASIBLE -- it solves the interior equation exactly, so it never
    # overclaims -- but it climbs nothing, so the objective rejects it without
    # help from the ties. Under the two-sided residual it was a perfect score.
    # Doubles as the end-to-end test of the derivative -> L -> Hamiltonian
    # pipeline. The concavity term must be EXACTLY silent on it (all learning
    # numbers are 0), with no clamp residue to tolerate.
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
        RidgeSample.exchangeable(128),
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

    # The symmetry term on an S3-INVARIANT premium: det T is invariant, so its
    # three learning numbers must agree at exchangeable states and the term
    # must read machine zero. This is the only check that pins the sign
    # convention of the pair-to-learning-number correspondence -- get it wrong
    # and an invariant premium scores nonzero.
    invariant_symmetry = symmetry_loss(
        DimensionlessValueFunction(_InvariantPremium()), RidgeSample.exchangeable(256)
    )

    assert invariant_symmetry.item() < 1e-10, invariant_symmetry.item()

    # And the sampler really is fixed by the group: equal pairwise precisions,
    # zero contrasts. Written in pair coordinates, where a relabel permutes.
    fixed = RidgeSample.exchangeable(64)
    pairs = torch.stack(
        [fixed.tau_bb + fixed.tau_bc, fixed.tau_cc + fixed.tau_bc, -fixed.tau_bc],
        dim=-1,
    )

    assert (fixed.mean == 0.0).all()
    assert (pairs.max(dim=-1).values - pairs.min(dim=-1).values).abs().max() < 1e-12

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

    # The violation only: the concavity term draws a fresh direction per call,
    # so per-draw values differ across relabels (only its zero set is
    # chart-invariant), and the climb is a premium mean that the relabel maps
    # onto itself trivially.
    original_loss, _, _ = subsolution_loss(invariant, clone(batch))
    swapped_loss, _, _ = subsolution_loss(invariant, clone(swapped))
    relabeled_loss, _, _ = subsolution_loss(invariant, clone(relabeled))

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

    from pathlib import Path

    champion = Path("data/three_arm.pt")

    if champion.exists():
        trained = DimensionlessValueFunction.load(champion)
        fit = subsolution_loss(trained, Sample.draw(8192).fold())

        # The LP objective ALONE -- no ties, which are the degeneracy breaker
        # the two-sided residual needs -- prefers the champion to the commit
        # envelope. This is what CLIMB_WEIGHT has to clear, and the margin is
        # only 10x: the champion's violation / climb is 4.1e-3.
        assert (
            fit[0] - CLIMB_WEIGHT * fit[1]
            < envelope_violation - CLIMB_WEIGHT * envelope_climb
        ), "the climb term must reject the commit envelope without the ties"
    print("ok")
