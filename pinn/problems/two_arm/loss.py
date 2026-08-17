"""
The two-arm objective: maximize the premium subject to not overclaiming,
rather than driving a two-sided residual to zero. In force since 2026-08-15;
kb/two_arm.md section 10 is the record, including the two-sided objective it
replaced and the variants that failed.

V* is the MAXIMAL subsolution of the HJB (learnings section 9), so

    maximize  u   subject to   v <= max H

has the true value function as its optimum, and any feasible point is a
proven lower bound: the greedy policy provably earns at least v. Two
consequences the two-sided residual does not give:

- The never-explore degeneracy dies in the OBJECTIVE. u = 0 zeroes the
  two-sided residual exactly, which is why two_arm needs RIDGE_WEIGHT 2e4 to
  outvote it; here u = 0 is merely feasible, and the climb term rejects it.
- The bound is what training optimizes, rather than something measured
  afterwards on a net that was never asked for it.

The cost: the climb term's gradient never vanishes, so the loss settles at a
BALANCE and its value stops being a quality measure. Judge a run by the
printed violation fraction, the mean gate, and sup(residual+) -- not by the
total.
"""

from __future__ import annotations

import torch

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import sample_ridge, sample_sobol

# Climb and violation are both in NATURAL UNITS, so only their RATIO matters:
# the violation carries weight 1 and this is the single knob. A flat climb
# (the gate u / nu) instead lost the floor decade, where the violation's
# tauhat**-1.5 reaches 3.2e4 -- the net bought slack by inflating the learning
# number rather than fixing the value (kb/two_arm.md section 10, which also
# holds the dual-ascent controller that was tried and removed).
#
# Diagnosis: violation PLATEAUS means this is too high; climb SAGS below the
# champion's 2476 means too low, feasibility being bought by shrinking the
# premium.
CLIMB_WEIGHT = 1.0e-7

# What one unit of SLACK costs, as a share of the residual budget: the two
# sides are priced (1 - SLACK_PRICE) and SLACK_PRICE, so they always sum to 1
# and moving this reallocates between them WITHOUT rescaling the objective --
# which is what keeps RIDGE_WEIGHT and every other weight calibrated against
# the residual valid across a sweep. This is the pinball loss at
# q = 1 - SLACK_PRICE: 0 is the pure subsolution objective (today's, exactly),
# 0.5 is the symmetric two-sided loss in L1, and above 0.5 prefers
# supersolutions, which is the wrong side for a lower bound.
#
# WHY SLACK IS PRICED AT ALL: the climb is a global MEAN, so it cannot say
# WHERE to climb -- the net climbs where that is cheap and sags where it is
# not. Slack is undershoot, `v < max H`, which is also exactly what INFLATING
# max H produces, so pricing it charges for the learning-number inflation that
# is otherwise free. FROM SCRATCH, ~35k iterations, against the polished
# champion's climb 2479 and mean learning number 5.44:
#
#     SLACK_PRICE    climb    L      overshoot   over%
#     0              1592    48.79    2.6e-4      5.8%
#     0.02           2487     5.18    9.1e-4      9.7%
#
# At 0 the premium sits 37% below V* and the Hamiltonian is 9x wrong while the
# OVERSHOOT metric looks better -- the loss was being gamed, not satisfied.
# 0 stays the default here because this problem's champion was polished at 0
# from a converged two-sided net; 0.02 is for COLD STARTS, annealed back to 0
# to tighten the bound once the solution is found.
SLACK_PRICE = 0.0

# BC1's share of the constraint. Calibrated against a SATISFIED ridge, which
# is the state it has to defend: two_arm's own long-standing value, kept
# because the climb term cannot see the wall at all. It is ALSO the
# never-explore breaker whenever SLACK_PRICE > 0: the commit envelope scores
# the priced residual exactly 0 on both sides, so only the ridge and the climb
# reject it.
RIDGE_WEIGHT = 2.0e4

# pos_learning is REMOVED from the objective and kept only as a printed
# diagnostic: the constraint subsumes it. Where L_ab < 0 the Hamiltonian is
# convex in alpha, so the max sits at a vertex and max H = e^s z (for z > 0),
# while v = e^s (z + g) with g >= 0 architectural -- so the violation is
# exactly e^s g / tauhat**1.5 = u. A negative learning operator is therefore
# infeasible wherever the premium is alive, penalized in proportion to it,
# and harmless where u = 0 (the commit region, where a vertex IS the right
# play). Measured on an untrained net, 12,030 states with L_ab < 0 and u > 0:
# violation/u in [0.94, 1.00] as derived, the only silent ones having
# u <= 1.5e-6, where the difference vanishes into float32 cancellation.
# The climb is what makes the repair go the right way: the penalty alone
# would restore feasibility by killing u, and only the pair admits the
# equilibrium L_ab >= 0 WITH u > 0.


def subsolution_loss(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns the priced residual, the climb and the learning-operator
    negative part.

    The residual is split by SIGN and the two sides priced against each other:
    `v > max H` is the overclaim that breaks the bound, `v < max H` is
    slack. Slack was free until 2026-08-16 and the two ways of exploiting that
    are measured in SLACK_PRICE's comment. Natural units and never scaled, as
    everywhere else.
    """
    lhs, best, l_ab = value.hamiltonian(muhat, tauhat)
    natural = tauhat.pow(1.5)
    residual = (lhs - best.value) / natural

    # LINEAR, not squared, for two reasons that point the same way. An L1
    # penalty is EXACT: the constrained optimum is reached at a finite
    # weight, with the constraint active and points sitting on it, while a
    # quadratic penalty only approaches feasibility as the weight grows and
    # at any finite weight overshoots by O(CLIMB_WEIGHT) -- fatal for a term
    # whose whole purpose is a proven bound. And it is what the house does for
    # every other sign condition: pos_learning and three_arm's concavity are
    # both linear, because squaring chases depth while the violating FRACTION
    # rises (two_arm_drift, 2026-08-08).
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()

    # Natural units, matching the violation above: u / tauhat**1.5 is the
    # premium in the units the residual is graded in.
    premium = value.premium(muhat, tauhat)
    climb = (premium / natural).mean()

    return violation, climb, torch.relu(-l_ab).mean()


def ridge_loss(value: DimensionlessValueFunction, ridge_tauhat: Tensor) -> Tensor:
    """
    BC1: du/dmuhat = -1/2 at muhat = 0, imposed on the premium (smooth there;
    the kink lives in the value's relu).
    """
    ridge_muhat = torch.zeros_like(ridge_tauhat).requires_grad_(True)
    u = value.premium(ridge_muhat, ridge_tauhat)
    (u_muhat,) = torch.autograd.grad(u.sum(), ridge_muhat, create_graph=True)

    return (u_muhat + 0.5).pow(2).mean()


def loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    ridge_tauhat: Tensor,
    iteration: int | None = None,
) -> Tensor:
    """
    The penalized LP: climb up, stay feasible, hold BC1.
    """
    violation, climb, pos_learning = subsolution_loss(value, muhat, tauhat)
    ridge = ridge_loss(value, ridge_tauhat)

    if iteration is not None:
        print(
            f"iter {iteration}: violation {violation.item():.3e}"
            f"  climb {climb.item():.4f}  ridge {ridge.item():.3e}"
            f"  pos_learning {pos_learning.item():.3e}"
        )

    return violation + RIDGE_WEIGHT * ridge - CLIMB_WEIGHT * climb


def draw(batch: int, device: str = "cpu") -> tuple:
    """
    One step's collocation tensors, in loss()'s argument order.
    """
    return (
        *(t.to(device) for t in sample_sobol(batch)),
        sample_ridge(batch // 4).to(device),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """
    The problem packaged for the generic trainer.
    """

    def step(value: DimensionlessValueFunction, iteration: int | None) -> Tensor:
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":
    import torch.nn as nn

    from pathlib import Path

    from .model import ExplorationPremium

    muhat, tauhat = sample_sobol(4096)
    ridge_tauhat = sample_ridge(1024)

    class _Dead(nn.Module):
        # The never-explore solution. SQUARED, not `0.0 * muhat`: the
        # CLAUDE.md stub rule covers one derivative, and this loss takes two
        # -- the first derivative of a z-linear stub no longer depends on z,
        # so the second grad call reports it unused. 0.0 * muhat**2 is still
        # identically zero and stays connected through both.
        def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
            return 0.0 * muhat**2

    dead = DimensionlessValueFunction(_Dead())
    dead_violation, dead_climb, _ = subsolution_loss(dead, muhat, tauhat)

    # THE POINT OF THIS OBJECTIVE: the dead solution is FEASIBLE -- it does
    # not overclaim -- but it climbs nothing, so the objective rejects it
    # without help from the ridge term. Under the two-sided residual it is a
    # perfect score, which is why two_arm must weight the ridge at 2e4.
    assert dead_violation.item() < 1e-12, dead_violation.item()
    assert dead_climb.item() == 0.0, dead_climb.item()

    live = ExplorationPremium([16, 16])
    live_value = DimensionlessValueFunction(live)
    _, live_climb, _ = subsolution_loss(live_value, muhat, tauhat)

    # Positive and finite on a live net. NOT bounded above any more: the
    # natural-units climb u / tauhat**1.5 grows like tauhat**-2 toward the
    # floor, which is the point -- it has to match the violation's scaling
    # there or the floor decade gets sacrificed.
    assert 0.0 < live_climb.item() < float("inf")

    # Gradients exist and are finite through every term.
    total = loss(live_value, muhat, tauhat, ridge_tauhat)
    total.backward()
    gradients = [p.grad for p in live_value.parameters() if p.grad is not None]

    assert gradients and all(g.isfinite().all() for g in gradients)

    champion = Path("data/two_arm.pt")

    if champion.exists():
        trained = DimensionlessValueFunction.load(champion)
        fit_violation, fit_climb, _ = subsolution_loss(trained, muhat, tauhat)

        # The LP objective ALONE -- no ridge, which is the degeneracy breaker
        # the two-sided residual needs -- prefers the champion to the dead
        # solution. That is the whole point: u = 0 is feasible and scores a
        # perfect two-sided residual, and only the climb term rejects it.
        assert (
            fit_violation - fit_climb < dead_violation - dead_climb
        ), "the climb term must reject the dead solution without the ridge"

        # ... and since 2026-08-15 the champion IS a subsolution net trained
        # by this objective, so it overclaims on almost nothing. This assert
        # is what promotion means and it is the one to break if a two-sided
        # net is ever promoted back: the predecessor scored 23.7% here, this
        # one 0.6%.
        lhs, best, _ = trained.hamiltonian(muhat, tauhat)
        residual = ((lhs - best.value) / tauhat.pow(1.5)).detach()
        overclaiming = (residual > 0).float().mean().item()

        assert overclaiming < 0.05, overclaiming
    print("ok")
