"""
The drift objective: maximize the premium subject to not overclaiming, rather
than driving a two-sided residual to zero. two_arm/loss.py holds the record
(kb/two_arm.md section 10); this is that objective with the drift term carried
through, so read it first.

One difference, from the drift term `etahat^2 e^(2s) tauhat_slope` sitting on
the left side: `pos_learning` *stays*, where two_arm deletes it by proof. That
proof needs the left side to be e^s (z + g) alone, and here the drift term can
pay for a negative learning operator, so feasibility no longer subsumes the
sign condition (kb/two_arm_drift.md section 10). The bound argument is
unchanged: the drift term is part of the generator, not an extra source.
"""

from __future__ import annotations

import torch

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import sample_ridge, sample_sobol

CLIMB_WEIGHT = 1.0e-7
"""
What the climb is worth against the violation, both in *natural units*, so only
their ratio matters and *small* is the exact-penalty side (why 1e-5 is
rejected: kb/two_arm_drift.md section 10). Violation rising against a flat
climb means this is too high; climb sagging below 1.73e4 means too low.
"""

RIDGE_WEIGHT = 2.0e4
"""
BC1's share of the constraint. Calibrated against a *satisfied* ridge, which is the
state it has to defend, because the climb term cannot see the wall at all.
"""

SLACK_PRICE = 0.0
"""
What one unit of *slack* costs as a share of the residual budget: the sides are
priced (1 - SLACK_PRICE) and SLACK_PRICE, so moving this reallocates between
them without rescaling the objective. Pinball at q = 1 - SLACK_PRICE. 0 here
because this champion was polished at 0; 0.02 is for *cold starts*.
"""

POSITIVITY_SCALE = 1.0e-7
"""
Saturation point of the L_ab >= 0 term, which is provable and not implied by
feasibility here (see the module docstring). relu is linear in depth, so one
deep violation dominates (max/mean 15,905x) and spikes the gradient norm
1.86 -> 1850 in a single step. Bounded per point, and not `tanh`.
"""

POSITIVITY_WEIGHT = 2.0e-2
"""
What that term is worth against the violation. The champion satisfies
L_ab >= 0 outright (term exactly 0), so it is calibrated where it has to work:
3k iterations in, where the term reads 1.2e-3 against a violation of 2.2e-4,
this holds it at ~11% of the anchor.
"""


def subsolution_loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    etahat: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns the violation, the climb and the saturated learning-operator negative
    part. The violation is the *positive* part of the natural-units residual:
    v > max H breaks the bound, v < max H is slack and left free for the climb to
    tighten. *Linear*, since an L1 penalty is exact at a finite weight.
    """
    lhs, best, l_ab = value.hamiltonian(muhat, tauhat, etahat)
    natural = tauhat.pow(1.5)

    residual = (lhs - best.value) / natural
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()
    climb = (value.premium(muhat, tauhat, etahat) / natural).mean()
    negative = torch.relu(-l_ab)

    return violation, climb, (negative / (POSITIVITY_SCALE + negative)).mean()


def ridge_loss(
    value: DimensionlessValueFunction, ridge_tauhat: Tensor, ridge_etahat: Tensor
) -> Tensor:
    """
    BC1: du/dmuhat = -1/2 at muhat = 0, imposed on the premium (smooth there; the kink
    lives in the value's relu). Holds at every etahat, so the drift coordinate is
    sampled here too.
    """
    ridge_muhat = torch.zeros_like(ridge_tauhat).requires_grad_(True)
    u = value.premium(ridge_muhat, ridge_tauhat, ridge_etahat)
    (u_muhat,) = torch.autograd.grad(u.sum(), ridge_muhat, create_graph=True)

    return (u_muhat + 0.5).pow(2).mean()


def loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    etahat: Tensor,
    ridge_tauhat: Tensor,
    ridge_etahat: Tensor,
    iteration: int | None = None,
) -> Tensor:
    """The penalized LP: climb up, stay feasible, hold BC1 and L_ab >= 0."""
    violation, climb, pos_learning = subsolution_loss(value, muhat, tauhat, etahat)
    ridge = ridge_loss(value, ridge_tauhat, ridge_etahat)

    if iteration is not None:
        print(
            f"iter {iteration}: violation {violation.item():.3e}"
            f"  climb {climb.item():.4e}  ridge {ridge.item():.3e}"
            f"  pos_learning {pos_learning.item():.3e}"
        )

    return (
        violation
        + RIDGE_WEIGHT * ridge
        + POSITIVITY_WEIGHT * pos_learning
        - CLIMB_WEIGHT * climb
    )


def draw(batch: int, device: str = "cpu") -> tuple[Tensor, ...]:
    """
    One step's sample points, in `loss`'s argument order. Split out of `objective`
    so the graphed trainer can hold them as fixed buffers and copy_ fresh draws
    in: a captured cuda graph replays the same tensor addresses.
    """
    return (
        *(t.to(device) for t in sample_sobol(batch)),
        *(t.to(device) for t in sample_ridge(batch // 4)),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """The problem packaged for the generic trainer."""

    def step(value: DimensionlessValueFunction, iteration: int | None) -> Tensor:
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":
    import torch.nn as nn

    from pathlib import Path

    from .model import ExplorationPremium

    muhat, tauhat, etahat = sample_sobol(4096)
    ridge_tauhat, ridge_etahat = sample_ridge(1024)

    class _Zero(nn.Module):
        """
        The never-explore solution. *Cubic* in muhat, not `zeros_like`: the hamiltonian
        takes two z-derivatives and autograd refuses to differentiate a constant tensor
        (CLAUDE.md's stub trap).
        """

        def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
            return 0.0 * muhat**3 * tauhat * (1.0 + etahat)

    dead = DimensionlessValueFunction(_Zero())
    dead_violation, dead_climb, dead_positivity = subsolution_loss(
        dead, muhat, tauhat, etahat
    )

    # **The point of this objective**: the dead solution is *feasible* at any etahat
    # but climbs nothing, so the objective rejects it without help from the ridge
    # term. Under the two-sided residual it is a perfect score.
    assert dead_violation.item() < 1e-12, dead_violation.item()
    assert dead_climb.item() == 0.0, dead_climb.item()
    assert dead_positivity.item() < 1e-12, dead_positivity.item()

    live = DimensionlessValueFunction(ExplorationPremium([16, 16]))
    _, live_climb, _ = subsolution_loss(live, muhat, tauhat, etahat)

    assert 0.0 < live_climb.item() < float("inf")

    total = loss(live, muhat, tauhat, etahat, ridge_tauhat, ridge_etahat)
    total.backward()
    gradients = [p.grad for p in live.parameters() if p.grad is not None]

    assert gradients and all(g.isfinite().all() for g in gradients)

    champion = Path("data/two_arm_drift.pt")

    if champion.exists():
        trained = DimensionlessValueFunction.load(champion)
        fit_violation, fit_climb, _ = subsolution_loss(trained, muhat, tauhat, etahat)

        # The LP objective *alone* (no ridge, which is the degeneracy breaker the
        # two-sided residual needs) prefers the champion to the dead solution.
        assert (
            fit_violation - CLIMB_WEIGHT * fit_climb
            < dead_violation - CLIMB_WEIGHT * dead_climb
        ), "the climb term must reject the dead solution without the ridge"
    print("ok")
