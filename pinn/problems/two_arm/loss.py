"""
The two-arm objective: maximize the premium subject to not overclaiming, rather than
driving a two-sided residual to zero. kb/two_arm.md section 10 is the record,
including the two-sided objective this replaced and the variants that failed.

`V*` is the *maximal* subsolution of the HJB (learnings section 9), so this has the
true value function as its optimum and any feasible point is a proven lower bound.
The never-explore degeneracy then dies in the *objective* rather than needing a
breaker, since `u = 0` is merely feasible and the climb rejects it.

The cost is that the climb's gradient never vanishes, so the loss settles at a
*balance* and its value stops being a quality measure: judge a run by the printed
violation, the mean gate and `sup(residual+)`, never by the total.
"""

from __future__ import annotations

import torch

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import sample_ridge, sample_sobol

CLIMB_WEIGHT = 1.0e-7
"""
What the climb is worth against the violation, both in natural units, so only
their ratio matters. How it was set and how to read it going wrong:
kb/two_arm.md section 10.
"""

SLACK_PRICE = 0.0
"""
What one unit of slack costs, as a share of the residual budget: the pinball
loss at `q = 1 - SLACK_PRICE`, 0 being the pure subsolution objective. Why
slack is priced at all, with the table: kb/two_arm.md section 10.
"""

RIDGE_WEIGHT = 2.0e4
"""
BC1's share of the constraint, calibrated against a *satisfied* ridge, the state it
defends. The climb cannot see the wall at all, and whenever `SLACK_PRICE > 0` this is
also the never-explore breaker: the commit envelope scores the priced residual exactly
0 on both sides, so only the ridge and the climb reject it.
"""


def subsolution_loss(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    The priced residual, the climb, and the learning-operator negative part, the last
    a printed diagnostic rather than a loss term because the constraint subsumes it
    (kb/two_arm.md section 10). The residual is split by *sign* and the two sides
    priced against each other. Natural units, never scaled, as everywhere else.
    """
    lhs, best, l_ab = value.hamiltonian(muhat, tauhat)
    natural = tauhat.pow(1.5)
    residual = (lhs - best.value) / natural

    # *Linear*, not squared: an L1 penalty is exact where a quadratic only approaches
    # feasibility as the weight grows. Squaring also chases depth while the violating
    # fraction rises (two_arm_drift, 2026-08-08). kb/two_arm.md section 10.
    violation = (1.0 - SLACK_PRICE) * torch.relu(residual).mean()

    if SLACK_PRICE > 0.0:
        violation = violation + SLACK_PRICE * torch.relu(-residual).mean()

    # Natural units, matching the violation above: `u / tauhat**1.5` is the premium
    # in the units the residual is graded in.
    premium = value.premium(muhat, tauhat)
    climb = (premium / natural).mean()

    return violation, climb, torch.relu(-l_ab).mean()


def ridge_loss(value: DimensionlessValueFunction, ridge_tauhat: Tensor) -> Tensor:
    """
    BC1, `du/dmuhat = -1/2` at `muhat = 0`, imposed on the premium, which is smooth
    there while the kink lives in the value's relu.
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
    """The penalized LP: climb up, stay feasible, hold BC1."""
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
    """One step's sample points, in `loss`'s argument order."""
    return (
        *(t.to(device) for t in sample_sobol(batch)),
        sample_ridge(batch // 4).to(device),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """The problem packaged for the generic trainer."""

    def step(value: DimensionlessValueFunction, iteration: int | None) -> Tensor:
        """One scored step on a fresh draw."""
        return loss(value, *draw(batch, device), iteration)

    return step


if __name__ == "__main__":
    import torch.nn as nn

    from pathlib import Path

    from .model import ExplorationPremium

    muhat, tauhat = sample_sobol(4096)
    ridge_tauhat = sample_ridge(1024)

    class _Dead(nn.Module):
        """
        The never-explore solution, *squared* rather than `0.0 * muhat`.

        CLAUDE.md's stub rule covers one derivative and this loss takes two: the
        first derivative of a `z`-linear stub no longer depends on `z`, so the second
        grad call reports it unused. `0.0 * muhat**2` is still identically zero and
        stays connected through both.
        """

        def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
            return 0.0 * muhat**2

    dead = DimensionlessValueFunction(_Dead())
    dead_violation, dead_climb, _ = subsolution_loss(dead, muhat, tauhat)

    # The point of this objective: the dead solution is *feasible*, since it does not
    # overclaim, but it climbs nothing, so the objective rejects it without help from
    # the ridge. Under the two-sided residual it scored perfectly.
    assert dead_violation.item() < 1e-12, dead_violation.item()
    assert dead_climb.item() == 0.0, dead_climb.item()

    live = ExplorationPremium([16, 16])
    live_value = DimensionlessValueFunction(live)
    _, live_climb, _ = subsolution_loss(live_value, muhat, tauhat)

    # Positive and finite on a live net, and *not* bounded above: the natural-units
    # climb grows like `tauhat**-2` toward the floor, which it has to, since matching
    # the violation's scaling there is what keeps the floor decade.
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

        # The LP objective *alone*, no ridge to break the degeneracy, prefers the
        # champion to the dead solution: `u = 0` is feasible and scores a perfect
        # two-sided residual, so only the climb rejects it.
        assert (
            fit_violation - fit_climb < dead_violation - dead_climb
        ), "the climb term must reject the dead solution without the ridge"

        # The champion is trained by this objective, so it overclaims on almost
        # nothing. This assert is what promotion means, and the one to break if a
        # two-sided net is promoted back: the predecessor scored 23.7%, this one 0.6%.
        lhs, best, _ = trained.hamiltonian(muhat, tauhat)
        residual = ((lhs - best.value) / tauhat.pow(1.5)).detach()
        overclaiming = (residual > 0).float().mean().item()

        assert overclaiming < 0.05, overclaiming
    print("ok")
