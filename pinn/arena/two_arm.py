"""
The two-arm problem in the arena: effect draw, observation model, and the
policy zoo (allocations are (control, treatment) simplex vectors).
"""

from __future__ import annotations

import torch

from functools import cache, cached_property
from math import sqrt
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.two_arm.model import DimensionlessValueFunction, ValueFunction
from .harness import Params, Policy, Run, Runner, optimal_deadline

CHECKPOINT = Path("data") / "two_arm.pt"
"""The champion the Pinn policy plays, repo-root relative like every other data path."""

_FLATTEST_TAUHAT = 1e-3
"""
The weakest prior the champion is trusted at, in dimensionless precision: the
sampler's own PRIOR_FLOOR, so the guard does not clamp inside the training support.
Measured 2026-08-13, and re-measure it whenever the champion changes
(kb/arena_results.md, the prior floor).
"""


@cache
def _champion() -> DimensionlessValueFunction:
    """
    One load per process: the arena constructs a fresh policy per run, and the
    checkpoint (plus its feature-scale calibration draw) must not be re-read every
    time. Read-only sharing, since nothing here trains.
    """
    return DimensionlessValueFunction.load(CHECKPOINT)


def draw_effect(runner: Runner) -> Tensor:
    """
    The environment's truth: (0, delta), one row per rep, the treatment
    effect drawn from the study's distribution.
    """
    delta = runner.normal(runner.params.effect, runner.params.effect_std)

    return torch.stack((torch.zeros_like(delta), delta), dim=1).float()


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    The truth between epochs. Static here by definition of the problem; drift
    lives in two_arm_drift, so --eta has no effect on this zoo.
    """
    return deltas


def observe(
    runner: Runner, allocation: Tensor, deltas: Tensor
) -> tuple[Tensor, Tensor]:
    """
    One epoch's evidence: a noisy estimate of the contrast at the precision the split
    bought, alpha_0 alpha_1 / sigma^2. A vertex rep buys nothing, consuming no draw
    (the mask skips its cursor) and reading zero precision, which the
    precision-weighted update multiplies away.
    """
    precision = (allocation[:, 0] * allocation[:, 1]).double() / runner.params.sigma**2
    live = precision > 0.0
    deviation = precision.masked_fill(~live, 1.0) ** -0.5

    return runner.normal(deltas[:, 1].double(), deviation, live), precision


def _split(treatment: Tensor) -> Tensor:
    """A treatment share as a (reps, 2) simplex row."""
    return torch.stack((1.0 - treatment, treatment), dim=1).float()


class Bayesian(Policy):
    """
    Flat-prior normal posterior on delta, accumulated in precision-weighted
    form over (reps,) float64 tensors.

    After n observations the posterior is N(total / total_precision, 1 /
    total_precision), so `mean` is the inverse-variance-weighted estimate and `z`
    is that estimate in posterior-standard-deviation units.
    """

    def __init__(self, params: Params, reps: int, device: str) -> None:
        self.params = params
        self.count = 0
        self.total = torch.zeros(reps, dtype=torch.float64, device=device)
        self.total_precision = torch.zeros(reps, dtype=torch.float64, device=device)

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(params, reps, device)

    def observe(self, observation: tuple[Tensor, Tensor]) -> None:
        delta, precision = observation

        self.count += 1
        self.total = self.total + delta * precision
        self.total_precision = self.total_precision + precision

    @property
    def mean(self) -> Tensor:
        """Posterior mean of delta; 0 before any evidence."""
        live = self.total_precision > 0.0

        return torch.where(
            live, self.total / self.total_precision.masked_fill(~live, 1.0), 0.0
        )

    @property
    def z(self) -> Tensor:
        """The posterior mean in posterior-standard-deviation units."""
        live = self.total_precision > 0.0

        return torch.where(
            live, self.total / self.total_precision.masked_fill(~live, 1.0).sqrt(), 0.0
        )

    @property
    def prob_positive(self) -> Tensor:
        """P(delta > 0) under the posterior."""
        return 0.5 * torch.special.erfc(-self.z / sqrt(2.0))


class ExploreThenCommit(Bayesian):
    """
    Explore at the most informative allocation, then commit on the sign of the
    estimate at a fixed deadline, however uncertain it is by then.
    """

    def __init__(self, params: Params, reps: int, device: str, deadline: int) -> None:
        super().__init__(params, reps, device)
        self.deadline = deadline

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(
            params, reps, device, deadline=optimal_deadline(params.rho, params.horizon)
        )

    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return _split(torch.full_like(self.total, 0.5))

        return _split((self.mean > 0.0).double())


class ProbabilityMatching(Bayesian):
    """
    Allocate the posterior probability that treatment wins: a = P(delta > 0).
    Thompson sampling, in its two-arm closed form.

    Exploration decays on its own as the posterior sharpens, with nothing to tune.
    It reaches a border (and so commits) only once |z| passes roughly 8.3, where
    the normal CDF saturates to exactly 0.0 or 1.0 in float64; before that it
    keeps a vanishing sliver of the traffic on the losing arm forever.
    """

    def propose(self) -> Tensor:
        return _split(self.prob_positive)


class ZTest(Bayesian):
    """
    Explore at 50/50 while the null holds, commit to the detected side once it is
    rejected at `p_value`, two-sided. No inference beyond reject / do not reject.

    The null is retested every epoch at the nominal level, so the real type I error
    is far above `p_value`: this rejects, and therefore commits to the wrong side,
    much more often than the number suggests.
    """

    # ponytail: fixed nominal level, no alpha spending. Swap `threshold` for an
    # O'Brien-Fleming or Pocock boundary in `count` if the peeking cost matters.
    p_value: float = 0.05
    """Two-sided nominal level the null is tested at, every epoch."""

    @cached_property
    def threshold(self) -> float:
        """The |z| that rejects at `p_value`, two-sided."""
        return float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))

    def propose(self) -> Tensor:
        commit = (self.z > 0.0).double()

        return _split(torch.where(self.z.abs() < self.threshold, 0.5, commit))


class Pinn(Bayesian):
    """
    The trained two_arm HJB policy: the arena's optimal-policy ceiling, mapped onto
    arena units with rate gamma = 1 - rho (exact is -log rho, O(gamma^2) apart).
    Commits are exact vertices of the simplex max, so the arena's absorbing border
    triggers honestly.

    The prior is a **policy parameter**, not environment knowledge: prior_std is its
    standard deviation on delta in arena units, and None means the flattest prior the
    checkpoint supports, which is legal against the prior-blind zoo.
    """

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        prior_std: float | None = None,
    ) -> None:
        super().__init__(params, reps, device)
        self.prior_std = prior_std
        self.value: ValueFunction | None = None

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        policy = cls(params, reps, device)
        policy.value = ValueFunction(
            _champion(), rho=1.0 - params.rho, sigma=params.sigma
        ).to(device)

        return policy

    def propose(self) -> Tensor:
        if self.prior_std is None:
            gamma = 1.0 - self.params.rho
            prior_tau = _FLATTEST_TAUHAT / (gamma * self.params.sigma**2)
        else:
            prior_tau = 1.0 / self.prior_std**2
        tau = prior_tau + self.total_precision
        mu = self.total / tau

        return _split(self.value.policy(mu.float(), tau.float()).double())


def demo() -> None:
    """Each policy reaches the border its docstring claims, on a clear winner."""
    import pinn.arena.two_arm as problem

    params = Params(
        rho=0.999, horizon=500, sigma=1.0, effect=0.5, effect_std=0.0, size=1
    )
    winner = torch.tensor([[0.0, 0.5]])
    loser = torch.tensor([[0.0, -0.5]])

    def play(cls: type[Policy], deltas: Tensor) -> Run:
        runner = Runner(params, [0])

        return runner.run(problem, cls.init(params, 1, "cpu"), deltas).runs()[0]

    # A clear winner: nearly all traffic should end up on it; the mirror
    # image holds nearly all traffic on control.
    matching = play(ProbabilityMatching, winner)
    assert matching.final_allocation[1] > 0.95, matching.final_allocation

    losing = play(ProbabilityMatching, loser)
    assert losing.final_allocation[1] < 0.1, losing.final_allocation

    # No effect at all: nothing to learn, no regret either way.
    null = play(ProbabilityMatching, torch.zeros(1, 2))
    assert abs(null.regret) < 1e-9, null.regret

    # Both committers reach the same border, by different routes; init
    # derives the deadline from params, so the commit epoch tracks it.
    etc = play(ExploreThenCommit, winner)
    assert etc.committed == 1, etc.committed
    assert etc.committed_at == ExploreThenCommit.init(params, 1, "cpu").deadline

    # Soft commit time reduces to the commit epoch exactly: every epoch before
    # it is a 50/50 split worth 1, every epoch after is a vertex worth 0.
    assert abs(etc.precision_time - etc.committed_at) < 1e-6, etc.precision_time

    ztest = play(ZTest, winner)
    assert ztest.committed == 1, ztest.committed

    # The PINN: at this gamma the effect is many prior sd, so it must commit
    # to the winner, and it opens at an even split (no evidence, near-flat
    # prior). Exercises the whole unit mapping end to end.
    pinn_policy = Pinn.init(params, 1, "cpu")
    assert abs(float(pinn_policy.propose()[0, 1]) - 0.5) < 0.05

    pinn = play(Pinn, winner)
    assert pinn.committed == 1, (pinn.committed, pinn.final_allocation)

    print(f"matching regret {matching.regret:.2f}, final a {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(f"ztest regret {ztest.regret:.2f}, committed at epoch {ztest.committed_at}")
    print(f"pinn regret {pinn.regret:.2f}, committed at epoch {pinn.committed_at}")


if __name__ == "__main__":
    demo()
