"""
The two-arm problem in the arena: effect draw, observation model, and the
policy zoo (allocations are (control, treatment) simplex vectors).
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from functools import cache, cached_property
from math import erfc, sqrt
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.two_arm import DimensionlessValueFunction, ValueFunction
from .harness import Params, Policy, Runner, optimal_deadline

# The champion checkpoint the Pinn policy plays; repo-root relative, like
# every other script's data path.
CHECKPOINT = Path("data") / "two_arm.pt"

# Internal (dimensionless) detail: the weakest prior the champion currently
# supports. Below this decade its L_ab goes negative at the ridge (the
# stiff-corner error, measured 2026-08-06), flipping the Hamiltonian convex
# and vertex-committing on zero evidence; revisit when the low-tau anchor
# lands. One decade above the sampler floor 1e-3.
_FLATTEST_TAUHAT = 1e-2


@cache
def _champion() -> DimensionlessValueFunction:
    """
    One load per process: the arena constructs a fresh policy per run job,
    and the checkpoint (plus its feature-scale calibration draw) must not be
    re-read every time. Read-only sharing -- nothing here trains.
    """
    return DimensionlessValueFunction.load(CHECKPOINT)


def draw_effect(runner: Runner) -> Tensor:
    """
    The environment's truth: (0, delta), the treatment effect drawn from the
    study's distribution.
    """
    delta = runner.normal(runner.params.effect, runner.params.effect_std)

    return torch.tensor([0.0, delta])


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    The truth between epochs. Static here by definition of the problem; drift
    lives in two_arm_drift, so --eta has no effect on this zoo.
    """
    return deltas


def observe(runner: Runner, allocation: Tensor, deltas: Tensor) -> tuple[float, float]:
    """
    One epoch's evidence: a noisy estimate of the contrast at the precision
    the split bought, alpha_0 alpha_1 / sigma^2.

    A vertex buys nothing, and the runner no longer stops there, so that case
    has to be finite rather than 1/0: zero precision, and a value the
    precision-weighted update multiplies away.
    """
    precision = float(allocation[0] * allocation[1]) / runner.params.sigma**2

    if precision == 0.0:
        return 0.0, 0.0

    return runner.normal(float(deltas[1]), precision**-0.5), precision


@dataclass(kw_only=True)
class Bayesian(Policy):
    """
    Flat-prior normal posterior on delta, accumulated in precision-weighted form.

    After n observations the posterior is N(total / total_precision, 1 /
    total_precision), so `mean` is the inverse-variance-weighted estimate and `z`
    is that estimate in posterior-standard-deviation units.
    """

    params: Params
    count: int = 0
    total: float = 0.0
    total_precision: float = 0.0

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(params=params)

    def observe(self, observation: tuple[float, float]) -> None:
        delta, precision = observation

        self.count += 1
        self.total += delta * precision
        self.total_precision += precision

    @property
    def mean(self) -> float:
        if self.total_precision == 0.0:
            return 0.0

        return self.total / self.total_precision

    @property
    def z(self) -> float:
        if self.total_precision == 0.0:
            return 0.0

        return self.total / sqrt(self.total_precision)

    @property
    def prob_positive(self) -> float:
        """
        P(delta > 0) under the posterior, via stdlib erfc: it runs every epoch.
        """
        return 0.5 * erfc(-self.z / sqrt(2.0))


def _split(treatment: float) -> Tensor:
    return torch.tensor([1.0 - treatment, treatment])


@dataclass
class ExploreThenCommit(Bayesian):
    """
    Explore at the most informative allocation, then commit on the sign of the
    estimate at a fixed deadline, however uncertain it is by then.
    """

    # `init` computes this from params. The default is only for direct construction,
    # which is how you sweep deadlines without going through `init`.
    deadline: int = 50

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(params=params, deadline=optimal_deadline(params.rho, params.horizon))

    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return _split(0.5)

        return _split(1.0 if self.mean > 0.0 else 0.0)


@dataclass
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


@dataclass
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

    @cached_property
    def threshold(self) -> float:
        return float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))

    def propose(self) -> Tensor:
        if abs(self.z) < self.threshold:
            return _split(0.5)

        return _split(1.0 if self.z > 0.0 else 0.0)


@dataclass
class Pinn(Bayesian):
    """
    The trained two_arm HJB policy: the arena's optimal-policy ceiling,
    mapped onto arena units with rate gamma = 1 - rho (exact is -log rho,
    O(gamma^2) apart). Commits are exact vertices of the simplex max, so the
    arena's absorbing border triggers honestly.

    The prior is a POLICY PARAMETER, not environment knowledge: prior_std is
    the prior standard deviation on delta, in the arena's own units. None
    means the flattest prior the checkpoint supports, computed per
    environment -- the near-flat entrant, legal against the prior-blind zoo.
    """

    prior_std: float | None = None
    value: ValueFunction | None = None

    @classmethod
    def init(cls, params: Params) -> Self:
        policy = cls(params=params)
        policy.value = ValueFunction(
            _champion(), rho=1.0 - params.rho, sigma=params.sigma
        )

        return policy

    def propose(self) -> Tensor:
        if self.prior_std is None:
            gamma = 1.0 - self.params.rho
            prior_tau = _FLATTEST_TAUHAT / (gamma * self.params.sigma**2)
        else:
            prior_tau = 1.0 / self.prior_std**2
        tau = prior_tau + self.total_precision
        mu = self.total / tau

        return _split(float(self.value.policy(torch.tensor([mu]), torch.tensor([tau]))))


def demo() -> None:
    import pinn.arena.two_arm as problem

    runner = Runner(
        Params(rho=0.999, horizon=500, sigma=1.0, effect=0.5, effect_std=0.0, size=1)
    )
    winner = torch.tensor([0.0, 0.5])
    loser = torch.tensor([0.0, -0.5])

    # A clear winner: nearly all traffic should end up on it; the mirror
    # image holds nearly all traffic on control.
    matching = runner.run(problem, ProbabilityMatching.init(runner.params), winner)
    assert matching.final_allocation[1] > 0.95, matching.final_allocation

    losing = runner.run(problem, ProbabilityMatching.init(runner.params), loser)
    assert losing.final_allocation[1] < 0.05, losing.final_allocation

    # No effect at all: nothing to learn, no regret either way.
    null = runner.run(problem, ProbabilityMatching.init(runner.params), torch.zeros(2))
    assert abs(null.regret) < 1e-9, null.regret

    # Both committers reach the same border, by different routes; init
    # derives the deadline from params, so the commit epoch tracks it.
    etc = runner.run(problem, ExploreThenCommit.init(runner.params), winner)
    assert etc.committed == 1, etc.committed
    assert etc.committed_at == ExploreThenCommit.init(runner.params).deadline

    # Soft commit time reduces to the commit epoch exactly: every epoch before
    # it is a 50/50 split worth 1, every epoch after is a vertex worth 0.
    assert abs(etc.precision_time - etc.committed_at) < 1e-6, etc.precision_time

    ztest = runner.run(problem, ZTest.init(runner.params), winner)
    assert ztest.committed == 1, ztest.committed

    # The PINN: at this gamma the effect is many prior sd, so it must commit
    # to the winner, and it opens at an even split (no evidence, near-flat
    # prior). Exercises the whole unit mapping end to end.
    pinn_policy = Pinn.init(runner.params)
    assert abs(float(pinn_policy.propose()[1]) - 0.5) < 0.05

    pinn = runner.run(problem, pinn_policy, winner)
    assert pinn.committed == 1, (pinn.committed, pinn.final_allocation)

    print(f"matching regret {matching.regret:.2f}, final a {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(f"ztest regret {ztest.regret:.2f}, committed at epoch {ztest.committed_at}")
    print(f"pinn regret {pinn.regret:.2f}, committed at epoch {pinn.committed_at}")


if __name__ == "__main__":
    demo()
