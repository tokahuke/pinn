"""
The two-arm problem with a drifting mean in the arena: effect draw, drift,
observation model, and the policy zoo.

ONE zoo, not two. Every policy carries its own `sigma` and `eta` as POLICY
parameters, so drift-blind is simply `eta = 0` and there is no separate
drift-unaware class to keep in step. `init` ties them to the environment's --
the correctly-specified case -- and direct construction unties them, which is
how the misspecification grid is swept (same pattern as ExploreThenCommit's
`deadline`).

The grid that motivates the split:

    eta_policy    = 0 < eta        the cost of ignoring real drift
    eta_policy    > eta = 0        the premium for insuring against none
    sigma_policy != sigma          the cost of misjudging the noise

with eta = eta_policy = 0 and sigma_policy = sigma reproducing the two_arm
arena exactly.
"""

from __future__ import annotations

import torch

from dataclasses import dataclass
from functools import cache, cached_property
from math import erfc, sqrt
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.two_arm_drift import DimensionlessValueFunction, ValueFunction
from .harness import Params, Policy, Runner, optimal_deadline

CHECKPOINT = Path("data") / "two_arm_drift.pt"

# The weakest prior the checkpoint is trusted at, in dimensionless precision.
# ponytail: inherited from two_arm's guard against the low-tauhat corner. Drift
# bounds tauhat from ABOVE (the ceiling 1/(2 etahat)) and says nothing about
# the floor, so this needs re-deriving against the drift checkpoint rather than
# trusting the two_arm number.
_FLATTEST_TAUHAT = 1e-2


@cache
def _champion() -> DimensionlessValueFunction:
    """
    One load per process; read-only sharing, nothing here trains.
    """
    return DimensionlessValueFunction.load(CHECKPOINT)


def draw_effect(runner: Runner) -> Tensor:
    """
    The environment's truth at epoch 0: (0, delta).
    """
    delta = runner.normal(runner.params.effect, runner.params.effect_std)

    return torch.tensor([0.0, delta])


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    One epoch of drift on the contrast. Control is the reference and stays at
    0 by definition, so the whole random walk lands on the treatment arm.

    Unconditional: at eta = 0 the draw is N(0, 0) = 0 and the world is static,
    which is the point -- the runner has no static/drifting branch. It does
    consume one variate per epoch either way, so this zoo's noise stream is
    its own; do not compare run-for-run against two_arm's.
    """
    drift = runner.normal(0.0, runner.params.eta)

    return torch.tensor([0.0, float(deltas[1]) + drift])


def observe(runner: Runner, allocation: Tensor, deltas: Tensor) -> tuple[float, float]:
    """
    One epoch's evidence: the noisy contrast estimate, and the DESIGN
    alpha_0 alpha_1 that bought it.

    The design, not the precision. Precision is alpha_0 alpha_1 / sigma^2, and
    that sigma is the policy's belief, not the environment's -- handing over a
    precision computed with the true sigma is what tied the two together in
    the two_arm zoo. The environment supplies how much design information was
    bought and a draw at the true noise; the policy supplies the scale it
    thinks that noise has.

    A vertex buys nothing, and the runner no longer stops there, so that case
    returns zero design and a value the precision-weighted update multiplies
    away.
    """
    design = float(allocation[0] * allocation[1])

    if design == 0.0:
        return 0.0, 0.0

    return runner.normal(float(deltas[1]), runner.params.sigma / sqrt(design)), design


@dataclass(kw_only=True)
class Filter(Policy):
    """
    Kalman posterior on a drifting contrast, carried as (mean, precision).

    Not two_arm's running sums: those are sufficient statistics only under a
    flat prior with perfect memory, and drift forgets. The recursion is
    forecast-then-update, and at eta = 0 it reduces to the same numbers.

    `sigma` and `eta` are POLICY parameters. `init` sets them to the truth;
    construct directly to misspecify either.
    """

    params: Params
    sigma: float
    eta: float
    count: int = 0
    mean: float = 0.0
    precision: float = 0.0

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(params=params, sigma=params.sigma, eta=params.eta)

    def observe(self, observation: tuple[float, float]) -> None:
        estimate, design = observation
        self.count += 1

        # Forecast: drift erodes precision before the update. Written as
        # tau / (1 + eta^2 tau) rather than 1 / (1/tau + eta^2) so a flat
        # prior (tau = 0) needs no special case.
        self.precision = self.precision / (1.0 + self.eta**2 * self.precision)

        gained = design / self.sigma**2
        total = self.precision + gained

        if total > 0.0:
            self.mean = (self.mean * self.precision + estimate * gained) / total
        self.precision = total

    @property
    def z(self) -> float:
        return self.mean * sqrt(self.precision)

    @property
    def prob_positive(self) -> float:
        """
        P(delta > 0) under the posterior, via stdlib erfc: it runs every epoch.
        """
        return 0.5 * erfc(-self.z / sqrt(2.0))


def _split(treatment: float) -> Tensor:
    return torch.tensor([1.0 - treatment, treatment])


@dataclass(kw_only=True)
class ExploreThenCommit(Filter):
    """
    Explore at the most informative allocation, then commit on the sign of the
    estimate at a fixed deadline. Under drift the commitment is permanent
    anyway -- the policy stops looking -- which is exactly the failure the
    drift-aware entrants should exploit.
    """

    deadline: int = 50

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(
            params=params,
            sigma=params.sigma,
            eta=params.eta,
            deadline=optimal_deadline(params.rho, params.horizon),
        )

    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return _split(0.5)

        return _split(1.0 if self.mean > 0.0 else 0.0)


@dataclass(kw_only=True)
class ProbabilityMatching(Filter):
    """
    Allocate the posterior probability that treatment wins: Thompson sampling
    in its two-arm closed form. With eta > 0 the posterior stops sharpening at
    the ceiling, so this keeps a permanent sliver on the loser instead of
    saturating -- it never reaches a vertex and never commits.
    """

    def propose(self) -> Tensor:
        return _split(self.prob_positive)


@dataclass(kw_only=True)
class ZTest(Filter):
    """
    Explore at 50/50 while the null holds, commit to the detected side once it
    is rejected at `p_value`, two-sided. Retested every epoch at the nominal
    level, so the real type I error is far above it.
    """

    p_value: float = 0.05

    @cached_property
    def threshold(self) -> float:
        return float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))

    def propose(self) -> Tensor:
        if abs(self.z) < self.threshold:
            return _split(0.5)

        return _split(1.0 if self.z > 0.0 else 0.0)


@dataclass(kw_only=True)
class Pinn(Filter):
    """
    The trained drift HJB policy, mapped onto arena units with rate
    gamma = 1 - rho (exact is -log rho, O(gamma^2) apart).

    Its etahat is the POLICY's eta, so one checkpoint plays every column of the
    misspecification grid -- that is what carrying etahat as a net input buys.
    """

    prior_std: float | None = None
    value: ValueFunction | None = None

    @classmethod
    def init(cls, params: Params) -> Self:
        policy = cls(params=params, sigma=params.sigma, eta=params.eta)
        policy.value = ValueFunction(
            _champion(),
            rho=1.0 - params.rho,
            sigma=policy.sigma,
            eta=policy.eta,
        )

        return policy

    def propose(self) -> Tensor:
        gamma = 1.0 - self.params.rho
        floor = _FLATTEST_TAUHAT / (gamma * self.sigma**2)

        if self.prior_std is not None:
            floor = 1.0 / self.prior_std**2
        tau = max(self.precision, floor)

        return _split(
            float(self.value.policy(torch.tensor([self.mean]), torch.tensor([tau])))
        )


def demo() -> None:
    import pinn.arena.two_arm_drift as problem

    static = Runner(
        Params(rho=0.999, horizon=500, sigma=1.0, effect=0.5, effect_std=0.0, size=1)
    )
    winner = torch.tensor([0.0, 0.5])

    # eta = 0 everywhere: the filter is two_arm's accumulator, and every
    # policy behaves as it does there.
    matching = static.run(problem, ProbabilityMatching.init(static.params), winner)
    assert matching.final_allocation[1] > 0.95, matching.final_allocation

    null = static.run(problem, ProbabilityMatching.init(static.params), torch.zeros(2))
    assert abs(null.regret) < 1e-9, null.regret

    etc = static.run(problem, ExploreThenCommit.init(static.params), winner)
    assert etc.committed == 1, etc.committed
    assert etc.committed_at == ExploreThenCommit.init(static.params).deadline
    assert abs(etc.precision_time - etc.committed_at) < 1e-6, etc.precision_time

    # Drift: the truth moves, so the effect at the end is not the effect at
    # the start, and the oracle moves with it.
    drifting = Runner(
        Params(
            rho=0.999,
            horizon=500,
            sigma=1.0,
            effect=0.5,
            effect_std=0.0,
            size=1,
            eta=0.05,
        )
    )
    walked = problem.draw_effect(drifting)
    for _ in range(500):
        walked = problem.advance(drifting, walked)

    assert float(walked[0]) == 0.0, walked
    assert abs(float(walked[1]) - 0.5) > 1e-3, walked

    # A drift-aware filter stops sharpening: its precision saturates at the
    # ceiling instead of growing without bound, which is the whole difference.
    aware = ProbabilityMatching.init(drifting.params)
    blind = ProbabilityMatching(params=drifting.params, sigma=1.0, eta=0.0)

    for _ in range(400):
        evidence = (0.5, 0.25)
        aware.observe(evidence)
        blind.observe(evidence)

    assert blind.precision > 5.0 * aware.precision, (blind.precision, aware.precision)

    # It converges to the recursion's own fixed point: tau = tau/(1 + eta^2 tau)
    # + p solves to p/2 + sqrt(p^2/4 + p/eta^2). That sits just ABOVE the
    # continuous ceiling sqrt(p)/eta = 1/(2 sigma eta), by the half-lump p/2 --
    # the discrete filter adds a whole epoch of information after the decay
    # where continuous time interleaves the two.
    gained, drift = 0.25, 0.05
    fixed_point = gained / 2.0 + sqrt(gained**2 / 4.0 + gained / drift**2)

    assert abs(aware.precision - fixed_point) < 1e-6, (aware.precision, fixed_point)
    assert fixed_point > 1.0 / (2.0 * 1.0 * drift)

    print(f"matching regret {matching.regret:.2f}, final a {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(f"drifted effect after 500 epochs: {float(walked[1]):.3f} (started 0.500)")
    print(
        f"precision after 400 epochs: aware {aware.precision:.2f}, blind {blind.precision:.2f}"
    )


if __name__ == "__main__":
    demo()
