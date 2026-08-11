"""
The arena's generic core: the policy contract, the environment runner, and
discounted-regret bookkeeping, N-arm (allocations are simplex vectors; arm 0
is control, its effect 0). Per-problem modules (two_arm, three_arm) supply
the effect draw, the observation model, and the policy zoo.
"""

from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torch import Tensor
from typing import Self


@dataclass
class Params:
    rho: float
    horizon: int
    sigma: float
    # effect and effect_std are the environment's draw of the true effects;
    # no policy reads them (priors are POLICY PARAMETERS, not environment
    # knowledge -- matching one to the other is a separate study).
    effect: float
    effect_std: float
    size: int
    # Drift volatility of the true effect per epoch. 0 is a static world and
    # recovers the pre-drift arena exactly. This is the ENVIRONMENT's eta; a
    # policy's belief about it is a policy parameter, so the two can differ.
    eta: float = 0.0


class Policy(ABC):
    @classmethod
    @abstractmethod
    def init(cls, params: Params) -> Self:
        pass

    @abstractmethod
    def observe(self, observation: object) -> None:
        pass

    @abstractmethod
    def propose(self) -> Tensor:
        pass


def optimal_deadline(rho: float, horizon: int) -> int:
    """
    Deadline for ExploreThenCommit, from the horizon alone.

    DESIGN RESTRICTION: this may not read effect, effect_std or sigma. The true
    regret-minimising deadline does depend on effect_std/sigma (and only on that
    ratio), but a policy does not know the distribution it is being tested against,
    so tuning to it would make the comparison against the other policies a lie.

    Blind to the effect size, the RATE T**(2/3) is the standard explore-then-commit
    result for an unknown gap: the known-gap optimum is (4/d**2)*log(T*d**2/4), and
    the worst case over d sits at d ~ T**(-1/3). T is the effective horizon, the
    shorter of the discount's 1/gamma and the hard horizon.

    THE LEADING CONSTANT OF 1 IS CALIBRATED, NOT DERIVED: swept against the
    exact optimum, c = 1 costs at most 1.13x the oracle deadline's regret
    over effect_std/sigma in 0.03..0.30.
    """
    effective = min(float(horizon), 1.0 / (1.0 - rho))

    return max(1, round(effective ** (2.0 / 3.0)))


@dataclass
class Run:
    """
    One simulated experiment: what was allocated and what it cost.

    `regret` is discounted at the runner's rho, so it is directly comparable across
    policies only within a single runner. `delta` is the effect at epoch 0; under
    drift it is the starting point, not the truth throughout. `committed` is the
    first arm the policy put all traffic on and `committed_at` the epoch it did
    so -- a record, not a stopping condition, since drift makes a vertex
    escapable and the policy may leave it again.
    """

    delta: list[float]
    policy: str = ""
    epochs: int = 0
    precision_time: float = 0.0
    final_allocation: list[float] = field(default_factory=list)
    regret: float = 0.0
    committed: int | None = None
    committed_at: int | None = None


@dataclass
class Study:
    """
    A whole simulation as pickled: the environment it ran under, plus every run.

    Params travels with the runs because regret is discounted at that rho, so the
    numbers are meaningless without it.
    """

    params: Params
    runs: list[Run]


@dataclass
class Runner:
    """
    The environment a policy is played against: noise scale, horizon, discount, rng.

    Regret at epoch t is discounted by rho**t, with rho = 1 - gamma for a decay rate
    gamma in epoch^-1.

    A single rng is drawn down across every run, so the seed reproduces a whole
    sweep rather than an individual experiment.
    """

    params: Params
    seed: int = 0
    rng: torch.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    @property
    def horizon(self) -> int:
        return self.params.horizon

    @property
    def rho(self) -> float:
        return self.params.rho

    @property
    def effective_horizon(self) -> float:
        return 1.0 / (1.0 - self.rho)

    def normal(self, mean: float, deviation: float) -> float:
        return float(torch.randn((), generator=self.rng)) * deviation + mean

    def run(self, problem, policy: Policy, deltas: Tensor) -> Run:
        """
        Play `policy` against the environment for the full horizon.

        Takes an already-built policy, since the caller constructs one per job via
        `Policy.init`. Reusing an instance across two runs carries state between them.

        The truth advances every epoch through `problem.advance`, which is the
        identity at eta = 0, so there is no static/drifting branch anywhere in
        the loop. Regret is measured against an oracle that puts everything on
        the best arm AT THAT EPOCH -- under drift the oracle switches with the
        world.

        Every epoch is simulated. The old shortcut (a vertex is absorbing, so
        fast-forward the discounted tail) is exact only when the truth is
        frozen; with drift a committed policy keeps accruing against a moving
        oracle and can profitably come back.
        """
        result = Run(delta=[float(d) for d in deltas])

        for epoch in range(self.horizon):
            allocation = policy.propose()
            result.epochs = epoch + 1
            result.final_allocation = [float(a) for a in allocation]
            best = float(deltas.max())
            reward = float((allocation * deltas).sum())
            result.regret += self.rho**epoch * (best - reward)

            # Soft commit time: each epoch counts for its share of the maximum
            # total pairwise learning rate, N (1 - sum a^2) / (N - 1) -- exactly
            # 4a(1-a) at two arms, 1 at uniform, 0 at a vertex. Summed, it is
            # the number of uniform-equivalent epochs of evidence bought.
            arms = len(deltas)
            result.precision_time += (
                arms * (1.0 - float((allocation**2).sum())) / (arms - 1)
            )

            top, arm = allocation.max(dim=-1)

            if float(top) >= 1.0 and result.committed is None:
                result.committed = int(arm)
                result.committed_at = epoch

            policy.observe(problem.observe(self, allocation, deltas))
            deltas = problem.advance(self, deltas)

        return result
