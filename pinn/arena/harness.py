"""
The arena's generic core: the policy contract, the environment runner, and
discounted-regret bookkeeping, N-arm (allocations are simplex vectors; arm 0
is control, its effect 0). Per-problem modules (two_arm, three_arm) supply
the effect draw, the observation model, and the policy zoo.

Everything is vectorized over REPS: state tensors carry a leading (reps,)
dimension, the epoch loop is the only sequential axis, and each rep's noise
stream is a function of its seed alone, so a rep's numbers do not depend on
the batch around it (the demo asserts it).
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
    """
    A policy over a batch of reps: state is (reps,)-shaped, `propose` returns
    (reps, arms), `observe` takes the batched observation. Rep i's numbers
    must not depend on the batch around it -- the demo asserts it.
    """

    @classmethod
    @abstractmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
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
class Batch:
    """
    `Run` over a whole batch: the same bookkeeping, each field a (reps,)- or
    (reps, arms)-shaped tensor, `committed`/`committed_at` carrying -1 for
    never. `runs()` explodes back to scalar Runs, the pickle format analyze
    reads.
    """

    delta: Tensor
    regret: Tensor
    precision_time: Tensor
    final_allocation: Tensor
    committed: Tensor
    committed_at: Tensor
    epochs: int
    policy: str

    def runs(self) -> list[Run]:
        delta = self.delta.cpu()
        regret = self.regret.cpu()
        precision_time = self.precision_time.cpu()
        final_allocation = self.final_allocation.cpu()
        committed = self.committed.cpu()
        committed_at = self.committed_at.cpu()
        out = []

        for i in range(delta.shape[0]):
            at = int(committed_at[i])
            out.append(
                Run(
                    delta=[float(d) for d in delta[i]],
                    policy=self.policy,
                    epochs=self.epochs,
                    precision_time=float(precision_time[i]),
                    final_allocation=[float(a) for a in final_allocation[i]],
                    regret=float(regret[i]),
                    committed=int(committed[i]) if at >= 0 else None,
                    committed_at=at if at >= 0 else None,
                )
            )

        return out


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
    The environment a batch of reps is played against: noise scale, horizon,
    discount, and one noise row per seed, pre-drawn and consumed through
    per-rep cursors. Regret at epoch t is discounted by rho**t, with
    rho = 1 - gamma for a decay rate gamma in epoch^-1.

    A masked `normal` advances only the consuming reps' cursors, so a rep at
    a vertex (which buys no draw) keeps its stream aligned with the same seed
    in any other batch.
    """

    params: Params
    seeds: list[int]
    device: str = "cpu"
    noise: Tensor = field(init=False)
    cursor: Tensor = field(init=False)

    def __post_init__(self) -> None:
        # 2 + 2 * horizon covers the hungriest zoo (three_arm: a two-draw
        # effect, then at most two draws per epoch).
        capacity = 2 + 2 * self.params.horizon
        rows = torch.empty(len(self.seeds), capacity)

        for i, seed in enumerate(self.seeds):
            generator = torch.Generator()
            generator.manual_seed(seed)
            rows[i] = torch.randn(capacity, generator=generator)

        self.noise = rows.to(self.device)
        self.cursor = torch.zeros(len(self.seeds), dtype=torch.long, device=self.device)

    def normal(
        self,
        mean: Tensor | float,
        deviation: Tensor | float,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        The next variate of every rep's stream, scaled per rep; masked-out
        reps consume nothing and read 0.
        """
        draw = self.noise.gather(1, self.cursor.unsqueeze(1)).squeeze(1).double()

        if mask is None:
            self.cursor += 1

            return draw * deviation + mean

        self.cursor += mask.long()

        return torch.where(mask, draw * deviation + mean, torch.zeros_like(draw))

    def run(self, problem, policy: Policy, deltas: Tensor) -> Batch:
        """
        Play `policy` against the environment for the full horizon, all reps
        at once; the epoch loop is the only sequential axis. Commit detection
        is a masked first crossing of an exact vertex. Allocations and deltas
        are float32, the accumulators float64.

        The truth advances every epoch through `problem.advance`, which is the
        identity at eta = 0, so there is no static/drifting branch anywhere in
        the loop. Regret is measured against an oracle that puts everything on
        the best arm AT THAT EPOCH -- under drift the oracle switches with the
        world, and a committed policy keeps accruing against it, so every
        epoch is simulated.
        """
        reps, arms = deltas.shape
        start = deltas
        regret = torch.zeros(reps, dtype=torch.float64, device=self.device)
        precision_time = torch.zeros(reps, dtype=torch.float64, device=self.device)
        committed = torch.full((reps,), -1, dtype=torch.long, device=self.device)
        committed_at = torch.full((reps,), -1, dtype=torch.long, device=self.device)

        for epoch in range(self.params.horizon):
            allocation = policy.propose()
            best = deltas.max(dim=1).values.double()
            reward = (allocation * deltas).sum(dim=1).double()
            regret += self.params.rho**epoch * (best - reward)

            # Soft commit time: each epoch counts for its share of the maximum
            # total pairwise learning rate, N (1 - sum a^2) / (N - 1) -- exactly
            # 4a(1-a) at two arms, 1 at uniform, 0 at a vertex. Summed, it is
            # the number of uniform-equivalent epochs of evidence bought.
            precision_time += (
                arms * (1.0 - (allocation**2).sum(dim=1).double()) / (arms - 1)
            )

            top, arm = allocation.max(dim=1)
            crossing = (top >= 1.0) & (committed_at < 0)
            committed = torch.where(crossing, arm, committed)
            committed_at = torch.where(crossing, epoch, committed_at)

            policy.observe(problem.observe(self, allocation, deltas))
            deltas = problem.advance(self, deltas)

        return Batch(
            delta=start,
            regret=regret,
            precision_time=precision_time,
            final_allocation=allocation,
            committed=committed,
            committed_at=committed_at,
            epochs=self.params.horizon,
            policy=type(policy).__name__,
        )


def demo() -> None:
    from importlib import import_module

    from .main import concrete_policies

    params = Params(
        rho=0.999,
        horizon=300,
        sigma=1.0,
        effect=0.3,
        effect_std=0.4,
        size=8,
        eta=0.02,
    )
    reps = 8

    # Paired seeds survive batching: a permuted sub-batch reproduces its
    # reps' numbers. Bitwise for the closed-form policies at any horizon; the
    # Pinn nets are float32, whose matmuls round batch-size-dependently, and
    # a chaotic trajectory amplifies that wobble (measured here: ~3e-7
    # relative static, ~1e-3 under drift; over thousands of epochs it can
    # decorrelate entirely -- same policy, same noise, different
    # micro-realization). The tolerances sit above the wobble and far below
    # the O(1) any cross-rep stream leakage produces; delta and the commit
    # fields stay exact, which is what a misaligned cursor breaks first.
    for name in ("two_arm", "two_arm_drift", "three_arm"):
        module = import_module(f"pinn.arena.{name}")

        for cls in concrete_policies():
            if cls.__module__ != module.__name__:
                continue

            runner = Runner(params, list(range(reps)))
            batch = runner.run(
                module, cls.init(params, reps, "cpu"), module.draw_effect(runner)
            )

            seeds = [5, 2, 7]
            runner = Runner(params, seeds)
            sub = runner.run(
                module, cls.init(params, len(seeds), "cpu"), module.draw_effect(runner)
            )

            label = cls.__name__
            assert torch.equal(sub.committed_at, batch.committed_at[seeds]), label
            assert torch.equal(sub.committed, batch.committed[seeds]), label
            assert torch.equal(sub.delta, batch.delta[seeds]), label
            assert torch.allclose(
                sub.regret, batch.regret[seeds], rtol=1e-2, atol=1e-6
            ), label
            assert torch.allclose(
                sub.precision_time, batch.precision_time[seeds], rtol=1e-2, atol=1e-6
            ), label
            assert torch.allclose(
                sub.final_allocation, batch.final_allocation[seeds], atol=2e-2
            ), label

        print(f"{name}: sub-batch == full batch, rep for rep")


if __name__ == "__main__":
    demo()
