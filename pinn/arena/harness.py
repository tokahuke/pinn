"""
The arena's generic core: the policy contract, the environment runner, and
discounted-regret bookkeeping, N-arm (allocations are simplex vectors; arm 0
is control, its effect 0). Per-problem modules (two_arm, three_arm) supply
the effect draw, the observation model, and the policy zoo.

Everything is vectorized over *reps*: state tensors carry a leading (reps,)
dimension, the epoch loop is the only sequential axis, and each rep's noise stream is a
function of its seed alone, so a rep's numbers do not depend on the batch around it
(the demo asserts it).
"""

from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from torch import Tensor
from types import ModuleType
from typing import Self


@dataclass
class Params:
    """One environment: what the world does, and how long and how hard it is played."""

    rho: float
    """Discount per epoch, so regret at epoch t is weighted rho**t."""

    horizon: int
    """Epochs simulated, every one of them, whether or not a policy has committed."""

    sigma: float
    """Noise scale of one observation."""

    effect: float
    """
    Mean of the environment's draw of the true effects. No policy reads it: priors are
    *policy parameters*, not environment knowledge, and matching one to the other is a
    separate study.
    """

    effect_std: float
    """Spread of that draw, read by the environment alone for the same reason."""

    size: int
    """Reps in the sweep."""

    eta: float = 0.0
    """
    Drift volatility of the true effect per epoch. 0 is a static world and recovers the
    pre-drift arena exactly. This is the *environment's* eta; a policy's belief about it
    is a policy parameter, so the two can differ.
    """


class Policy(ABC):
    """
    A policy over a batch of reps: state is (reps,)-shaped, `propose` returns
    (reps, arms), `observe` takes the batched observation. Rep i's numbers must not
    depend on the batch around it, which the demo asserts.
    """

    @classmethod
    @abstractmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        """The policy as the sweep builds it, with every parameter tied to `params`."""

    @abstractmethod
    def observe(self, observation: object) -> None:
        """Fold one epoch's evidence in. Its shape is the problem module's own."""

    @abstractmethod
    def propose(self) -> Tensor:
        """This epoch's allocation, one simplex row per rep."""


def optimal_deadline(rho: float, horizon: int) -> int:
    """
    Deadline for ExploreThenCommit: T**(2/3) with a leading constant of 1, T the
    shorter of the discount's 1/gamma and the hard horizon. It reads the horizon and
    the discount and nothing else, which is a design restriction rather than an
    oversight; kb/arena_results.md has it, with the calibration behind the constant.
    """
    effective = min(float(horizon), 1.0 / (1.0 - rho))

    return max(1, round(effective ** (2.0 / 3.0)))


@dataclass
class Run:
    """
    One simulated experiment: what was allocated and what it cost.

    `regret` is discounted at the runner's rho, so it compares across policies only
    within one runner. `delta` is the effect at epoch 0, which under drift is the
    starting point rather than the truth throughout. `committed` and `committed_at` are
    a record, not a stopping condition: drift makes a vertex escapable.
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
        """One scalar `Run` per rep, which is the pickle format `analyze` reads."""
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
    The environment a batch of reps is played against: noise scale, horizon, discount,
    and one noise row per seed, pre-drawn and consumed through per-rep cursors. Regret
    at epoch t is discounted by rho**t, with rho = 1 - gamma for a decay rate gamma in
    epoch^-1.

    A masked `normal` advances only the consuming reps' cursors, which is what keeps a
    rep's stream independent of its batch (kb/arena_results.md, harness invariants).
    """

    params: Params
    """The environment being played."""

    seeds: list[int]
    """One seed per rep, which is the whole of a rep's noise stream."""

    device: str = "cpu"
    """Where the batch lives."""

    draws_per_epoch: int = 2
    """
    Variates one epoch can consume, per rep. 2 is the static zoos' appetite and the
    default **on purpose**: raising it for everyone would move every arena number ever
    recorded, so a zoo needing more declares DRAWS_PER_EPOCH (kb/arena_results.md).
    """

    noise: Tensor = field(init=False)
    """(reps, capacity) of pre-drawn standard normals, one row per seed."""

    cursor: Tensor = field(init=False)
    """Per-rep index into `noise`, advanced only by the reps that consume a draw."""

    def __post_init__(self) -> None:
        # Two for the effect draw, then the zoo's per-epoch appetite.
        capacity = 2 + self.draws_per_epoch * self.params.horizon
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
        The next variate of every rep's stream, scaled per rep; masked-out reps consume
        nothing and read 0.
        """
        draw = self.noise.gather(1, self.cursor.unsqueeze(1)).squeeze(1).double()

        if mask is None:
            self.cursor += 1

            return draw * deviation + mean

        self.cursor += mask.long()

        return torch.where(mask, draw * deviation + mean, torch.zeros_like(draw))

    def run(self, problem: ModuleType, policy: Policy, deltas: Tensor) -> Batch:
        """
        Play `policy` for the full horizon, all reps at once; the epoch loop is the only
        sequential axis, commit detection a masked first crossing of an exact vertex.
        Regret is measured every epoch against an oracle on the best arm *at that
        epoch*, and `advance` is the identity at eta = 0, so nothing branches on drift.
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

            # Soft commit time: N (1 - sum a^2) / (N - 1) is 4a(1-a) at two arms, 1 at
            # uniform, 0 at a vertex. Summed, the uniform-equivalent epochs of evidence
            # bought (kb/arena_results.md, Readings).
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
    """A rep's numbers do not depend on the batch around it, for every zoo."""
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

    # Paired seeds survive batching: a permuted sub-batch reproduces its reps' numbers,
    # bitwise for the closed-form policies. The trajectory tolerances below cover the
    # float32 wobble (kb/arena_results.md, harness invariants).
    for name in ("two_arm", "two_arm_drift", "three_arm", "three_arm_drift"):
        module = import_module(f"pinn.arena.{name}")
        appetite = getattr(module, "DRAWS_PER_EPOCH", 2)

        for cls in concrete_policies():
            if cls.__module__ != module.__name__:
                continue

            runner = Runner(params, list(range(reps)), draws_per_epoch=appetite)
            batch = runner.run(
                module, cls.init(params, reps, "cpu"), module.draw_effect(runner)
            )

            seeds = [5, 2, 7]
            runner = Runner(params, seeds, draws_per_epoch=appetite)
            sub = runner.run(
                module, cls.init(params, len(seeds), "cpu"), module.draw_effect(runner)
            )

            label = cls.__name__
            assert torch.equal(sub.committed_at, batch.committed_at[seeds]), label
            assert torch.equal(sub.committed, batch.committed[seeds]), label
            assert torch.equal(sub.delta, batch.delta[seeds]), label

            # A *net-carrying* policy is exempt from the trajectory comparisons: no
            # tolerance can be set for it, since the wobble is a property of the loaded
            # checkpoint. The exact fields above carry the test.
            if cls.__name__ == "Pinn":
                continue

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
