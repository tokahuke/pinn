"""
The two-arm problem with a drifting mean in the arena: effect draw, drift, observation
model, and the policy zoo.

**One** zoo, not two. Every policy carries its own `sigma` and `eta` as *policy*
parameters, so drift-blind is simply `eta = 0`, `init` ties them to the environment's,
and direct construction unties them to sweep the misspecification grid. At
`eta = eta_policy = 0` and `sigma_policy = sigma` this reproduces the two_arm arena
exactly, Pinn included, which the demo asserts. kb/arena_results.md has the grid.
"""

from __future__ import annotations

import torch

from functools import cache, cached_property
from math import sqrt
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.two_arm_drift.model import DimensionlessValueFunction, ValueFunction
from .harness import Params, Policy, Run, Runner, optimal_deadline

CHECKPOINT = Path("data") / "two_arm_drift.pt"
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
    """One load per process; read-only sharing, since nothing here trains."""
    return DimensionlessValueFunction.load(CHECKPOINT)


def draw_effect(runner: Runner) -> Tensor:
    """The environment's truth at epoch 0: (0, delta), one row per rep."""
    delta = runner.normal(runner.params.effect, runner.params.effect_std)

    return torch.stack((torch.zeros_like(delta), delta), dim=1).float()


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    One epoch of drift on the contrast. Control is the reference and stays at 0, so the
    whole random walk lands on the treatment arm. Unconditional, so nothing branches on
    drift, which costs one variate per epoch either way: this zoo's noise stream is its
    own and is not comparable run-for-run against two_arm's.
    """
    drift = runner.normal(0.0, runner.params.eta)

    return torch.stack(
        (torch.zeros_like(drift), deltas[:, 1].double() + drift), dim=1
    ).float()


def observe(
    runner: Runner, allocation: Tensor, deltas: Tensor
) -> tuple[Tensor, Tensor]:
    """
    One epoch's evidence: the noisy contrast estimate, and the *design* alpha_0 alpha_1
    that bought it. The design, not the precision, because precision divides by a sigma
    the policy owns rather than the environment: here the environment supplies the
    design and a draw at the true noise. A vertex rep buys nothing and reads (0, 0).
    """
    design = (allocation[:, 0] * allocation[:, 1]).double()
    live = design > 0.0
    deviation = runner.params.sigma / design.masked_fill(~live, 1.0).sqrt()

    return runner.normal(deltas[:, 1].double(), deviation, live), design


def _split(treatment: Tensor) -> Tensor:
    """A treatment share as a (reps, 2) simplex row."""
    return torch.stack((1.0 - treatment, treatment), dim=1).float()


class Filter(Policy):
    """
    Kalman posterior on a drifting contrast, carried as (mean, precision), both
    (reps,) float64.

    Not two_arm's running sums: those are sufficient statistics only under a flat prior
    with perfect memory, and drift forgets. The recursion is forecast-then-update, and
    at eta = 0 it reduces to the same numbers. `sigma` and `eta` are *policy*
    parameters, set to the truth by `init` and misspecified by constructing directly.
    """

    def __init__(
        self, params: Params, reps: int, device: str, sigma: float, eta: float
    ) -> None:
        self.params = params
        self.sigma = sigma
        self.eta = eta
        self.count = 0
        self.mean = torch.zeros(reps, dtype=torch.float64, device=device)
        self.precision = torch.zeros(reps, dtype=torch.float64, device=device)

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(params, reps, device, sigma=params.sigma, eta=params.eta)

    def observe(self, observation: tuple[Tensor, Tensor]) -> None:
        estimate, design = observation
        self.count += 1

        # Forecast: drift erodes precision before the update. Written as
        # tau / (1 + eta^2 tau) rather than 1 / (1/tau + eta^2) so a flat
        # prior (tau = 0) needs no special case.
        self.precision = self.precision / (1.0 + self.eta**2 * self.precision)

        gained = design / self.sigma**2
        total = self.precision + gained

        self.mean = torch.where(
            total > 0.0,
            (self.mean * self.precision + estimate * gained)
            / total.masked_fill(total == 0.0, 1.0),
            self.mean,
        )
        self.precision = total

    @property
    def z(self) -> Tensor:
        """The posterior mean in posterior-standard-deviation units."""
        return self.mean * self.precision.sqrt()

    @property
    def prob_positive(self) -> Tensor:
        """P(delta > 0) under the posterior."""
        return 0.5 * torch.special.erfc(-self.z / sqrt(2.0))


class ExploreThenCommit(Filter):
    """
    Explore at the most informative allocation, then commit on the sign of the
    estimate at a fixed deadline. Under drift the commitment is permanent anyway
    (the policy stops looking), which is exactly the failure the drift-aware entrants
    should exploit.
    """

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        sigma: float,
        eta: float,
        deadline: int,
    ) -> None:
        super().__init__(params, reps, device, sigma, eta)
        self.deadline = deadline

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(
            params,
            reps,
            device,
            sigma=params.sigma,
            eta=params.eta,
            deadline=optimal_deadline(params.rho, params.horizon),
        )

    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return _split(torch.full_like(self.mean, 0.5))

        return _split((self.mean > 0.0).double())


class ProbabilityMatching(Filter):
    """
    Allocate the posterior probability that treatment wins: Thompson sampling
    in its two-arm closed form. With eta > 0 the posterior stops sharpening at
    the ceiling, so this keeps a permanent sliver on the loser instead of saturating:
    it never reaches a vertex and never commits.
    """

    def propose(self) -> Tensor:
        return _split(self.prob_positive)


class ZTest(Filter):
    """
    Explore at 50/50 while the null holds, commit to the detected side once it
    is rejected at `p_value`, two-sided. Retested every epoch at the nominal
    level, so the real type I error is far above it.
    """

    p_value: float = 0.05
    """Two-sided nominal level the null is tested at, every epoch."""

    @cached_property
    def threshold(self) -> float:
        """The |z| that rejects at `p_value`, two-sided."""
        return float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))

    def propose(self) -> Tensor:
        commit = (self.z > 0.0).double()

        return _split(torch.where(self.z.abs() < self.threshold, 0.5, commit))


class Pinn(Filter):
    """
    The trained drift HJB policy, mapped onto arena units with rate
    gamma = 1 - rho (exact is -log rho, O(gamma^2) apart).

    Its etahat is the *policy's* eta, so one checkpoint plays every column of the
    misspecification grid, which is what carrying etahat as a net input buys.
    """

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        sigma: float,
        eta: float,
        prior_std: float | None = None,
    ) -> None:
        super().__init__(params, reps, device, sigma, eta)
        self.prior_std = prior_std
        self.value: ValueFunction | None = None
        self.precision = torch.full_like(self.precision, self.prior_tau)

    @property
    def prior_tau(self) -> float:
        """The prior precision the filter starts at, in arena units."""
        if self.prior_std is not None:
            return 1.0 / self.prior_std**2

        return _FLATTEST_TAUHAT / ((1.0 - self.params.rho) * self.sigma**2)

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        policy = cls(params, reps, device, sigma=params.sigma, eta=params.eta)
        policy.value = ValueFunction(
            _champion(),
            rho=1.0 - params.rho,
            sigma=policy.sigma,
            eta=policy.eta,
        ).to(device)

        return policy

    def propose(self) -> Tensor:
        # The trust guard on the net *input* only: erosion can pull the eroded prior
        # below the flattest tauhat the checkpoint supports.
        floor = _FLATTEST_TAUHAT / ((1.0 - self.params.rho) * self.sigma**2)
        tau = self.precision.clamp_min(floor)

        return _split(self.value.policy(self.mean.float(), tau.float()).double())


def demo() -> None:
    """At eta = 0 this zoo is two_arm's, and with drift the filter stops sharpening."""
    import pinn.arena.two_arm_drift as problem

    static = Params(
        rho=0.999, horizon=500, sigma=1.0, effect=0.5, effect_std=0.0, size=1
    )
    winner = torch.tensor([[0.0, 0.5]])

    def play(cls: type[Policy], deltas: Tensor) -> Run:
        runner = Runner(static, [0])

        return runner.run(problem, cls.init(static, 1, "cpu"), deltas).runs()[0]

    # eta = 0 everywhere: the filter is two_arm's accumulator, and every
    # policy behaves as it does there.
    matching = play(ProbabilityMatching, winner)
    assert matching.final_allocation[1] > 0.95, matching.final_allocation

    null = play(ProbabilityMatching, torch.zeros(1, 2))
    assert abs(null.regret) < 1e-9, null.regret

    etc = play(ExploreThenCommit, winner)
    assert etc.committed == 1, etc.committed
    assert etc.committed_at == ExploreThenCommit.init(static, 1, "cpu").deadline
    assert abs(etc.precision_time - etc.committed_at) < 1e-6, etc.precision_time

    # Drift: the truth moves, so the effect at the end is not the effect at
    # the start, and the oracle moves with it.
    drifting = Params(
        rho=0.999,
        horizon=500,
        sigma=1.0,
        effect=0.5,
        effect_std=0.0,
        size=1,
        eta=0.05,
    )
    runner = Runner(drifting, [0])
    walked = problem.draw_effect(runner)
    for _ in range(500):
        walked = problem.advance(runner, walked)

    assert float(walked[0, 0]) == 0.0, walked
    assert abs(float(walked[0, 1]) - 0.5) > 1e-3, walked

    # A drift-aware filter stops sharpening: its precision saturates at the
    # ceiling instead of growing without bound, which is the whole difference.
    aware = ProbabilityMatching.init(drifting, 1, "cpu")
    blind = ProbabilityMatching(drifting, 1, "cpu", sigma=1.0, eta=0.0)

    evidence = (
        torch.full((1,), 0.5, dtype=torch.float64),
        torch.full((1,), 0.25, dtype=torch.float64),
    )
    for _ in range(400):
        aware.observe(evidence)
        blind.observe(evidence)

    aware_precision = float(aware.precision[0])
    blind_precision = float(blind.precision[0])
    assert blind_precision > 5.0 * aware_precision, (blind_precision, aware_precision)

    # The recursion's own fixed point: tau = tau/(1 + eta^2 tau) + p solves to
    # p/2 + sqrt(p^2/4 + p/eta^2), which sits a half-lump p/2 *above* the continuous
    # ceiling sqrt(p)/eta, the discrete filter adding a whole epoch after the decay.
    gained, drift = 0.25, 0.05
    fixed_point = gained / 2.0 + sqrt(gained**2 / 4.0 + gained / drift**2)

    assert abs(aware_precision - fixed_point) < 1e-6, (aware_precision, fixed_point)
    assert fixed_point > 1.0 / (2.0 * 1.0 * drift)

    # The prior fold: at eta = 0 the Pinn filter carries exactly two_arm's conjugate
    # posterior, same shrunk mean and same precision. sigma = 1, where the two zoos'
    # observation conventions (design vs precision) coincide.
    from .two_arm import _FLATTEST_TAUHAT as TWO_ARM_FLATTEST, Pinn as TwoArmPinn

    assert TWO_ARM_FLATTEST == _FLATTEST_TAUHAT

    drift_pinn = Pinn(static, 1, "cpu", sigma=1.0, eta=0.0)
    conjugate = TwoArmPinn(static, 1, "cpu")

    for estimate in (0.8, -0.3, 1.4, 0.1):
        pair = (
            torch.full((1,), estimate, dtype=torch.float64),
            torch.full((1,), 0.25, dtype=torch.float64),
        )
        drift_pinn.observe(pair)
        conjugate.observe(pair)

    prior_tau = _FLATTEST_TAUHAT / (1.0 - static.rho)
    total_tau = prior_tau + float(conjugate.total_precision[0])

    assert abs(float(drift_pinn.precision[0]) - total_tau) < 1e-12
    assert (
        abs(float(drift_pinn.mean[0]) - float(conjugate.total[0]) / total_tau) < 1e-12
    )

    print(f"matching regret {matching.regret:.2f}, final a {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(f"drifted effect after 500 epochs: {float(walked[0, 1]):.3f} (started 0.500)")
    print(
        f"precision after 400 epochs: aware {aware_precision:.2f}, blind {blind_precision:.2f}"
    )


if __name__ == "__main__":
    demo()
