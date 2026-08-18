"""
The three-arm problem with drifting effects in the arena: the drift, the filter that
forecasts through it, and the policy zoo.

**Assembled**, not written: the effect draw, the observation model and every policy's
`propose` are three_arm's unchanged, because drift changes neither what an epoch buys
nor how a policy reads a posterior. Only `advance`, which walks the *arms*, and
`Filter`, which forecasts before it updates, are new, both the two_arm_drift move
carried to a 2x2 posterior. At eta = 0 every number here is three_arm's, which the demo
asserts. kb/arena_results.md has the erosion matrix and the recursion's form.
"""

from __future__ import annotations

import torch

from functools import cache
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.three_arm_drift.model import DimensionlessValueFunction, ValueFunction
from . import three_arm as static
from .harness import Params, Policy, Runner
from .three_arm import draw_effect, observe

CHECKPOINT = Path("data") / "three_arm_drift.pt"
"""The champion the Pinn policy plays, repo-root relative like every other data path."""

DRAWS_PER_EPOCH = 5
"""
Two observation draws plus one walk per arm. Declared for the harness, whose default of
2 is what keeps the static zoos' recorded numbers reproducible.
"""

_FLATTEST_TAUHAT = 1e-3
"""
The weakest prior the champion is trusted at, in dimensionless precision. It is
three_arm's value, the sampler's own PRIOR_FLOOR, so the guard does not clamp inside
the training support.
"""


@cache
def _champion() -> DimensionlessValueFunction:
    """One load per process; read-only sharing, since nothing here trains."""
    return DimensionlessValueFunction.load(CHECKPOINT)


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    One epoch of drift. Each *arm's* truth takes an independent step at volatility eta,
    and the contrasts inherit the control's step with a minus sign, which correlates
    them. Unconditional, so nothing branches on drift, at three variates per epoch
    either way: this zoo's stream is not comparable run-for-run against three_arm's.
    """
    walk_a = runner.normal(0.0, runner.params.eta)
    walk_b = runner.normal(0.0, runner.params.eta)
    walk_c = runner.normal(0.0, runner.params.eta)

    return torch.stack(
        (
            torch.zeros_like(walk_a),
            deltas[:, 1].double() + walk_b - walk_a,
            deltas[:, 2].double() + walk_c - walk_a,
        ),
        dim=1,
    ).float()


class Filter(static.Bayesian):
    """
    three_arm's posterior with erosion: carried as (mean, precision) rather than
    (evidence, precision), because forecasting needs the covariance and a flat prior
    cannot be inverted.

    `t_bb`, `t_bc`, `t_cc` and `mean()` keep three_arm's meaning, which is what lets
    every policy there be reused verbatim. `eta` is a *policy* parameter, as in
    two_arm_drift: tied to the environment's by `init`, untied by direct construction.
    """

    def __init__(
        self, params: Params, reps: int, device: str, eta: float | None = None
    ) -> None:
        super().__init__(params, reps, device)
        self.eta = params.eta if eta is None else eta
        self.m_b = torch.zeros(reps, dtype=torch.float64, device=device)
        self.m_c = torch.zeros(reps, dtype=torch.float64, device=device)

    def mean(self) -> tuple[Tensor, Tensor]:
        """The posterior mean of the contrasts, carried rather than solved for."""
        return self.m_b, self.m_c

    def _erode(self) -> None:
        """
        T <- T (I + E T)^-1 with E = eta^2 (I + 11'), the exact one-epoch
        forecast. The mean is a martingale under the walk, so it does not move.
        """
        if self.eta == 0.0:
            return

        step = self.eta**2
        a_bb = 2.0 * step * self.t_bb + step * self.t_bc
        a_bc = 2.0 * step * self.t_bc + step * self.t_cc
        a_cb = step * self.t_bb + 2.0 * step * self.t_bc
        a_cc = step * self.t_bc + 2.0 * step * self.t_cc

        det = (1.0 + a_bb) * (1.0 + a_cc) - a_bc * a_cb
        t_bb = (self.t_bb * (1.0 + a_cc) - self.t_bc * a_cb) / det
        t_bc = (-self.t_bb * a_bc + self.t_bc * (1.0 + a_bb)) / det
        t_cb = (self.t_bc * (1.0 + a_cc) - self.t_cc * a_cb) / det
        t_cc = (-self.t_bc * a_bc + self.t_cc * (1.0 + a_bb)) / det

        self.t_bb, self.t_cc = t_bb, t_cc
        # Symmetric in exact arithmetic; averaged so float64 rounding cannot
        # tilt the matrix and make the policies read an asymmetric posterior.
        self.t_bc = 0.5 * (t_bc + t_cb)

    def observe(self, observation: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
        dq, g_bb, g_bc, g_cc = observation

        self.count += 1
        self._erode()

        # Information-form update on the *forecast* posterior: T' = T + G and
        # T' m' = T m + dq.
        q_b = self.t_bb * self.m_b + self.t_bc * self.m_c + dq[:, 0].double()
        q_c = self.t_bc * self.m_b + self.t_cc * self.m_c + dq[:, 1].double()
        self.t_bb = self.t_bb + g_bb
        self.t_bc = self.t_bc + g_bc
        self.t_cc = self.t_cc + g_cc

        det = self.determinant
        live = det > 1e-12
        safe = det.masked_fill(~live, 1.0)
        self.m_b = torch.where(live, (self.t_cc * q_b - self.t_bc * q_c) / safe, 0.0)
        self.m_c = torch.where(live, (-self.t_bc * q_b + self.t_bb * q_c) / safe, 0.0)


class ExploreThenCommit(static.ExploreThenCommit, Filter):
    """three_arm's, reading the eroding posterior."""


class ProbabilityMatching(static.ProbabilityMatching, Filter):
    """three_arm's, reading the eroding posterior."""


class Elimination(static.Elimination, Filter):
    """three_arm's, reading the eroding posterior."""


class Pinn(Filter):
    """
    The trained three_arm_drift HJB policy, mapped onto arena units with rate
    gamma = 1 - rho. Its etahat is the *policy's* eta, so one checkpoint plays every
    column of the misspecification grid.
    """

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        eta: float | None = None,
        prior_std: float | None = None,
    ) -> None:
        super().__init__(params, reps, device, eta)
        self.prior_std = prior_std
        self.value: ValueFunction | None = None

        prior = self.prior_precision
        self.t_bb = torch.full_like(self.t_bb, prior)
        self.t_bc = torch.full_like(self.t_bc, -prior / 2.0)
        self.t_cc = torch.full_like(self.t_cc, prior)

    @property
    def prior_precision(self) -> float:
        """The arm-symmetric prior (off-diagonal -1/2), three_arm's convention."""
        if self.prior_std is not None:
            return 4.0 / (3.0 * self.prior_std**2)

        return _FLATTEST_TAUHAT / ((1.0 - self.params.rho) * self.params.sigma**2)

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        policy = cls(params, reps, device)
        policy.value = ValueFunction(
            _champion(),
            rho=1.0 - params.rho,
            sigma=params.sigma,
            eta=policy.eta,
        ).to(device)

        return policy

    def propose(self) -> Tensor:
        return self.value.policy(
            self.m_b.float(),
            self.m_c.float(),
            self.t_bb.float(),
            self.t_bc.float(),
            self.t_cc.float(),
        )


def demo() -> None:
    """At eta = 0 the filter is three_arm's, and erosion is the matrix it claims."""
    import pinn.arena.three_arm_drift as problem

    reps = 64
    seeds = list(range(reps))

    # eta = 0 is three_arm **exactly**, checked on the *filter* rather than on a run,
    # since this zoo's noise stream is deliberately not aligned with three_arm's. Drive
    # both posteriors with the same observations instead.
    static_params = Params(
        rho=0.999, horizon=60, sigma=1.0, effect=0.0, effect_std=0.4, size=reps, eta=0.0
    )
    # ProbabilityMatching purely to have a concrete class: the check is on
    # `observe`, which Filter supplies and the policy never touches.
    eroding = ProbabilityMatching(static_params, reps, "cpu")
    plain = static.ProbabilityMatching(static_params, reps, "cpu")
    generator = torch.Generator().manual_seed(11)

    for _ in range(20):
        allocation = torch.rand(reps, 3, generator=generator)
        allocation = allocation / allocation.sum(dim=1, keepdim=True)
        g_bb, g_bc, g_cc = static._rates(allocation, static_params.sigma)
        dq = torch.randn(reps, 2, generator=generator)
        observation = (dq, g_bb, g_bc, g_cc)
        eroding.observe(observation)
        plain.observe(observation)

    # rtol 1e-4, not tighter: three_arm accumulates q in *float32* while this filter
    # carries the mean in float64, and that gap lands at ~1e-7 relative (this side
    # being the more precise one).
    for got, want in zip(eroding.mean(), plain.mean()):
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-9), (got[:3], want[:3])
    assert torch.allclose(eroding.t_bb, plain.t_bb, rtol=1e-12)
    assert torch.allclose(eroding.t_bc, plain.t_bc, rtol=1e-12)

    # Erosion is a one-way ratchet on information: with eta > 0 and the *same*
    # allocations, the posterior is strictly less precise than the static one.
    drift_params = Params(
        rho=0.999, horizon=60, sigma=1.0, effect=0.0, effect_std=0.4, size=reps, eta=0.2
    )
    runner = Runner(drift_params, seeds, draws_per_epoch=DRAWS_PER_EPOCH)
    eroded = ExploreThenCommit.init(drift_params, reps, "cpu")
    runner.run(problem, eroded, problem.draw_effect(runner))

    runner = Runner(static_params, seeds, draws_per_epoch=DRAWS_PER_EPOCH)
    kept = ExploreThenCommit.init(static_params, reps, "cpu")
    runner.run(problem, kept, problem.draw_effect(runner))

    assert (eroded.t_bb < kept.t_bb).all(), "erosion must cost precision"
    assert (eroded.determinant < kept.determinant).all()

    # The forecast is exactly the matrix recursion it claims to be, checked
    # against an explicit inverse on a posterior with a real off-diagonal.
    probe = ProbabilityMatching(drift_params, 4, "cpu", eta=0.3)
    probe.t_bb = torch.tensor([2.0, 5.0, 1.0, 9.0], dtype=torch.float64)
    probe.t_bc = torch.tensor([-0.5, -1.0, -0.25, -3.0], dtype=torch.float64)
    probe.t_cc = torch.tensor([3.0, 4.0, 2.0, 7.0], dtype=torch.float64)
    before = [probe.t_bb.clone(), probe.t_bc.clone(), probe.t_cc.clone()]
    probe._erode()

    erosion = 0.3**2 * torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)

    for i in range(4):
        precision = torch.tensor(
            [[before[0][i], before[1][i]], [before[1][i], before[2][i]]],
            dtype=torch.float64,
        )
        want = torch.linalg.inv(torch.linalg.inv(precision) + erosion)

        assert abs(probe.t_bb[i] - want[0, 0]) < 1e-9, (i, probe.t_bb[i], want[0, 0])
        assert abs(probe.t_bc[i] - want[0, 1]) < 1e-9, (i, probe.t_bc[i], want[0, 1])
        assert abs(probe.t_cc[i] - want[1, 1]) < 1e-9, (i, probe.t_cc[i], want[1, 1])

    # A flat start survives the forecast: T = 0 stays 0 rather than dividing.
    flat = ProbabilityMatching(drift_params, 4, "cpu", eta=0.5)
    flat._erode()

    assert flat.t_bb.abs().max() == 0.0 and flat.determinant.abs().max() == 0.0

    # The drift really moves the truth, and the contrasts pick up the
    # control's walk: variance 2 eta^2 each, covariance eta^2.
    big = Params(
        rho=0.999, horizon=1, sigma=1.0, effect=0.0, effect_std=0.0, size=20000, eta=0.5
    )
    runner = Runner(big, list(range(20000)), draws_per_epoch=DRAWS_PER_EPOCH)
    walked = advance(runner, draw_effect(runner))
    b, c = walked[:, 1].double(), walked[:, 2].double()

    assert abs(b.var().item() - 2.0 * 0.25) < 0.03, b.var().item()
    assert abs(c.var().item() - 2.0 * 0.25) < 0.03, c.var().item()
    assert abs(((b * c).mean() - b.mean() * c.mean()).item() - 0.25) < 0.03

    print("ok")


if __name__ == "__main__":
    demo()
