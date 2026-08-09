"""
The three-arm problem in the arena: effect draw, observation model in
information form, and the policy zoo (allocations are (a, b, c) simplex
vectors, arm a the control). State lives in contrast coordinates
(theta_b - theta_a, theta_c - theta_a): q the accumulated precision-weighted
evidence, T the 2x2 data precision as (t_bb, t_bc, t_cc).
"""

from __future__ import annotations

import torch

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.three_arm import DimensionlessValueFunction, ValueFunction
from ..utils.gaussian import _bivariate_ndtr
from .harness import Params, Policy, Runner, optimal_deadline

CHECKPOINT = Path("data") / "value_3a_64:64:64.pt"

# The weakest prior the three-arm champion supports, in the dimensionless
# units of its chart; the symmetric-point probes are clean from tauhat ~ 0.03
# down to 0.01 but the floor decade below is unprobed -- same guard as
# two_arm's, same revisit-when-the-anchor-lands.
_FLATTEST_TAUHAT = 1e-2


@cache
def _champion() -> DimensionlessValueFunction:
    return DimensionlessValueFunction.load(CHECKPOINT)


def draw_effect(runner: Runner) -> Tensor:
    """
    The environment's truth: (0, delta_b, delta_c), the challenger effects
    drawn iid from the study's distribution.
    """
    return torch.tensor(
        [
            0.0,
            runner.normal(runner.params.effect, runner.params.effect_std),
            runner.normal(runner.params.effect, runner.params.effect_std),
        ]
    )


def _rates(allocation: Tensor, sigma: float) -> tuple[float, float, float]:
    """
    The epoch's information matrix G / sigma^2 in contrast coordinates
    (doc: dT/dt = G, det G = a_a a_b a_c).
    """
    a_a, a_b, a_c = (float(a) for a in allocation)

    return (
        (a_a * a_b + a_b * a_c) / sigma**2,
        -(a_b * a_c) / sigma**2,
        (a_a * a_c + a_b * a_c) / sigma**2,
    )


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    The truth between epochs. Static here by definition of the problem; drift
    lives in two_arm_drift, so --eta has no effect on this zoo.
    """
    return deltas


def observe(
    runner: Runner, allocation: Tensor, deltas: Tensor
) -> tuple[Tensor, float, float, float]:
    """
    One epoch's evidence in information form: dq = G delta + noise with
    noise ~ N(0, G), plus the precision increment G itself -- the natural
    parameters, so no inversion of a possibly singular G is ever needed
    (an edge allocation is rank one, a vertex rank zero).
    """
    g_bb, g_bc, g_cc = _rates(allocation, runner.params.sigma)

    # A vertex buys nothing. Return early rather than drawing two variates and
    # multiplying them away: the runner no longer stops at a vertex, so those
    # draws would shift the shared rng stream the seed pairing relies on.
    if g_bb == 0.0 and g_cc == 0.0:
        return torch.zeros(2), 0.0, 0.0, 0.0

    # Closed-form 2x2 Cholesky of G, guarded for the rank-deficient edges.
    root_bb = g_bb**0.5
    root_bc = g_bc / root_bb if root_bb > 0.0 else 0.0
    root_cc = max(g_cc - root_bc**2, 0.0) ** 0.5

    noise_1 = runner.normal(0.0, 1.0)
    noise_2 = runner.normal(0.0, 1.0)
    dq = torch.tensor(
        [
            g_bb * float(deltas[1]) + g_bc * float(deltas[2]) + root_bb * noise_1,
            g_bc * float(deltas[1])
            + g_cc * float(deltas[2])
            + root_bc * noise_1
            + root_cc * noise_2,
        ]
    )

    return dq, g_bb, g_bc, g_cc


@dataclass(kw_only=True)
class Bayesian(Policy):
    """
    Flat-prior posterior on the contrasts, accumulated in natural parameters:
    posterior precision T (data only) and evidence q, mean = T^-1 q.
    """

    params: Params
    count: int = 0
    q: Tensor = field(default_factory=lambda: torch.zeros(2))
    t_bb: float = 0.0
    t_bc: float = 0.0
    t_cc: float = 0.0

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(params=params)

    def observe(self, observation: tuple[Tensor, float, float, float]) -> None:
        dq, g_bb, g_bc, g_cc = observation

        self.count += 1
        self.q = self.q + dq
        self.t_bb += g_bb
        self.t_bc += g_bc
        self.t_cc += g_cc

    @property
    def determinant(self) -> float:
        return self.t_bb * self.t_cc - self.t_bc**2

    def mean(self) -> tuple[float, float]:
        """
        Flat-prior posterior mean of the contrasts; (0, 0) before the data
        precision is invertible.
        """
        if self.determinant <= 1e-12:
            return 0.0, 0.0

        q_b, q_c = float(self.q[0]), float(self.q[1])

        return (
            (self.t_cc * q_b - self.t_bc * q_c) / self.determinant,
            (-self.t_bc * q_b + self.t_bb * q_c) / self.determinant,
        )


def _one_hot(arm: int) -> Tensor:
    allocation = torch.zeros(3)
    allocation[arm] = 1.0

    return allocation


def _commit_arm(m_b: float, m_c: float) -> int:
    levels = (0.0, m_b, m_c)

    return max(range(3), key=lambda arm: levels[arm])


THIRDS = torch.full((3,), 1.0 / 3.0)


@dataclass
class ExploreThenCommit(Bayesian):
    """
    Explore at thirds (the most informative allocation), then commit to the
    posterior leader at a fixed deadline, however uncertain it is by then.
    """

    deadline: int = 50

    @classmethod
    def init(cls, params: Params) -> Self:
        return cls(params=params, deadline=optimal_deadline(params.rho, params.horizon))

    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return THIRDS.clone()

        return _one_hot(_commit_arm(*self.mean()))


@dataclass
class ProbabilityMatching(Bayesian):
    """
    Allocate each arm the posterior probability it is best: three-arm
    Thompson sampling, exact via the bivariate normal cdf (each arm's
    win probability is one orthant of a transformed contrast pair). Commits
    only when a probability saturates to exactly 1 in float64.
    """

    def propose(self) -> Tensor:
        if self.determinant <= 1e-12:
            return THIRDS.clone()

        m_b, m_c = self.mean()
        s_bb = self.t_cc / self.determinant
        s_bc = -self.t_bc / self.determinant
        s_cc = self.t_bb / self.determinant

        def orthant(
            mean_x: float, var_x: float, mean_y: float, var_y: float, cov: float
        ) -> float:
            correlation = cov / (var_x * var_y) ** 0.5

            return float(
                _bivariate_ndtr(
                    torch.tensor(mean_x / var_x**0.5),
                    torch.tensor(mean_y / var_y**0.5),
                    torch.tensor(max(min(correlation, 0.999), -0.999)),
                )
            )

        # P(arm best) = P(both its contrasts positive), each an orthant of a
        # Gaussian pair: b beats a is (m_b, s_bb); b beats c is
        # (m_b - m_c, s_bb + s_cc - 2 s_bc), covariance s_bb - s_bc.
        gap = m_b - m_c
        var_gap = s_bb + s_cc - 2.0 * s_bc
        win_b = orthant(m_b, s_bb, gap, var_gap, s_bb - s_bc)
        win_c = orthant(m_c, s_cc, -gap, var_gap, s_cc - s_bc)
        win_a = orthant(-m_b, s_bb, -m_c, s_cc, s_bc)

        allocation = torch.tensor([win_a, win_b, win_c])

        return allocation / allocation.sum()


@dataclass
class Elimination(Bayesian):
    """
    Successive elimination, the z-test's three-arm analog (ported from the
    posterior-space benchmark): split evenly among surviving arms, drop any
    arm pairwise-significantly worse at `p_value`, commit when one survives.
    Stateless in the eliminations -- every epoch retests from the accumulated
    data, so the peeking caveat of the two-arm ZTest applies squared.
    """

    p_value: float = 0.05

    def propose(self) -> Tensor:
        if self.determinant <= 1e-12 or self.t_bb <= 1e-9 or self.t_cc <= 1e-9:
            return THIRDS.clone()

        threshold = float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))
        m_b, m_c = self.mean()
        s_bb = self.t_cc / self.determinant
        s_bc = -self.t_bc / self.determinant
        s_cc = self.t_bb / self.determinant

        z_b = m_b / s_bb**0.5
        z_c = m_c / s_cc**0.5
        z_bc = (m_b - m_c) / max(s_bb + s_cc - 2.0 * s_bc, 1e-12) ** 0.5

        out_a = z_b > threshold or z_c > threshold
        out_b = z_b < -threshold or z_bc < -threshold
        out_c = z_c < -threshold or z_bc > threshold
        survivors = torch.tensor([not out_a, not out_b, not out_c]).float()

        # Cyclic eliminations can in principle empty the set; fall back to
        # the posterior leader.
        if float(survivors.sum()) < 0.5:
            return _one_hot(_commit_arm(m_b, m_c))

        return survivors / survivors.sum()


@dataclass
class Pinn(Bayesian):
    """
    The trained three_arm HJB policy, mapped onto arena units with rate
    gamma = 1 - rho; commits are exact vertices of the simplex max.

    The prior is a POLICY PARAMETER: prior_std is the prior standard
    deviation on each contrast, in arena units, realized as the arm-symmetric
    prior (off-diagonal -1/2, today's "correlated" convention). None means
    the flattest prior the checkpoint supports, computed per environment.
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
            prior = _FLATTEST_TAUHAT / (gamma * self.params.sigma**2)
        else:
            # The arm-symmetric prior with marginal contrast variance
            # prior_std^2: inverting [[p, -p/2], [-p/2, p]] gives marginal
            # variance 4 / (3 p).
            prior = 4.0 / (3.0 * self.prior_std**2)

        tau_bb = prior + self.t_bb
        tau_bc = -prior / 2.0 + self.t_bc
        tau_cc = prior + self.t_cc
        det = tau_bb * tau_cc - tau_bc**2
        q_b, q_c = float(self.q[0]), float(self.q[1])
        m_b = (tau_cc * q_b - tau_bc * q_c) / det
        m_c = (-tau_bc * q_b + tau_bb * q_c) / det

        allocation = self.value.policy(
            torch.tensor([m_b]),
            torch.tensor([m_c]),
            torch.tensor([tau_bb]),
            torch.tensor([tau_bc]),
            torch.tensor([tau_cc]),
        )

        return allocation.squeeze(0)


def demo() -> None:
    import pinn.arena.three_arm as problem

    runner = Runner(
        Params(rho=0.999, horizon=500, sigma=1.0, effect=0.0, effect_std=0.3, size=1)
    )
    b_wins = torch.tensor([0.0, 0.6, -0.2])
    null = torch.zeros(3)

    # A clear winner: matching concentrates on it; the null pays no regret.
    matching = runner.run(problem, ProbabilityMatching.init(runner.params), b_wins)
    assert matching.final_allocation[1] > 0.9, matching.final_allocation

    nothing = runner.run(problem, ProbabilityMatching.init(runner.params), null)
    assert abs(nothing.regret) < 1e-9, nothing.regret

    # The committers commit to the winner.
    etc = runner.run(problem, ExploreThenCommit.init(runner.params), b_wins)
    assert etc.committed == 1, etc.committed

    # Soft commit time reduces to the deadline: thirds is the uniform
    # allocation (share 1, up to float32 thirds), and the committing epoch
    # buys nothing.
    assert abs(etc.precision_time - etc.committed_at) < 1e-3, etc.precision_time

    elimination = runner.run(problem, Elimination.init(runner.params), b_wins)
    assert elimination.committed == 1, elimination.committed

    # The PINN: opens near thirds on no evidence, commits to the winner.
    pinn_policy = Pinn.init(runner.params)
    assert float(pinn_policy.propose().min()) > 0.2, pinn_policy.propose()

    pinn = runner.run(problem, pinn_policy, b_wins)
    assert pinn.committed == 1, (pinn.committed, pinn.final_allocation)

    print(f"matching regret {matching.regret:.2f}, final {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(
        f"elimination regret {elimination.regret:.2f}, committed at epoch {elimination.committed_at}"
    )
    print(f"pinn regret {pinn.regret:.2f}, committed at epoch {pinn.committed_at}")


if __name__ == "__main__":
    demo()
