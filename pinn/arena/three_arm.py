"""
The three-arm problem in the arena: effect draw, observation model in
information form, and the policy zoo (allocations are (a, b, c) simplex
vectors, arm a the control). State lives in contrast coordinates
(theta_b - theta_a, theta_c - theta_a): q the accumulated precision-weighted
evidence, T the 2x2 data precision as (t_bb, t_bc, t_cc).
"""

from __future__ import annotations

import torch

from functools import cache
from pathlib import Path
from torch import Tensor
from typing import Self

from ..problems.three_arm import DimensionlessValueFunction, ValueFunction
from ..utils.gaussian import _bivariate_ndtr
from .harness import Params, Policy, Runner, optimal_deadline

CHECKPOINT = Path("data") / "three_arm.pt"

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
    The environment's truth: (0, delta_b, delta_c), one row per rep, the
    challenger effects drawn iid from the study's distribution.
    """
    delta_b = runner.normal(runner.params.effect, runner.params.effect_std)
    delta_c = runner.normal(runner.params.effect, runner.params.effect_std)

    return torch.stack((torch.zeros_like(delta_b), delta_b, delta_c), dim=1).float()


def advance(runner: Runner, deltas: Tensor) -> Tensor:
    """
    The truth between epochs. Static here by definition of the problem; drift
    lives in two_arm_drift, so --eta has no effect on this zoo.
    """
    return deltas


def _rates(allocation: Tensor, sigma: float) -> tuple[Tensor, Tensor, Tensor]:
    """
    The epoch's information matrix G / sigma^2 in contrast coordinates
    (doc: dT/dt = G, det G = a_a a_b a_c).
    """
    a_a, a_b, a_c = (allocation[:, i].double() for i in range(3))

    return (
        (a_a * a_b + a_b * a_c) / sigma**2,
        -(a_b * a_c) / sigma**2,
        (a_a * a_c + a_b * a_c) / sigma**2,
    )


def observe(
    runner: Runner, allocation: Tensor, deltas: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    One epoch's evidence in information form: dq = G delta + noise with
    noise ~ N(0, G), plus the precision increment G itself -- the natural
    parameters, so no inversion of a possibly singular G is ever needed
    (an edge allocation is rank one, a vertex rank zero).

    A vertex rep buys nothing: it consumes no draws (the mask skips its
    cursor twice, keeping the stream aligned across batches) and reads zeros.
    """
    g_bb, g_bc, g_cc = _rates(allocation, runner.params.sigma)
    live = ~((g_bb == 0.0) & (g_cc == 0.0))

    # Closed-form 2x2 Cholesky of G, guarded for the rank-deficient edges.
    root_bb = g_bb.sqrt()
    root_bc = torch.where(
        root_bb > 0.0, g_bc / root_bb.masked_fill(root_bb == 0.0, 1.0), 0.0
    )
    root_cc = (g_cc - root_bc**2).clamp_min(0.0).sqrt()

    noise_1 = runner.normal(0.0, 1.0, live)
    noise_2 = runner.normal(0.0, 1.0, live)
    delta_b, delta_c = deltas[:, 1].double(), deltas[:, 2].double()
    dq = torch.stack(
        (
            g_bb * delta_b + g_bc * delta_c + root_bb * noise_1,
            g_bc * delta_b + g_cc * delta_c + root_bc * noise_1 + root_cc * noise_2,
        ),
        dim=1,
    ).float()

    return dq, g_bb, g_bc, g_cc


class Bayesian(Policy):
    """
    Flat-prior posterior on the contrasts, accumulated in natural parameters:
    posterior precision T (data only, (reps,) float64 entries) and evidence
    q ((reps, 2) float32), mean = T^-1 q.
    """

    def __init__(self, params: Params, reps: int, device: str) -> None:
        self.params = params
        self.count = 0
        self.q = torch.zeros(reps, 2, device=device)
        self.t_bb = torch.zeros(reps, dtype=torch.float64, device=device)
        self.t_bc = torch.zeros(reps, dtype=torch.float64, device=device)
        self.t_cc = torch.zeros(reps, dtype=torch.float64, device=device)

    @classmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(params, reps, device)

    def observe(self, observation: tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
        dq, g_bb, g_bc, g_cc = observation

        self.count += 1
        self.q = self.q + dq
        self.t_bb = self.t_bb + g_bb
        self.t_bc = self.t_bc + g_bc
        self.t_cc = self.t_cc + g_cc

    @property
    def determinant(self) -> Tensor:
        return self.t_bb * self.t_cc - self.t_bc**2

    def mean(self) -> tuple[Tensor, Tensor]:
        """
        Flat-prior posterior mean of the contrasts; (0, 0) before the data
        precision is invertible.
        """
        det = self.determinant
        live = det > 1e-12
        safe = det.masked_fill(~live, 1.0)
        q_b, q_c = self.q[:, 0].double(), self.q[:, 1].double()

        return (
            torch.where(live, (self.t_cc * q_b - self.t_bc * q_c) / safe, 0.0),
            torch.where(live, (-self.t_bc * q_b + self.t_bb * q_c) / safe, 0.0),
        )


def _one_hot(arm: Tensor) -> Tensor:
    return torch.eye(3, device=arm.device)[arm]


def _commit_arm(m_b: Tensor, m_c: Tensor) -> Tensor:
    """
    First-maximal arm of (0, m_b, m_c): strictly-greater wins, ties keep the
    earlier arm.
    """
    arm = torch.zeros_like(m_b, dtype=torch.long)
    arm = torch.where(m_b > 0.0, 1, arm)

    return torch.where(m_c > torch.maximum(m_b, torch.zeros_like(m_b)), 2, arm)


def _thirds(reps: int, device: torch.device) -> Tensor:
    return torch.full((reps, 3), 1.0 / 3.0, device=device)


class ExploreThenCommit(Bayesian):
    """
    Explore at thirds (the most informative allocation), then commit to the
    posterior leader at a fixed deadline, however uncertain it is by then.
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
            return _thirds(len(self.t_bb), self.t_bb.device)

        return _one_hot(_commit_arm(*self.mean()))


class ProbabilityMatching(Bayesian):
    """
    Allocate each arm the posterior probability it is best: three-arm
    Thompson sampling, exact via the bivariate normal cdf (each arm's
    win probability is one orthant of a transformed contrast pair). Commits
    only when a probability saturates to exactly 1 in float64.
    """

    def propose(self) -> Tensor:
        det = self.determinant
        live = det > 1e-12
        safe = det.masked_fill(~live, 1.0)
        m_b, m_c = self.mean()
        # Dead rows get the harmless (mean 0, variance 1, covariance 0)
        # stats; their orthants are computed anyway and discarded below.
        s_bb = torch.where(live, self.t_cc / safe, 1.0)
        s_bc = torch.where(live, -self.t_bc / safe, 0.0)
        s_cc = torch.where(live, self.t_bb / safe, 1.0)

        def orthant(
            mean_x: Tensor, var_x: Tensor, mean_y: Tensor, var_y: Tensor, cov: Tensor
        ) -> Tensor:
            correlation = cov / (var_x * var_y) ** 0.5

            return _bivariate_ndtr(
                (mean_x / var_x**0.5).float(),
                (mean_y / var_y**0.5).float(),
                correlation.clamp(-0.999, 0.999).float(),
            )

        # P(arm best) = P(both its contrasts positive), each an orthant of a
        # Gaussian pair: b beats a is (m_b, s_bb); b beats c is
        # (m_b - m_c, s_bb + s_cc - 2 s_bc), covariance s_bb - s_bc.
        gap = m_b - m_c
        var_gap = s_bb + s_cc - 2.0 * s_bc
        win_b = orthant(m_b, s_bb, gap, var_gap, s_bb - s_bc)
        win_c = orthant(m_c, s_cc, -gap, var_gap, s_cc - s_bc)
        win_a = orthant(-m_b, s_bb, -m_c, s_cc, s_bc)

        wins = torch.stack((win_a, win_b, win_c), dim=1)

        return torch.where(
            live.unsqueeze(1),
            wins / wins.sum(dim=1, keepdim=True),
            _thirds(len(det), det.device),
        )


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
        det = self.determinant
        live = (det > 1e-12) & (self.t_bb > 1e-9) & (self.t_cc > 1e-9)
        safe = det.masked_fill(~live, 1.0)

        threshold = float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))
        m_b, m_c = self.mean()
        s_bb = torch.where(live, self.t_cc / safe, 1.0)
        s_bc = torch.where(live, -self.t_bc / safe, 0.0)
        s_cc = torch.where(live, self.t_bb / safe, 1.0)

        z_b = m_b / s_bb**0.5
        z_c = m_c / s_cc**0.5
        z_bc = (m_b - m_c) / (s_bb + s_cc - 2.0 * s_bc).clamp_min(1e-12) ** 0.5

        out_a = (z_b > threshold) | (z_c > threshold)
        out_b = (z_b < -threshold) | (z_bc < -threshold)
        out_c = (z_c < -threshold) | (z_bc > threshold)
        survivors = (~torch.stack((out_a, out_b, out_c), dim=1)).float()

        # Cyclic eliminations can in principle empty the set; fall back to
        # the posterior leader.
        count = survivors.sum(dim=1)
        empty = count < 0.5
        allocation = torch.where(
            empty.unsqueeze(1),
            _one_hot(_commit_arm(m_b, m_c)),
            survivors / count.masked_fill(empty, 1.0).unsqueeze(1),
        )

        return torch.where(live.unsqueeze(1), allocation, _thirds(len(det), det.device))


class Pinn(Bayesian):
    """
    The trained three_arm HJB policy, mapped onto arena units with rate
    gamma = 1 - rho; commits are exact vertices of the simplex max.

    The prior is a POLICY PARAMETER: prior_std is the prior standard
    deviation on each contrast, in arena units, realized as the arm-symmetric
    prior (off-diagonal -1/2, today's "correlated" convention). None means
    the flattest prior the checkpoint supports, computed per environment.
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
        q_b, q_c = self.q[:, 0].double(), self.q[:, 1].double()
        m_b = (tau_cc * q_b - tau_bc * q_c) / det
        m_c = (-tau_bc * q_b + tau_bb * q_c) / det

        return self.value.policy(
            m_b.float(), m_c.float(), tau_bb.float(), tau_bc.float(), tau_cc.float()
        )


def demo() -> None:
    import pinn.arena.three_arm as problem

    from .harness import Run

    params = Params(
        rho=0.999, horizon=500, sigma=1.0, effect=0.0, effect_std=0.3, size=1
    )
    b_wins = torch.tensor([[0.0, 0.6, -0.2]])

    def play(cls: type[Policy], deltas: Tensor) -> Run:
        runner = Runner(params, [0])

        return runner.run(problem, cls.init(params, 1, "cpu"), deltas).runs()[0]

    # A clear winner: matching concentrates on it; the null pays no regret.
    matching = play(ProbabilityMatching, b_wins)
    assert matching.final_allocation[1] > 0.9, matching.final_allocation

    nothing = play(ProbabilityMatching, torch.zeros(1, 3))
    assert abs(nothing.regret) < 1e-9, nothing.regret

    # The committers commit to the winner.
    etc = play(ExploreThenCommit, b_wins)
    assert etc.committed == 1, etc.committed

    # Soft commit time reduces to the deadline: thirds is the uniform
    # allocation (share 1, up to float32 thirds), and the committing epoch
    # buys nothing.
    assert abs(etc.precision_time - etc.committed_at) < 1e-3, etc.precision_time

    elimination = play(Elimination, b_wins)
    assert elimination.committed == 1, elimination.committed

    # The PINN: opens near thirds on no evidence, commits to the winner.
    pinn_policy = Pinn.init(params, 1, "cpu")
    assert float(pinn_policy.propose().min()) > 0.2, pinn_policy.propose()

    pinn = play(Pinn, b_wins)
    assert pinn.committed == 1, (pinn.committed, pinn.final_allocation)

    print(f"matching regret {matching.regret:.2f}, final {matching.final_allocation}")
    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(
        f"elimination regret {elimination.regret:.2f}, committed at epoch {elimination.committed_at}"
    )
    print(f"pinn regret {pinn.regret:.2f}, committed at epoch {pinn.committed_at}")


if __name__ == "__main__":
    demo()
