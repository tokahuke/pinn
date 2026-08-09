"""
Losses for the drift problem: the interior HJB residual, the ridge condition
BC1, and the trainer-facing objective. Mirrors two_arm/loss.py.
"""

from __future__ import annotations

import torch

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import sample_ridge, sample_sobol

RIDGE_WEIGHT = 10.0
POWER = 2.0

# Weight on the learning operator's negative part. L_ab >= 0 is provable for
# the answer and INVISIBLE to the residual: it enters the equation only through
# alpha(1-alpha) <= 1/4, while the policy is its argmax and moves like 1/L^2.
# So the residual can be excellent while the policy is ill-posed -- measured
# 2026-08-08, L_ab < 0 on 39.6% of the cloud at pde 1.5e-6, rising to 60% at
# etahat > 10, with a re-entrant commit region to match.
#
# Zero at the answer, so it cannot move the fixed point, only the path. The
# term is LINEAR in the violation (see pde_loss); at the pre-positivity
# checkpoint its mean is ~2e-3 against pde 1.3e-6, so 6e-4 would make the two
# comparable. Deliberately two orders above that: the point of the experiment
# is to see whether the SIGN pattern moves at all, and a term merely comparable
# to pde was not going to shift a 38% violating fraction. At this weight
# positivity is ~200x the pde term and is driving; pde will give ground.
POSITIVITY_WEIGHT = 6.0e-2


def pde_loss(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor, etahat: Tensor
) -> Tensor:
    """
    Returns TWO numbers: the squared residual of the HJB in similarity coordinates
    (docs/two_arm_drift.md section 6), maximization kept explicit:

        e^s (z + g) + etahat^2 e^(2s) tauhat_slope
            = max over alpha of { alpha e^s z + alpha(1-alpha) L_ab[g] }

    Graded in this chart for two_arm's reason: the 1/(2 tauhat^2) that
    multiplied curvature is cancelled algebraically. The leaves are (z, s) and
    autograd does the chain rule; __main__ checks it against the raw form.

    KNOWN DEGENERATE on its own: g = 0 (the never-explore solution) zeroes this
    residual exactly, at every etahat -- the drift extension does NOT break the
    degeneracy. The ridge loss still breaks it.

    The second number is the plain mean of relu(-L_ab), the learning operator's
    negative part. LINEAR, not squared, and not p-meaned: the policy reads the
    SIGN of L_ab, and a squared term chases depth instead. Measured 2026-08-08,
    the squared version cut its own magnitude 6.3x while the violating FRACTION
    rose 38.1% -> 40.2% and the worst policy drop doubled -- it dragged deep
    violations toward zero and pushed marginal points across it. A linear
    penalty has the same gradient at every depth, so it has no reason to prefer
    the deep ones.

    Note g = 0 zeroes this too, so it neither helps nor hurts the degeneracy --
    unlike the rejected complementarity term, it does not pay to spread
    deadness. Watch the dead fraction anyway: flattening curvature is the cheap
    way to buy positivity.
    """
    lhs, best, l_ab = value.hamiltonian(muhat, tauhat, etahat)

    return _graded(lhs - best.value), torch.relu(-l_ab).mean()


def _graded(residual: Tensor) -> Tensor:
    """
    The problem's own grading: no relative scale, power-mean attention
    (two_arm/loss.py explains both). The HJB residual only.
    """
    graded = residual.pow(2)
    scale = graded.mean().detach().clamp_min(1e-30)

    return scale * (graded / scale).pow(POWER).mean().pow(1.0 / POWER)


def ridge_loss(
    value: DimensionlessValueFunction, ridge_tauhat: Tensor, ridge_etahat: Tensor
) -> Tensor:
    """
    BC1, the ridge condition that rules out the never-explore solution:
    du/dmuhat = -1/2 at muhat = 0, imposed on the premium (smooth there; the
    kink lives in the value's relu). Unchanged by drift -- the arm swap
    theta -> -theta still maps the problem to itself, so it holds at every
    etahat, and the drift coordinate is sampled along with tauhat.
    """
    ridge_muhat = torch.zeros_like(ridge_tauhat).requires_grad_(True)
    u = value.premium(ridge_muhat, ridge_tauhat, ridge_etahat)
    (u_muhat,) = torch.autograd.grad(u.sum(), ridge_muhat, create_graph=True)

    return (u_muhat + 0.5).pow(2).mean()


def loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    etahat: Tensor,
    ridge_tauhat: Tensor,
    ridge_etahat: Tensor,
    iteration: int | None = None,
) -> Tensor:
    """
    Full training loss: interior residual, the weighted ridge condition, and
    the learning operator's negative part.
    """
    pde, pos_learning = pde_loss(value, muhat, tauhat, etahat)
    ridge = ridge_loss(value, ridge_tauhat, ridge_etahat)

    if iteration is not None:
        print(
            f"iter {iteration}: pde {pde.item():.3e}  ridge {ridge.item():.3e}"
            f"  pos_learning {pos_learning.item():.3e}"
        )

    return pde + RIDGE_WEIGHT * ridge + POSITIVITY_WEIGHT * pos_learning


def objective(batch: int = 1024) -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol + ridge draws,
    scored by loss.
    """

    def step(value: DimensionlessValueFunction, iteration: int | None) -> Tensor:
        muhat, tauhat, etahat = sample_sobol(batch)
        ridge_tauhat, ridge_etahat = sample_ridge(batch // 4)

        return loss(value, muhat, tauhat, etahat, ridge_tauhat, ridge_etahat, iteration)

    return step


if __name__ == "__main__":
    import torch.nn as nn

    from ..two_arm.simplex import maximize_quadratic
    from .model import ExplorationPremium

    class _RidgeExact(nn.Module):
        # u = -muhat / 2 satisfies the ridge condition exactly; graph-connected
        # (0.0 * x, never zeros_like) so autograd has a path.
        def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
            return -0.5 * muhat + 0.0 * tauhat * etahat

    ridge_tauhat, ridge_etahat = sample_ridge(64)

    assert (
        ridge_loss(
            DimensionlessValueFunction(_RidgeExact()), ridge_tauhat, ridge_etahat
        ).item()
        < 1e-12
    )

    value = DimensionlessValueFunction(ExplorationPremium([16, 16]))
    objective_value = objective(batch=256)(value, None)

    assert (
        objective_value.dim() == 0 and objective_value.item() == objective_value.item()
    )

    objective_value.backward()
    grads = [p.grad for p in value.parameters() if p.grad is not None]

    assert len(grads) > 0 and all(g.isfinite().all() for g in grads)

    # The never-explore degeneracy, at every etahat: u = 0 zeroes the interior
    # residual exactly. Doubles as the end-to-end test of the derivative chain.
    class _Zero(nn.Module):
        # Exactly zero in value, but CUBIC in muhat: hamiltonian takes two
        # z-derivatives, and the derivative of an identically-zero tensor is a
        # constant zero with no path back to z, which autograd refuses to
        # differentiate again (the graph-connected-stub trap, CLAUDE.md).
        # 0.0 * muhat**3 keeps a live path through both.
        def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
            return 0.0 * muhat**3 * tauhat * (1.0 + etahat)

    dead = DimensionlessValueFunction(_Zero())
    muhat, tauhat, _ = sample_sobol(512)

    for etahat in [0.0, 1.0]:
        residual, negative = pde_loss(
            dead, muhat, tauhat, torch.full_like(muhat, etahat)
        )

        assert residual.item() < 1e-20, (etahat, residual.item())

        # u = 0 zeroes the learning operator too (one line, no switching), so
        # the positivity term is silent on the degenerate solution -- it
        # neither breaks nor deepens it.
        assert negative.item() < 1e-20, (etahat, negative.item())

    # Transcription identity, two_arm's check with the drift term carried
    # through: the similarity residual is tauhat**(3/2) times the raw one.
    # Any future edit of the similarity grading must keep it.
    muhat = torch.rand(128, dtype=torch.float64) * 3.0 + 0.05
    tauhat = torch.exp(torch.rand(128, dtype=torch.float64) * 6.0 - 3.0)
    value = value.double()

    for etahat_value in [0.0, 0.3, 1.0, 4.0]:
        etahat = torch.full_like(muhat, etahat_value)

        lhs, best_sim, _ = value.hamiltonian(muhat, tauhat, etahat)
        residual_sim = lhs - best_sim.value

        raw_muhat = muhat.clone().requires_grad_(True)
        raw_tauhat = tauhat.clone().requires_grad_(True)
        v = value(raw_muhat, raw_tauhat, etahat)
        v_m, v_t = torch.autograd.grad(
            v.sum(), [raw_muhat, raw_tauhat], create_graph=True
        )
        (v_mm,) = torch.autograd.grad(v_m.sum(), raw_muhat, create_graph=True)
        lhat = v_t + v_mm / (2 * raw_tauhat**2)
        best_raw = maximize_quadratic(-lhat, raw_muhat + lhat)
        residual_raw = v + etahat**2 * raw_tauhat**2 * v_t - best_raw.value

        gap = (residual_sim - tauhat**1.5 * residual_raw).abs().max().item()

        assert gap < 1e-8, (etahat_value, gap)
    print("ok")
