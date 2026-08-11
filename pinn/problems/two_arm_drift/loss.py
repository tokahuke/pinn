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

# Dead-solution floor: u = 0 scores pde exactly 0 and ridge exactly 0.25, so
# 100 * pde / 0.25 leaves the dead branch 100x worse than the live one.
# Re-derive after any large pde move -- the previous 2.0e5 was set at pde 483
# and left standing at pde 7e-2, i.e. 7000x its own criterion. At 28 the ridge
# settles near 3e-5 rather than the 5e-8 it held at 2.0e5: enforced weakly, not
# diverging.
RIDGE_WEIGHT = 2.8e1

# Plain mean-of-squares. P = 2 compensated for the chart weight's suppressed
# tail; in natural units it over-corrects, dropping the effective sample size
# at batch 4096 to 2.1 points on three_arm (54 at P = 1).
POWER = 1.0

# L_ab >= 0 is provable and INVISIBLE to the residual, which sees it only
# through alpha(1-alpha) <= 1/4 while the policy is its argmax: the residual
# can be excellent with the policy ill-posed. Zero at the answer, so it moves
# the path, not the fixed point. Bounded per point by the saturation in
# pde_loss, so not comparable to the pre-2026-08-10 weights; holds ~1% of pde.
POSITIVITY_SCALE = 1.0e-7
POSITIVITY_WEIGHT = 6.0e-2


def pde_loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    etahat: Tensor,
) -> Tensor:
    """
    Returns TWO numbers: the squared residual of the HJB in similarity coordinates
    (kb/two_arm_drift.md section 6), maximization kept explicit:

        e^s (z + g) + etahat^2 e^(2s) tauhat_slope
            = max over alpha of { alpha e^s z + alpha(1-alpha) L_ab[g] }

    DERIVED in this chart for two_arm's reason -- the 1/(2 tauhat^2) that
    multiplied curvature is cancelled algebraically, and the leaves are (z, s)
    with autograd doing the chain rule -- but GRADED in natural units: the
    chart residual is tauhat**(3/2) times the equation's own, so it is divided
    back out before grading. See two_arm/loss.py for the record and the
    standing rule; __main__ checks the identity against the raw form.

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
    # Natural units: undo the chart's tauhat**(3/2). L_ab carries the same
    # factor (L_ab = tauhat**(3/2) lhat), so the positivity term is divided
    # too -- same sign, natural magnitude.
    natural = tauhat.pow(1.5)

    # SATURATED, and not in natural units: relu is linear in depth, so one deep
    # violation dominates (max/mean 15,905x raw, 116,628x divided) and spiked
    # the gradient norm 1.86 -> 1850 in a single step. y / (s + y) is bounded
    # per point, linear below s, 1/y^2 above -- NOT tanh (gradient underflows).
    violation = torch.relu(-l_ab)

    return (
        _graded((lhs - best.value) / natural),
        (violation / (POSITIVITY_SCALE + violation)).mean(),
    )


def _graded(residual: Tensor) -> Tensor:
    """
    The problem's own grading: NEVER a scale on the residual, power-mean
    attention
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


def draw(batch: int, device: str = "cpu") -> tuple[Tensor, ...]:
    """
    One step's collocation tensors, in loss()'s argument order.

    Split out of objective so the graphed trainer can hold them as fixed
    buffers and copy_ fresh draws in: a captured cuda graph replays the same
    tensor addresses, so the sampling has to live outside it.
    """
    return (
        *(t.to(device) for t in sample_sobol(batch)),
        *(t.to(device) for t in sample_ridge(batch // 4)),
    )


def objective(batch: int = 1024, device: str = "cpu") -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol + ridge draws,
    scored by loss.

    `device` is the ONE place the trainer's device enters the problem. Sobol
    draws on CPU (SobolEngine ignores the default device) and the batch moves
    once; everything downstream inherits from its inputs, so no loss, model or
    sampler needs to know a device exists. Defaults to CPU, which is what the
    arena, probes and every module self-check rely on.
    """

    def step(value: DimensionlessValueFunction, iteration: int | None) -> Tensor:
        return loss(value, *draw(batch, device), iteration)

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
