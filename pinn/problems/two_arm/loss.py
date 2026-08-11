"""
Losses for the two-arm problem: the interior HJB residual, the ridge condition
BC1, and the trainer-facing objective.
"""

from __future__ import annotations

import torch

from torch import Tensor

from ...train import Objective
from .model import DimensionlessValueFunction
from .sample import sample_ridge, sample_sobol
from .simplex import maximize_quadratic

# Dead-solution floor: u = 0 scores pde exactly 0 and ridge exactly 0.25, so
# 100 * pde / 0.25 leaves the dead branch 100x worse. Set at pde 48.5; the pde
# is now 2.5e-4, so this stands ~200,000x above its own criterion and the ridge
# is ~24% of what best-checkpointing selects on. It holds the ridge at ~4e-9
# and trains smoothly, so it is left alone -- but re-derive it before trusting
# the number.
RIDGE_WEIGHT = 2.0e4

# Plain mean-of-squares. P = 2 compensated for the chart weight's suppressed
# tail; in natural units it over-corrects, dropping the effective sample size
# at batch 4096 to 2.1 points on three_arm (54 at P = 1).
POWER = 1.0

# Zero on the dead solution, so the floor above says nothing about it; a
# multiple of pde instead, which the units change leaves alone. This net has
# had no violations since the natural-units retrain, so the term is inert.
POSITIVITY_WEIGHT = 6.5e2


def pde_loss(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Returns TWO numbers. The first is the squared residual of the HJB in
    similarity coordinates (kb/two_arm.md
    section 8), maximization kept explicit. With z = muhat sqrt(tauhat),
    s = log tauhat, g = sqrt(tauhat) u, and the O(1) operator

        L_ab[g] = g_s + (1/2) g_zz + (z/2) g_z - (1/2) g

    the HJB v = max alpha muhat + alpha(1-alpha) Lhat[v] becomes

        e^s (z + g) = max over alpha in [0, 1] of alpha e^s z + alpha(1-alpha) L_ab[g]

    Grading in this form is what kills the raw-coordinate stiffness: the
    1/(2 tauhat^2) that multiplied curvature (9 decades of coefficient spread)
    is cancelled algebraically, so autograd error is never amplified; e^s
    touches only derivative-free terms. The leaves are (z, s) and autograd
    performs the chain rule through muhat = z e^(-s/2), tauhat = e^s -- the
    section 8 algebra never appears in code.

    KNOWN DEGENERATE on its own: g = 0 (the never-explore solution) zeroes
    this residual exactly. The ridge loss breaks the degeneracy.

    The second is the mean of relu(-L_ab). L_ab >= 0 is provable, and where it
    fails the Hamiltonian turns convex and the max jumps to a vertex -- the
    policy commits on no evidence. LINEAR, not squared: squaring chases depth
    while the violating fraction rises (two_arm_drift, 2026-08-08).
    """
    # The derivation (similarity chart, L_ab[g], interval max) lives on the
    # model: one chain serves training and policy readout.
    lhs, best, l_ab = value.hamiltonian(muhat, tauhat)

    # NATURAL UNITS, NEVER SCALED. The chart multiplies the equation by
    # tauhat**(3/2) (the identity __main__ asserts), so grading its residual
    # squared applied an undeclared tauhat**3 weight (learnings section 3).
    # The positivity term is NOT divided by it: that rule governs the RESIDUAL,
    # and a sign condition's depth weighting is a free choice.
    natural = tauhat.pow(1.5)
    graded = ((lhs - best.value) / natural).pow(2)
    scale = graded.mean().detach().clamp_min(1e-30)

    return (
        scale * (graded / scale).pow(POWER).mean().pow(1.0 / POWER),
        torch.relu(-l_ab).mean(),
    )


def ridge_loss(value: DimensionlessValueFunction, ridge_tauhat: Tensor) -> Tensor:
    """
    BC1, the ridge condition that rules out the never-explore solution:
    du/dmuhat = -1/2 at muhat = 0, imposed on the premium (smooth there; the
    kink lives in the value's relu).
    """
    ridge_muhat = torch.zeros_like(ridge_tauhat).requires_grad_(True)
    u = value.premium(ridge_muhat, ridge_tauhat)
    (u_muhat,) = torch.autograd.grad(u.sum(), ridge_muhat, create_graph=True)

    return (u_muhat + 0.5).pow(2).mean()


def loss(
    value: DimensionlessValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    ridge_tauhat: Tensor,
    iteration: int | None = None,
) -> Tensor:
    """
    Full training loss: interior residual, the weighted ridge condition, and
    the learning operator's negative part.
    """
    pde, pos_learning = pde_loss(value, muhat, tauhat)
    ridge = ridge_loss(value, ridge_tauhat)

    if iteration is not None:
        print(
            f"iter {iteration}: pde {pde.item():.3e}  ridge {ridge.item():.3e}"
            f"  pos_learning {pos_learning.item():.3e}"
        )

    return pde + RIDGE_WEIGHT * ridge + POSITIVITY_WEIGHT * pos_learning


def draw(batch: int, device: str = "cpu") -> tuple:
    """
    One step's collocation tensors, in loss()'s argument order.

    Split out of objective so the graphed trainer can hold them as fixed
    buffers: a captured cuda graph replays the same tensor addresses, so the
    sampling has to live outside it.
    """
    return (
        *(t.to(device) for t in sample_sobol(batch)),
        sample_ridge(batch // 4).to(device),
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

    from .model import ExplorationPremium

    class _RidgeExact(nn.Module):
        # u = -muhat / 2 satisfies the ridge condition exactly; graph-connected
        # (0.0 * x, never zeros_like) so autograd has a path.
        def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
            return -0.5 * muhat + 0.0 * tauhat

    assert (
        ridge_loss(
            DimensionlessValueFunction(_RidgeExact()), torch.rand(64) + 0.1
        ).item()
        < 1e-12
    )

    # The never-explore solution scores exactly 0 on both numbers.
    class _Dead(nn.Module):
        # SQUARED in muhat: hamiltonian takes g_zz without allow_unused.
        def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
            return 0.0 * muhat**2 * tauhat

    dead_pde, dead_positivity = pde_loss(
        DimensionlessValueFunction(_Dead()),
        torch.rand(256) + 0.05,
        torch.rand(256) + 0.05,
    )

    assert dead_pde.item() == 0.0 and dead_positivity.item() == 0.0

    value = DimensionlessValueFunction(ExplorationPremium([16, 16]))
    objective_value = objective(batch=256)(value, None)

    assert (
        objective_value.dim() == 0 and objective_value.item() == objective_value.item()
    )

    objective_value.backward()
    grads = [p.grad for p in value.parameters() if p.grad is not None]

    assert len(grads) > 0 and all(g.isfinite().all() for g in grads)

    # Transcription identity: the similarity residual equals tauhat**(3/2)
    # times the raw-coordinate residual, per point. Recompute both ways for a
    # random net; any future edit of the similarity grading must keep this.
    muhat = torch.rand(128, dtype=torch.float64) * 3.0 + 0.05
    tauhat = torch.exp(torch.rand(128, dtype=torch.float64) * 6.0 - 3.0)
    value = value.double()

    z = (muhat * tauhat.sqrt()).detach().requires_grad_(True)
    s = tauhat.log().detach().requires_grad_(True)
    g = (s / 2).exp() * value.premium(z * (-s / 2).exp(), s.exp())
    g_z, g_s = torch.autograd.grad(g.sum(), [z, s], create_graph=True)
    (g_zz,) = torch.autograd.grad(g_z.sum(), z, create_graph=True)
    l_ab = g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g
    best_sim = maximize_quadratic(-l_ab, s.exp() * z + l_ab)
    residual_sim = s.exp() * (z + g) - best_sim.value

    raw_muhat = muhat.clone().requires_grad_(True)
    raw_tauhat = tauhat.clone().requires_grad_(True)
    v = value(raw_muhat, raw_tauhat)
    v_m, v_t = torch.autograd.grad(v.sum(), [raw_muhat, raw_tauhat], create_graph=True)
    (v_mm,) = torch.autograd.grad(v_m.sum(), raw_muhat, create_graph=True)
    lhat = v_t + v_mm / (2 * raw_tauhat**2)
    best_raw = maximize_quadratic(-lhat, raw_muhat + lhat)
    residual_raw = v - best_raw.value

    gap = (residual_sim - tauhat**1.5 * residual_raw).abs().max().item()

    assert gap < 1e-9, gap
    print("ok")
