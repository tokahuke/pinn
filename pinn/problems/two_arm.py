"""
The two-armed Bayesian allocation problem of docs/two_arm.md: models, loss, and
collocation samplers. One module, one problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Exponential
from torch.quasirandom import SobolEngine

from ..net import GainedTanh
from ..train import Objective, log_cosh


_SOBOL = SobolEngine(dimension=2, scramble=True)


class ExplorationPremium(nn.Module):
    """
    Dense tanh MLP times a tauhat**(-1/2) decay envelope:

        u = exp(log_scale) * tauhat**(-1/2)
            * N(muhat, log tauhat, muhat sqrt(tauhat), muhat tauhat)

    log_scale is a learnable scalar for the unknown decay constant. Feature
    choices: log tauhat gives every decade of precision equal resolution;
    muhat sqrt(tauhat) is the posterior z-score (the corridor is a near-vertical
    band in it); muhat tauhat is the tail similarity coordinate (the far-field
    free boundary is its level set ~ 1/2). Kinky structures become near
    axis-aligned in feature space, so the net buys them cheaply.

    Fourier features on z were tried (4 sin/cos harmonics, 2026-08-04) and
    REVERTED: the sinusoids imprinted their level sets on the residual, doubled
    the exterior ripple, and even L-BFGS could not make the basis pay.

    Only trained on muhat >= 0; the true premium is even, so evaluate at
    |muhat| yourself if you must go left. Output at muhat < 0 is garbage.
    """

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()

        sizes = [4, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            # Xavier with the tanh gain (PyTorch's default is relu-flavored and
            # ~4x too small for tanh here); plain gain for the linear head.
            head = i == len(sizes) - 2
            nn.init.xavier_uniform_(
                linear.weight, gain=1.0 if head else nn.init.calculate_gain("tanh")
            )
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        envelope = self.log_scale.exp() * tauhat.rsqrt()
        features = torch.stack(
            [muhat, tauhat.log(), muhat * tauhat.sqrt(), muhat * tauhat], dim=-1
        )

        return envelope * self.net(features).squeeze(-1)


class ValueFunction(nn.Module):
    """
    Dimensionless value on top of the premium: v = max(muhat, 0) + u.

    The commit-value term max(mu, 0)/rho rescales to exactly max(muhat, 0), so
    the value costs one relu. Backprop through v; read u off when convenient.
    """

    def __init__(self, premium: ExplorationPremium) -> None:
        super().__init__()

        self.premium = premium

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat)


def sample(n: int) -> tuple[Tensor, Tensor]:
    """
    Long-tailed random draw: tauhat ~ Exp(mean 2), muhat ~ Exp(mean
    2 / sqrt(tauhat)) given tauhat.
    """
    tauhat = Exponential(torch.full((n,), 0.5)).sample()
    muhat = Exponential(tauhat.sqrt() / 2).sample()

    return muhat, tauhat


def sample_sobol(n: int) -> tuple[Tensor, Tensor]:
    """
    Scrambled Sobol points pushed through the long-tail law: tauhat ~ 0.1 +
    Exp(mean 2) (floored away from the singular corner), muhat ~ Exp(mean
    2 / sqrt(tauhat)) given tauhat. Same tails as sample(), but low-discrepancy:
    grid-grade spread with no clumps, in any dimension. Successive calls
    continue one sequence, so coverage keeps refining across iterations.
    """
    t = _SOBOL.draw(n).clamp(1e-7, 1.0 - 1e-7)
    tauhat = 0.1 - 2.0 * (1.0 - t[:, 0]).log()
    muhat = -(2.0 / tauhat.sqrt()) * (1.0 - t[:, 1]).log()

    return muhat, tauhat


def sample_ridge(n: int) -> Tensor:
    """
    Draw n ridge points (muhat = 0 implied): tauhat ~ Exp(mean 2), matching the
    interior draw's tail.
    """
    return Exponential(torch.full((n,), 0.5)).sample()


def loss(
    value: ValueFunction,
    muhat: Tensor,
    tauhat: Tensor,
    ridge_tauhat: Tensor,
    iteration: int | None = None,
) -> Tensor:
    """
    Squared residual of the HJB in value form, maximization kept explicit:

        v = max over alpha in [0, 1] of alpha * muhat + alpha * (1 - alpha) * Lhat[v]

    The constrained max of the quadratic in alpha is closed form: the clamped
    vertex when the Hamiltonian is concave (Lhat > 0), else the best endpoint.

    Plus BC1, the ridge condition that rules out the never-explore solution:
    du/dmuhat = -1/2 at muhat = 0, imposed on the premium (smooth there; the
    kink lives in the value's relu).
    """
    muhat.requires_grad_(True)
    tauhat.requires_grad_(True)

    v = value(muhat, tauhat)
    v_muhat, v_tauhat = torch.autograd.grad(v.sum(), [muhat, tauhat], create_graph=True)
    (v_muhat_muhat,) = torch.autograd.grad(v_muhat.sum(), muhat, create_graph=True)

    lhat = v_tauhat + v_muhat_muhat / (2 * tauhat**2)
    concave = lhat > 0

    # The dead branch of a torch.where still computes (and backprops), so the
    # denominator gets a dummy 1.0 wherever the vertex formula is not selected.
    safe_lhat = lhat.masked_fill(~concave, 1.0)
    alpha = torch.where(
        concave,
        ((muhat + lhat) / (2 * safe_lhat)).clamp(0.0, 1.0),
        (muhat > 0).float(),
    )
    hamiltonian = alpha * muhat + alpha * (1 - alpha) * lhat

    # Relative residual: each point is scored on relative error against its own
    # detached scale (floor of 1), so big-coefficient regions lose their bully vote.
    scale = 1.0 + hamiltonian.detach().abs()
    pde_loss = log_cosh((v - hamiltonian) / scale).mean()

    # Boundary condition loss: at the ridge premium derivative is known.
    ridge_muhat = torch.zeros_like(ridge_tauhat).requires_grad_(True)
    u = value.premium(ridge_muhat, ridge_tauhat)
    (u_muhat,) = torch.autograd.grad(u.sum(), ridge_muhat, create_graph=True)
    bc_loss = (u_muhat + 0.5).pow(2).mean()

    if iteration is not None:
        print(f"iter {iteration}: pde {pde_loss.item():.3e}  bc {bc_loss.item():.3e}")

    return pde_loss + 10.0 * bc_loss


def objective(batch: int = 1024) -> Objective:
    """
    The problem packaged for the generic trainer: fresh Sobol + ridge draws,
    scored by loss.
    """

    def step(value: ValueFunction, iteration: int | None) -> Tensor:
        muhat, tauhat = sample_sobol(batch)
        ridge_tauhat = sample_ridge(batch // 4)

        return loss(value, muhat, tauhat, ridge_tauhat, iteration)

    return step


if __name__ == "__main__":
    muhat, tauhat = sample_sobol(1000)

    assert muhat.shape == tauhat.shape == (1000,)
    assert (muhat > 0).all() and (tauhat >= 0.1).all()
    assert (sample_ridge(100) > 0).all()

    value = ValueFunction(ExplorationPremium([16, 16]))
    objective_value = objective(batch=256)(value, None)

    assert (
        objective_value.dim() == 0 and objective_value.item() == objective_value.item()
    )
    print("ok")
