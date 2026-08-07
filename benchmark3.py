"""
Monte Carlo shoot-out for three_arm: the trained PINN policy vs optimally-timed
explore-then-commit (thirds until the commit time maximizing the analytic
nu2 value, then bang-bang) and successive elimination by pairwise z-test.
Dimensionless units (rho = sigma = 1), posterior-space simulation from the
arm-symmetric prior (m = 0, tau_bb = tau_cc = TAU0, tau_bc = -TAU0/2).

Unlike two_arm's table, the 5-D policy cannot be tabulated: it is evaluated
by autograd every step on the folded state and un-permuted back to physical
arms via Sample.fold_ordered. That prices the simulation.
ponytail: DT 5x coarser and PATHS 2.5x fewer than two_arm's; refine when the
per-step autograd cost is worth optimizing.
"""

from __future__ import annotations

import click
import math
import torch

from pathlib import Path
from torch import Tensor

from pinn.problems.three_arm import ExplorationPremium, DimensionlessValueFunction
from pinn.problems.three_arm.sample import Sample
from pinn.problems.three_arm.simplex import maximize_quadratic
from pinn.utils import nu2

TAU0 = 1.0
DT = 5e-3
HORIZON = 12.0
PATHS = 20_000

# Optimally-timed explore-then-commit at THIS symmetric prior: under thirds
# every pairwise information rate is 1/9, so T(t) = T0 + (t/9) [[2,-1],[-1,2]]
# and the commit value is exp(-t) nu2 over the accumulated mean movement
# (prior covariance minus posterior covariance).
_T0 = torch.tensor([[TAU0, -TAU0 / 2.0], [-TAU0 / 2.0, TAU0]])


def _etc_value(times: Tensor) -> Tensor:
    information = times / 9.0
    tau_bb = _T0[0, 0] + 2.0 * information
    tau_bc = _T0[0, 1] - information
    det = tau_bb**2 - tau_bc**2
    det0 = _T0[0, 0] ** 2 - _T0[0, 1] ** 2

    # C = inv(T0) - inv(T), symmetric in b/c at the symmetric prior.
    c_diag = _T0[0, 0] / det0 - tau_bb / det
    c_off = -_T0[0, 1] / det0 + tau_bc / det
    correlation = (c_off / c_diag.clamp_min(1e-12)).clamp(-0.999, 0.999)

    return times.neg().exp() * nu2(
        torch.zeros_like(times),
        torch.zeros_like(times),
        c_diag.clamp_min(1e-12).sqrt(),
        c_diag.clamp_min(1e-12).sqrt(),
        correlation,
    )


_times = torch.linspace(0.01, 4.0, 4000)
COMMIT_TIME = float(_times[_etc_value(_times).argmax()])


def load(path: Path) -> DimensionlessValueFunction:
    state = torch.load(path)
    hidden = [
        w.shape[0]
        for k, w in state.items()
        if k.startswith("premium.net.") and k.endswith(".weight")
    ][:-1]
    kinks = (
        state["premium.kink_in.weight"].shape[0]
        if "premium.kink_in.weight" in state
        else 0
    )
    value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))
    value.load_state_dict(state)

    return value


def wedge_alpha(value: DimensionlessValueFunction, folded: Sample) -> Tensor:
    """
    Argmax allocation on wedge states, roles (a, b, c), mirroring
    loss.pde_loss's derivative chain.
    """
    m_b, m_c = folded.m_b.requires_grad_(True), folded.m_c.requires_grad_(True)
    tau_bb = folded.tau_bb.requires_grad_(True)
    tau_bc = folded.tau_bc.requires_grad_(True)
    tau_cc = folded.tau_cc.requires_grad_(True)

    v = value(m_b, m_c, tau_bb, tau_bc, tau_cc)
    v_mb, v_mc, v_tbb, v_tbc, v_tcc = torch.autograd.grad(
        v.sum(),
        [m_b, m_c, tau_bb, tau_bc, tau_cc],
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )
    v_mbmb, v_mbmc = torch.autograd.grad(
        v_mb.sum(),
        [m_b, m_c],
        create_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )
    (v_mcmc,) = torch.autograd.grad(
        v_mc.sum(), [m_c], allow_unused=True, materialize_grads=True
    )

    det = tau_bb * tau_cc - tau_bc**2

    def mean_diffusion(w_b: Tensor, w_c: Tensor) -> Tensor:
        return 0.5 * (w_b**2 * v_mbmb + 2.0 * w_b * w_c * v_mbmc + w_c**2 * v_mcmc)

    l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + v_tbb
    l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + v_tcc
    l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
        v_tbb + v_tcc - v_tbc
    )
    best = maximize_quadratic(-l_ab, -l_ac, l_bc - l_ab - l_ac, m_b + l_ab, m_c + l_ac)

    return torch.stack([1.0 - best.x - best.y, best.x, best.y], dim=-1).detach()


def pinn_policy(value: DimensionlessValueFunction):
    def policy(
        m_b: Tensor,
        m_c: Tensor,
        tau_bb: Tensor,
        tau_bc: Tensor,
        tau_cc: Tensor,
        t: float,
    ) -> Tensor:
        folded, order = Sample(
            m_b.clone(), m_c.clone(), tau_bb.clone(), tau_bc.clone(), tau_cc.clone()
        ).fold_ordered()
        roles = wedge_alpha(value, folded)

        return torch.zeros_like(roles).scatter_(1, order, roles)

    return policy


def etc_policy(
    m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor, t: float
) -> Tensor:
    if t < COMMIT_TIME:
        return torch.full((len(m_b), 3), 1.0 / 3.0)

    levels = torch.stack([torch.zeros_like(m_b), m_b, m_c], dim=-1)

    return torch.nn.functional.one_hot(levels.argmax(dim=-1), 3).float()


def elimination_policy(
    m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor, t: float
) -> Tensor:
    """
    Successive elimination, stateless: recover the data-only posterior
    (prior subtracted matrix-style, the two_arm z-test's trick), drop any arm
    pairwise-significantly worse at 5%, split evenly among survivors. A
    single survivor is an allocation vertex, absorbed by simulate().
    """
    d11 = tau_bb - _T0[0, 0]
    d12 = tau_bc - _T0[0, 1]
    d22 = tau_cc - _T0[1, 1]
    det = (d11 * d22 - d12**2).clamp_min(1e-12)
    ready = (d11 > 1e-9) & (d22 > 1e-9)

    q_b = tau_bb * m_b + tau_bc * m_c
    q_c = tau_bc * m_b + tau_cc * m_c
    data_b = (d22 * q_b - d12 * q_c) / det
    data_c = (-d12 * q_b + d11 * q_c) / det

    s11, s12, s22 = d22 / det, -d12 / det, d11 / det
    z_b = data_b / s11.clamp_min(1e-12).sqrt()
    z_c = data_c / s22.clamp_min(1e-12).sqrt()
    z_bc = (data_b - data_c) / (s11 + s22 - 2.0 * s12).clamp_min(1e-12).sqrt()

    out_a = ready & ((z_b > 1.96) | (z_c > 1.96))
    out_b = ready & ((z_b < -1.96) | (z_bc < -1.96))
    out_c = ready & ((z_c < -1.96) | (z_bc > 1.96))
    survivors = 1.0 - torch.stack([out_a, out_b, out_c], dim=-1).float()

    # Cyclic eliminations can in principle empty the set; fall back to the
    # posterior leader.
    empty = survivors.sum(dim=-1) < 0.5
    levels = torch.stack([torch.zeros_like(m_b), m_b, m_c], dim=-1)
    leader = torch.nn.functional.one_hot(levels.argmax(dim=-1), 3).float()
    survivors = torch.where(empty.unsqueeze(-1), leader, survivors)

    return survivors / survivors.sum(dim=-1, keepdim=True)


def simulate(policy) -> Tensor:
    """
    Discounted value per path. An allocation vertex freezes learning, so
    committed paths close in exact arithmetic: e^-t times the chosen arm's
    level. Mean noise has covariance T^-1 G T^-1 dt (doc section 3), G the
    pairwise information-rate matrix with det G = a_a a_b a_c.
    """
    torch.manual_seed(0)
    m_b = torch.zeros(PATHS)
    m_c = torch.zeros(PATHS)
    tau_bb = torch.full((PATHS,), _T0[0, 0].item())
    tau_bc = torch.full((PATHS,), _T0[0, 1].item())
    tau_cc = torch.full((PATHS,), _T0[1, 1].item())
    payoff = torch.zeros(PATHS)
    active = torch.ones(PATHS, dtype=torch.bool)
    t = 0.0

    while t < HORIZON and bool(active.any()):
        alpha = policy(m_b, m_c, tau_bb, tau_bc, tau_cc, t)
        discount = math.exp(-t)

        levels = torch.stack([torch.zeros_like(m_b), m_b, m_c], dim=-1)
        top, arm = alpha.max(dim=-1)
        committed = active & (top >= 1.0 - 1e-6)
        payoff = payoff + committed * discount * levels.gather(
            -1, arm.unsqueeze(-1)
        ).squeeze(-1)
        active = active & ~committed

        a_a, a_b, a_c = alpha[:, 0], alpha[:, 1], alpha[:, 2]
        g11 = a_a * a_b + a_b * a_c
        g12 = -a_b * a_c
        g22 = a_a * a_c + a_b * a_c

        det = tau_bb * tau_cc - tau_bc**2
        i11, i12, i22 = tau_cc / det, -tau_bc / det, tau_bb / det
        h11, h21 = g11 * i11 + g12 * i12, g12 * i11 + g22 * i12
        h12, h22 = g11 * i12 + g12 * i22, g12 * i12 + g22 * i22
        a11 = (i11 * h11 + i12 * h21).clamp_min(0.0)
        a12 = i11 * h12 + i12 * h22
        a22 = i12 * h12 + i22 * h22

        l11 = a11.sqrt()
        l21 = a12 / l11.clamp_min(1e-12)
        l22 = (a22 - l21**2).clamp_min(0.0).sqrt()

        payoff = payoff + active * discount * (a_b * m_b + a_c * m_c) * DT
        noise_1, noise_2 = torch.randn(PATHS), torch.randn(PATHS)
        root_dt = math.sqrt(DT)
        m_b = m_b + active * l11 * noise_1 * root_dt
        m_c = m_c + active * (l21 * noise_1 + l22 * noise_2) * root_dt
        tau_bb = tau_bb + active * g11 * DT
        tau_bc = tau_bc + active * g12 * DT
        tau_cc = tau_cc + active * g22 * DT
        t += DT

    tail = torch.stack([torch.zeros_like(m_b), m_b, m_c], -1).max(dim=-1).values

    return payoff + active * math.exp(-HORIZON) * tail


@click.command()
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data") / "value_3a_64:64:64.pt",
    help="Checkpoint to benchmark.",
)
def main(in_path: Path) -> None:
    value = load(in_path)

    with torch.no_grad():
        predicted = float(
            value(
                torch.zeros(1),
                torch.zeros(1),
                torch.tensor([TAU0]),
                torch.tensor([-TAU0 / 2.0]),
                torch.tensor([TAU0]),
            )
        )

    pinn_values = simulate(pinn_policy(value))
    etc_values = simulate(etc_policy)
    elimination_values = simulate(elimination_policy)

    analytic = float(_etc_value(torch.tensor([COMMIT_TIME])))
    stderr = 2.0 / math.sqrt(PATHS)

    print(f"ETC commit time (optimized)  = {COMMIT_TIME:.3f}")
    print(f"net's own claim v(0, sym)    = {predicted:.4f}")
    print(
        f"PINN policy (simulated)      = {pinn_values.mean():.4f}"
        f" +/- {stderr * pinn_values.std():.4f}"
    )
    print(
        f"explore-then-commit          = {etc_values.mean():.4f}"
        f" +/- {stderr * etc_values.std():.4f}"
    )
    print(f"explore-then-commit (exact)  = {analytic:.4f}")
    print(
        f"successive elimination at 5% = {elimination_values.mean():.4f}"
        f" +/- {stderr * elimination_values.std():.4f}"
    )
    print(
        f"PINN - ETC                   = {pinn_values.mean() - etc_values.mean():.4f}"
        f"   (ratio {pinn_values.mean() / etc_values.mean():.3f})"
    )
    print(
        f"PINN - elimination           = {pinn_values.mean() - elimination_values.mean():.4f}"
        f"   (ratio {pinn_values.mean() / elimination_values.mean():.3f})"
    )


if __name__ == "__main__":
    main()
