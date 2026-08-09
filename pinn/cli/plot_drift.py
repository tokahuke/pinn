"""
Field-and-slice plots of a trained two_arm_drift checkpoint: premium, policy,
and PDE residual, one png each in data/.

Similarity coordinates, not raw: the sampler couples tauhat to etahat, so a
raw (muhat, tauhat) window is mostly unreachable. x is z = muhat sqrt(tauhat),
where the commit boundary stands still; y runs to just past the ceiling
1/(2 etahat), which is drawn. Slices sit at a tauhat the whole family reaches.
"""

from __future__ import annotations

import click
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch import Tensor

from ..problems.two_arm_drift import DimensionlessValueFunction
from ..problems.two_arm_drift.sample import ETAHAT_MAX, PRIOR_FLOOR

# Field etahat: near deployment (3% of baseline per horizon is 10.7). TAUHAT
# is low enough that 2 etahat tauhat <= 1 for the whole family.
ETAHAT = 10.0
FAMILY = [0.0, 0.1, 1.0, 10.0, 30.0]
TAUHAT = 0.01

# tauhat stops AT the ceiling: above it the drift drains precision faster than
# any allocation buys it, nothing is sampled there and the net extrapolates.
CEILING = 1.0 / (2.0 * ETAHAT)

GRID = 301
z_axis = torch.linspace(1e-3, 3.0, GRID)
tauhat_axis = torch.linspace(PRIOR_FLOOR, CEILING, GRID)
z_grid, tauhat_grid = torch.meshgrid(z_axis, tauhat_axis, indexing="xy")
slice_z = torch.linspace(1e-3, 3.0, 401)


def policy_and_residual(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor, etahat: float
) -> tuple[Tensor, Tensor]:
    """
    alpha* and the SIMILARITY-chart residual, the one the loss grades.

    Not the raw one: raw = similarity / tauhat**1.5, which varies by orders
    across a panel, so no single colour scale means anything.
    """
    drift = torch.full_like(muhat, etahat)
    left, best, _ = value.hamiltonian(muhat, tauhat, drift)

    return best.x.detach(), (left - best.value).detach()


def double_plot(
    field: Tensor,
    slices: dict[float, Tensor],
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> plt.Axes:
    """
    Field at etahat = ETAHAT left, the etahat family at tauhat = TAUHAT right.
    Callers decorate the returned slice axis, then save(ax, filename).
    """
    fig, (ax_field, ax_slice) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), width_ratios=[1.15, 1]
    )

    mesh = ax_field.pcolormesh(
        z_grid, tauhat_grid, field, cmap=cmap, vmin=vmin, vmax=vmax
    )
    fig.colorbar(mesh, ax=ax_field, label=label)
    ax_field.set_xlabel("z = muhat sqrt(tauhat)")
    ax_field.set_ylabel("tauhat")
    ax_field.set_title(
        f"{label}(z, tauhat), etahat = {ETAHAT}, up to the ceiling {CEILING:g}"
    )

    for etahat, values in slices.items():
        reference = etahat == 0.0
        ax_slice.plot(
            slice_z,
            values,
            linewidth=2 if reference else 1.5,
            linestyle="--" if reference else "-",
            color="#9ea3ab" if reference else None,
            label=f"etahat = {etahat}" + (" (two_arm)" if reference else ""),
        )

    ax_slice.set_xlabel("z = muhat sqrt(tauhat)")
    ax_slice.set_ylabel(label)
    ax_slice.set_title(f"{label}(z, {TAUHAT}, etahat)")
    ax_slice.grid(True, alpha=0.25, linewidth=0.5)
    ax_slice.spines[["top", "right"]].set_visible(False)
    ax_slice.legend(frameon=False, fontsize=8)

    return ax_slice


def save(ax: plt.Axes, filename: str) -> None:
    ax.figure.tight_layout()
    ax.figure.savefig(Path("data") / filename, dpi=150)


@click.command(name="plot-drift")
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data") / "two_arm_drift.pt",
    show_default=True,
    help="Checkpoint to plot.",
)
def main(in_path: Path) -> None:
    """
    Render premium, policy, and residual for a two_arm_drift checkpoint.
    """
    value = DimensionlessValueFunction.load(in_path)
    flat_tauhat = tauhat_grid.flatten()
    flat_muhat = z_grid.flatten() / flat_tauhat.sqrt()
    slice_tauhat = torch.full_like(slice_z, TAUHAT)
    slice_muhat = slice_z / TAUHAT**0.5

    with torch.no_grad():
        u_field = value.premium(
            flat_muhat, flat_tauhat, torch.full_like(flat_muhat, ETAHAT)
        ).reshape(GRID, GRID)
        u_slices = {
            etahat: value.premium(
                slice_muhat, slice_tauhat, torch.full_like(slice_muhat, etahat)
            )
            for etahat in FAMILY
        }

    alpha_field, residual_field = policy_and_residual(
        value, flat_muhat, flat_tauhat, ETAHAT
    )
    alpha_field = alpha_field.reshape(GRID, GRID)
    residual_field = residual_field.reshape(GRID, GRID)
    policies = {}
    residuals = {}

    for etahat in FAMILY:
        policies[etahat], residuals[etahat] = policy_and_residual(
            value, slice_muhat, slice_tauhat, etahat
        )

    reachable = tauhat_grid <= CEILING
    limit = float(u_field[reachable].abs().max())
    ax = double_plot(u_field, u_slices, "u", "RdBu_r", -limit, limit)
    ax.axhline(0.0, color="#9ea3ab", linewidth=1)
    save(ax, "premium_drift.png")

    ax = double_plot(alpha_field, policies, "alpha*", "Blues", 0.5, 1.0)
    ax.axhline(0.5, color="#9ea3ab", linewidth=1, linestyle=":")
    ax.axhline(1.0, color="#9ea3ab", linewidth=1, linestyle=":")
    ax.set_ylim(0.45, 1.05)
    save(ax, "policy_drift.png")

    limit = float(residual_field[reachable].abs().quantile(0.99))
    ax = double_plot(
        residual_field, residuals, "residual (sim)", "RdBu_r", -limit, limit
    )
    ax.axhline(0.0, color="#9ea3ab", linewidth=1)
    save(ax, "residual_drift.png")

    print(
        f"{'etahat':>8} {'2 eta tau':>10} {'u(z=0)':>9} {'ridge slope':>12} {'z commit':>9}"
    )

    for etahat in FAMILY:
        u = u_slices[etahat]
        ridge_muhat = torch.zeros(64).requires_grad_(True)
        ridge_u = value.premium(
            ridge_muhat, torch.full((64,), TAUHAT), torch.full((64,), etahat)
        )
        (slope,) = torch.autograd.grad(ridge_u.sum(), ridge_muhat)
        below = (policies[etahat] < 1.0 - 1e-6).nonzero()
        commit = float(slice_z[below[-1] + 1]) if len(below) else float("nan")
        print(
            f"{etahat:>8} {2.0 * etahat * TAUHAT:>10.3f} {float(u[0]):>9.4f}"
            f" {float(slope.mean()):>12.4f} {commit:>9.4f}"
        )

    print(
        "(BC1 says the ridge slope is -0.5 at every etahat; etahat <= "
        f"{ETAHAT_MAX:g} and 2 etahat tauhat <= 1 are the sampled domain)"
    )
    print("saved data/premium_drift.png data/policy_drift.png data/residual_drift.png")


if __name__ == "__main__":
    main()
