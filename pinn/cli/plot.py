"""
Field-and-slice plots of a trained checkpoint: premium, policy, and PDE
residual, one png each in data/.
"""

from __future__ import annotations

import click
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch import Tensor

from ..problems.two_arm.model import DimensionlessValueFunction, init_model

TAUHAT = 0.1
"""The tauhat every slice is cut at."""

muhat_axis = torch.linspace(1e-3, 2.5, 301)
tauhat_axis = torch.linspace(0.1, 4.0, 301)
muhat_grid, tauhat_grid = torch.meshgrid(muhat_axis, tauhat_axis, indexing="xy")
slice_muhat = torch.linspace(1e-3, 4.0, 401)
slice_tauhat = torch.full_like(slice_muhat, TAUHAT)


def policy_and_residual(
    value: DimensionlessValueFunction, muhat: Tensor, tauhat: Tensor
) -> tuple[Tensor, Tensor]:
    """
    alpha*(muhat, tauhat) and the HJB residual v - max_alpha H, same closed-form
    max as the training loss.
    """
    muhat = muhat.clone().requires_grad_(True)
    tauhat = tauhat.clone().requires_grad_(True)

    v = value(muhat, tauhat)
    v_muhat, v_tauhat = torch.autograd.grad(v.sum(), [muhat, tauhat], create_graph=True)
    (v_muhat_muhat,) = torch.autograd.grad(v_muhat.sum(), muhat)

    lhat = v_tauhat.detach() + v_muhat_muhat / (2 * tauhat.detach() ** 2)
    concave = lhat > 0
    safe_lhat = lhat.masked_fill(~concave, 1.0)
    alpha = torch.where(
        concave,
        ((muhat.detach() + lhat) / (2 * safe_lhat)).clamp(0.0, 1.0),
        (muhat.detach() > 0).float(),
    )
    residual = v.detach() - (alpha * muhat.detach() + alpha * (1 - alpha) * lhat)

    return alpha.detach(), residual.detach()


def ansatz_policy(muhat: Tensor, tauhat: float) -> Tensor:
    """
    Approximate self-similar policy (kb/two_arm.md section 7), scaled score
    q = muhat sqrt(tauhat) / sqrt(2 h(8 tauhat)).
    """
    t = torch.linspace(0.0, 80.0, 200001)
    h = float(torch.trapezoid(t * torch.exp(-t) / (8.0 * tauhat + t), t))

    q = muhat * tauhat**0.5 / (2.0 * h) ** 0.5
    b = 1.0 / (8.0 * tauhat * h) - 1.0
    interior = 0.5 + q / (2.0 * (1.0 + b * q**2))

    return torch.where(q <= 1.0, interior, torch.ones_like(q))


def double_plot(
    field: Tensor,
    slice_values: Tensor,
    label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> plt.Axes:
    """
    The house pattern: field over the (muhat, tauhat) window on the left, slice at
    tauhat = TAUHAT on the right. Callers decorate the returned slice axis, then
    save(ax, filename).
    """
    fig, (ax_field, ax_slice) = plt.subplots(
        1, 2, figsize=(12, 4.6), width_ratios=[1.15, 1]
    )

    mesh = ax_field.pcolormesh(
        muhat_grid, tauhat_grid, field, cmap=cmap, vmin=vmin, vmax=vmax
    )
    fig.colorbar(mesh, ax=ax_field, label=label)
    ax_field.set_xlabel("muhat")
    ax_field.set_ylabel("tauhat")
    ax_field.set_title(f"{label}(muhat, tauhat)")

    ax_slice.plot(slice_muhat, slice_values, color="#4269d0", linewidth=2)
    ax_slice.set_xlabel("muhat")
    ax_slice.set_ylabel(label)
    ax_slice.set_title(f"{label}(muhat, {TAUHAT})")
    ax_slice.grid(True, alpha=0.25, linewidth=0.5)
    ax_slice.spines[["top", "right"]].set_visible(False)

    return ax_slice


def save(ax: plt.Axes, filename: str) -> None:
    """Write the decorated figure into data/."""
    ax.figure.tight_layout()
    ax.figure.savefig(Path("data") / filename, dpi=150)


@click.command()
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data") / "two_arm.pt",
    help="Checkpoint to plot.",
)
def main(in_path: Path) -> None:
    """Render premium, policy, and residual as field-and-slice pngs in data/."""
    value = init_model(state=torch.load(in_path))

    with torch.no_grad():
        u_field = value.premium(muhat_grid.flatten(), tauhat_grid.flatten()).reshape(
            301, 301
        )
        u_slice = value.premium(slice_muhat, slice_tauhat)

    alpha_field, residual_field = policy_and_residual(
        value, muhat_grid.flatten(), tauhat_grid.flatten()
    )
    alpha_field = alpha_field.reshape(301, 301)
    residual_field = residual_field.reshape(301, 301)
    alpha_slice, residual_slice = policy_and_residual(value, slice_muhat, slice_tauhat)

    limit = float(u_field.abs().max())
    ax = double_plot(u_field, u_slice, "u", "RdBu_r", -limit, limit)
    ax.axhline(0.0, color="#9ea3ab", linewidth=1)
    save(ax, "premium.png")

    ax = double_plot(alpha_field, alpha_slice, "alpha*", "Blues", 0.5, 1.0)
    ax.plot(
        slice_muhat,
        ansatz_policy(slice_muhat, TAUHAT),
        color="#ff725c",
        linewidth=1.5,
        linestyle="--",
        label="self-similar ansatz",
    )
    ax.axhline(0.5, color="#9ea3ab", linewidth=1, linestyle="--")
    ax.axhline(1.0, color="#9ea3ab", linewidth=1, linestyle="--")
    ax.set_ylim(0.45, 1.05)
    ax.legend(frameon=False)
    save(ax, "policy.png")

    limit = float(residual_field.abs().flatten().quantile(0.99))
    ax = double_plot(
        residual_field, residual_slice, "residual", "RdBu_r", -limit, limit
    )
    ax.axhline(0.0, color="#9ea3ab", linewidth=1)
    save(ax, "residual.png")

    ridge_slope = float((u_slice[1] - u_slice[0]) / (slice_muhat[1] - slice_muhat[0]))
    print(f"u(0, {TAUHAT}) = {float(u_slice[0]):.4f}")
    print(f"ridge slope = {ridge_slope:.4f}  (BC1 says -0.5)")
    print(f"min u on slice = {float(u_slice.min()):.4f}")
    print(
        f"|residual| field: mean {residual_field.abs().mean():.3e}"
        f"  max {residual_field.abs().max():.3e}"
    )
    print("saved data/premium.png data/policy.png data/residual.png")


if __name__ == "__main__":
    main()
