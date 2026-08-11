"""
Consistency checks (kb/two_arm.md section 6) for a trained checkpoint, plus an
off-grid residual audit. Reports numbers; judges nothing.
"""

from __future__ import annotations

import click
import torch

from pathlib import Path
from torch import Tensor

from ..problems.two_arm import init_model, sample_sobol


@click.command()
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data") / "two_arm.pt",
    show_default=True,
    help="Checkpoint to validate.",
)
def main(in_path: Path) -> None:
    """
    The section 6 identities of kb/two_arm.md, on a two_arm checkpoint.
    """
    value = init_model(state=torch.load(in_path))

    def pde_residual(muhat: Tensor, tauhat: Tensor) -> Tensor:
        """
        v - max_alpha H, same closed-form max as the training loss.
        """
        muhat = muhat.clone().requires_grad_(True)
        tauhat = tauhat.clone().requires_grad_(True)

        v = value(muhat, tauhat)
        v_muhat, v_tauhat = torch.autograd.grad(
            v.sum(), [muhat, tauhat], create_graph=True
        )
        (v_muhat_muhat,) = torch.autograd.grad(v_muhat.sum(), muhat, create_graph=True)

        lhat = v_tauhat + v_muhat_muhat / (2 * tauhat**2)
        concave = lhat > 0
        safe_lhat = lhat.masked_fill(~concave, 1.0)
        alpha = torch.where(
            concave,
            ((muhat + lhat) / (2 * safe_lhat)).clamp(0.0, 1.0),
            (muhat > 0).float(),
        )

        return v - (alpha * muhat + alpha * (1 - alpha) * lhat)

    print("== residual audit ==")
    train_muhat, train_tauhat = sample_sobol(1024)
    on_measure = pde_residual(train_muhat, train_tauhat).abs()

    off_muhat = torch.rand(1024) * 2.5
    off_tauhat = 0.1 + torch.rand(1024) * 9.9
    off_grid = pde_residual(off_muhat, off_tauhat).abs()

    print(
        f"training measure |residual|: mean {on_measure.mean():.3e}  max {on_measure.max():.3e}"
    )
    print(
        f"uniform window   |residual|: mean {off_grid.mean():.3e}  max {off_grid.max():.3e}"
    )

    print("\n== ridge (muhat -> 0+) ==")
    ridge_tauhat = torch.linspace(0.2, 4.0, 20)
    ridge_muhat = torch.full_like(ridge_tauhat, 1e-3).requires_grad_(True)
    u_ridge = value.premium(ridge_muhat, ridge_tauhat)
    (slope,) = torch.autograd.grad(u_ridge.sum(), ridge_muhat)
    print(
        f"BC1 slope: mean {slope.mean():.4f}  worst {slope.min():.4f} / {slope.max():.4f}  (want -0.5)"
    )

    decay = value.premium(torch.full((20,), 1e-3), ridge_tauhat)
    monotone = bool((decay[1:] < decay[:-1]).all())
    print(f"u(0, tauhat) monotone decreasing: {monotone}")

    print("\n== free boundary (u < 1% of ridge value) ==")
    print("tauhat      G(tauhat)   2*tauhat*G   u_mm(G-)/(tauhat^2 G)")

    for t in [0.5, 1.0, 2.0, 3.0, 4.0]:
        scan = torch.linspace(1e-3, 2.5, 2000)
        tau = torch.full_like(scan, t)

        with torch.no_grad():
            profile = value.premium(scan, tau)

        below = (profile < 0.01 * profile[0]).nonzero()

        if len(below) == 0:
            print(f"{t:6.2f}      no boundary inside the window")

            continue

        boundary = float(scan[below[0]])
        inside = torch.tensor([0.95 * boundary]).requires_grad_(True)
        u_inside = value.premium(inside, torch.tensor([t]))
        (du,) = torch.autograd.grad(u_inside.sum(), inside, create_graph=True)
        (ddu,) = torch.autograd.grad(du.sum(), inside)
        curvature_law = float(ddu) / (t**2 * boundary)
        print(
            f"{t:6.2f}      {boundary:8.4f}   {2 * t * boundary:8.4f}     {curvature_law:8.4f}"
        )


if __name__ == "__main__":
    main()
