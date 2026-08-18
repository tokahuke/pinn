"""
Diagnostic probes for a three_arm checkpoint: wedge slices, tau lines, and the
symmetric-point allocation table, one set of pngs per checkpoint in data/.

Convention: "correlated" means tau_bc = -tau/2, the correlation a shared control arm
induces, which is the arm-symmetric state where the allocation must tend to
(1/3, 1/3, 1/3). tau_bc = 0 means the control is known perfectly (the off-diagonal of
the difference covariance is the control variance), a weird state for this problem;
probes do not use it.
"""

from __future__ import annotations

import click
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from pinn.problems.three_arm.model import ExplorationPremium, DimensionlessValueFunction
from pinn.problems.three_arm.simplex import maximize_quadratic
from pinn.utils import nu
from torch import Tensor

CORRELATION = -0.5
"""tau_bc as a share of tau: the correlation a shared control arm induces."""


def probe(
    value: DimensionlessValueFunction,
    m_b: Tensor,
    m_c: Tensor,
    tau_bb: Tensor,
    tau_bc: Tensor,
    tau_cc: Tensor,
) -> dict[str, Tensor]:
    """
    Pointwise diagnostics: premium, envelope utilization, relative HJB residual, and
    the argmax allocation. Mirrors loss.subsolution_loss.
    """
    m_b, m_c = m_b.requires_grad_(True), m_c.requires_grad_(True)
    tau_bb = tau_bb.requires_grad_(True)
    tau_bc = tau_bc.requires_grad_(True)
    tau_cc = tau_cc.requires_grad_(True)

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
        v_mc.sum(), [m_c], create_graph=True, allow_unused=True, materialize_grads=True
    )

    det = tau_bb * tau_cc - tau_bc**2

    def mean_diffusion(w_b: Tensor, w_c: Tensor) -> Tensor:
        """The Ito term for a pair direction, given that pair's weights."""
        return 0.5 * (w_b**2 * v_mbmb + 2.0 * w_b * w_c * v_mbmc + w_c**2 * v_mcmc)

    l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + v_tbb
    l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + v_tcc
    l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
        v_tbb + v_tcc - v_tbc
    )
    best = maximize_quadratic(-l_ab, -l_ac, l_bc - l_ab - l_ac, m_b + l_ab, m_c + l_ac)

    commit = torch.relu(torch.maximum(m_b, m_c))
    scale = 1.0 + (best.value - commit).detach().abs()

    precision_b = tau_bb - tau_bc**2 / tau_cc
    precision_c = tau_cc - tau_bc**2 / tau_bb
    envelope = value.premium.log_scale.exp() * (
        nu(m_b, precision_b.rsqrt()) + nu(m_c, precision_c.rsqrt())
    )
    u = v - commit

    return {
        k: t.detach()
        for k, t in {
            "premium": u,
            "utilization": u / envelope,
            "residual": (v - best.value) / scale,
            "alpha_b": best.x,
            "alpha_c": best.y,
            "alpha_a": 1.0 - best.x - best.y,
        }.items()
    }


def wedge_slice(value: DimensionlessValueFunction, tau: float, path: Path) -> None:
    """
    2-D slice over means at tau_bb = tau_cc = tau, correlated tau_bc, masked to the
    wedge m_c <= m_b <= 0 (strictly inside: the relu kink sits at m_b = 0). The mean
    axis spans 3 posterior sd, so slices are comparable across tau.
    """
    axis = torch.linspace(-3.0 / tau**0.5, -0.02, 150)
    m_b, m_c = torch.meshgrid(axis, axis, indexing="xy")
    m_b, m_c = m_b.flatten(), m_c.flatten()
    ones = torch.full_like(m_b, tau)
    d = probe(
        value, m_b.clone(), m_c.clone(), ones.clone(), CORRELATION * ones, ones.clone()
    )

    wedge = m_c <= m_b
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)

    for ax, (name, field) in zip(axes.flat, d.items()):
        z = field.masked_fill(~wedge, torch.nan).reshape(150, 150)
        diverging = name == "residual"
        limit = z[wedge.reshape(150, 150)].abs().max().item()
        # The three alpha panels share the [0, 1] scale so they read as one allocation;
        # the top row keeps per-panel autoscale.
        alpha_panel = name.startswith("alpha")
        image = ax.pcolormesh(
            axis,
            axis,
            z,
            cmap="RdBu_r" if diverging else "viridis",
            vmin=-limit if diverging else 0.0 if alpha_panel else None,
            vmax=limit if diverging else 1.0 if alpha_panel else None,
        )
        fig.colorbar(image, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("m_b")
        ax.set_ylabel("m_c")

    fig.suptitle(f"wedge slice at tau_bb = tau_cc = {tau}, tau_bc = {CORRELATION} tau")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def tau_lines(value: DimensionlessValueFunction, path: Path) -> None:
    """
    Premium and relative residual along tau_bb = tau_cc = t at the correlated tau_bc,
    through chosen mean points; the first is the triple point.
    """
    points = [(-0.05, -0.05), (-0.5, -0.5), (-1.0, -1.0), (-0.05, -2.0)]
    t = torch.logspace(-2.5, 0.9, 200)
    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for (mb, mc), color in zip(points, plt.cm.tab10.colors):
        d = probe(
            value,
            torch.full_like(t, mb),
            torch.full_like(t, mc),
            t.clone(),
            CORRELATION * t,
            t.clone(),
        )
        ax_u.plot(t, d["premium"], color=color, label=f"m = ({mb}, {mc})")
        ax_r.plot(t, d["residual"], color=color, label=f"m = ({mb}, {mc})")

    for ax, title in [(ax_u, "premium u"), (ax_r, "relative residual")]:
        ax.set_xscale("log")
        ax.set_xlabel("tau_bb = tau_cc = t")
        ax.set_title(f"{title} (tau_bc = {CORRELATION} t)")
        ax.axhline(0.0, color="gray", lw=0.5)
        ax.legend()

    fig.savefig(path, dpi=110)
    plt.close(fig)


def symmetric_point(value: DimensionlessValueFunction) -> None:
    """
    Allocation at the near-symmetric point m_b = m_c = -eps, correlated tau_bc: should
    tend to (1/3, 1/3, 1/3).
    """
    print("tau      alpha_a  alpha_b  alpha_c   residual")
    for tau in [0.03, 0.1, 0.3, 1.0, 3.0, 8.0]:
        t = torch.full((1,), tau)
        m = torch.full((1,), -0.02)
        d = probe(value, m.clone(), m.clone(), t.clone(), CORRELATION * t, t.clone())
        print(
            f"{tau:<8g} {d['alpha_a'].item():.4f}   {d['alpha_b'].item():.4f}"
            f"   {d['alpha_c'].item():.4f}   {d['residual'].item():+.2e}"
        )


@click.command()
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data") / "three_arm.pt",
    help="Checkpoint to probe.",
)
def main(in_path: Path) -> None:
    """
    Render wedge slices and tau lines as pngs in data/, print the symmetric-point
    allocation table.
    """
    value = DimensionlessValueFunction.load(in_path)

    stem = in_path.stem
    for tau in [1.0, 0.1]:
        wedge_slice(value, tau, Path("data") / f"wedge_{stem}_tau{tau:g}.png")
    tau_lines(value, Path("data") / f"tau_lines_{stem}.png")
    symmetric_point(value)
    print(f"saved data/wedge_{stem}_tau{{1,0.1}}.png data/tau_lines_{stem}.png")


if __name__ == "__main__":
    main()
