"""
Hero image: the three-arm policy field in sd-normalized coordinates, with
real simulated experiments wandering over it until the free boundaries stop
them. Background field at tau = 1 (the chart where the policy is
quasi-stationary in sd units); trajectories simulated with the true
posterior dynamics under the champion's policy.
"""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.collections import LineCollection


from pinn.problems.three_arm import DimensionlessValueFunction, ValueFunction

ANCHORS = np.array(
    [
        [42 / 255, 120 / 255, 214 / 255],  # control - blue
        [235 / 255, 104 / 255, 52 / 255],  # B - orange
        [27 / 255, 175 / 255, 122 / 255],  # C - aqua
    ]
)
DOT_COLORS = ["#1b4f8f", "#a63c16", "#0e7a53"]

PATHS = 30
DT = 5e-3
HORIZON = 12.0
TAU0 = 0.3
SPAN = 2.9


def field(value: ValueFunction, n: int = 480) -> np.ndarray:
    tau = 1.0
    deviation = (0.75 * tau) ** -0.5
    axis = torch.linspace(-SPAN * deviation, SPAN * deviation, n)
    m_b, m_c = torch.meshgrid(axis, axis, indexing="xy")
    alpha = value.policy(
        m_b.flatten(),
        m_c.flatten(),
        torch.full((n * n,), tau),
        torch.full((n * n,), -tau / 2.0),
        torch.full((n * n,), tau),
    )

    return (alpha.numpy() @ ANCHORS).reshape(n, n, 3)


def simulate(value: ValueFunction) -> tuple[list, list]:
    torch.manual_seed(11)
    m_b = torch.zeros(PATHS)
    m_c = torch.zeros(PATHS)
    tau_bb = torch.full((PATHS,), TAU0)
    tau_bc = torch.full((PATHS,), -TAU0 / 2.0)
    tau_cc = torch.full((PATHS,), TAU0)
    active = torch.ones(PATHS, dtype=torch.bool)
    frozen_arm = torch.full((PATHS,), -1, dtype=torch.long)

    trails: list[list[tuple[float, float]]] = [[] for _ in range(PATHS)]
    t = 0.0

    while t < HORIZON and bool(active.any()):
        p_b = tau_bb - tau_bc**2 / tau_cc
        p_c = tau_cc - tau_bc**2 / tau_bb

        for i in range(PATHS):
            if bool(active[i]):
                trails[i].append(
                    (float(m_b[i] * p_b[i] ** 0.5), float(m_c[i] * p_c[i] ** 0.5))
                )

        alpha = value.policy(m_b, m_c, tau_bb, tau_bc, tau_cc)
        top, arm = alpha.max(dim=-1)
        committed = active & (top >= 1.0)
        frozen_arm[committed] = arm[committed]
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

        noise_1, noise_2 = torch.randn(PATHS), torch.randn(PATHS)
        root = DT**0.5
        m_b = m_b + active * l11 * noise_1 * root
        m_c = m_c + active * (l21 * noise_1 + l22 * noise_2) * root
        tau_bb = tau_bb + active * g11 * DT
        tau_bc = tau_bc + active * g12 * DT
        tau_cc = tau_cc + active * g22 * DT
        t += DT

    return trails, frozen_arm.tolist()


def main() -> None:
    value = ValueFunction(
        DimensionlessValueFunction.load("data/value_3a_64:64:64.pt"), rho=1.0, sigma=1.0
    )
    image = field(value)
    trails, arms = simulate(value)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(
        np.clip(image, 0, 1) ** 0.92,
        origin="lower",
        extent=[-SPAN, SPAN, -SPAN, SPAN],
        interpolation="bilinear",
    )

    for trail, arm in zip(trails, arms):
        if len(trail) < 3:
            continue

        # Decimate and lightly smooth: the raw SDE steps are visual noise.
        points = np.array(trail)[::10]
        points = np.vstack([points, trail[-1]])

        if len(points) > 4:
            kernel = np.ones(3) / 3.0
            for column in range(2):
                interior = np.convolve(points[:, column], kernel, "same")
                points[1:-1, column] = interior[1:-1]

        segments = np.stack([points[:-1], points[1:]], axis=1)
        colors = np.ones((len(segments), 4))
        colors[:, 3] = np.linspace(0.10, 0.85, len(segments))
        ax.add_collection(
            LineCollection(segments, colors=colors, linewidths=2.5, capstyle="round")
        )

        if arm >= 0:
            ax.plot(
                points[-1, 0],
                points[-1, 1],
                marker="o",
                markersize=13,
                color=DOT_COLORS[arm],
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=5,
            )

    ax.set_xlim(-1.78, 1.97)
    ax.set_ylim(-0.92, 0.96)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig("docs/hero.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("saved docs/hero.png")


if __name__ == "__main__":
    main()
