"""
Hero image: the three-arm policy field in sd-normalized coordinates, with
real simulated experiments wandering over it until the free boundaries stop
them. Background field at tau = 1 (the chart where the policy is
quasi-stationary in sd units); trajectories simulated with the true
posterior dynamics under the champion's policy.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.collections import LineCollection
from PIL import Image


from pinn.problems.three_arm import DimensionlessValueFunction, ValueFunction
from pinn.problems.two_arm import (
    DimensionlessValueFunction as TwoArmDimensionless,
    ValueFunction as TwoArmValue,
)
from pinn.problems.two_arm_drift import (
    DimensionlessValueFunction as DriftDimensionless,
    ValueFunction as DriftValue,
)

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
        DimensionlessValueFunction.load("data/three_arm.pt"), rho=1.0, sigma=1.0
    )
    image = field(value)
    trails, arms = simulate(value)
    shown = np.clip(image, 0, 1) ** 0.92

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(
        shown,
        origin="lower",
        extent=[-SPAN, SPAN, -SPAN, SPAN],
        interpolation="bilinear",
        aspect="auto",
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

        # Evolving thickness carries the arrow of time: each test is born as
        # a hairline and commits at full weight. One opaque color, so there
        # is no gradient for the eye to average and no compositing to bead.
        widths = np.linspace(0.35, 2.6, len(segments))
        ax.add_collection(
            LineCollection(
                segments, colors="white", linewidths=widths, capstyle="round"
            )
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

    # Supersample 2x and downsample with Lanczos: the hairline ends of the
    # tapers live below one output pixel, where direct rendering aliases.
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    Image.open(buffer).resize((2400, 1200), Image.LANCZOS).save("docs/hero.png")
    print("saved docs/hero.png")


def _draw_trail(ax, trail: list[tuple[float, float]]) -> np.ndarray:
    """
    The shared trail treatment: decimate, smooth, taper. Returns the final
    point so the caller can place the decision dot.
    """
    points = np.array(trail)[::10]
    points = np.vstack([points, trail[-1]])

    if len(points) > 4:
        kernel = np.ones(3) / 3.0
        for column in range(2):
            interior = np.convolve(points[:, column], kernel, "same")
            points[1:-1, column] = interior[1:-1]

    segments = np.stack([points[:-1], points[1:]], axis=1)
    widths = np.linspace(0.35, 2.6, len(segments))
    ax.add_collection(
        LineCollection(segments, colors="white", linewidths=widths, capstyle="round")
    )

    return points[-1]


def two_arm_hero() -> None:
    """
    The funnel of doubt: x is accumulated precision (time flows right), y is
    the belief about the treatment effect, the field is the policy (blue =
    all-in control, orange = all-in treatment), and each test rides the
    funnel until a wall commits it.
    """
    value = TwoArmValue(TwoArmDimensionless.load("data/two_arm.pt"), rho=1.0, sigma=1.0)

    torch.manual_seed(5)
    m = torch.zeros(PATHS)
    tau = torch.full((PATHS,), TAU0)
    active = torch.ones(PATHS, dtype=torch.bool)
    chosen = torch.full((PATHS,), -1, dtype=torch.long)
    trails: list[list[tuple[float, float]]] = [[] for _ in range(PATHS)]
    t = 0.0

    while t < HORIZON and bool(active.any()):
        for i in range(PATHS):
            if bool(active[i]):
                trails[i].append((float(tau[i]), float(m[i])))

        share = value.policy(m, tau)
        up = active & (share >= 1.0)
        down = active & (share <= 0.0)
        chosen[up] = 1
        chosen[down] = 0
        active = active & ~(up | down)

        rate = share * (1.0 - share)
        m = m + active * (rate.sqrt() / tau) * DT**0.5 * torch.randn(PATHS)
        tau = tau + active * rate * DT
        t += DT

    tau_max = 1.05 * max(trail[-1][0] for trail in trails if trail)
    mu_max = 1.10 * max(abs(point[1]) for trail in trails for point in trail)

    # Breathing room left of the birth point; floored above the decade where
    # the field is untrained.
    tau_left = max(0.1, TAU0 - 0.04 * (tau_max - TAU0))

    columns, rows = 760, 380
    tau_axis = torch.linspace(tau_left, tau_max, columns)
    mu_axis = torch.linspace(-mu_max, mu_max, rows)
    mu_grid, tau_grid = torch.meshgrid(mu_axis, tau_axis, indexing="ij")
    share_grid = value.policy(mu_grid.flatten(), tau_grid.flatten()).reshape(
        rows, columns
    )
    image = (
        share_grid.numpy()[..., None] * ANCHORS[1]
        + (1.0 - share_grid.numpy())[..., None] * ANCHORS[0]
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(
        np.clip(image, 0, 1) ** 0.92,
        origin="lower",
        extent=[tau_left, tau_max, -mu_max, mu_max],
        interpolation="bilinear",
        aspect="auto",
    )

    for trail, arm in zip(trails, chosen):
        if len(trail) < 3:
            continue
        end = _draw_trail(ax, trail)

        if arm >= 0:
            ax.plot(
                end[0],
                end[1],
                marker="o",
                markersize=13,
                color=DOT_COLORS[int(arm)],
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=5,
            )

    ax.set_xlim(tau_left, tau_max)
    ax.set_ylim(-mu_max, mu_max)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=600, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    Image.open(buffer).resize((2400, 1200), Image.LANCZOS).save("docs/hero2.png")
    print("saved docs/hero2.png")


def drift_hero() -> None:
    """
    The funnel of doubt with a wandering truth: same chart as two_arm_hero
    (x accumulated precision, y belief, field the policy), but the world
    drifts, so precision erodes at eta^2 tau^2. Exploring paths climb toward
    the drift ceiling instead of running right forever; a committed path buys
    nothing, slides LEFT as its knowledge rots, and the boundary re-opens it
    -- no decision is final, and the picture shows the cycle.
    """
    etahat = 0.7
    paths, horizon = 18, 8.0
    birth_tau = 0.55
    value = DriftValue(
        DriftDimensionless.load("data/two_arm_drift.pt"),
        rho=1.0,
        sigma=1.0,
        eta=etahat,
    )

    torch.manual_seed(3)
    m = torch.zeros(paths)
    tau = torch.full((paths,), birth_tau)
    trails: list[list[tuple[float, float]]] = [[] for _ in range(paths)]

    # A dot per committed EPISODE, placed where it began -- under drift a
    # commit is an event on the way, not a fate: the slide left that follows
    # is the commitment rotting until the boundary re-opens it. Episodes
    # shorter than the debounce are wall chatter, not decisions.
    debounce = int(0.4 / DT)
    entry: list[tuple[float, float, int] | None] = [None] * paths
    streak = [0] * paths
    commits: list[tuple[float, float, int]] = []
    t = 0.0

    while t < horizon:
        for i in range(paths):
            trails[i].append((float(tau[i]), float(m[i])))

        share = value.policy(m, tau)

        for i in range(paths):
            # Near-vertex counts as committed: the boundary chatters by
            # epsilon while a path rides the wall, and exact-vertex tests
            # would split one ride into confetti.
            if float(share[i]) >= 0.995 or float(share[i]) <= 0.005:
                if entry[i] is None:
                    entry[i] = (float(tau[i]), float(m[i]), int(float(share[i]) >= 0.5))
                    streak[i] = 0
                streak[i] += 1
            else:
                if entry[i] is not None and streak[i] >= debounce:
                    commits.append(entry[i])
                entry[i] = None

        rate = share * (1.0 - share)

        # Belief dynamics under drift: same innovation as the static funnel,
        # while precision gains the erosion. At a vertex rate = 0, so the mean
        # freezes and the path slides left until the boundary re-opens it.
        m = m + (rate.sqrt() / tau) * DT**0.5 * torch.randn(paths)
        tau = (tau + (rate - etahat**2 * tau**2) * DT).clamp_min(0.05)
        t += DT

    commits.extend(
        held for held, run in zip(entry, streak) if held is not None and run >= debounce
    )
    tau_max = 1.06 * max(point[0] for trail in trails for point in trail)
    mu_max = 1.10 * max(abs(point[1]) for trail in trails for point in trail)
    tau_left = 0.98 * min(point[0] for trail in trails for point in trail)

    columns, rows = 760, 380
    tau_axis = torch.linspace(tau_left, tau_max, columns)
    mu_axis = torch.linspace(-mu_max, mu_max, rows)
    mu_grid, tau_grid = torch.meshgrid(mu_axis, tau_axis, indexing="ij")
    share_grid = value.policy(mu_grid.flatten(), tau_grid.flatten()).reshape(
        rows, columns
    )
    image = (
        share_grid.numpy()[..., None] * ANCHORS[1]
        + (1.0 - share_grid.numpy())[..., None] * ANCHORS[0]
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(
        np.clip(image, 0, 1) ** 0.92,
        origin="lower",
        extent=[tau_left, tau_max, -mu_max, mu_max],
        interpolation="bilinear",
        aspect="auto",
    )

    for trail in trails:
        if len(trail) < 3:
            continue
        _draw_trail(ax, trail)

    for commit_tau, commit_m, arm in commits:
        ax.plot(
            commit_tau,
            commit_m,
            marker="o",
            markersize=11,
            color=DOT_COLORS[arm],
            markeredgecolor="white",
            markeredgewidth=1.1,
            zorder=5,
        )

    ax.set_xlim(tau_left, tau_max)
    ax.set_ylim(-mu_max, mu_max)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=600, transparent=True)
    plt.close(fig)
    buffer.seek(0)
    Image.open(buffer).resize((2400, 1200), Image.LANCZOS).save("docs/hero_drift.png")
    print("saved docs/hero_drift.png")


if __name__ == "__main__":
    main()
    two_arm_hero()
    drift_hero()
