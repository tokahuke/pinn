"""
README visuals: the three-arm policy atlas (allocation as barycentric colour over the
mean plane, information growing left to right) and the two-arm arena result (regret
and information spend relative to Thompson sampling). Palette: dataviz default
categorical slots 1-3 (all-pairs validated).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

from collections.abc import Callable
from matplotlib.patches import Patch
from pathlib import Path
from pinn.problems.three_arm.model import DimensionlessValueFunction, ValueFunction

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
"""Categorical slots 1-4, one per entity, held across every chart here."""

INK, MUTED = "#1a1f26", "#5b6472"
"""Text and axis-furniture greys."""

ANCHORS = np.array(
    [
        [42 / 255, 120 / 255, 214 / 255],  # arm a (control): blue
        [235 / 255, 104 / 255, 52 / 255],  # arm b: orange
        [27 / 255, 175 / 255, 122 / 255],  # arm c: aqua
    ]
)
"""One colour per arm; the allocation mixes them into the atlas field."""


def atlas() -> None:
    """Render the three-arm policy atlas to docs/policy_atlas.png."""
    value = ValueFunction(
        DimensionlessValueFunction.load("data/three_arm.pt"), rho=1.0, sigma=1.0
    )
    taus = [0.03, 0.3, 1.0, 3.0]
    n = 220

    fig, axes = plt.subplots(1, len(taus), figsize=(16, 4.6))

    for ax, tau in zip(axes, taus):
        deviation = (0.75 * tau) ** -0.5
        axis = torch.linspace(-3.0 * deviation, 3.0 * deviation, n)
        m_b, m_c = torch.meshgrid(axis, axis, indexing="xy")
        alpha = value.policy(
            m_b.flatten(),
            m_c.flatten(),
            torch.full((n * n,), tau),
            torch.full((n * n,), -tau / 2.0),
            torch.full((n * n,), tau),
        )
        image = (alpha.numpy() @ ANCHORS).reshape(n, n, 3)
        ax.imshow(
            image, origin="lower", extent=[-3, 3, -3, 3], interpolation="bilinear"
        )
        ax.set_title(f"information = {tau:g}", fontsize=12, color=INK)
        ax.set_xlabel("mean of B - control  (posterior sd)", fontsize=9, color=MUTED)
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        ax.tick_params(colors=MUTED, labelsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("mean of C - control  (posterior sd)", fontsize=9, color=MUTED)
    fig.suptitle(
        "The learned allocation policy: color = traffic split over (control, B, C)",
        fontsize=15,
        color=INK,
        y=1.02,
    )
    fig.legend(
        handles=[
            Patch(color=BLUE, label="all-in on control"),
            Patch(color=ORANGE, label="all-in on B"),
            Patch(color=AQUA, label="all-in on C"),
            Patch(color="#5e6b6e", label="blends = exploration"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(
        "docs/policy_atlas.png", dpi=150, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)
    print("saved docs/policy_atlas.png")


def arena(
    study_path: str = "data/frontier4k.pkl",
    out_path: str = "docs/arena_two_arm.png",
    suptitle: str = "A/B test policy arena",
    entities: list[tuple[str, str, str]] | None = None,
    caption: str | None = None,
    xlim: float = 2.6,
) -> None:
    """
    Render one arena study as paired regret and evidence bars, each normalized to
    Thompson sampling.
    """
    # Colour follows the entity across charts: PINN blue, Thompson orange,
    # explore-then-commit aqua, the significance-testing family yellow.
    entities = entities or [
        ("Pinn", "PINN (this repo)", BLUE),
        ("ProbabilityMatching", "Thompson sampling", ORANGE),
        ("ExploreThenCommit", "explore-then-commit", AQUA),
        ("ZTest", "z-test at 5%", YELLOW),
    ]
    study = pickle.loads(Path(study_path).read_bytes())
    by_policy: dict[str, list] = {}

    for run in study.runs:
        by_policy.setdefault(run.policy, []).append(run)

    order = [name for name, _, _ in entities]
    labels = [label for _, label, _ in entities]
    colors = [color for _, _, color in entities]

    def stats(name: str, field: Callable[[object], float]) -> tuple[float, float]:
        """One policy's mean of `field` over its runs, with a 95% interval."""
        values = np.array([field(r) for r in by_policy[name]])

        return values.mean(), 1.96 * values.std(ddof=1) / len(values) ** 0.5

    ts_regret, _ = stats("ProbabilityMatching", lambda r: r.regret)
    ts_info, _ = stats(
        "ProbabilityMatching", lambda r: getattr(r, "precision_time", 0.0)
    )

    # Every caption number is **computed** from this study: the 2026-08-13 redraw still
    # claimed "4,000 tests" and "less than half the evidence" against a 6,000-rep run
    # measuring 0.57. Interpretation belongs in the README, not here.
    if caption is None:
        pinn_regret, _ = stats("Pinn", lambda r: r.regret)
        pinn_info, _ = stats("Pinn", lambda r: getattr(r, "precision_time", 0.0))
        caption = (
            f"Each strategy plays the same {len(by_policy['Pinn'])} random tests. "
            "Regret: profit left on the table vs an oracle that\n"
            "picks the winner from day one (lower is better). Evidence: total "
            "measurement bought: an even split measures at\n"
            "full power, sending everyone to one arm measures nothing, and the sum "
            "counts perfect-split epochs of data.\n"
            f"The PINN loses {1 - pinn_regret / ts_regret:.0%} less than Thompson "
            f"sampling on {pinn_info / ts_info:.0%} of the evidence."
        )

    fig, (ax_regret, ax_info) = plt.subplots(1, 2, figsize=(12, 3.6))

    for ax, field, base, title in [
        (
            ax_regret,
            lambda r: r.regret,
            ts_regret,
            "discounted regret  (Thompson = 1.00, lower is better)",
        ),
        (
            ax_info,
            lambda r: getattr(r, "precision_time", 0.0),
            ts_info,
            "evidence bought, in perfect-split epochs  (Thompson = 1.00)",
        ),
    ]:
        means, cis = zip(*(stats(name, field) for name in order))
        y = np.arange(len(order))[::-1]
        ax.barh(
            y,
            [m / base for m in means],
            xerr=[c / base for c in cis],
            color=colors,
            height=0.62,
            error_kw={"ecolor": INK, "capsize": 3, "elinewidth": 1},
        )

        for position, mean, ci in zip(y, means, cis):
            ax.text(
                (mean + ci) / base + 0.07,
                position,
                f"{mean / base:.2f}",
                va="center",
                fontsize=10,
                color=INK,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlim(0, xlim)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(suptitle, fontsize=16, color=INK, x=0.02, ha="left")
    fig.text(
        0.02, -0.01, caption, fontsize=8, color="#8a93a1", va="top", linespacing=1.45
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


def arena_three() -> None:
    """Render the three-arm arena to docs/arena_three_arm.png."""
    arena(
        study_path="data/frontier3a.pkl",
        out_path="docs/arena_three_arm.png",
        suptitle="A/B/C test policy arena",
        entities=[
            ("Pinn", "PINN (this repo)", BLUE),
            ("ProbabilityMatching", "Thompson sampling", ORANGE),
            ("Elimination", "elimination at 5%\n(generalized z-test)", YELLOW),
            ("ExploreThenCommit", "explore-then-commit", AQUA),
        ],
        xlim=2.8,
    )


def arena_drift() -> None:
    """
    The drifting-world arena: the fixed drift entrants (drift_grid_fixed.pkl) beside
    the static-net transplant from the original grid, restricted to the reps both
    sweeps share so every bar plays the same tests. Mixing the two pickles is licensed
    by the worlds being identical: paired TS old-vs-new differences are zero.
    """
    fixed = pickle.loads(Path("data/drift_grid_fixed.pkl").read_bytes())
    original = pickle.loads(Path("data/drift_grid.pkl").read_bytes())

    def drifting(study: object, wanted: str) -> dict[tuple, object]:
        """One study's runs of a named drifting policy, keyed by the world they played."""
        return {
            tuple(r.delta): r for r in study.runs if r.policy == f"drifting/{wanted}"
        }

    kept = {
        name: drifting(fixed, name)
        for name in ("Pinn-aware", "Pinn-blind", "TS-aware", "TS-blind")
    }
    kept["Pinn-static"] = drifting(original, "Pinn-twoarm")
    shared = set.intersection(*(set(runs) for runs in kept.values()))

    merged = pickle.loads(Path("data/drift_grid_fixed.pkl").read_bytes())
    merged.runs = []

    for name, runs in kept.items():
        for key in shared:
            run = runs[key]
            # arena() normalizes by the literal name "ProbabilityMatching".
            run.policy = "ProbabilityMatching" if name == "TS-blind" else name
            merged.runs.append(run)

    Path("data/drift_grid_chart.pkl").write_bytes(pickle.dumps(merged))

    arena(
        study_path="data/drift_grid_chart.pkl",
        out_path="docs/arena_two_arm_drift.png",
        suptitle="A/B test policy arena, drifting world",
        entities=[
            ("Pinn-static", "PINN, static net\n(transplanted unchanged)", BLUE),
            ("Pinn-blind", "PINN, drift net\n(told there is no drift)", "#7aa8e0"),
            ("Pinn-aware", "PINN, drift net\n(told the true drift)", "#164a8a"),
            ("ProbabilityMatching", "Thompson sampling", ORANGE),
            ("TS-aware", "Thompson, drift-aware\n(forgetful posterior)", YELLOW),
        ],
        xlim=2.2,
    )


if __name__ == "__main__":
    atlas()
    arena()
    arena_three()
    arena_drift()
