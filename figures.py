"""
README visuals: the three-arm policy atlas (allocation as barycentric color
over the mean plane, information growing left to right) and the two-arm arena
result (regret and information spend relative to Thompson sampling).
Palette: dataviz default categorical slots 1-3 (all-pairs validated).
"""

from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


from matplotlib.patches import Patch

from pinn.problems.three_arm import DimensionlessValueFunction, ValueFunction

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#1a1f26", "#5b6472"

ANCHORS = np.array(
    [
        [42 / 255, 120 / 255, 214 / 255],  # arm a (control) - blue
        [235 / 255, 104 / 255, 52 / 255],  # arm b - orange
        [27 / 255, 175 / 255, 122 / 255],  # arm c - aqua
    ]
)


def atlas() -> None:
    value = ValueFunction(
        DimensionlessValueFunction.load("data/value_3a_64:64:64.pt"), rho=1.0, sigma=1.0
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
    # Color follows the entity across charts: PINN blue, Thompson orange,
    # explore-then-commit aqua, the significance-testing family yellow.
    entities = entities or [
        ("Pinn", "PINN (this repo)", BLUE),
        ("ProbabilityMatching", "Thompson sampling", ORANGE),
        ("ExploreThenCommit", "explore-then-commit", AQUA),
        ("ZTest", "z-test at 5%", YELLOW),
    ]
    caption = caption or (
        "Each strategy plays the same 4,000 random A/B tests. Regret: profit left on the table vs an oracle that picks the\n"
        "winner from day one (lower is better). Evidence: total measurement bought - an even split measures at full power,\n"
        "sending everyone to one arm measures nothing, and the sum counts perfect-split epochs' worth of data collected.\n"
        "The PINN loses 20% less than Thompson sampling on less than half the evidence - it knows when measuring stops paying."
    )
    study = pickle.loads(open(study_path, "rb").read())
    by_policy: dict[str, list] = {}

    for run in study.runs:
        by_policy.setdefault(run.policy, []).append(run)

    order = [name for name, _, _ in entities]
    labels = [label for _, label, _ in entities]
    colors = [color for _, _, color in entities]

    def stats(name: str, field) -> tuple[float, float]:
        values = np.array([field(r) for r in by_policy[name]])

        return values.mean(), 1.96 * values.std(ddof=1) / len(values) ** 0.5

    ts_regret, _ = stats("ProbabilityMatching", lambda r: r.regret)
    ts_info, _ = stats(
        "ProbabilityMatching", lambda r: getattr(r, "precision_time", 0.0)
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
        caption=(
            "Each strategy plays the same 4,000 random A/B/C tests (two challengers vs a control). Regret: profit left on\n"
            "the table vs an oracle that picks the winner from day one (lower is better). Evidence: total measurement\n"
            "bought, in perfect-thirds epochs. The PINN loses 22% less than Thompson sampling on 38% of the evidence -\n"
            "the margin over every other policy grows with the number of arms."
        ),
        xlim=2.8,
    )


if __name__ == "__main__":
    atlas()
    arena()
    arena_three()
