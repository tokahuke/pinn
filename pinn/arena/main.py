"""
The arena's CLI: sweep every concrete policy of the chosen problem against a
drawn effect, pickle the Study. Mounted by poetry as `arena` (pyproject
[project.scripts]): `poetry run arena simulate ...`.
"""

from __future__ import annotations

import click
import pickle
import sys
import torch

from collections.abc import Callable, Iterator
from importlib import import_module
from inspect import isabstract
from pathlib import Path

from .harness import Params, Policy, Run, Runner, Study

CHUNK = 4096
"""Reps per batched chunk, which is what caps the noise buffer."""


def concrete_policies(cls: type[Policy] = Policy) -> Iterator[type[Policy]]:
    """
    Every instantiable Policy at any depth, the intermediates being abstract.

    The caller **must** filter by module: a drifting zoo imports its static sibling, so
    reaching either registers both and an unfiltered sweep runs two classes named Pinn.
    """
    # Deduped: a drifting zoo's policy inherits from both the static policy and the
    # drifting filter, so the recursion reaches it down two paths and would otherwise
    # yield it twice.
    seen = []

    for sub in cls.__subclasses__():
        for found in concrete_policies(sub):
            if found not in seen:
                seen.append(found)
                yield found

        if isabstract(sub) is False and sub not in seen:
            seen.append(sub)
            yield sub


@click.group()
def cli() -> None:
    """Simulate a policy sweep, then report it."""


@cli.command()
@click.argument("runs", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--problem",
    type=click.Choice(["two_arm", "two_arm_drift", "three_arm", "three_arm_drift"]),
    default="two_arm",
    help="Which problem's zoo to sweep.",
)
@click.option("--rho", type=float, required=True)
@click.option("--horizon", type=int, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--effect", type=float, required=True)
@click.option("--effect-std", type=float, default=0.0, show_default=True)
@click.option(
    "--eta",
    type=float,
    default=0.0,
    show_default=True,
    help="Drift volatility of the true effect per epoch (two_arm_drift only).",
)
@click.option("--size", type=int, required=True)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Torch CPU threads; default all cores. Fewer keeps the laptop usable.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device for the batched simulation.",
)
def simulate(
    runs: Path,
    problem: str,
    rho: float,
    horizon: int,
    sigma: float,
    effect: float,
    effect_std: float,
    eta: float,
    size: int,
    workers: int | None,
    device: str,
) -> None:
    """
    Sweep every policy of the chosen problem against one drawn environment, and pickle
    the Study for `analyze`.
    """
    if workers is not None:
        torch.set_num_threads(workers)

    module = import_module(f"pinn.arena.{problem}")
    params = Params(
        rho=rho,
        horizon=horizon,
        sigma=sigma,
        effect=effect,
        effect_std=effect_std,
        size=size,
        eta=eta,
    )
    # Seeded by *rep*, so cross-policy comparisons are paired and a rep's stream is
    # independent of its chunk. A drifting zoo eats more of that stream, which is why
    # it declares its own appetite (kb/arena_results.md, harness invariants).
    appetite = getattr(module, "DRAWS_PER_EPOCH", 2)
    classes = [c for c in concrete_policies() if c.__module__ == module.__name__]
    results: list[Run] = []
    total = size * len(classes)

    for cls in classes:
        for chunk_start in range(0, size, CHUNK):
            seeds = list(range(chunk_start, min(chunk_start + CHUNK, size)))
            runner = Runner(params, seeds, device, appetite)
            policy = cls.init(params, len(seeds), device)
            batch = runner.run(module, policy, module.draw_effect(runner))
            results.extend(batch.runs())
            print(f"{len(results)}/{total}", file=sys.stderr, flush=True)

    runs.write_bytes(pickle.dumps(Study(params=params, runs=results)))


@cli.command()
@click.argument("runs", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyze(runs: Path) -> None:
    """
    The report table: per policy, mean regret with 95% CI, ratio vs the best,
    wrong-commit share, commit share, and the median commit epoch. Under drift read
    regret, not `wrong%` (kb/arena_results.md, reading the report).
    """
    study: Study = pickle.loads(runs.read_bytes())
    by_policy: dict[str, list[Run]] = {}

    for run in study.runs:
        by_policy.setdefault(run.policy, []).append(run)

    def mean_ci(values: list[float]) -> tuple[float, float]:
        """The mean and its 95% half-width, normal approximation."""
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)

        return mean, 1.96 * (variance / len(values)) ** 0.5

    best = min(mean_ci([r.regret for r in runs_])[0] for runs_ in by_policy.values())

    print(study.params)
    print(
        f"{'policy':<22} {'regret':>9} {'95% CI':>8} {'vs best':>8}"
        f" {'wrong%':>7} {'commit%':>8} {'median epoch':>13} {'precision time':>15}"
    )

    for name, runs_ in sorted(
        by_policy.items(), key=lambda item: mean_ci([r.regret for r in item[1]])[0]
    ):
        mean, ci = mean_ci([r.regret for r in runs_])
        committed = [r for r in runs_ if r.committed is not None]
        wrong = [r for r in committed if r.delta[r.committed] < max(r.delta)]
        # committed_at, not epochs: the runner plays the full horizon, so epochs says
        # nothing about commitment. Studies predating the field read as None and drop
        # out of the median, like precision_time below.
        epochs = sorted(r.committed_at for r in committed if r.committed_at is not None)
        median = epochs[len(epochs) // 2] if len(epochs) > 0 else None
        # Old studies predate the field; they read as 0.
        info, info_ci = mean_ci([getattr(r, "precision_time", 0.0) for r in runs_])
        print(
            f"{name:<22} {mean:>9.1f} {ci:>8.1f} {mean / best:>8.2f}"
            f" {100 * len(wrong) / len(runs_):>6.1f}%"
            f" {100 * len(committed) / len(runs_):>7.1f}%"
            f" {median if median is not None else 'never':>13}"
            f" {info:>9.1f} +/-{info_ci:<4.1f}"
        )

    _paired(by_policy, mean_ci)


def _paired(
    by_policy: dict[str, list[Run]],
    mean_ci: Callable[[list[float]], tuple[float, float]],
) -> None:
    """
    The same comparison, paired by rep, which is the one to read: the per-rep
    difference cancels the environment, and that is nearly all of the variance. Also
    prints the reps needed for a 2-sigma read on a 2% effect, which is how the *next*
    sweep should be sized (kb/arena_results.md, reading the report).
    """
    ranked = sorted(
        by_policy.items(), key=lambda item: mean_ci([r.regret for r in item[1]])[0]
    )
    best_name, best_runs = ranked[0]

    print(f"\npaired against {best_name}, per rep (same effects, same noise)")
    print(
        f"{'policy':<22} {'difference':>12} {'95% CI':>9} {'unpaired CI':>12} {'reps for 2%':>12}"
    )

    for name, runs_ in ranked[1:]:
        # Identical draws are the whole premise; if the reps do not line up,
        # say so rather than quietly differencing unrelated runs.
        if len(runs_) != len(best_runs) or any(
            a.delta != b.delta for a, b in zip(runs_, best_runs)
        ):
            print(f"{name:<22} {'reps do not align, not paired':>50}")
            continue

        gaps = [a.regret - b.regret for a, b in zip(runs_, best_runs)]
        mean, ci = mean_ci(gaps)
        _, loose = mean_ci([r.regret for r in runs_])
        deviation = ci / 1.96 * len(gaps) ** 0.5
        target = 0.02 * sum(r.regret for r in best_runs) / len(best_runs)
        needed = (2.0 * deviation / target) ** 2

        print(f"{name:<22} {mean:>12.1f} {ci:>9.1f} {loose:>12.1f} {needed:>12.0f}")


if __name__ == "__main__":
    cli()
