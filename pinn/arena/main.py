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

from importlib import import_module
from inspect import isabstract
from pathlib import Path
from typing import Iterator

from .harness import Params, Policy, Run, Runner, Study

# Reps per batched chunk: caps the noise buffer (~120 MB at the production horizon)
# and keeps the progress line moving.
CHUNK = 4096


def concrete_policies(cls: type[Policy] = Policy) -> Iterator[type[Policy]]:
    """
    Every instantiable Policy at any depth. __subclasses__ only sees direct
    children, and the intermediates (the Bayesian bases) are abstract. Only
    the chosen problem module is imported, so only its zoo is registered.
    """
    for sub in cls.__subclasses__():
        yield from concrete_policies(sub)

        if not isabstract(sub):
            yield sub


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("runs", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--problem",
    type=click.Choice(["two_arm", "two_arm_drift", "three_arm"]),
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
    # Vectorized over reps, chunked to bound memory. Seeded by REP: every
    # policy sees the same per-rep noise stream (until allocations diverge),
    # so cross-policy comparisons are paired -- and a rep's stream is
    # independent of its chunk, so chunking does not move any number.
    classes = list(concrete_policies())
    results: list[Run] = []
    total = size * len(classes)

    for cls in classes:
        for chunk_start in range(0, size, CHUNK):
            seeds = list(range(chunk_start, min(chunk_start + CHUNK, size)))
            runner = Runner(params, seeds, device)
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
    wrong-commit share, commit share, and the median commit epoch.

    CAVEAT under drift: `wrong%` scores the committed arm against `delta`,
    which is the effect at epoch 0. When eta > 0 the truth moves afterwards, so
    this reads "committed against the arm that was best when the run started",
    not "against the arm that was best while committed". The honest drift
    metric is the regret column, which is measured per epoch against the
    moving oracle.
    """
    study: Study = pickle.loads(runs.read_bytes())
    by_policy: dict[str, list[Run]] = {}

    for run in study.runs:
        by_policy.setdefault(run.policy, []).append(run)

    def mean_ci(values: list[float]) -> tuple[float, float]:
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
        # committed_at, not epochs: the runner plays the full horizon now, so
        # epochs is the horizon for every run and says nothing about commitment.
        # Old studies predate the field; their runs read as None and are
        # dropped from the median, like precision_time below.
        epochs = sorted(r.committed_at for r in committed if r.committed_at is not None)
        median = epochs[len(epochs) // 2] if epochs else None
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


def _paired(by_policy: dict[str, list[Run]], mean_ci) -> None:
    """
    The same comparison, paired by rep.

    Every policy plays the SAME drawn effects and the same noise, so the
    difference in regret on one rep cancels the environment -- which is nearly
    all of the variance. Comparing the unpaired means throws that away and
    reports a confidence interval dominated by how hard the draws were, not by
    how the policies differ.

    Prints what the pairing costs to buy: the reps needed for a 2-sigma read on
    a 2% effect, which is how the NEXT sweep should be sized. Sizing from the
    unpaired spread is how a 50k sweep gets run to resolve something a few
    thousand paired reps would have settled.
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
            print(f"{name:<22} {'reps do not align -- not paired':>50}")
            continue

        gaps = [a.regret - b.regret for a, b in zip(runs_, best_runs)]
        mean, ci = mean_ci(gaps)
        _, loose = mean_ci([r.regret for r in runs_])
        deviation = ci / 1.96 * len(gaps) ** 0.5
        target = 0.02 * sum(r.regret for r in best_runs) / len(best_runs)
        needed = (2.0 * deviation / target) ** 2

        print(f"{name:<22} {mean:>12.1f} {ci:>9.1f} {loose:>12.1f} {needed:>12,.0f}")


if __name__ == "__main__":
    cli()
