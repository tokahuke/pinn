"""
The arena's CLI: sweep every concrete policy of the chosen problem against a
drawn effect, pickle the Study. Mounted by poetry as `arena` (pyproject
[project.scripts]): `poetry run arena simulate ...`.
"""

from __future__ import annotations

import click
import pickle

import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from inspect import isabstract
from pathlib import Path
from typing import Iterator

from .harness import Params, Policy, Run, Runner, Study


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


def run_job(job: tuple[int, str, Policy]) -> Run:
    """
    One simulation in a worker process. Top level so it pickles; the problem
    travels by module name for the same reason.
    """
    import torch

    # One intra-op thread per worker: the pool already fills the cores, and
    # torch's default threading on top oversubscribes them (observed: laptop
    # as lap-grill).
    torch.set_num_threads(1)

    seed, problem_name, policy = job
    problem = import_module(problem_name)
    runner = Runner(policy.params, seed=seed)
    result = runner.run(problem, policy, problem.draw_effect(runner))
    result.policy = type(policy).__name__

    return result


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("runs", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--problem",
    type=click.Choice(["two_arm", "three_arm"]),
    default="two_arm",
    help="Which problem's zoo to sweep.",
)
@click.option("--rho", type=float, required=True)
@click.option("--horizon", type=int, required=True)
@click.option("--sigma", type=float, required=True)
@click.option("--effect", type=float, required=True)
@click.option("--effect-std", type=float, default=0.0, show_default=True)
@click.option("--size", type=int, required=True)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Pool size; default all cores. Fewer keeps the laptop usable.",
)
def simulate(
    runs: Path,
    problem: str,
    rho: float,
    horizon: int,
    sigma: float,
    effect: float,
    effect_std: float,
    size: int,
    workers: int | None,
) -> None:
    module = import_module(f"pinn.arena.{problem}")
    params = Params(
        rho=rho,
        horizon=horizon,
        sigma=sigma,
        effect=effect,
        effect_std=effect_std,
        size=size,
    )
    # Interleaved, not grouped by policy: a committing policy stops after tens of
    # epochs where ProbabilityMatching runs all of them, so grouping would hand whole
    # chunks of all-slow jobs to one worker. Seeded by REP, not by job: every
    # policy in a rep sees the same effect draw and noise stream (until
    # allocations diverge), so cross-policy comparisons are paired.
    jobs = [
        (rep, module.__name__, cls.init(params))
        for rep in range(size)
        for cls in concrete_policies()
    ]

    # Unordered collection (imap_unordered, executor-style): results are
    # grouped by Run.policy at analyze time, so completion order is fine and
    # the progress line reflects actual work done, not the slowest prefix.
    step = max(1, len(jobs) // 50)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        results = []

        for future in as_completed(futures):
            results.append(future.result())

            if len(results) % step == 0 or len(results) == len(jobs):
                print(f"{len(results)}/{len(jobs)}", file=sys.stderr, flush=True)

    runs.write_bytes(pickle.dumps(Study(params=params, runs=results)))


@cli.command()
@click.argument("runs", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyze(runs: Path) -> None:
    """
    The report table: per policy, mean regret with 95% CI, ratio vs the best,
    wrong-commit share, commit share, and the median commit epoch.
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
        epochs = sorted(r.epochs for r in committed)
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


if __name__ == "__main__":
    cli()
