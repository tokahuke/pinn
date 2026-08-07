"""
Entry point: train the PINN for one of the problems in pinn/problems/.
"""

from __future__ import annotations

import click
import importlib
import itertools
import torch

from pathlib import Path

from pinn.train import decay, train

HIDDEN = [
    64,
    64,
    64,
]
KINKS = 16


@click.command()
@click.option(
    "--problem",
    type=click.Choice(["two_arm", "three_arm"]),
    default="two_arm",
    help="Which problem module to train.",
)
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Checkpoint to continue from; a fresh model when omitted.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save; defaults to --in, or data/<problem>.pt for a fresh model.",
)
def main(problem: str, in_path: Path | None, out_path: Path | None) -> None:
    """
    Train (Ctrl-C to stop and save), printing the loss now and then.
    """
    module = importlib.import_module(f"pinn.problems.{problem}")
    out_path = out_path if out_path is not None else in_path

    if out_path is None:
        out_path = Path("data") / f"{problem}.pt"

    extra = {"kinks": KINKS} if problem == "three_arm" else {}

    if in_path is None:
        value = module.DimensionlessValueFunction(
            module.ExplorationPremium(HIDDEN, **extra)
        )
    else:
        value = module.DimensionlessValueFunction.load(in_path, **extra)

    try:
        for _ in itertools.islice(
            train(value, module.objective(batch=2048), lr=decay(3e-4, 100_000)), None
        ):
            pass
    except KeyboardInterrupt:
        print("Interrupted by user")

    out_path.parent.mkdir(exist_ok=True)
    torch.save(value.state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
