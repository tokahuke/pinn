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

HIDDEN = [512, 128]


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

    if in_path is None:
        value = module.ValueFunction(module.ExplorationPremium(HIDDEN))
    else:
        state = torch.load(in_path)
        hidden = [w.shape[0] for k, w in state.items() if k.endswith(".weight")][:-1]
        value = module.ValueFunction(module.ExplorationPremium(hidden))
        value.load_state_dict(state)

    try:
        for _ in itertools.islice(
            train(value, module.objective(batch=1024), lr=decay(1e-4, 10_000)), None
        ):
            pass
    except KeyboardInterrupt:
        print("Interrupted by user")

    out_path.parent.mkdir(exist_ok=True)
    torch.save(value.state_dict(), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
