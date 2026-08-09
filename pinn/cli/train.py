"""
`pinn train`: run the generic trainer on a problem.
"""

from __future__ import annotations

import click
import torch

from importlib import import_module
from pathlib import Path

from ..problems import PROBLEMS

from ..train import decay, train as run_training

# Batch 2048 is the defended floor at POWER = 2 (the p-mean's gradient rides
# ~13% of the batch; 1024 oscillates).
BATCH = 1024
DECAY_OVER = 100_000


@click.command()
@click.option(
    "--problem", type=click.Choice(PROBLEMS), required=True, help="Which problem."
)
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Checkpoint to continue from; make one with `pinn init`.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to save; defaults to --in, training the checkpoint in place.",
)
@click.option(
    "--lr",
    type=float,
    required=True,
    help=f"Initial learning rate, decayed over {DECAY_OVER} iterations.",
)
@click.option("--batch", type=int, default=BATCH, show_default=True)
def train(
    problem: str, in_path: Path, out_path: Path | None, lr: float, batch: int
) -> None:
    """
    Train (Ctrl-C to stop and save), printing the loss now and then.

    Every resume restarts the lr schedule at the hot end, so a short resumed
    run first SMEARS a polished checkpoint before improving it. Never judge a
    resume by its first prints.
    """
    out_path = out_path if out_path is not None else in_path
    module = import_module(f"pinn.problems.{problem}")
    state = torch.load(in_path)
    value = module.init_model(state=state)

    try:
        for _ in run_training(
            value, module.objective(batch=batch), lr=decay(lr, DECAY_OVER)
        ):
            pass
    except KeyboardInterrupt:
        click.echo("Interrupted by user")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.state_dict(), out_path)
    click.echo(f"saved {out_path}")
