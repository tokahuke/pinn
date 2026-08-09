"""
`pinn init`: create an untrained checkpoint.
"""

from __future__ import annotations

import click
import torch

from importlib import import_module
from pathlib import Path

from ..problems import PROBLEMS


@click.command(name="init")
@click.option(
    "--problem", type=click.Choice(PROBLEMS), required=True, help="Which problem."
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Where to write the new checkpoint.",
)
@click.option(
    "--topology",
    type=str,
    default=None,
    help='Fresh net shape, e.g. "32:512" or "64:64:64k16" (k = kink units).',
)
@click.option(
    "--from",
    "from_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Adapt an existing checkpoint instead of starting fresh.",
)
def init(
    problem: str, out_path: Path, topology: str | None, from_path: Path | None
) -> None:
    """
    Create an untrained checkpoint, fresh or adapted from another.

    At least one of --topology and --from. Given both, --topology is the target
    shape and --from the source adapted into it -- which is how a kink branch
    is grafted onto a trained smooth net. Reading the file is this command's
    job; the problem module is handed the state dict and decides what it means
    -- for two_arm_drift a two_arm checkpoint stitches in as the etahat = 0
    slice, for three_arm a kinked state keeps its kinks.
    """
    if topology is None and from_path is None:
        raise click.UsageError("pass at least one of --topology and --from")

    module = import_module(f"pinn.problems.{problem}")
    state = torch.load(from_path) if from_path is not None else None

    try:
        value = module.init_model(state=state, topology=topology)
    except ValueError as bad:
        raise click.UsageError(str(bad)) from bad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.state_dict(), out_path)
    source = " ".join(
        part
        for part in (
            f"from {from_path}" if from_path is not None else "",
            f"topology {topology}" if topology is not None else "",
        )
        if part
    )
    click.echo(f"initialised {problem} {source} -> {out_path}")
