"""
The `pinn` command: one module per command, assembled here.

`init` is the only command that creates a checkpoint; `train` only continues
one.
"""

from __future__ import annotations

import click

from .init import init
from .plot import main as plot
from .plot_drift import main as plot_drift
from .train import train
from .validate import main as validate


@click.group()
def cli() -> None:
    """
    Physics-informed nets for Bayesian allocation.
    """


cli.add_command(init)
cli.add_command(train)
cli.add_command(plot, name="plot")
cli.add_command(plot_drift, name="plot-drift")
cli.add_command(validate, name="validate")


if __name__ == "__main__":
    cli()
