"""
The `jobq` command: run trainings on an ephemeral RunPod pod.
"""

from __future__ import annotations

import click

from .cp import cp
from .down import down
from .run import run
from .up import up


@click.group()
def cli() -> None:
    """
    Poor man's job queue.
    """


cli.add_command(up)
cli.add_command(cp)
cli.add_command(run)
cli.add_command(down)
