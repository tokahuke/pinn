"""
`jobq push`: send the working tree to the pod.
"""

from __future__ import annotations

import click

from .pod import Pod


@click.command()
@click.option(
    "--pod", "name", default="pinn", show_default=True, help="Pod to push to."
)
def push(name: str) -> None:
    """
    Send the repo to the pod, and nothing else.

    `jobq up` also pushes, but it reinstalls and restarts the idle timer,
    which is not what you want while a job is running. The install is
    editable, so a pushed file is live -- no reinstall after a code change.
    """
    pod = Pod.require(name)
    pod.send_repo()
    click.echo(f"pushed to {name}")
