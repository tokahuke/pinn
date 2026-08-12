"""
`jobq sync`: push the working tree to the pod.
"""

from __future__ import annotations

import click

from .pod import find, ssh_info, sync_repo


@click.command()
@click.option("--name", default="pinn", show_default=True, help="Pod to sync to.")
def sync(name: str) -> None:
    """
    Rsync the repo onto the pod, and nothing else.

    `jobq up` also syncs, but it reinstalls and restarts the idle timer, which
    is not what you want while a job is running. The install is editable, so a
    synced file is live -- no reinstall after a code change.
    """
    pod = find(name)

    if pod is None:
        raise click.ClickException(f"no pod named {name}; `jobq up` first")

    detail = ssh_info(pod["id"])
    sync_repo(detail["ip"], detail["port"])
    click.echo(f"synced to {name}")
