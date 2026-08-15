"""
`jobq down`: fetch the checkpoints back, then destroy the pod.
"""

from __future__ import annotations

import click

from pathlib import Path

from .daemon import Daemon
from .pod import Pod

# NOT data/ itself: the champions live there under the same names a pod would
# write, and a fetch that clobbers two_arm.pt is unrecoverable.
LANDING = Path("data/pod")


def fetch(pod: Pod) -> str:
    """Whatever is on the pod, into data/pod. Content-blind, like backup."""
    before = set(LANDING.rglob("*"))
    # Not fatal: a pod whose ssh has died still has to be destroyable, or it
    # bills forever.
    moved, code = pod.fetch(local=str(LANDING))
    arrived = len(set(LANDING.rglob("*")) - before)

    if code != 0:
        return f"fetch FAILED (rsync {code}) -- {arrived} new files in {LANDING}"

    return f"{moved} files synced, {arrived} new, into {LANDING}"


@click.command()
@click.option("--pod", "name", default="pinn", show_default=True)
@click.option("--yes", is_flag=True, help="Skip the confirmation.")
@click.option("--no-fetch", is_flag=True, help="Destroy without pulling checkpoints.")
def down(name: str, yes: bool, no_fetch: bool) -> None:
    """
    Destroy the pod. Checkpoints land in data/pod first.

    Destroying is the only thing that stops the billing -- a stopped pod still
    costs.
    """
    pod = Pod.find(name, resolve=False)

    if pod is None:
        click.echo(f"no pod named {name}")
        stopped = Daemon(name).stop()

        if stopped is not None:
            click.echo(f"  stopped an orphaned backup daemon (pid {stopped})")

        return

    click.echo(f"{name} ({pod.id}) at ${pod.cost}/hr")

    if not no_fetch:
        try:
            click.echo(f"  {fetch(Pod.require(name))}")
        except click.ClickException as unreachable:
            click.echo(f"  cannot fetch: {unreachable.format_message()}")

    if not yes:
        click.confirm(f"destroy {name}?", abort=True)

    # Stopped BEFORE the pod goes, so it is not left retrying against a
    # corpse -- it reconnects with backoff and would never give up on its own.
    stopped = Daemon(name).stop()
    click.echo(
        f"  backup daemon stopped (pid {stopped})"
        if stopped is not None
        else "  no backup daemon was running"
    )
    pod.destroy()
    click.echo(
        f"destroyed {name}"
        if Pod.find(name, resolve=False) is None
        else f"{name} STILL PRESENT"
    )
