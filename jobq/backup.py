"""`jobq backup`: keep a pod directory backed up locally as it changes."""

from __future__ import annotations

import click
import time

from .pod import OURS, Pod

BACKOFF: tuple[int, ...] = (5, 10, 20, 40, 60)
"""
Reconnect delays. A pod restart changes its ssh port, so every attempt re-resolves
the pod rather than reusing the old address.
"""


@click.command()
@click.option(
    "--pod", "name", default="pinn", show_default=True, help="Pod to back up."
)
@click.option("--remote", default="/workspace", show_default=True)
@click.option("--to", "local", default="data/pod", show_default=True)
@click.option(
    "--exclude",
    multiple=True,
    default=OURS,
    show_default=True,
    help="Repeatable. Defaults to what jobq itself put there.",
)
@click.option(
    "--settle",
    default=15,
    show_default=True,
    help="Seconds to coalesce events over. A busy directory emits dozens per "
    "second and they all move the same bytes.",
)
def backup(
    name: str, remote: str, local: str, exclude: tuple[str, ...], settle: int
) -> None:
    """
    Back REMOTE up into --to continuously, event-driven rather than polled. Ctrl-C to
    stop; a dead connection ends it loudly, since a backup you believe in but that is
    not running is worse than none. It accumulates rather than mirrors, and does not
    keep the pod alive: kb/jobq.md, "Backups accumulate".
    """
    attempt = 0

    while True:
        try:
            pod = Pod.require(name)
            moved, _ = pod.fetch(remote, local, exclude)
            click.echo(
                f"backing up {name} {pod.address}:{pod.port}:{remote} -> {local}"
            )
            click.echo(f"  initial sync: {moved} files")
            attempt = 0

            for events in pod.changes(remote, exclude, settle):
                moved, code = pod.fetch(remote, local, exclude)
                stamp = time.strftime("%H:%M:%S")

                if code != 0:
                    click.echo(f"  {stamp}  SYNC FAILED (rsync {code})")
                else:
                    click.echo(f"  {stamp}  {events} events -> {moved} files")
        except KeyboardInterrupt:
            click.echo("\nstopped")

            return
        except click.ClickException as broke:
            # A blip, a pod restart (which moves the ssh port), or a pod that is gone
            # for good are indistinguishable from here, so keep trying. The daemon
            # costs nothing while it waits and a pod that returns gets backed up again.
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            click.echo(f"  {time.strftime('%H:%M:%S')}  {broke.format_message()}")
            click.echo(f"  retrying in {wait}s")
            attempt += 1
            time.sleep(wait)
