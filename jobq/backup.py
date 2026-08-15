"""
`jobq backup`: keep a pod directory backed up locally as it changes.
"""

from __future__ import annotations

import click
import time

from .pod import OURS, Pod

# Reconnect delays. A pod restart changes its ssh port, so every attempt
# re-resolves the pod rather than reusing the old address.
BACKOFF: tuple[int, ...] = (5, 10, 20, 40, 60)


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
    Back REMOTE up into --to continuously, event-driven rather than polled.

    Content-blind on purpose: it copies whatever changed. The only things
    skipped by default are the venv and the repo, because jobq put those there
    itself and would be copying them back to their source.

    Deliberately NOT a mirror: there is no --delete, so a file removed on the
    pod survives here. The pod is disposable and its files only grow, so "the
    latest copy of everything that ever existed there" is the useful shape. It
    keeps no history, though -- one version per file.

    Ctrl-C to stop. A dead connection ends the backup loudly -- one you
    believe in but that is not running is worse than none. Only useful while
    this machine is awake, and it does NOT keep the pod alive: the seppuku
    counts training processes and tty sessions, and rsync is neither.
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
            # A blip, a pod restart (which moves the ssh port), or a pod that
            # is gone for good -- indistinguishable from here, so keep trying.
            # The daemon costs nothing while it waits and a pod that returns
            # gets backed up again.
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            click.echo(f"  {time.strftime('%H:%M:%S')}  {broke.format_message()}")
            click.echo(f"  retrying in {wait}s")
            attempt += 1
            time.sleep(wait)
