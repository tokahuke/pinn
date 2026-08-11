"""
`jobq down`: fetch the checkpoints back, then destroy the pod.
"""

from __future__ import annotations

import click

from pathlib import Path

from .pod import REMOTE, find, runpodctl, shell, ssh_flags

# NOT data/ itself: the champions live there under the same names a pod would
# write, and a fetch that clobbers two_arm.pt is unrecoverable.
LANDING = Path("data/pod")


def fetch(address: str, port: int) -> str:
    """Checkpoints off the pod into data/pod."""
    LANDING.mkdir(parents=True, exist_ok=True)
    before = set(LANDING.glob("*.pt"))
    # Not fatal: a pod whose ssh has died still has to be destroyable, or it
    # bills forever.
    code = shell(
        [
            "rsync",
            "-az",
            "-e",
            f"ssh {' '.join(ssh_flags(port))}",
            "--include",
            "*.pt",
            "--exclude",
            "*",
            f"root@{address}:/workspace/",
            f"root@{address}:{REMOTE}/",
            str(LANDING),
        ],
        "fetch",
        fatal=False,
    )
    arrived = len(set(LANDING.glob("*.pt")) - before)

    if code != 0:
        return f"rsync failed ({code}), {arrived} new .pt in {LANDING}"

    return f"{arrived} new .pt in {LANDING}"


@click.command()
@click.option("--name", default="pinn", show_default=True)
@click.option("--yes", is_flag=True, help="Skip the confirmation.")
@click.option("--no-fetch", is_flag=True, help="Destroy without pulling checkpoints.")
def down(name: str, yes: bool, no_fetch: bool) -> None:
    """
    Destroy the pod. Checkpoints land in data/pod first.

    Destroying is the only thing that stops the billing -- a stopped pod still
    costs.
    """
    pod = find(name)

    if pod is None:
        click.echo(f"no pod named {name}")

        return

    click.echo(f"{name} ({pod['id']}) at ${pod.get('costPerHr', '?')}/hr")

    if not no_fetch:
        detail = runpodctl("ssh", "info", pod["id"], timeout=120)

        if "error" in detail:
            click.echo(f"  cannot fetch: {detail['error']}")
        else:
            click.echo(f"  {fetch(detail['ip'], detail['port'])}")

    if not yes:
        click.confirm(f"destroy {name}?", abort=True)

    runpodctl("pod", "delete", pod["id"], timeout=120)
    click.echo(f"destroyed {name}" if find(name) is None else f"{name} STILL PRESENT")
