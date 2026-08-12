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
    """
    Every checkpoint under /workspace, at any depth, into data/pod.

    Two traps, both found by a --dry-run before a real teardown. `--exclude
    '*'` excludes DIRECTORIES too, so without `--include '*/'` rsync never
    descends and a sweep writing to /workspace/sweep/ is silently left behind.
    And rsync refuses two remote sources in one call -- it printed its usage
    and returned non-zero, which this treats as a failed fetch. One recursive
    source covers both, since REMOTE lives under /workspace.
    """
    LANDING.mkdir(parents=True, exist_ok=True)
    before = set(LANDING.rglob("*.pt"))
    # Not fatal: a pod whose ssh has died still has to be destroyable, or it
    # bills forever.
    code = shell(
        [
            "rsync",
            "-az",
            "--prune-empty-dirs",
            "-e",
            f"ssh {' '.join(ssh_flags(port))}",
            "--exclude",
            "venv/",
            "--include",
            "*/",
            "--include",
            "*.pt",
            "--exclude",
            "*",
            f"root@{address}:/workspace/",
            str(LANDING),
        ],
        "fetch",
        fatal=False,
    )
    arrived = len(set(LANDING.rglob("*.pt")) - before)

    if code != 0:
        return f"rsync FAILED ({code}) -- {arrived} new .pt in {LANDING}"

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
