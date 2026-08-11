"""
`jobq run`: a command on the pod, attached to this terminal.
"""

from __future__ import annotations

import click
import shlex
import sys

from .pod import REMOTE, VENV, find, runpodctl, shell, ssh_flags, ssh_info


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--name", default="pinn", show_default=True, help="Pod to run on.")
@click.option(
    "--no-start",
    is_flag=True,
    help="Fail instead of starting a stopped pod.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
def run(name: str, no_start: bool, args: tuple[str, ...]) -> None:
    """
    Run ARGS on the pod, output streaming here.

    Anything runs: `jobq run nvidia-smi`, `jobq run python probes.py --in x.pt`.
    `pinn` and `arena` are the venv's real console scripts, on PATH, and the
    working directory is the repo.

    One session per job: the run is attached, so closing this terminal ends
    it and ctrl-c reaches the trainer (which saves before exiting). For
    several at once, open several terminals.

    A stopped pod is resumed first (--no-start to refuse instead); a pod that
    does not exist at all is `jobq up`'s business, since creating one needs
    its gpu, image and idle settings.

    Push code with `jobq up`, which is idempotent and rsyncs. Nothing else
    puts a checkpoint on the pod, so a fresh net starts with

        jobq run pinn init --problem three_arm --topology 64:64 --out /workspace/x.pt
    """
    pod = find(name)

    if pod is None:
        raise click.ClickException(f"no pod named {name}; `jobq up` first")

    if pod.get("desiredStatus") != "RUNNING":
        if no_start is True:
            raise click.ClickException(
                f"{name} is {pod.get('desiredStatus')}; drop --no-start to resume it"
            )

        click.echo(f"{name} is {pod.get('desiredStatus')}, starting...")
        runpodctl("pod", "start", pod["id"], timeout=300)

    detail = ssh_info(pod["id"])

    remote = f"cd {REMOTE} && export PATH={VENV}/bin:$PATH && {shlex.join(args)}"
    # -t forces a tty: without one, ctrl-c kills the local ssh and leaves the
    # trainer running on a gpu nobody is watching.
    code = shell(
        ["ssh", "-t", *ssh_flags(detail["port"]), f"root@{detail['ip']}", remote],
        "run",
        fatal=False,
    )

    sys.exit(code)
