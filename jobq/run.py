"""
`jobq run`: a command on the pod, attached to this terminal.
"""

from __future__ import annotations

import click
import shlex
import sys

from .pod import REMOTE, VENV, Pod, runpodctl


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--name", default="pinn", show_default=True, help="Pod to run on.")
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
def run(name: str, args: tuple[str, ...]) -> None:
    """
    Run ARGS on the pod, output streaming here.

    Anything runs: `jobq run nvidia-smi`, `jobq run python probes.py --in x.pt`.
    `pinn` and `arena` are the venv's real console scripts, on PATH, and the
    working directory is the repo.

    One session per job: the run is attached, so closing this terminal ends
    it and ctrl-c reaches the trainer (which saves before exiting). For
    several at once, open several terminals.

    Does not create or start anything: `jobq up` owns making a pod ready,
    including adopting a stopped one. This only runs.

    Push code with `jobq up`, which is idempotent and rsyncs. Nothing else
    puts a checkpoint on the pod, so a fresh net starts with

        jobq run pinn init --problem three_arm --topology 64:64 --out /workspace/x.pt
    """
    pod = Pod.require(name, resolve=False)
    state = runpodctl("pod", "get", pod.id, timeout=120)
    status = state.get("desiredStatus") if isinstance(state, dict) else None

    if status not in (None, "RUNNING"):
        raise click.ClickException(f"{name} is {status}; `jobq up` to make it ready")

    pod = Pod.require(name)

    remote = f"cd {REMOTE} && export PATH={VENV}/bin:$PATH && {shlex.join(args)}"
    # -t forces a tty: without one, ctrl-c kills the local ssh and leaves the
    # trainer running on a gpu nobody is watching.
    sys.exit(pod.ssh(remote, tty=True))
