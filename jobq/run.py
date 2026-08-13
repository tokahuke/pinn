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
@click.option(
    "--log",
    default=None,
    help="Append output to this file ON THE POD and detach. Without it the "
    "run is attached to this terminal.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
def run(name: str, log: str | None, args: tuple[str, ...]) -> None:
    """
    Run ARGS on the pod, output streaming here.

    Anything runs: `jobq run nvidia-smi`, `jobq run python probes.py --in x.pt`.
    `pinn` and `arena` are the venv's real console scripts, on PATH, and the
    working directory is the repo.

    One session per job: the run is attached, so closing this terminal ends
    it and ctrl-c reaches the trainer (which saves before exiting). For
    several at once, open several terminals.

    `--log FILE` detaches instead: the job outlives this command under nohup,
    output APPENDS to FILE (so a log keeps its history across lr changes), and
    the pid is printed because that is the only safe way to stop it later --
    a pattern match on "pinn train" also matches the command doing the
    matching, and once cost a running job.

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

    # PYTHONUNBUFFERED because a detached job's stdout is a FILE, and python
    # block-buffers those: without it a log stays empty for hours while the
    # job runs fine, which is indistinguishable from a hang.
    prefix = f"cd {REMOTE} && export PATH={VENV}/bin:$PATH PYTHONUNBUFFERED=1"

    if log is not None:
        remote = (
            f"{prefix} && nohup {shlex.join(args)} >> {shlex.quote(log)} 2>&1 & echo $!"
        )
        # No tty: -t and a backgrounded nohup fight over the terminal, and the
        # pid never comes back.
        sys.exit(pod.ssh(remote))

    # -t forces a tty: without one, ctrl-c kills the local ssh and leaves the
    # trainer running on a gpu nobody is watching.
    sys.exit(pod.ssh(f"{prefix} && {shlex.join(args)}", tty=True))
