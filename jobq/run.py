"""`jobq run`: a command on the pod, attached to this terminal."""

from __future__ import annotations

import click
import shlex
import sys

from .pod import REMOTE, VENV, Pod, runpodctl


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--pod", "name", default="pinn", show_default=True, help="Pod to run on.")
@click.option(
    "--log",
    default=None,
    help="Append output to this file on the pod and detach. Without it the "
    "run is attached to this terminal.",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED, required=True)
def run(name: str, log: str | None, args: tuple[str, ...]) -> None:
    """
    Run ARGS on the pod, output streaming here (`jobq run nvidia-smi`). Attached, so
    ctrl-c reaches the trainer; `--log FILE` detaches and prints the pid to stop it
    by. Pods, code and installs come from `jobq up`, checkpoints from `jobq cp`.
    Detach mechanics: kb/jobq.md.
    """
    pod = Pod.require(name, resolve=False)
    state = runpodctl("pod", "get", pod.id, timeout=120)
    status = state.get("desiredStatus") if isinstance(state, dict) else None

    if status not in (None, "RUNNING"):
        raise click.ClickException(f"{name} is {status}; `jobq up` to make it ready")

    pod = Pod.require(name)

    # PYTHONUNBUFFERED because a detached job's stdout is a *file* and python
    # block-buffers those, so the log stays empty for hours and reads as a hang.
    prefix = f"cd {REMOTE} && export PATH={VENV}/bin:$PATH PYTHONUNBUFFERED=1"

    if log is not None:
        # The parentheses are load-bearing: without them `&` binds the whole and-list,
        # ssh waits for the job, and $! names a wrapper rather than the job.
        # kb/jobq.md, "Detaching a run".
        remote = (
            f"{prefix} && (setsid nohup {shlex.join(args)} < /dev/null"
            f" >> {shlex.quote(log)} 2>&1 & echo $!)"
        )
        # No tty: -t and a backgrounded nohup fight over the terminal, and the pid
        # never comes back.
        sys.exit(pod.ssh(remote))

    # -t forces a tty: without one, ctrl-c kills the local ssh and leaves the trainer
    # running on a gpu nobody is watching.
    sys.exit(pod.ssh(f"{prefix} && {shlex.join(args)}", tty=True))
