"""
Talking to RunPod and to a pod: the plumbing every command shares.
"""

from __future__ import annotations

import click
import dotenv
import json
import os
import subprocess
import sys
import time

from pathlib import Path

REMOTE = "/workspace/pinn"
VENV = "/workspace/venv"
KEY = Path.home() / ".ssh" / "id_rsa"


def env(name: str) -> str:
    """One value out of .env (gitignored) or the shell."""
    dotenv.load_dotenv()
    value = os.environ.get(name)

    if not value:
        raise click.ClickException(f"{name} missing from .env")

    return value


def runpodctl(*args: str, timeout: int = 600) -> dict | list:
    # env() loads .env into os.environ, which the subprocess inherits; the
    # call is here for the clear failure when the key is absent.
    env("RUNPOD_API_KEY")
    done = subprocess.run(
        ["runpodctl", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = done.stdout.strip().splitlines()

    # The cli prefixes advisory notes before the json body.
    start = next((i for i, line in enumerate(tail) if line[:1] in "[{"), None)

    if start is None:
        raise click.ClickException(
            f"runpodctl {' '.join(args)}\n{done.stderr.strip() or done.stdout.strip()}"
        )

    return json.loads("\n".join(tail[start:]))


def find(name: str) -> dict | None:
    pods = runpodctl("pod", "list", timeout=120)
    pods = pods if isinstance(pods, list) else pods.get("pods", [])

    return next((p for p in pods if p.get("name") == name), None)


def ssh_info(pod_id: str, tries: int = 30) -> dict:
    """
    Address and port once the host has published the mapping.

    A pod reports RUNNING before its ssh port is mapped -- straight after a
    create or a start there is a window of tens of seconds, sometimes minutes,
    where `ssh info` only returns an error. Prints a dot per attempt: this is
    the long wait in `jobq up`, and a silent one reads as a hang.
    """
    for attempt in range(tries):
        detail = runpodctl("ssh", "info", pod_id, timeout=120)

        if "error" not in detail:
            if attempt > 0:
                click.echo("")

            return detail

        if attempt == 0:
            click.echo("waiting for ssh ", nl=False)

        click.echo(".", nl=False)
        sys.stdout.flush()
        time.sleep(10)

    click.echo("")

    raise click.ClickException(f"pod {pod_id} never published ssh: {detail['error']}")


def ssh_flags(port: int) -> list[str]:
    return [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-i",
        str(KEY),
        "-p",
        str(port),
    ]


def shell(command: list[str], what: str, fatal: bool = True) -> int:
    done = subprocess.run(command, text=True)

    if done.returncode != 0 and fatal is True:
        raise click.ClickException(f"{what} failed ({done.returncode})")

    return done.returncode
