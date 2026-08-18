"""Talking to RunPod, and to one pod: the plumbing every command shares."""

from __future__ import annotations

import click
import dotenv
import json
import os
import subprocess
import sys
import time

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

REMOTE = "/workspace/pinn"
"""Where the repo lives on the pod."""

VENV = "/workspace/venv"
"""Where the pod's virtualenv lives."""

KEY = Path.home() / ".ssh" / "id_rsa"
"""The key every ssh, rsync and scp here authenticates with."""

OURS = ("venv/", REMOTE.rsplit("/", 1)[-1] + "/")
"""
What jobq itself put on the pod and would only be copying back to itself. Everything
else is the user's and gets moved without jobq knowing or caring what it is.
"""


def env(name: str) -> str:
    """One value out of .env (gitignored) or the shell."""
    dotenv.load_dotenv()
    value = os.environ.get(name)

    if value is None or len(value) == 0:
        raise click.ClickException(f"{name} missing from .env")

    return value


def runpodctl(*args: str, timeout: int = 600) -> dict | list:
    """One runpodctl call, with its json body parsed out of the chatter around it."""
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


def shell(command: list[str], what: str, fatal: bool = True) -> int:
    """
    Run a command with its output going straight to the terminal, returning its exit
    code. `what` names it in the exception a fatal failure raises.
    """
    done = subprocess.run(command, text=True)

    if done.returncode != 0 and fatal is True:
        raise click.ClickException(f"{what} failed ({done.returncode})")

    return done.returncode


@dataclass
class Pod:
    """
    One reachable pod. Commands take this rather than an (address, port) pair, since
    every one of them was doing the same find-then-resolve-ssh dance.
    """

    name: str

    id: str
    """RunPod's id for the pod, which is what runpodctl addresses it by."""

    address: str
    port: int

    cost: float = 0.0
    """Dollars per hour, as RunPod reports it."""

    @classmethod
    def find(cls, name: str, resolve: bool = True) -> Self | None:
        """
        The pod of that name, or None. `resolve=False` skips waiting for ssh, for
        callers that only need to know it exists (down, for one).
        """
        pods = runpodctl("pod", "list", timeout=120)
        pods = pods if isinstance(pods, list) else pods.get("pods", [])
        found = next((p for p in pods if p.get("name") == name), None)

        if found is None:
            return None

        if resolve is False:
            return cls(name, found["id"], "", 0, found.get("costPerHr") or 0.0)
        detail = ssh_info(found["id"])

        return cls(
            name,
            found["id"],
            detail["ip"],
            detail["port"],
            found.get("costPerHr") or 0.0,
        )

    @classmethod
    def require(cls, name: str, resolve: bool = True) -> Self:
        """The pod of that name, or a clear failure telling the caller to bring one up."""
        pod = cls.find(name, resolve)

        if pod is None:
            raise click.ClickException(f"no pod named {name}; `jobq up` first")

        return pod

    @property
    def host(self) -> str:
        """The user and address ssh, rsync and scp all address the pod by."""
        return f"root@{self.address}"

    @property
    def flags(self) -> list[str]:
        """The ssh options every call here shares: the key, the port, and no host keys."""
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
            str(self.port),
        ]

    @property
    def ssh_command(self) -> str:
        """What rsync and scp want for their -e / --rsh argument."""
        return f"ssh {' '.join(self.flags)}"

    def ssh(self, command: str, tty: bool = False, stdin: str | None = None) -> int:
        """Run one command on the pod, returning its exit code."""
        argv = ["ssh", *(["-t"] if tty else []), *self.flags, self.host, command]
        done = subprocess.run(argv, text=True, input=stdin)

        return done.returncode

    def write(self, path: str, body: str) -> None:
        """A rendered artifact onto the pod, never touching the local disk."""
        if self.ssh(f"cat > {path}", stdin=body) != 0:
            raise click.ClickException(f"writing {path} failed")

    def send_repo(self) -> None:
        """The working tree up. data/ stays here: it holds the champions."""
        shell(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                self.ssh_command,
                "--exclude",
                ".git",
                "--exclude",
                "data",
                "--exclude",
                ".venv",
                "--exclude",
                "__pycache__",
                "./",
                f"{self.host}:{REMOTE}/",
            ],
            "rsync",
        )

    def fetch(
        self,
        remote: str = "/workspace",
        local: str = "data/pod",
        exclude: tuple[str, ...] = OURS,
    ) -> tuple[int, int]:
        """
        Pull `remote` down to `local`. Returns (files transferred, exit code).

        Content-blind, and no --delete: kb/jobq.md, "Backups accumulate".
        """
        Path(local).mkdir(parents=True, exist_ok=True)
        done = subprocess.run(
            [
                "rsync",
                "-az",
                "--prune-empty-dirs",
                "--stats",
                "-e",
                self.ssh_command,
                *(arg for pattern in exclude for arg in ("--exclude", pattern)),
                f"{self.host}:{remote.rstrip('/')}/",
                local,
            ],
            capture_output=True,
            text=True,
        )
        # rsync words this differently across versions ("Number of files transferred"
        # vs "Number of regular files transferred"), so match the part they share.
        moved = [
            line
            for line in done.stdout.splitlines()
            if "files transferred:" in line.lower()
        ]

        return (
            int(moved[0].split(":")[1].strip().replace(",", "")) if moved else -1,
            done.returncode,
        )

    def changes(
        self,
        remote: str = "/workspace",
        exclude: tuple[str, ...] = OURS,
        settle: int = 15,
    ) -> Iterator[int]:
        """
        Yield once per `settle` seconds in which `remote` changed, giving the number of
        file events coalesced into that window. inotify runs on the pod, so the kernel
        there decides what changed. Why modify and not close_write alone, and why one
        regex --exclude: kb/jobq.md, "Backups accumulate".
        """
        skip = (
            "--exclude '(" + "|".join(p.rstrip("/") for p in exclude) + ")'"
            if exclude
            else ""
        )
        watch = (
            "inotifywait -m -r -q -e close_write -e modify --format '%w%f' "
            f"{remote} {skip}"
        )
        stream = subprocess.Popen(
            ["ssh", *self.flags, self.host, watch],
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        pending, last = 0, 0.0

        try:
            for _ in stream.stdout:
                pending += 1
                now = time.monotonic()

                if now - last < settle:
                    continue

                yield pending
                pending, last = 0, now
        finally:
            stream.terminate()

        # -15 is the terminate above; anything else means it fell over.
        if stream.poll() not in (None, 0, -15):
            raise click.ClickException(
                f"the change stream died (ssh {stream.poll()}): nothing is watching"
            )

    def destroy(self) -> None:
        """Delete the pod, which is what stops it costing anything."""
        runpodctl("pod", "delete", self.id, timeout=120)


def ssh_info(pod_id: str, tries: int = 30) -> dict:
    """
    Address and port once the host has published the mapping. A pod reports RUNNING
    tens of seconds, sometimes minutes, before its ssh port is mapped, so this polls.
    Prints a dot per attempt: it is the long wait in `jobq up`, and a silent one reads
    as a hang.
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
