"""
Talking to RunPod, and to one pod: the plumbing every command shares.
"""

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
VENV = "/workspace/venv"
KEY = Path.home() / ".ssh" / "id_rsa"

# What jobq itself put on the pod and would only be copying back to itself.
# Everything else is the user's and gets moved without jobq knowing or caring
# what it is.
OURS = ("venv/", REMOTE.rsplit("/", 1)[-1] + "/")


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


def shell(command: list[str], what: str, fatal: bool = True) -> int:
    done = subprocess.run(command, text=True)

    if done.returncode != 0 and fatal is True:
        raise click.ClickException(f"{what} failed ({done.returncode})")

    return done.returncode


@dataclass
class Pod:
    """
    One reachable pod. Commands take this rather than an (address, port) pair
    -- every one of them was doing the same find-then-resolve-ssh dance.
    """

    name: str
    id: str
    address: str
    port: int
    cost: float = 0.0

    @classmethod
    def find(cls, name: str, resolve: bool = True) -> Self | None:
        """
        The pod of that name, or None. `resolve=False` skips waiting for ssh,
        for callers that only need to know it exists (down, for one).
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
        pod = cls.find(name, resolve)

        if pod is None:
            raise click.ClickException(f"no pod named {name}; `jobq up` first")

        return pod

    @property
    def host(self) -> str:
        return f"root@{self.address}"

    @property
    def flags(self) -> list[str]:
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

        Deliberately content-blind: it copies what changed, not what jobq
        thinks is interesting. No --delete, so a file removed there survives
        here -- this accumulates rather than mirrors.
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
        # rsync words this differently across versions ("Number of files
        # transferred" vs "Number of regular files transferred"), so match the
        # part they share.
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
        Yield once per `settle` seconds in which `remote` changed, giving the
        number of file events coalesced into that window.

        inotify runs on the pod and its output is the stream: the kernel there
        decides when something changed, so there is no polling interval to get
        wrong. BOTH events are watched and modify is the load-bearing one --
        close_write fires only when a writer CLOSES the file, so a log held
        open by a running process never emits it. Watching close_write alone
        produced zero events in a minute against eight live jobs.
        """
        # inotifywait takes ONE --exclude and it is a regex, not a repeatable
        # flag -- passing several makes it silently honour only the last.
        # rsync's IS repeatable, which is why fetch() takes them as a list.
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
                f"the change stream died (ssh {stream.poll()}) -- nothing is watching"
            )

    def destroy(self) -> None:
        runpodctl("pod", "delete", self.id, timeout=120)


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
