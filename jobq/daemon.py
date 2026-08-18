"""
The local backup daemon: one detached `jobq backup` per pod, started by `up` and
stopped by `down`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

STATE = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "jobq"
"""
Where the pid files live: runtime state, so it goes where the OS clears it, and
never in ~/.jobq or the repo. kb/jobq.md, "Where the pid file lives".
"""


@dataclass
class Daemon:
    """
    The backup process for one pod, identified by a pid file under STATE.

    Outside the repo on purpose: it is a local process, not a pod artifact, and a pid
    file inside the directory being backed up would be copied to the pod on the next
    push. See STATE for where it does live and why.
    """

    name: str

    @property
    def pidfile(self) -> Path:
        """This pod's pid file, whether or not anything is running."""
        return STATE / f"{self.name}.pid"

    @property
    def pid(self) -> int | None:
        """
        This pod's daemon pid, or None. Checks the *command line*, not just that the
        pid exists: pids are reused, so a stale file would point at whatever inherited
        the number. Killing an unidentified pid cost a training run on 2026-08-11.
        """
        if self.pidfile.exists() is False:
            return None

        try:
            pid = int(self.pidfile.read_text().strip())
        except ValueError:
            self.pidfile.unlink(missing_ok=True)

            return None

        command = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
        ).stdout

        if "jobq" in command and "backup" in command and self.name in command:
            return pid

        self.pidfile.unlink(missing_ok=True)

        return None

    def start(self) -> int | None:
        """
        Spawn it unless one is already up. Returns the pid. Output is discarded, since
        a log inside the directory it backs up would feed itself.
        """
        existing = self.pid

        if existing is not None:
            return existing

        STATE.mkdir(parents=True, exist_ok=True)
        child = subprocess.Popen(
            [sys.executable, "-m", "jobq", "backup", "--pod", self.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=Path.cwd(),
        )
        self.pidfile.write_text(str(child.pid))

        return child.pid

    def stop(self) -> int | None:
        """Stop it if it is ours. Returns the pid it stopped, or None."""
        pid = self.pid

        if pid is None:
            self.pidfile.unlink(missing_ok=True)

            return None

        os.kill(pid, signal.SIGTERM)
        self.pidfile.unlink(missing_ok=True)

        return pid
