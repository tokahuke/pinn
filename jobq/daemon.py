"""
The local backup daemon: one detached `jobq backup` per pod, started by `up`
and stopped by `down`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

# A pid file is RUNTIME state, so it belongs where the OS clears it. On Linux
# that is XDG_RUNTIME_DIR (/run/user/$UID): mode 0700, local filesystem, and
# removed at logout by specification, which makes a stale pid impossible.
# macOS sets no XDG_RUNTIME_DIR but TMPDIR is per-user and 0700
# (/var/folders/../T, NOT the shared /tmp) -- cleaned on a schedule rather
# than at logout, so there the pid check below is the real guarantee.
#
# Not ~/.jobq: that persists across reboots, which is the one thing a live
# pid must not do. Not the repo: data/ gets rsynced to the pod on push.
STATE = Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()) / "jobq"


@dataclass
class Daemon:
    """
    The backup process for one pod, identified by a pid file under ~/.jobq.

    Outside the repo on purpose: it is a local process, not a pod artifact,
    and a pid file inside the directory being backed up would be copied to
    the pod on the next push. See STATE for where it does live and why.
    """

    name: str

    @property
    def pidfile(self) -> Path:
        return STATE / f"{self.name}.pid"

    @property
    def pid(self) -> int | None:
        """
        This pod's daemon pid, or None.

        Checks the COMMAND LINE, not just that the pid exists: pids are
        reused, so a stale file from a reboot would otherwise point at
        whatever inherited the number. Never kill what you have not
        identified -- that mistake cost a training run on 2026-08-11.
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
        Spawn it unless one is already up. Returns the pid.

        Output is discarded: the daemon is judged by whether the backup
        directory is fresh, and a log inside the directory it backs up would
        feed itself.
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
