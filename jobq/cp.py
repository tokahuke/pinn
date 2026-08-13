"""
`jobq cp`: files to or from the pod.
"""

from __future__ import annotations

import click

from .pod import Pod, shell


@click.command()
@click.option("--name", default="pinn", show_default=True, help="Pod to copy with.")
@click.argument("paths", nargs=-1, required=True)
def cp(name: str, paths: tuple[str, ...]) -> None:
    """
    Copy PATHS, scp-style: a leading `:` marks the pod side.

    `jobq up` rsyncs the repo but excludes data/, so this is how a checkpoint
    travels -- continuing a champion rather than starting cold:

        jobq cp data/two_arm_drift.pt :/workspace/
        jobq run pinn train --problem two_arm_drift --in /workspace/two_arm_drift.pt
        jobq cp :/workspace/two_arm_drift.pt data/pod/

    One direction per call: either the destination is remote or the sources
    are, never both.

    -L, not plain -a: the canonical checkpoint names are symlinks onto the
    topology-tagged file, and rsync's default is to send the link itself,
    which arrives dangling.
    """
    if len(paths) < 2:
        raise click.ClickException("need at least a source and a destination")

    *sources, destination = paths
    uploading = destination.startswith(":")

    if uploading is any(source.startswith(":") for source in sources):
        raise click.ClickException(
            "exactly one side must be remote; mark it with a leading `:`"
        )

    pod = Pod.require(name)

    def resolve(path: str) -> str:
        return f"{pod.host}:{path[1:]}" if path.startswith(":") else path

    shell(
        [
            "rsync",
            "-azL",
            "--progress",
            "-e",
            pod.ssh_command,
            *(resolve(source) for source in sources),
            resolve(destination),
        ],
        "cp",
    )
