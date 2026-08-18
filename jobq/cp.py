"""`jobq cp`: files to or from the pod."""

from __future__ import annotations

import click

from .pod import Pod, shell


@click.command()
@click.option(
    "--pod", "name", default="pinn", show_default=True, help="Pod to copy with."
)
@click.argument("paths", nargs=-1, required=True)
def cp(name: str, paths: tuple[str, ...]) -> None:
    """
    Copy PATHS, scp-style: a leading `:` marks the pod side, one direction per call
    (`jobq cp data/two_arm.pt :/workspace/`). This is how a checkpoint travels, since
    `jobq up` excludes data/. Sent with -L, not plain -a: the canonical names are
    symlinks and rsync's default sends the link itself, which arrives dangling.
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
        """One path as rsync wants it, with a leading `:` becoming the pod's host."""
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
