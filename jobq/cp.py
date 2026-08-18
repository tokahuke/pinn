"""`jobq cp`: files to or from the pod."""

from __future__ import annotations

import click

from .pod import REMOTE, Pod, shell


@click.command()
@click.option(
    "--pod", "name", default="pinn", show_default=True, help="Pod to copy with."
)
@click.argument("paths", nargs=-1, required=True)
def cp(name: str, paths: tuple[str, ...]) -> None:
    """
    Copy PATHS, scp-style: a leading `:` marks the pod side, one direction per call.
    A relative pod path is anchored at the repo, where `jobq run` commands run, so
    `jobq cp data/two_arm.pt :data/` lands where `pinn train --in data/...` looks
    (bare `:host-relative` would mean root's HOME). This is how a checkpoint
    travels, since `jobq up` excludes data/. Sent with -L, not plain -a: the
    canonical names are symlinks and rsync's default sends the link itself, which
    arrives dangling.
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
        """
        One path as rsync wants it: a leading `:` becomes the pod's host, and a
        relative pod path is anchored at the repo.
        """
        if path.startswith(":") is False:
            return path

        remote = path[1:]

        if remote.startswith("/") is False:
            remote = f"{REMOTE}/{remote}"

        return f"{pod.host}:{remote}"

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
