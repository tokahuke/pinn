"""
`jobq up`: a RunPod pod with the repo on it, ready to train.
"""

from __future__ import annotations

import click
import subprocess

from importlib.resources import files

from .pod import KEY, find, runpodctl, ssh_flags, ssh_info, sync_repo

# Ships a working CUDA torch on python 3.12; setup.sh inherits both through
# a --system-site-packages venv rather than reinstalling 2.5GB of torch.
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"

# Tried in order: stock moves hour to hour and a create against a sold-out
# type just errors. Measured 2026-08-11: the step is dispatch-bound, so the
# cheapest card is as good as the dearest -- order by price, not by FLOPs.
GPUS = (
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA A40",
)


def remote_write(address: str, port: int, path: str, body: str) -> None:
    """A rendered artifact onto the pod, never touching the local disk."""
    done = subprocess.run(
        ["ssh", *ssh_flags(port), f"root@{address}", f"cat > {path}"],
        input=body,
        text=True,
    )

    if done.returncode != 0:
        raise click.ClickException(f"writing {path} failed ({done.returncode})")


@click.command()
@click.option("--name", default="pinn", show_default=True)
@click.option(
    "--gpu",
    default=None,
    help="Exact gpu id; default walks a cheap-first list until one has stock.",
)
@click.option("--image", default=IMAGE, show_default=True)
@click.option("--disk", default=25, show_default=True, help="Container disk, GiB.")
@click.option(
    "--cloud",
    type=click.Choice(["SECURE", "COMMUNITY"]),
    default="SECURE",
    show_default=True,
    help="COMMUNITY is cheaper but only maps ssh on hosts with a public ip.",
)
@click.option(
    "--idle",
    default=30,
    show_default=True,
    help="Self-destruct after this many minutes with no ssh session; 0 disables.",
)
def up(
    name: str, gpu: str | None, image: str, disk: int, cloud: str, idle: int
) -> None:
    """
    Create a pod, push the repo, install what it imports.

    Idempotent: run it again and it prints the existing pod's ssh line. The
    pod is billed while it exists, so `jobq down` when finished.
    """
    if not KEY.exists():
        raise click.ClickException(f"no ssh key at {KEY}")

    keys = runpodctl("ssh", "list-keys", timeout=120)

    if not (keys.get("keys") if isinstance(keys, dict) else keys):
        raise click.ClickException(
            "no ssh key on the runpod account; add one with "
            f"`runpodctl ssh add-key --key-file {KEY}.pub`"
        )

    pod = find(name)

    if pod is None:
        for candidate in [gpu] if gpu else GPUS:
            click.echo(f"trying {candidate}...")

            try:
                pod = runpodctl(
                    "pod",
                    "create",
                    "--name",
                    name,
                    "--image",
                    image,
                    "--gpu-id",
                    candidate,
                    "--cloud-type",
                    cloud,
                    "--container-disk-in-gb",
                    str(disk),
                    # Declared at CREATE: adding 22/tcp later restarts the
                    # container.
                    "--ports",
                    "22/tcp",
                    "--ssh",
                )
            except click.ClickException:
                continue

            if "error" not in pod:
                break
        else:
            raise click.ClickException("no gpu type had stock; try --cloud COMMUNITY")

    detail = ssh_info(pod["id"])
    address, port = detail["ip"], detail["port"]
    click.echo(f"{name} ({pod['id']}) at {address}:{port}")

    sync_repo(address, port)

    if idle > 0:
        seppuku = (
            files("jobq.artifacts")
            .joinpath("seppuku.sh")
            .read_text()
            .replace("@@IDLE_MINUTES@@", str(idle))
        )
        remote_write(address, port, "/usr/local/bin/jobq-seppuku", seppuku)

    setup = files("jobq.artifacts").joinpath("setup.sh").read_text()
    done = subprocess.run(
        ["ssh", *ssh_flags(port), f"root@{address}", "bash -s"],
        input=setup,
        text=True,
    )

    if done.returncode != 0:
        raise click.ClickException(f"setup failed ({done.returncode})")

    click.echo(f"\nready:  ssh {' '.join(ssh_flags(port))} root@{address}")
    click.echo("        jobq run pinn train --problem ... --device cuda")

    if idle > 0:
        click.echo(f"        self-destructs after {idle}m with no ssh session")
