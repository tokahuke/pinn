"""`jobq up`: a RunPod pod with the repo on it, ready to train."""

from __future__ import annotations

import click

from importlib.resources import files

from .daemon import Daemon
from .pod import KEY, Pod, runpodctl

IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
"""
Ships a working CUDA torch on python 3.12; setup.sh inherits both through a
--system-site-packages venv rather than reinstalling 2.5GB of torch.
"""

GPUS = (
    "NVIDIA RTX 4000 Ada Generation",  # $0.20, measured 20.5 ms/step
    "NVIDIA GeForce RTX 4090",  # $0.34, measured 14.5 ms/step
    "NVIDIA GeForce RTX 4070 Ti",  # $0.19
    "NVIDIA GeForce RTX 4080 SUPER",  # $0.28
    "NVIDIA GeForce RTX 5080",  # $0.39, Blackwell
    "NVIDIA GeForce RTX 5090",  # $0.69, Blackwell
    "NVIDIA RTX A6000",  # $0.33, measured 46.7 ms/step
    "NVIDIA A40",
    "NVIDIA L4",
)
"""
Tried in order, first with capacity winning, since stock moves hour to hour and a
create against a sold-out type errors. Ordered by *measured* work per dollar rather
than price, which is why Ampere is last and the unmeasured entries trail: kb/jobq.md,
"Card choice is work per dollar".
"""


@click.command()
@click.option("--pod", "name", default="pinn", show_default=True)
@click.option(
    "--gpu",
    default=None,
    help="Exact gpu id; default walks a cheap-first list until one has stock.",
)
@click.option("--image", default=IMAGE, show_default=True)
@click.option("--disk", default=25, show_default=True, help="Container disk, GiB.")
@click.option(
    "--cloud",
    type=click.Choice(["COMMUNITY", "SECURE"]),
    default="COMMUNITY",
    show_default=True,
    help="COMMUNITY is other people's machines at ~half the price; the backup "
    "daemon and best-EMA saves make interruption survivable (kb/jobq.md). "
    "SECURE buys dedicated hosts at ~2x.",
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
    Create a pod, push the repo, install what it imports. Idempotent: run it again and
    it prints the existing pod's ssh line. The pod is billed while it exists, so
    `jobq down` when finished.
    """
    if KEY.exists() is False:
        raise click.ClickException(f"no ssh key at {KEY}")

    keys = runpodctl("ssh", "list-keys", timeout=120)
    listed = keys.get("keys") if isinstance(keys, dict) else keys

    if listed is None or len(listed) == 0:
        raise click.ClickException(
            "no ssh key on the runpod account; add one with "
            f"`runpodctl ssh add-key --key-file {KEY}.pub`"
        )

    pod = Pod.find(name, resolve=False)

    if pod is None:
        for candidate in [gpu] if gpu is not None else GPUS:
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
                    # Community hosts only publish an ssh port when they have a
                    # public ip, and without this the create waits for a mapping that
                    # never appears. Harmless on secure.
                    *(["--public-ip"] if cloud == "COMMUNITY" else []),
                    "--container-disk-in-gb",
                    str(disk),
                    # Declared at create time: adding 22/tcp later restarts it.
                    "--ports",
                    "22/tcp",
                    "--ssh",
                )
            except click.ClickException:
                continue

            if "error" not in pod:
                break
        else:
            raise click.ClickException(
                "no gpu type had stock; try --gpu with an exact id, "
                "or --cloud SECURE"
            )
    else:
        # `up` means "make a working pod exist", so a stopped one is adopted rather
        # than refused. Nothing else starts a pod: run, cp and backup all assume
        # RUNNING and say so.
        state = runpodctl("pod", "get", pod.id, timeout=120)
        status = state.get("desiredStatus") if isinstance(state, dict) else None

        if status not in (None, "RUNNING"):
            click.echo(f"{name} is {status}, starting it")
            runpodctl("pod", "start", pod.id, timeout=300)

    pod = Pod.require(name)
    click.echo(f"{name} ({pod.id}) at {pod.address}:{pod.port}")

    pod.send_repo()

    if idle > 0:
        seppuku = (
            files("jobq.artifacts")
            .joinpath("seppuku.sh")
            .read_text()
            .replace("@@IDLE_MINUTES@@", str(idle))
        )
        pod.write("/usr/local/bin/jobq-seppuku", seppuku)

    setup = files("jobq.artifacts").joinpath("setup.sh").read_text()

    if pod.ssh("bash -s", stdin=setup) != 0:
        raise click.ClickException("setup failed")

    pid = Daemon(name).start()
    click.echo(
        f"  backup daemon pid {pid}"
        if pid is not None
        else "  BACKUP DAEMON FAILED TO START: nothing is backing this pod up"
    )
    click.echo(f"\nready:  ssh {' '.join(pod.flags)} {pod.host}")
    click.echo("        jobq run pinn train --problem ... --device cuda")

    if idle > 0:
        click.echo(f"        self-destructs after {idle}m with no ssh session")
