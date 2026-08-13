"""
`jobq up`: a RunPod pod with the repo on it, ready to train.
"""

from __future__ import annotations

import click

from importlib.resources import files

from .daemon import Daemon
from .pod import KEY, Pod, runpodctl

# Ships a working CUDA torch on python 3.12; setup.sh inherits both through
# a --system-site-packages venv rather than reinstalling 2.5GB of torch.
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"

# Tried in order: stock moves hour to hour and a create against a sold-out
# type just errors, so the first with capacity wins.
#
# Ordered by MEASURED work per dollar, not by price. The step is
# dispatch-bound, which made "the cheapest card is as good as the dearest"
# look obvious -- and it is wrong. Same benchmark (three_arm, graphed, batch
# 16384, idle card), 2026-08-12:
#
#     RTX 4090    14.5 ms/step   $0.34 community / $0.74 secure
#     RTX A6000   46.7 ms/step   $0.33 community / $0.53 secure
#
# 3.2x faster for the same community price -- about 5x the work per dollar
# against the A6000 on secure. Clock explains only part of it (3105 MHz
# against 2100); Ada's scheduler does the rest on small kernels. L4 and A40
# are UNMEASURED fallbacks, listed only so a create still lands when the
# first two are dry.
GPUS = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L4",
)


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
    help="COMMUNITY is ~40% cheaper for the same card; it is other people's machines, so treat interruption as possible.",
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

    pod = Pod.find(name, resolve=False)

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
                    # Community hosts only publish an ssh port when they have
                    # a public ip, and without this the create waits for a
                    # mapping that never appears. Harmless on secure.
                    *(["--public-ip"] if cloud == "COMMUNITY" else []),
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
    else:
        # `up` means "make a working pod exist", so a stopped one is adopted
        # rather than refused. Nothing else starts a pod: run, cp and backup
        # all assume RUNNING and say so.
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
        else "  BACKUP DAEMON FAILED TO START -- nothing is backing this pod up"
    )
    click.echo(f"\nready:  ssh {' '.join(pod.flags)} {pod.host}")
    click.echo("        jobq run pinn train --problem ... --device cuda")

    if idle > 0:
        click.echo(f"        self-destructs after {idle}m with no ssh session")
