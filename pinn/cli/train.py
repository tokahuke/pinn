"""
`pinn train`: run the generic trainer on a problem.
"""

from __future__ import annotations

import click
import torch

from importlib import import_module
from pathlib import Path

from ..problems import PROBLEMS

from ..train import train as run_training
from ..train import train_graphed as graphed_training

# In natural units the residual spans six orders, so even at POWER = 1 the
# gradient rides ~1.3% of the batch: effective sample size at 4096 is 69
# (two_arm), 177 (two_arm_drift), 54 (three_arm), 36 (three_arm_drift).
# Raise it if a run oscillates.
BATCH = 4096
# Best available, not hardcoded: mps is ~2x CPU here, but the command has to
# keep working where there is no Metal.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@click.command()
@click.option(
    "--problem", type=click.Choice(PROBLEMS), required=True, help="Which problem."
)
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Checkpoint to continue from; make one with `pinn init`.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to save; defaults to --in, training the checkpoint in place.",
)
@click.option(
    "--lr",
    type=float,
    required=True,
    help="Learning rate, constant.",
)
@click.option("--batch", type=int, default=BATCH, show_default=True)
@click.option(
    "--device",
    type=str,
    default=DEVICE,
    show_default=True,
    help="Torch device to train on. The checkpoint is saved on CPU whatever "
    "this is, and the arena always runs on CPU.",
)
@click.option(
    "--compile/--no-compile",
    "compile_it",
    default=True,
    show_default=True,
    help="Wrap the net in torch.compile: fuses the double-backward graph into "
    "fewer, larger kernels. Costs a one-off pause at step 0. Pass "
    "--no-compile if a backend cannot handle it, which shows up immediately "
    "rather than mid-run.",
)
@click.option(
    "--graph/--no-graph",
    "graph_it",
    default=True,
    show_default=True,
    help="On cuda, capture the step as a cuda graph and replay it. The step "
    "is dispatch-bound, so this is the large win; needs the problem to "
    "expose draw() and refreshes the cloud every --refresh steps instead "
    "of every step.",
)
@click.option(
    "--refresh",
    default=100,
    show_default=True,
    help="Steps between collocation redraws under --graph. The redraw also "
    "runs eagerly, which is what prints the loss breakdown.",
)
def train(
    problem: str,
    in_path: Path,
    out_path: Path | None,
    lr: float,
    batch: int,
    device: str,
    compile_it: bool,
    graph_it: bool,
    refresh: int,
) -> None:
    """
    Train (Ctrl-C to stop), printing the loss now and then.

    Saves the BEST net, not the last one: the loss oscillates around its floor
    and the excursions hand back the descent between them. Scored on a 100-step
    EMA so a lucky batch cannot win, checked at the print cadence. A run
    shorter than 100 iterations writes nothing.

    The rate is CONSTANT: no schedule, so a resume picks up where it left off
    instead of restarting one at its hot end, which is what used to smear a
    polished checkpoint for its first ~1k iterations.
    """
    out_path = out_path if out_path is not None else in_path

    # On an accelerator the cpu only issues kernels, and torch's default of
    # ncpu/2 threads makes every small op in that path pay a fan-out it never
    # earns back. Measured 2026-08-11 on a 96-core cuda box, two_arm_drift at
    # batch 65536: 48 threads 388 ms/step, 8 threads 200, 1 thread 181. The
    # bigger the box, the worse the default. Neutral on mps, so gate on cpu
    # only, where the threads do real work.
    if device != "cpu":
        torch.set_num_threads(1)

    module = import_module(f"pinn.problems.{problem}")
    state = torch.load(in_path, map_location="cpu")
    value = module.init_model(state=state).to(device)
    graphing = graph_it and device.startswith("cuda") and hasattr(module, "draw")

    if graph_it is True and graphing is False and device.startswith("cuda"):
        click.echo(f"--graph ignored: {problem} exposes no draw()")

    # Train the wrapper, SAVE the original: a compiled module's state_dict
    # carries `_orig_mod.` key prefixes, which every loader here would reject.
    # torch.compile only ever captured the forward anyway (it refuses higher
    # order gradients), and it has no business inside a graph capture.
    trained = torch.compile(value) if compile_it and graphing is False else value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best, smoothed = float("inf"), None
    steps = (
        graphed_training(
            trained,
            lambda: module.draw(batch, device),
            module.loss,
            lr=lr,
            refresh=refresh,
        )
        if graphing is True
        else run_training(trained, module.objective(batch=batch, device=device), lr=lr)
    )

    try:
        for step, score in enumerate(steps):
            smoothed = score if smoothed is None else 0.99 * smoothed + 0.01 * score

            if step % 100 == 99 and smoothed < best:
                best = smoothed
                # Saved on CPU whatever we trained on: the arena, probes and
                # every loader read these without a map_location, and an
                # mps-tensor checkpoint would fail for them.
                torch.save(
                    {k: v.cpu() for k, v in value.state_dict().items()}, out_path
                )
    except KeyboardInterrupt:
        click.echo("Interrupted by user")

    click.echo(
        f"saved {out_path} (best 100-step mean {best:.3e})"
        if best < float("inf")
        else f"nothing saved: fewer than 100 iterations"
    )
