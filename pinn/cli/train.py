"""
`pinn train`: run the generic trainer on a problem.
"""

from __future__ import annotations

import click
import math
import torch

from pathlib import Path

from ..problems import Problem
from ..train import train as run_training
from ..train import train_graphed as graphed_training

BATCH = 4096
"""
In natural units the residual spans six orders, so even at POWER = 1 the gradient rides
~1.3% of the batch: effective sample size at 4096 is 69 (two_arm), 177 (two_arm_drift),
54 (three_arm), 36 (three_arm_drift). Raise it if a run oscillates.
"""

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
"""
Best available, not hardcoded: mps is ~2x CPU here, but the command has to keep working
where there is no Metal.
"""


@click.command()
@click.option(
    "--problem",
    type=click.Choice(Problem.names()),
    required=True,
    help="Which problem.",
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
    Train (Ctrl-C to stop), printing the loss now and then, and saving the **best** net
    rather than the last: the loss oscillates around its floor. Scored on a 100-step
    EMA so a lucky batch cannot win, so a run under 100 iterations writes nothing. The
    rate is constant, no schedule, so a resume picks up where it left off.
    """
    out_path = out_path if out_path is not None else in_path

    # On an accelerator the cpu only issues kernels, so torch's ncpu/2 default buys a
    # fan-out it never earns back: 96-core cuda box, batch 65536, 48 threads 388
    # ms/step against 1 thread 181 (2026-08-11). Neutral on mps, so gate on cpu.
    if device != "cpu":
        torch.set_num_threads(1)

    chosen = Problem.named(problem)
    state = torch.load(in_path, map_location="cpu")
    value = chosen.init_model(state=state).to(device)
    graphing = graph_it and device.startswith("cuda")

    # Train the wrapper, **save** the original: a compiled module's state_dict carries
    # `_orig_mod.` prefixes every loader here rejects. torch.compile only ever captured
    # the forward (it refuses higher order gradients) and has none inside a capture.
    trained = torch.compile(value) if compile_it and graphing is False else value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best, smoothed, collapsed = float("inf"), None, 0
    steps = (
        graphed_training(
            trained,
            lambda: chosen.draw(batch, device),
            chosen.loss,
            lr=lr,
            refresh=refresh,
        )
        if graphing is True
        else run_training(trained, chosen.objective(batch=batch, device=device), lr=lr)
    )

    try:
        for step, score in enumerate(steps):
            # Exactly zero, or non-finite, is a broken *measurement*, and "smaller is
            # better" would file it as the best result ever and overwrite a champion
            # with the wreck. It happened: kb/learnings.md section 15.
            if score == 0.0 or math.isfinite(score) is False:
                collapsed += 1

                if collapsed in (1, 10, 100, 1000):
                    click.echo(
                        f"iter {step}: score {score} is not a measurement "
                        f"({collapsed} so far). Not saving, the run is dead."
                    )

                continue

            smoothed = score if smoothed is None else 0.99 * smoothed + 0.01 * score

            if step % 100 == 99 and smoothed < best:
                best = smoothed
                # Saved on CPU whatever we trained on: the arena, probes and every
                # loader read these without a map_location, and an mps-tensor
                # checkpoint would fail for them.
                torch.save(
                    {k: v.cpu() for k, v in value.state_dict().items()}, out_path
                )
    except KeyboardInterrupt:
        click.echo("Interrupted by user")

    if collapsed > 0:
        click.echo(f"{collapsed} steps scored zero or non-finite and were ignored")

    click.echo(
        f"saved {out_path} (best 100-step mean {best:.3e})"
        if best < float("inf")
        else "nothing saved: fewer than 100 iterations"
    )
