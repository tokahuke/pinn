"""
Generic training machinery: the Adam loop and learning-rate schedules. Knows
nothing about any particular problem; feed it a model and an objective.
"""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn

from collections.abc import Callable, Iterator
from torch import Tensor

type LearningRate = float | Iterator[float]
type Objective = Callable[[nn.Module, int | None], Tensor]


def decay(initial: float, half_life: int) -> Iterator[float]:
    """
    Endless exponential decay: initial * 0.5 ** (step / half_life).
    """
    factor = 0.5 ** (1.0 / half_life)

    while True:
        yield initial

        initial *= factor


def train(
    model: nn.Module,
    objective: Objective,
    lr: LearningRate = 1e-3,
    verbose: bool = True,
) -> Iterator[float]:
    """
    Adam on objective(model, iteration), yielding the loss after each step.
    lr is a constant or a generator of per-step rates (see decay); a finite
    generator ends the training when it runs out, otherwise endless and the
    consumer decides when to stop (itertools.islice is your friend).
    """
    rates = itertools.repeat(lr) if isinstance(lr, (int, float)) else lr
    optimizer = torch.optim.Adam(model.parameters())

    for iteration in itertools.count():
        rate = next(rates, None)

        if rate is None:
            return

        for group in optimizer.param_groups:
            group["lr"] = rate

        objective_value = objective(
            model, iteration if verbose and iteration % 100 == 0 else None
        )

        optimizer.zero_grad()
        objective_value.backward()
        optimizer.step()

        yield objective_value.item()
