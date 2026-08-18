"""
Generic training machinery: the Adam loop and learning-rate schedules. Knows
nothing about any particular problem; feed it a model and an objective.
"""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn

from collections.abc import Callable, Iterator
from dataclasses import fields, is_dataclass
from torch import Tensor

type LearningRate = float | Iterator[float]
"""A constant rate, or a generator handing out one rate per step."""

type Objective = Callable[[nn.Module, int | None], Tensor]
"""What the trainer minimises: `(model, iteration or None) -> loss`."""


def train(
    model: nn.Module,
    objective: Objective,
    lr: LearningRate = 1e-3,
    verbose: bool = True,
) -> Iterator[float]:
    """
    Adam on objective(model, iteration), yielding the loss after each step. `lr` is a
    constant or a generator of per-step rates, and a finite generator ends the run;
    otherwise this is endless and the consumer stops it (`itertools.islice`).
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


def leaves(structure: object) -> Iterator[Tensor]:
    """
    Every tensor in a structure of tensors, dataclasses and sequences, in a stable
    order. Problems hand their loss whatever shape suits them (five bare tensors, or a
    Sample and two RidgeSamples), and walking the structure is what lets one graphed
    trainer serve all of them without a loss signature changing.
    """
    if isinstance(structure, Tensor):
        yield structure
    elif is_dataclass(structure) is True:
        for field in fields(structure):
            yield from leaves(getattr(structure, field.name))
    elif isinstance(structure, (tuple, list)):
        for item in structure:
            yield from leaves(item)


def clone(structure: object) -> object:
    """The same structure with every tensor cloned."""
    if isinstance(structure, Tensor):
        return structure.clone()

    if is_dataclass(structure) is True:
        return type(structure)(
            *(clone(getattr(structure, field.name)) for field in fields(structure))
        )

    if isinstance(structure, (tuple, list)):
        return type(structure)(clone(item) for item in structure)

    return structure


def train_graphed(
    model: nn.Module,
    draw: Callable[[], tuple],
    score: Callable[..., Tensor],
    lr: float = 1e-3,
    refresh: int = 100,
) -> Iterator[float]:
    """
    train(), but the step is captured as a cuda graph and replayed. The step is
    dispatch-bound (47k aten calls, gpu 24% idle, 10x on an RTX 4090, 2026-08-11) and
    capture survives create_graph, which torch.compile does not. Replay reuses tensor
    **addresses**, so `draw()` refills fixed buffers every `refresh` eager steps.
    """
    static = clone(draw())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)

    # Warmup on a side stream, or capture fails outright.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=False)
            score(model, *static, None).backward()
            optimizer.step()

    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()

    # zero_grad **inside** the capture: backward accumulates, so without it every
    # replay would add to the previous replay's gradient.
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        captured = score(model, *static, None)
        captured.backward()
        optimizer.step()

    for iteration in itertools.count():
        if iteration % refresh == 0:
            for buffer, fresh in zip(leaves(static), leaves(draw())):
                buffer.copy_(fresh)

            optimizer.zero_grad(set_to_none=False)
            eager = score(model, *static, iteration)
            eager.backward()
            optimizer.step()

            yield eager.item()
            continue

        graph.replay()

        yield captured.item()


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _Pair:
        """A two-tensor dataclass, to check that leaves() walks one."""

        left: Tensor
        right: Tensor

    nested = (torch.ones(3), _Pair(torch.ones(2), torch.ones(4)), [torch.ones(5)])
    found = list(leaves(nested))

    # Order is what makes the copy_ zip in train_graphed correct: buffers and
    # fresh draws must line up leaf for leaf.
    assert [tensor.numel() for tensor in found] == [3, 2, 4, 5]

    copied = clone(nested)

    assert [tensor.numel() for tensor in leaves(copied)] == [3, 2, 4, 5]
    assert type(copied[1]) is _Pair and type(copied[2]) is list
    assert all(a is not b for a, b in zip(leaves(nested), leaves(copied)))
    assert all(torch.equal(a, b) for a, b in zip(leaves(nested), leaves(copied)))

    # A clone must not alias: writing through one may not touch the other.
    for tensor in leaves(copied):
        tensor.zero_()

    assert all(tensor.abs().sum() > 0 for tensor in leaves(nested))
    print("ok")
