"""
Generic building blocks, problem-agnostic. The problem packages under
`pinn/problems` assemble these into models.
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from pathlib import Path
from torch import Tensor
from typing import Self


class DimensionlessValue(nn.Module, ABC):
    """
    A problem's value function on its own chart: dimensionless, and inert until it
    is bound to the rates of one experiment.

    This is the boundary the four problems share, so anything loading a net can be
    typed against it rather than against whatever a module lookup happened to
    return. Each subclass narrows `bind` to the rates its own problem needs, which
    is how a missing or foreign one is caught.
    """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """The trained net in a file, at the architecture that file declares."""

    @abstractmethod
    def bind(self, **kwargs: float) -> nn.Module:
        """This net tied to one experiment: the problem's deployment adapter."""


class GainedTanh(nn.Module):
    """
    tanh(gain * x) with a trainable per-unit gain (init 1). Same function class
    as plain tanh; the gain gives each unit's sharpness its own optimization
    coordinate, which plain weight vectors grow too slowly to provide.
    """

    def __init__(self, width: int) -> None:
        super().__init__()

        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return (self.gain * x).tanh()


class DeclaresTopology(nn.Module):
    """
    Base for premium nets: records its shape as buffers, so it travels inside the
    state dict and a loader **reads** the architecture instead of reverse-engineering
    it from weight geometry. Subclass it and pass the shape up.

    The buffers describe **this** module, never the file: a declaration decides what to
    build, and the module is the authority after that, which is what stops a graft
    saving a shape its own weights disprove. The bugs behind the rule: learnings 12.
    """

    def __init__(self, features: int, hidden: list[int], kinks: int = 0) -> None:
        super().__init__()

        self.register_buffer("features", torch.tensor(features, dtype=torch.long))
        self.register_buffer("topology", torch.tensor(hidden, dtype=torch.long))
        self.register_buffer("kink_count", torch.tensor(kinks, dtype=torch.long))

    def _load_from_state_dict(
        self, state_dict: dict, prefix: str, *rest: object
    ) -> None:
        for name in ("features", "topology", "kink_count"):
            state_dict[prefix + name] = getattr(self, name)

        super()._load_from_state_dict(state_dict, prefix, *rest)


def read_topology(state: dict, prefix: str = "premium.") -> tuple[list[int], int]:
    """
    The (hidden widths, kink count) a checkpoint declares. Every checkpoint does,
    the ones predating `DeclaresTopology` having been rewritten 2026-08-13, so a
    KeyError here is a file to migrate rather than a case to handle.
    """
    return state[prefix + "topology"].tolist(), int(state[prefix + "kink_count"])


def read_features(state: dict, prefix: str = "premium.") -> int:
    """
    How wide a checkpoint's feature stack is, as it declares. The drift problems graft
    from a static sibling exactly one feature narrower, so the graft needs the width
    declared rather than measured off the first layer's input.
    """
    return int(state[prefix + "features"])


def parse_topology(topology: str) -> tuple[list[int], int]:
    """
    A net's shape from the string the CLI takes: colon-separated hidden widths and an
    optional trailing k<count> of kink units, one grammar for every problem, so a
    problem without kinks ignores the second element. `"32:512"` is `([32, 512], 0)`
    and `"64:64:64k16"` is `([64, 64, 64], 16)`.
    """
    if re.fullmatch(r"\d+(:\d+)*(k\d+)?", topology) is None:
        raise ValueError(
            f"bad topology {topology!r}: widths colon-separated, optional k<count>"
        )

    body, _, kinks = topology.partition("k")

    return [int(width) for width in body.split(":")], int(kinks or 0)


if __name__ == "__main__":
    assert parse_topology("32:512") == ([32, 512], 0)
    assert parse_topology("64:64:64k16") == ([64, 64, 64], 16)
    assert parse_topology("8") == ([8], 0)

    # The grammar rejects garbage instead of half-parsing it: without the check,
    # "junk" splits on its own "k" and complains about an int named "jun".
    for bad in ["junk", "32:512king", "", "32::8", "k4"]:
        try:
            parse_topology(bad)
            raise AssertionError(bad)
        except ValueError:
            pass

    activation = GainedTanh(4)

    assert activation(torch.zeros(3, 4)).shape == (3, 4)
    assert torch.allclose(activation(torch.ones(4)), torch.ones(4).tanh())
    print("ok")
