"""
Generic building blocks, problem-agnostic. Problem modules (bandit.py, ...)
assemble these into models.
"""

from __future__ import annotations

import re
import torch
import torch.nn as nn

from torch import Tensor


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
    Base for premium nets: records its own shape as buffers, so the shape
    travels inside the state dict and a loader READS the architecture instead
    of reverse-engineering it from weight geometry. Subclass it and pass the
    shape up -- `super().__init__(hidden, kinks)`.

    Buffers, not a separate config file or a wrapped save format: `state_dict()`
    already carries them, so nothing about how checkpoints are written changes.

    The buffers always describe THIS module. A checkpoint's declaration is read
    (read_topology) to decide what to BUILD; once built, the module is the
    authority and a loaded state dict cannot contradict it. GRAFTS depend on
    that: stitching a smooth checkpoint into a kinked net would otherwise let
    the source's kink_count = 0 overwrite the target's 8, and the net would
    save a declaration its own weights disprove.
    """

    def __init__(self, features: int, hidden: list[int], kinks: int = 0) -> None:
        super().__init__()

        self.register_buffer("features", torch.tensor(features, dtype=torch.long))
        self.register_buffer("topology", torch.tensor(hidden, dtype=torch.long))
        self.register_buffer("kink_count", torch.tensor(kinks, dtype=torch.long))

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *rest) -> None:
        for name in ("features", "topology", "kink_count"):
            state_dict[prefix + name] = getattr(self, name)

        super()._load_from_state_dict(state_dict, prefix, *rest)


def read_topology(state: dict, prefix: str = "premium.") -> tuple[list[int], int]:
    """
    The (hidden widths, kink count) a checkpoint declares.

    Every checkpoint declares: they are written by DeclaresTopology and the
    ones predating it were rewritten 2026-08-13. A KeyError here means a file
    from before that, which is a file to migrate, not a case to handle.
    """
    return state[prefix + "topology"].tolist(), int(state[prefix + "kink_count"])


def read_features(state: dict, prefix: str = "premium.") -> int:
    """
    How wide a checkpoint's feature stack is, as it declares.

    The drift problems graft from their static sibling, which is exactly one
    feature narrower, and used to detect that by measuring the first layer's
    input width. Same reverse-engineering as the old hidden_widths, one
    dimension over.
    """
    return int(state[prefix + "features"])


def parse_topology(topology: str) -> tuple[list[int], int]:
    """
    A net's shape from the string the CLI takes: hidden widths colon-separated,
    with an optional trailing k<count> for kink units.

        "32:512"      -> ([32, 512], 0)
        "64:64:64k16" -> ([64, 64, 64], 16)

    Problems without a kink branch ignore the second element; keeping one
    grammar means one thing to document and no per-problem drift in what a
    topology string means.
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

    # A CLI grammar rejects garbage instead of half-parsing it: "junk" used to
    # split on its own "k" and complain about an int named 'jun'.
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
