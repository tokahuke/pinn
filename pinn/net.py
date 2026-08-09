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


def hidden_widths(state: dict) -> list[int]:
    """
    A checkpoint's hidden widths, from the premium net's weight shapes. The
    head's output width is not a hidden layer, hence the trailing drop.
    """
    return [
        w.shape[0]
        for k, w in state.items()
        if k.startswith("premium.net.") and k.endswith(".weight")
    ][:-1]


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
