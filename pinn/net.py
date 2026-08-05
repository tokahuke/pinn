"""
Generic building blocks, problem-agnostic. Problem modules (bandit.py, ...)
assemble these into models.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    activation = GainedTanh(4)

    assert activation(torch.zeros(3, 4)).shape == (3, 4)
    assert torch.allclose(activation(torch.ones(4)), torch.ones(4).tanh())
    print("ok")
