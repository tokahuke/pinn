"""
Two-armed Bayesian allocation (A/B tests). Maths in kb/two_arm.md.

Layout: sample.py (long-tailed collocation and ridge draws), simplex.py
(plain calculus: quadratic max over an interval), model.py (premium net with
the proven free-information envelope and the value wrapper), loss.py (the
subsolution objective: climb subject to v <= max H, plus BC1).

The champion is graded one-sided rather than fitted two-sided: it overclaims on
0.60% of the cloud against its predecessor's 23.7%, at no measurable regret
(kb/two_arm.md section 10). It is not itself a subsolution; making the bound
rigorous means subtracting its worst point, which section 10 prices.
"""

from torch import Tensor

from ...net import DimensionlessValue
from ...train import Objective
from ..problem import Problem
from .loss import draw, loss, objective
from .model import DimensionlessValueFunction, init_model


class TwoArm(Problem):
    """The A/B-test problem: kb/two_arm.md."""

    name = "two_arm"
    net = DimensionlessValueFunction

    def init_model(
        self, state: dict | None = None, topology: str | None = None
    ) -> DimensionlessValueFunction:
        return init_model(state, topology)

    def objective(self, batch: int = 1024, device: str = "cpu") -> Objective:
        return objective(batch, device)

    def draw(self, batch: int, device: str = "cpu") -> tuple:
        return draw(batch, device)

    def loss(self, value: DimensionlessValue, *args: object) -> Tensor:
        return loss(value, *args)
