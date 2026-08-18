"""
Three-armed Bayesian allocation (ABC tests). Maths in kb/three_arm.md.

Layout: sample.py (state space, wall families, the wedge fold), simplex.py
(plain calculus: quadratic max over a triangle), model.py (premium net and
value wrapper), loss.py (the subsolution objective and both tie losses of doc
section 12).

The objective maximizes the premium subject to `v <= max H` since 2026-08-16
(kb/three_arm.md section 18), so a trained net is a proven lower bound
rather than a two-sided fit.
"""

from torch import Tensor

from ...net import DimensionlessValue
from ...train import Objective
from ..problem import Problem
from .loss import draw, loss, objective
from .model import DimensionlessValueFunction, init_model


class ThreeArm(Problem):
    """The A/B/C-test problem: kb/three_arm.md."""

    name = "three_arm"
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
