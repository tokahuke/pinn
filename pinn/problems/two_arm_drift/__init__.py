"""
Two-armed allocation with a drifting mean. Maths in kb/two_arm_drift.md. Layout mirrors
two_arm: sample.py (collocation draws, plus the etahat law), envelope.py (the discounted
perfect-information bound, closed form), model.py (premium net, value, deployment
adapter), loss.py (the subsolution objective, ridge condition BC1). simplex.py is
two_arm's, reused unchanged, since the drift term never touches the maximization.

etahat = 0 is two_arm exactly, in every formula and in the code: a two_arm checkpoint
stitches in through `ExplorationPremium.stitch` and reproduces itself bitwise on that
slice.
"""

from torch import Tensor

from ...net import DimensionlessValue
from ...train import Objective
from ..problem import Problem
from .loss import draw, loss, objective
from .model import DimensionlessValueFunction, init_model


class TwoArmDrift(Problem):
    """The A/B test with an effect that drifts: kb/two_arm_drift.md."""

    name = "two_arm_drift"
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
