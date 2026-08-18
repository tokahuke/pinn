"""
Three-armed Bayesian allocation with drifting arm means. Maths in
kb/three_arm_drift.md; read kb/three_arm.md first.

Layout mirrors three_arm, minus simplex.py: the erosion drift adds is
control-free, so it never touches the maximization and three_arm's simplex is
imported unchanged. envelope.py is the one genuinely new module, and it reuses
two_arm_drift's formula once per pair of arms.

etahat = 0 is three_arm exactly: in every formula and in the code, bitwise,
which is what makes grafting the three_arm champion an exact bootstrap rather
than a warm start.
"""

from torch import Tensor

from ...net import DimensionlessValue
from ...train import Objective
from ..problem import Problem
from .loss import draw, loss, objective
from .model import DimensionlessValueFunction, init_model


class ThreeArmDrift(Problem):
    """The A/B/C test with an effect that drifts: kb/three_arm_drift.md."""

    name = "three_arm_drift"
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
