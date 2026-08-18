"""
One package per problem, each answering `Problem`.

Importing this package is what makes the four problems exist: a subclass
registers itself as it is defined, and `Problem.named` resolves from there.
"""

from .problem import Problem
from .three_arm import ThreeArm
from .three_arm_drift import ThreeArmDrift
from .two_arm import TwoArm
from .two_arm_drift import TwoArmDrift

__all__ = ["Problem", "ThreeArm", "ThreeArmDrift", "TwoArm", "TwoArmDrift"]
