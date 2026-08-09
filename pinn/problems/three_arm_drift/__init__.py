"""
Three-armed Bayesian allocation with drifting arm means. Maths in
docs/three_arm_drift.md; read docs/three_arm.md first.

Layout mirrors three_arm, minus simplex.py: the erosion drift adds is
control-free, so it never touches the maximization and three_arm's simplex is
imported unchanged. envelope.py is the one genuinely new module, and it reuses
two_arm_drift's formula once per pair of arms.

etahat = 0 is three_arm exactly -- in every formula and in the code, bitwise,
which is what makes grafting the three_arm champion an exact bootstrap rather
than a warm start.
"""

from .envelope import envelope
from .loss import control_tie_loss, loss, objective, pde_loss, treatment_tie_loss
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import RidgeSample, Sample
