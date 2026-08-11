"""
Two-armed allocation with a drifting mean. Maths in kb/two_arm_drift.md.

Layout mirrors two_arm: sample.py (collocation draws, plus the etahat law),
envelope.py (the discounted perfect-information bound, closed form),
model.py (premium net, value, deployment adapter), loss.py (HJB residual in
similarity coordinates, ridge condition BC1, objective). simplex.py is
two_arm's, reused unchanged -- the drift term never touches the maximization.

etahat = 0 is two_arm exactly, in every formula and in the code: a two_arm
checkpoint stitches in through DimensionlessValueFunction.bootstrap and
reproduces itself bitwise on that slice.
"""

from .envelope import envelope
from .loss import draw, loss, objective, pde_loss, ridge_loss
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import sample_ridge, sample_sobol
