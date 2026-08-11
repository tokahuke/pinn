"""
Three-armed Bayesian allocation (ABC tests). Maths in kb/three_arm.md.

Layout: sample.py (state space, wall families, the wedge fold), simplex.py
(plain calculus: quadratic max over a triangle), model.py (premium net and
value wrapper), loss.py (HJB residual, tie losses pending, objective).

Complete and training-ready: samplers, fold, models with the proven
free-information envelope (doc section 13), simplex max, and the full loss
(interior HJB residual + both tie losses of doc section 12).
"""

from .loss import (
    control_tie_loss,
    draw,
    loss,
    objective,
    pde_loss,
    treatment_tie_loss,
)
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import RidgeSample, Sample
