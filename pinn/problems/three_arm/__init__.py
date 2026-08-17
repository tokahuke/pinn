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

from .loss import (
    control_tie_loss,
    draw,
    loss,
    objective,
    subsolution_loss,
    treatment_tie_loss,
)
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import RidgeSample, Sample
