"""
Two-armed Bayesian allocation (A/B tests). Maths in kb/two_arm.md.

Layout: sample.py (long-tailed collocation and ridge draws), simplex.py
(plain calculus: quadratic max over an interval), model.py (premium net with
the proven free-information envelope and the value wrapper), loss.py (the
subsolution objective: climb subject to v <= max H, plus BC1).

The champion is a certified lower bound rather than a two-sided fit: it
overclaims on 0.60% of the cloud against its predecessor's 23.7%, at no
measurable regret (kb/two_arm.md section 10).
"""

from .loss import draw, loss, objective, ridge_loss, subsolution_loss
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import sample_ridge, sample_sobol
