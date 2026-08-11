"""
Two-armed Bayesian allocation (A/B tests). Maths in kb/two_arm.md.

Layout: sample.py (long-tailed collocation and ridge draws), simplex.py
(plain calculus: quadratic max over an interval), model.py (premium net with
the proven free-information envelope and the value wrapper), loss.py (HJB
residual, ridge condition BC1, objective).

Solved to loss ~8e-8 in the pre-envelope architecture (that checkpoint is
data/two_arm.2026-08-04-legacy-arch.pt and loads only at the initial commit's
class; the 8e-8 is on the old chart-weighted grading, not an error magnitude);
retraining under the nu envelope in progress.
"""

from .loss import draw, loss, objective, pde_loss, ridge_loss
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import sample_ridge, sample_sobol
