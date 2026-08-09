"""
Two-armed Bayesian allocation (A/B tests). Maths in docs/two_arm.md.

Layout: sample.py (long-tailed collocation and ridge draws), simplex.py
(plain calculus: quadratic max over an interval), model.py (premium net with
the proven free-information envelope and the value wrapper), loss.py (HJB
residual, ridge condition BC1, objective).

Solved to loss ~8e-8 in the pre-envelope architecture (champion checkpoint
data/value_champion_8e-8.pt loads only at the initial commit's class);
retraining under the nu envelope in progress.
"""

from .loss import loss, objective, pde_loss, ridge_loss
from .model import (
    DimensionlessValueFunction,
    init_model,
    ExplorationPremium,
    ValueFunction,
)
from .sample import sample_ridge, sample_sobol
