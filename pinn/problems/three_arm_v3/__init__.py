"""
Three-armed Bayesian allocation, v3: three_arm's problem and losses on the
drop-one subsolution architecture of kb/three_arm.md section 17 -- the frozen
two_arm champion as a hard-max base, a positive learned interpolation toward
the nu2 envelope as the correction. Same contract as v2: promote or scrub.

A CLONE of three_arm (sample.py, simplex.py, loss.py verbatim; model.py
replaced), not a re-export: the champion pipeline stays untouched while this
lives or dies. A fresh net needs a base checkpoint:
`pinn init --problem three_arm_v3 --topology 64:64:64 --from data/two_arm.pt`.
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
