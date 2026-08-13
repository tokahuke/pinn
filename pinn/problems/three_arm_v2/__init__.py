"""
The ABC-test problem again, solved through a pairwise basis instead of one
monolithic net. Maths unchanged: kb/three_arm.md.

Only the PREMIUM differs, so everything else is three_arm's -- the wedge
sampler, the simplex maximization, the HJB residual, both tie losses, the
concavity term and the objective. What changes is that the response also sees
a frozen two_arm net's premium for each pair of the wedge (model.py).

`_v2` IS A PROMISE: this either supersedes `three_arm` and takes its name, or
it is deleted. See CLAUDE.md -- there is never a `three_arm` and a
`three_arm_v2` in the tree at rest.
"""

from ..three_arm.loss import (
    control_tie_loss,
    draw,
    loss,
    objective,
    pde_loss,
    treatment_tie_loss,
)
from ..three_arm.sample import RidgeSample, Sample
from ..three_arm.simplex import Maximum, maximize_quadratic
from .model import (
    DimensionlessValueFunction,
    ExplorationPremium,
    ValueFunction,
    init_model,
)
