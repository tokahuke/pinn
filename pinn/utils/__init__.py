"""
Shared, problem-agnostic math.

`gaussian` holds the expectations the free-information envelopes are built
from; `laws` holds the inverse-cdf distributions the quasi-random samplers
push Sobol points through.
"""

from .gaussian import nu, nu2
from .laws import (
    chi_squared_1,
    decade_scale,
    exponential,
    laplace,
    truncated_pareto,
)

__all__ = [
    "chi_squared_1",
    "decade_scale",
    "exponential",
    "laplace",
    "nu",
    "nu2",
    "truncated_pareto",
]
