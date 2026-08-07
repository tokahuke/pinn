"""
The policy arena: discrete-epoch
simulation, discounted regret against an oracle, per-problem policy zoos
(two_arm, three_arm), and the PINN entrants. Generic core in harness.py;
import a problem module explicitly to reach its zoo -- the CLI's reflection
discovers whichever zoo is loaded.
"""

from .harness import Params, Policy, Run, Runner, Study, optimal_deadline

__all__ = [
    "Params",
    "Policy",
    "Run",
    "Runner",
    "Study",
    "optimal_deadline",
]
