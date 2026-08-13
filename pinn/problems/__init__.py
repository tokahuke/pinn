"""
One module per problem; each exports its models and an objective() for the
generic trainer.
"""

# The registry the CLI offers. Adding a problem means adding it here and
# exposing init_model() and objective() from its package.
PROBLEMS = [
    "two_arm",
    "two_arm_drift",
    "three_arm",
    "three_arm_drift",
    # A promise, not a name: it supersedes three_arm or it goes (CLAUDE.md).
    "three_arm_v2",
]
