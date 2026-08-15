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
    "three_arm_v3",
    "three_arm_drift",
]
