"""
2D plot of the trained exploration premium over the (muhat, tauhat) half-strip,
with the u = 0 level set dashed (the emergent free boundary).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from pathlib import Path

from pinn.problems.two_arm import ExplorationPremium, ValueFunction

state = torch.load(Path("data") / "value.pt")
hidden = [weight.shape[0] for key, weight in state.items() if key.endswith(".weight")][
    :-1
]

value = ValueFunction(ExplorationPremium(hidden))
value.load_state_dict(state)

muhat = torch.linspace(0.0, 2.5, 301)
tauhat = torch.linspace(0.1, 4.0, 301)
muhat_grid, tauhat_grid = torch.meshgrid(muhat, tauhat, indexing="xy")

with torch.no_grad():
    u = value.premium(muhat_grid.flatten(), tauhat_grid.flatten()).reshape(301, 301)

fig, ax = plt.subplots(figsize=(7, 5))
limit = float(u.abs().max())
mesh = ax.pcolormesh(muhat_grid, tauhat_grid, u, cmap="RdBu_r", vmin=-limit, vmax=limit)
ax.contour(muhat_grid, tauhat_grid, u, levels=[0.0], colors="#1a1a1a", linestyles="--")
fig.colorbar(mesh, ax=ax, label="u")
ax.set_xlabel("muhat")
ax.set_ylabel("tauhat")
ax.set_title("Trained exploration premium (dashed: u = 0)")

fig.tight_layout()
fig.savefig(Path("data") / "field.png", dpi=150)
print("saved data/field.png")
