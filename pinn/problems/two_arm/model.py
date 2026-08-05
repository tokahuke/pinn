"""
Models for the two-arm problem: the premium net and the value wrapper that
carries the commit envelope's kink.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor

from ...net import GainedTanh
from ...utils import nu

# Width of the feature stack in ExplorationPremium.forward.
FEATURE_COUNT = 4


class ExplorationPremium(nn.Module):
    """
    Dense GainedTanh MLP times the free-information envelope:

        u = exp(log_scale) * nu(-muhat, tauhat**(-1/2)) * y / (1 + y),
        y = relu(response)**2

    nu(-muhat, sd) is the proven upper bound on the premium (docs/three_arm.md
    section 13, specialized to one challenger); the original tauhat**(-1/2)
    envelope is exactly its ridge slice, nu(0, sd) = sd / sqrt(2 pi), so this
    upgrade adds the muhat decay the old envelope lacked. The response head is
    linear and y/(1+y) maps it into [0, 1), so 0 <= u < envelope holds by
    construction; log_scale init 0 makes the envelope the proven bound
    itself. Rational saturation, NOT tanh(y): tanh is float32-exactly 1
    beyond r ~ 2.5 and its gradient underflows, a cliff with no way back
    (three_arm's first start died there); y/(1+y) saturates with a
    polynomial tail, so gradients survive any overshoot.

    The squared relu is the free-boundary trick: in the commit region the true
    premium is exactly 0 (v = muhat solves the HJB there), and at the boundary
    the solution pastes smoothly, u = u_m = 0 with a jump only in u_mm (the
    curvature law). relu(r)**2 has exactly that regularity on the learned zero
    set of r: the commit region is solved exactly, and the curvature kink a
    smooth response had to fake is built in. A plain clip was rejected: it
    kinks the FIRST derivative, breaking smooth pasting.

    Feature choices: log tauhat gives every decade of precision equal
    resolution; muhat sqrt(tauhat) is the posterior z-score (the corridor is a
    near-vertical band in it); muhat tauhat is the tail similarity coordinate
    (the far-field free boundary is its level set ~ 1/2). Kinky structures
    become near axis-aligned in feature space, so the net buys them cheaply.

    Fourier features on z were tried (4 sin/cos harmonics, 2026-08-04) and
    REVERTED: the sinusoids imprinted their level sets on the residual, doubled
    the exterior ripple, and even L-BFGS could not make the basis pay.

    -muhat, NOT -|muhat|: on the muhat >= 0 domain they are the same bound,
    but the abs would put a kink at exactly muhat = 0, where the ridge loss
    differentiates -- autograd's sign(0) = 0 would silently drop the
    envelope's one-sided slope and train BC1 against a derivative the
    muhat > 0 side never sees. nu is smooth in its mean, so the smooth form
    keeps the ridge trap honored: the premium stays smooth at the ridge.

    Only trained on muhat >= 0; the true premium is even, so evaluate at
    |muhat| yourself if you must go left. Output at muhat < 0 is garbage
    (and the envelope grows there instead of decaying).
    """

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            # Xavier with the tanh gain (PyTorch's default is relu-flavored and
            # ~4x too small for tanh here); plain gain for the linear head.
            head = i == len(sizes) - 2
            nn.init.xavier_uniform_(
                linear.weight, gain=1.0 if head else nn.init.calculate_gain("tanh")
            )

            # Head bias 1: start alive everywhere. An all-dead relu**2 net
            # has zero loss gradient and never recovers (CLAUDE.md traps).
            if head is True:
                nn.init.ones_(linear.bias)
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        features = torch.stack(
            [muhat, tauhat.log(), muhat * tauhat.sqrt(), muhat * tauhat], dim=-1
        )
        response = self.net(features).squeeze(-1)
        envelope = self.log_scale.exp() * nu(-muhat, tauhat.rsqrt())

        response_squared = torch.relu(response) ** 2

        return envelope * response_squared / (1.0 + response_squared)


class ValueFunction(nn.Module):
    """
    Dimensionless value on top of the premium: v = max(muhat, 0) + u.

    The commit-value term max(mu, 0)/rho rescales to exactly max(muhat, 0), so
    the value costs one relu. Backprop through v; read u off when convenient.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat)


if __name__ == "__main__":
    from .sample import sample_sobol

    premium = ExplorationPremium([32, 16])
    muhat, tauhat = sample_sobol(1000)
    u = premium(muhat, tauhat)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == muhat.shape and u.isfinite().all()

    # At log_scale = 0 the premium obeys both proven properties by
    # construction: 0 <= u <= the free-information bound.
    bound = nu(-muhat, tauhat.rsqrt())

    assert (u >= 0).all() and (u <= bound + 1e-6).all()

    v = ValueFunction(premium)(muhat, tauhat)

    assert torch.allclose(v, torch.relu(muhat) + u)
    print("ok")
