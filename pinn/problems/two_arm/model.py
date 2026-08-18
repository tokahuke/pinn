"""
Models for the two-arm problem: the premium net and the value wrapper that
carries the commit envelope's kink.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import (
    DeclaresTopology,
    DimensionlessValue,
    GainedTanh,
    parse_topology,
    read_features,
    read_topology,
)
from ...utils import nu
from .sample import PRIOR_FLOOR, sample_sobol
from .simplex import Maximum, maximize_quadratic

FEATURE_COUNT = 4
"""Width of the feature stack in `ExplorationPremium.forward`."""


class ExplorationPremium(DeclaresTopology):
    """
    Dense GainedTanh MLP times the free-information envelope, `y = relu(response)**2`:

        u = exp(log_scale) * nu(-muhat, tauhat**(-1/2)) * y / (1 + y)

    `0 <= u < envelope` is architectural and `log_scale = 0` starts at the proven
    bound. Why each factor has its shape: kb/two_arm.md section 11. Trained on
    `muhat >= 0` only; the premium is even, so evaluate at `|muhat|` to go left.
    """

    def __init__(self, hidden: list[int]) -> None:
        super().__init__(FEATURE_COUNT, hidden)

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            # Xavier with the tanh gain, since PyTorch's default is relu-flavored and
            # ~4x too small for tanh here. Plain gain for the linear head.
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

        # Xavier assumes unit-variance inputs and the raw feature stack breaks that,
        # so calibrate a scale from the law once at init, as a buffer so checkpoints
        # carry it (kb/two_arm.md section 11).
        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        muhat, tauhat = sample_sobol(4096)
        self.feature_scale = self._features(muhat, tauhat).std(dim=0).clamp_min(1e-3)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *rest) -> None:
        super()._load_from_state_dict(state_dict, prefix, *rest)

    def _features(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        """The raw (uncalibrated) feature stack."""
        return torch.stack(
            [muhat, tauhat.log(), muhat * tauhat.sqrt(), muhat * tauhat], dim=-1
        )

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        # Sub-floor states get the floor's shape continued self-similarly: response at
        # the z-preserving floor state, envelope at the true one (kb/two_arm.md
        # section 9). Binds only off the sampling law.
        tau_eff = tauhat.clamp_min(PRIOR_FLOOR)
        response = self.net(
            self._features(muhat * (tauhat / tau_eff).sqrt(), tau_eff)
            / self.feature_scale
        )
        response = response.squeeze(-1)

        response_squared = torch.relu(response) ** 2
        gated = self.log_scale.exp() * response_squared / (1.0 + response_squared)

        return nu(-muhat, tauhat.rsqrt()) * gated


class DimensionlessValueFunction(DimensionlessValue):
    """
    Dimensionless value on top of the premium: v = max(muhat, 0) + u.

    The commit-value term max(mu, 0)/rho rescales to exactly max(muhat, 0), so
    the value costs one relu. Backprop through v; read u off when convenient.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        A trained checkpoint as a model, at the architecture the checkpoint
        *declares*. A file that declares nothing raises a KeyError, which means it
        predates the declaration and is a file to migrate.
        """
        state = torch.load(path)
        hidden, _ = read_topology(state)
        value = cls(ExplorationPremium(hidden))
        value.load_state_dict(state)

        return value

    def bind(self, rho: float, sigma: float) -> ValueFunction:
        """
        This net tied to one experiment, which is what makes it usable: means
        and precisions go in and come back in your own units.
        """
        return ValueFunction(self, rho, sigma)

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat)

    def hamiltonian(
        self, muhat: Tensor, tauhat: Tensor
    ) -> tuple[Tensor, Maximum, Tensor]:
        """
        The HJB's two sides on the similarity chart (z, s), O(1)-conditioned at every
        information level (kb/two_arm.md section 8). Returns the left side, the
        maximization and `L_ab`, graph-connected so policy reads the same argmax.
        """
        z = (muhat * tauhat.sqrt()).detach().requires_grad_(True)
        s = tauhat.log().detach().requires_grad_(True)

        g = (s / 2).exp() * self.premium(z * (-s / 2).exp(), s.exp())
        g_z, g_s = torch.autograd.grad(g.sum(), [z, s], create_graph=True)
        (g_zz,) = torch.autograd.grad(g_z.sum(), z, create_graph=True)

        l_ab = g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g
        best = maximize_quadratic(-l_ab, s.exp() * z + l_ab)

        return s.exp() * (z + g), best, l_ab

    def policy(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        """
        The argmax allocation (treatment share). `muhat >= 0` only, like forward,
        because `ValueFunction.policy` handles the arm swap.
        """
        _, best, _ = self.hamiltonian(muhat, tauhat)

        return best.x.detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, either sign of the mean.

    Wraps a trained `DimensionlessValueFunction` with the readout dictionary
    (`muhat = mu / (sigma sqrt(rho))`, `tauhat = rho sigma^2 tau`,
    `V = (sigma / sqrt(rho)) vhat`) and evaluates the premium at `|muhat|`, since the
    true premium is even and the net is only trained on `muhat >= 0`. The commit term
    keeps the sign.
    """

    def __init__(
        self, dimensionless: DimensionlessValueFunction, rho: float, sigma: float
    ) -> None:
        super().__init__()

        self.dimensionless = dimensionless
        self.rho = rho
        self.sigma = sigma

    def forward(self, mu: Tensor, tau: Tensor) -> Tensor:
        muhat = mu / (self.sigma * self.rho**0.5)
        tauhat = self.rho * self.sigma**2 * tau
        vhat = torch.relu(muhat) + self.dimensionless.premium(muhat.abs(), tauhat)

        return (self.sigma / self.rho**0.5) * vhat

    def policy(self, mu: Tensor, tau: Tensor) -> Tensor:
        """
        The argmax allocation (treatment share): the dimensionless policy evaluated
        at `|muhat|` and reflected back by the arm swap.
        """
        muhat = mu / (self.sigma * self.rho**0.5)
        tauhat = self.rho * self.sigma**2 * tau
        alpha = self.dimensionless.policy(muhat.abs(), tauhat)

        return torch.where(muhat >= 0, alpha, 1.0 - alpha)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from: fresh at `topology`, or adapted from an existing
    checkpoint's `state`. Exactly one of the two. Takes the state dict rather than a
    path, so `pinn init --from` stays the only place that reads a source file.
    """
    if (state is None) == (topology is None):
        raise ValueError("pass exactly one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)

        if kinks > 0:
            raise ValueError(f"{__package__.rsplit('.', 1)[-1]} has no kink branch")

        return DimensionlessValueFunction(ExplorationPremium(hidden))

    hidden, _ = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(hidden))
    value.load_state_dict(state)

    return value


if __name__ == "__main__":
    premium = ExplorationPremium([32, 16])
    muhat, tauhat = sample_sobol(1000)
    u = premium(muhat, tauhat)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == muhat.shape and u.isfinite().all()

    # At `log_scale = 0` the premium obeys both proven properties by construction: it
    # sits between 0 and the free-information bound.
    bound = nu(-muhat, tauhat.rsqrt())

    assert (u >= 0).all() and (u <= bound + 1e-6).all()

    # The self-similar continuation below the floor: there the premium in shape units
    # is a function of z *alone* (the floor's own shape, frozen), so it is identical
    # at every `tauhat <= PRIOR_FLOOR` and continuous into the floor itself.
    z_grid = torch.linspace(0.0, 3.0, 16)

    def shape(tauhat_value: float) -> Tensor:
        """The premium in shape units along the z grid, at one tauhat."""
        deep = torch.full((16,), tauhat_value)

        return deep.sqrt() * premium(z_grid / deep.sqrt(), deep)

    assert torch.allclose(shape(PRIOR_FLOOR), shape(PRIOR_FLOOR / 16.0), atol=1e-6)
    assert torch.allclose(shape(PRIOR_FLOOR), shape(PRIOR_FLOOR / 1024.0), atol=1e-6)

    v = DimensionlessValueFunction(premium)(muhat, tauhat)

    assert torch.allclose(v, torch.relu(muhat) + u)

    # The deployment wrapper: at `rho = sigma = 1` on `muhat >= 0` it equals the
    # dimensionless form, across the ridge `V(mu) - V(-mu)` is exactly the
    # commit-value gap `mu / rho`, and units scale by the readout dictionary.
    dimensionless = DimensionlessValueFunction(premium)
    wrapper = ValueFunction(dimensionless, rho=1.0, sigma=1.0)

    assert torch.allclose(wrapper(muhat, tauhat), v, atol=1e-6)
    assert torch.allclose(
        wrapper(muhat, tauhat) - wrapper(-muhat, tauhat), muhat, atol=1e-5
    )

    rho, sigma = 0.04, 2.5
    real = ValueFunction(dimensionless, rho=rho, sigma=sigma)

    assert torch.allclose(
        real(muhat * sigma * rho**0.5, tauhat / (rho * sigma**2)),
        (sigma / rho**0.5) * v,
        rtol=1e-5,
    )

    # The policy: a valid allocation, antisymmetric across the ridge by the
    # arm swap.
    alpha = wrapper.policy(muhat, tauhat)

    assert (alpha >= 0).all() and (alpha <= 1).all()
    assert torch.allclose(
        alpha + wrapper.policy(-muhat, tauhat), torch.ones_like(alpha)
    )
    print("ok")
