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

from ...net import DeclaresTopology, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ...utils import nu
from .sample import PRIOR_FLOOR, sample_sobol
from .simplex import Maximum, maximize_quadratic

# Width of the feature stack in ExplorationPremium.forward.
FEATURE_COUNT = 4


class ExplorationPremium(DeclaresTopology):
    """
    Dense GainedTanh MLP times the free-information envelope:

        u = exp(log_scale) * nu(-muhat, tauhat**(-1/2)) * y / (1 + y),
        y = relu(response)**2

    nu(-muhat, sd) is the proven upper bound on the premium (kb/three_arm.md
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
        super().__init__(FEATURE_COUNT, hidden)

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

        # Xavier assumes unit-variance inputs; the raw feature stack breaks
        # that under the general sampling law, railing first-layer tanh units
        # (see three_arm's model for the measurement). Calibrate a fixed
        # per-feature scale from the law once at init; a buffer, so
        # checkpoints carry it.
        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        muhat, tauhat = sample_sobol(4096)
        self.feature_scale = self._features(muhat, tauhat).std(dim=0).clamp_min(1e-3)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *rest) -> None:
        super()._load_from_state_dict(state_dict, prefix, *rest)

    def _features(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        """
        The raw (uncalibrated) feature stack.
        """
        return torch.stack(
            [muhat, tauhat.log(), muhat * tauhat.sqrt(), muhat * tauhat], dim=-1
        )

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        # Sub-floor states get the floor's shape continued self-similarly:
        # response at the z-preserving floor state, envelope at the true one
        # (kb/two_arm.md section 9, which also holds the four constructed
        # targets this replaced). Binds only off the sampling law.
        tau_eff = tauhat.clamp_min(PRIOR_FLOOR)
        response = self.net(
            self._features(muhat * (tauhat / tau_eff).sqrt(), tau_eff)
            / self.feature_scale
        )
        response = response.squeeze(-1)

        response_squared = torch.relu(response) ** 2
        gated = self.log_scale.exp() * response_squared / (1.0 + response_squared)

        return nu(-muhat, tauhat.rsqrt()) * gated


class DimensionlessValueFunction(nn.Module):
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
        DECLARES; older ones that declare nothing fall back to inferring the
        widths from the premium net's weight shapes.
        """
        state = torch.load(path)
        hidden, _ = read_topology(state)
        value = cls(ExplorationPremium(hidden))
        value.load_state_dict(state)

        return value

    def forward(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat)

    def hamiltonian(
        self, muhat: Tensor, tauhat: Tensor
    ) -> tuple[Tensor, Maximum, Tensor]:
        """
        The HJB's two sides on the similarity chart (z, s), where the operator

            L_ab[g] = g_s + (1/2) g_zz + (z/2) g_z - (1/2) g

        is O(1)-conditioned at every information level (kb/two_arm.md
        section 8): the equation reads e^s (z + g) = max over alpha of
        alpha e^s z + alpha(1-alpha) L_ab[g]. Returns the left side, the
        maximization, and L_ab itself -- all graph-connected to the premium's
        parameters, so pde_loss grades the gap AND the operator's sign, and
        policy reads the argmax off the same derivation. muhat >= 0 only,
        like forward.
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
        The argmax allocation (treatment share). muhat >= 0 only, like
        forward; ValueFunction.policy handles the arm swap.
        """
        _, best, _ = self.hamiltonian(muhat, tauhat)

        return best.x.detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, either sign of the mean.

    Wraps a trained DimensionlessValueFunction with the readout dictionary
    (muhat = mu / (sigma sqrt(rho)), tauhat = rho sigma^2 tau,
    V = (sigma / sqrt(rho)) vhat) and evaluates the premium at |muhat| -- the
    true premium is even, and the net is only trained on muhat >= 0 (its
    docstring's warning) -- while the commit term keeps the sign.
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
        The argmax allocation (treatment share): the dimensionless policy
        evaluated at |muhat| and reflected back by the arm swap.
        """
        muhat = mu / (self.sigma * self.rho**0.5)
        tauhat = self.rho * self.sigma**2 * tau
        alpha = self.dimensionless.policy(muhat.abs(), tauhat)

        return torch.where(muhat >= 0, alpha, 1.0 - alpha)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from: fresh at `topology`, or adapted from an
    existing checkpoint's `state`. Exactly one of the two.

    The CLI reads the file; this takes the state dict. A problem module has no
    business knowing where checkpoints live, and passing the dict keeps
    `pinn init --from` the only place that decides what a source file means.
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

    # At log_scale = 0 the premium obeys both proven properties by
    # construction: 0 <= u <= the free-information bound.
    bound = nu(-muhat, tauhat.rsqrt())

    assert (u >= 0).all() and (u <= bound + 1e-6).all()

    # The self-similar continuation below the floor: there the premium in
    # shape units is a function of z ALONE -- the floor's own shape, frozen
    # -- so it is identical at every tauhat <= PRIOR_FLOOR and continuous
    # into the floor itself.
    z_grid = torch.linspace(0.0, 3.0, 16)

    def shape(tauhat_value: float) -> Tensor:
        deep = torch.full((16,), tauhat_value)

        return deep.sqrt() * premium(z_grid / deep.sqrt(), deep)

    assert torch.allclose(shape(PRIOR_FLOOR), shape(PRIOR_FLOOR / 16.0), atol=1e-6)
    assert torch.allclose(shape(PRIOR_FLOOR), shape(PRIOR_FLOOR / 1024.0), atol=1e-6)

    v = DimensionlessValueFunction(premium)(muhat, tauhat)

    assert torch.allclose(v, torch.relu(muhat) + u)

    # The deployment wrapper: at rho = sigma = 1 on muhat >= 0 it equals the
    # dimensionless form; across the ridge V(mu) - V(-mu) is exactly the
    # commit-value gap mu / rho; and units scale by the readout dictionary.
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
