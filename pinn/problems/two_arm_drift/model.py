"""
Models for the drift problem: the premium net, the trainer-facing value, and the
deployment adapter. Structured in deliberate parallel to two_arm/model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import DeclaresTopology, DimensionlessValue, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ..two_arm.simplex import Maximum, maximize_quadratic
from .envelope import envelope
from .sample import sample_sobol

FEATURE_COUNT = 5
"""
Width of the feature stack in `ExplorationPremium.forward`: two_arm's four, plus the
drift coordinate.
"""


class ExplorationPremium(DeclaresTopology):
    """
    two_arm's premium net with one extra feature and a wider envelope:
    `u = exp(log_scale) * u_env(muhat, tauhat, etahat) * y/(1 + y)`, `y = relu(r)**2`.

    Read two_arm's docstring first: the response map, the features and the
    rational saturation are all its. The envelope is kb/two_arm_drift.md section
    7; the fifth feature and what the response costs under drift are section 8.
    Only trained on muhat >= 0, and the true premium is even, so evaluate at
    |muhat| to go left.
    """

    def __init__(self, hidden: list[int], kinks: int = 0) -> None:
        super().__init__(FEATURE_COUNT, hidden, kinks)

        sizes = [FEATURE_COUNT, *hidden, 1]
        layers: list[nn.Module] = []

        for i in range(len(sizes) - 1):
            linear = nn.Linear(sizes[i], sizes[i + 1])

            head = i == len(sizes) - 2
            nn.init.xavier_uniform_(
                linear.weight, gain=1.0 if head else nn.init.calculate_gain("tanh")
            )

            # Head bias 1, two_arm's: relu**2 has an all-dead absorbing state (u = 0
            # zeroes every gradient), so every unit must start alive.
            if head is True:
                nn.init.constant_(linear.bias, 1.0)
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])

        # Kink units: curvature-jump primitives for the commit/explore surface. Zero-init
        # out (a graft is a no-op at step 0), bias +0.5 (start alive), y/(1+y) (cannot
        # colonise the bulk). Stitch, never co-train (kb/two_arm_drift.md section 8).
        self.kinks = kinks

        if kinks > 0:
            self.kink_in = nn.Linear(FEATURE_COUNT, kinks)
            self.kink_out = nn.Linear(kinks, 1)
            nn.init.zeros_(self.kink_out.weight)
            nn.init.zeros_(self.kink_out.bias)
            nn.init.constant_(self.kink_in.bias, 0.5)

        self.log_scale = nn.Parameter(torch.zeros(()))

        self.register_buffer("feature_scale", torch.ones(FEATURE_COUNT))

        muhat, tauhat, etahat = sample_sobol(4096)
        self.feature_scale = (
            self._features(muhat, tauhat, etahat).std(dim=0).clamp_min(1e-3)
        )

    def stitch(self, source: dict) -> None:
        """
        Adopt another premium's parameters: a two_arm checkpoint (one feature
        narrower) or a drift one with no kink branch. Bitwise at etahat = 0 and
        function-preserving into a wider target; the rules are kb/two_arm_drift.md
        section 8.
        """
        state = dict(source)

        if read_features(state, prefix="") == FEATURE_COUNT - 1:
            state["feature_scale"] = torch.cat(
                [state["feature_scale"], self.feature_scale[-1:]]
            )

        grafted = {}

        for name, want in self.state_dict().items():
            have = state.get(name)

            # The shape buffers describe *this* net, never the source: a graft changes
            # the feature width and can change the kink count, so copying the source's
            # declaration would make the net save a shape its own weights disprove.
            if name in ("topology", "kink_count"):
                grafted[name] = want
                continue

            # Defaulting the kink branch by name, not by a `kink_` prefix that would also
            # catch kink_count. Anything else missing is a real mismatch and should fail
            # loudly. The graft rules are kb/two_arm_drift.md section 8.
            if have is None:
                if name.split(".")[0] not in ("kink_in", "kink_out"):
                    raise KeyError(f"source is missing {name}")

                grafted[name] = want
                continue

            if have.shape == want.shape:
                grafted[name] = have
                continue

            room = want.clone()

            if room.dim() == 2:
                room[:, have.shape[1] :] = 0.0
                room[: have.shape[0], : have.shape[1]] = have
            else:
                room[: have.shape[0]] = have

            grafted[name] = room

        self.load_state_dict(grafted)

    def _features(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
        """
        The raw (uncalibrated) feature stack: two_arm's four, then the drift coordinate.
        Order matters, since `stitch` pads on the right.
        """
        return torch.stack(
            [
                muhat,
                tauhat.log(),
                muhat * tauhat.sqrt(),
                muhat * tauhat,
                torch.log1p(2.0 * etahat * tauhat),
            ],
            dim=-1,
        )

    def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
        features = self._features(muhat, tauhat, etahat) / self.feature_scale
        response = self.net(features).squeeze(-1)

        if self.kinks > 0:
            bumps = torch.relu(self.kink_in(features)) ** 2
            response = response + self.kink_out(bumps / (1.0 + bumps)).squeeze(-1)
        y = torch.relu(response) ** 2
        gated = self.log_scale.exp() * y / (1.0 + y)

        # Grouped as two_arm groups it, envelope *last*: float multiplication does not
        # associate, so another grouping is the same function to 1e-7 but breaks the
        # bitwise etahat = 0 bootstrap (kb section 9, item 6). The self-check catches it.
        return envelope(muhat, tauhat, etahat) * gated


class DimensionlessValueFunction(DimensionlessValue):
    """
    Dimensionless value on top of the premium: v = max(muhat, 0) + u, exactly two_arm's
    split. The commit-value term rescales to max(muhat, 0), so the value costs one relu.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """A trained checkpoint as a model, at the architecture it declares."""
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        value = cls(ExplorationPremium(hidden, kinks=kinks))
        value.load_state_dict(state)

        return value

    def bind(self, rho: float, sigma: float, eta: float) -> ValueFunction:
        """
        This net tied to one experiment, which is what makes it usable: means and
        precisions go in and come back in your own units. `eta` is the drift rate of the
        effect itself.
        """
        return ValueFunction(self, rho, sigma, eta)

    def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat, etahat)

    def hamiltonian(
        self, muhat: Tensor, tauhat: Tensor, etahat: Tensor
    ) -> tuple[Tensor, Maximum, Tensor]:
        """
        The HJB's two sides on the similarity chart (z, s), transcribed in
        kb/two_arm_drift.md section 6, where etahat = 0 is two_arm's equation exactly.
        Returns the left side, the maximization and L_ab, all graph-connected to the
        premium, so `subsolution_loss` grades the gap and `policy` reads the argmax.
        """
        z = (muhat * tauhat.sqrt()).detach().requires_grad_(True)
        s = tauhat.log().detach().requires_grad_(True)

        g = (s / 2).exp() * self.premium(z * (-s / 2).exp(), s.exp(), etahat)
        g_z, g_s = torch.autograd.grad(g.sum(), [z, s], create_graph=True)
        (g_zz,) = torch.autograd.grad(g_z.sum(), z, create_graph=True)

        l_ab = g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g
        tauhat_slope = l_ab - 0.5 * g_zz
        best = maximize_quadratic(-l_ab, s.exp() * z + l_ab)
        left = s.exp() * (z + g) + etahat**2 * (2.0 * s).exp() * tauhat_slope

        return left, best, l_ab

    def policy(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
        """
        The argmax allocation (treatment share). muhat >= 0 only, like `forward`, since
        `ValueFunction.policy` handles the arm swap.
        """
        _, best, _ = self.hamiltonian(muhat, tauhat, etahat)

        return best.x.detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, either sign of the mean.

    Wraps a trained `DimensionlessValueFunction` with the readout dictionary
    (muhat = mu/(sigma sqrt(rho)), tauhat = rho sigma^2 tau,
    etahat = eta/(rho sigma), V = (sigma/sqrt(rho)) vhat) and evaluates the premium at
    |muhat| (the true premium is even, and the net is only trained on muhat >= 0) while
    the commit term keeps the sign.
    """

    def __init__(
        self,
        dimensionless: DimensionlessValueFunction,
        rho: float,
        sigma: float,
        eta: float,
    ) -> None:
        super().__init__()

        self.dimensionless = dimensionless
        self.rho = rho
        self.sigma = sigma
        self.eta = eta

    def forward(self, mu: Tensor, tau: Tensor) -> Tensor:
        muhat = mu / (self.sigma * self.rho**0.5)
        tauhat = self.rho * self.sigma**2 * tau
        etahat = torch.full_like(muhat, self.eta / (self.rho * self.sigma))
        vhat = torch.relu(muhat) + self.dimensionless.premium(
            muhat.abs(), tauhat, etahat
        )

        return (self.sigma / self.rho**0.5) * vhat

    def policy(self, mu: Tensor, tau: Tensor) -> Tensor:
        """
        The argmax allocation (treatment share): the dimensionless policy evaluated at
        |muhat| and reflected back by the arm swap.
        """
        muhat = mu / (self.sigma * self.rho**0.5)
        tauhat = self.rho * self.sigma**2 * tau
        etahat = torch.full_like(muhat, self.eta / (self.rho * self.sigma))
        alpha = self.dimensionless.policy(muhat.abs(), tauhat, etahat)

        return torch.where(muhat >= 0, alpha, 1.0 - alpha)


def init_model(
    state: dict | None = None, topology: str | None = None
) -> DimensionlessValueFunction:
    """
    A model to start training from: fresh at `topology`, or adapted from an
    existing checkpoint's `state`. Exactly one of the two. The CLI reads the file
    and hands over the dict, so `pinn init --from` stays the only place that
    decides what a source file means.
    """
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))

        # Both: topology is the *target* shape, state the source to adapt into it. This
        # is how a kink branch is grafted onto a trained smooth net.
        if state is not None:
            value.premium.stitch(
                {k.removeprefix("premium."): v for k, v in state.items()}
            )

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))
    features = read_features(state)

    # A two_arm checkpoint is one feature narrower: stitch it as the etahat = 0 slice.
    if features == FEATURE_COUNT - 1:
        value.premium.stitch({k.removeprefix("premium."): v for k, v in state.items()})

        return value
    value.load_state_dict(state)

    return value


if __name__ == "__main__":
    from ..two_arm.model import ExplorationPremium as TwoArmPremium
    from .envelope import envelope as bound_of

    premium = ExplorationPremium([32, 16])
    muhat, tauhat, etahat = sample_sobol(1000)
    u = premium(muhat, tauhat, etahat)

    assert premium.net[0].in_features == FEATURE_COUNT
    assert u.shape == muhat.shape and u.isfinite().all()

    # 0 <= u < envelope, architectural: the free-information argument *proves* the
    # upper bound and no loss term is as tight as a construction. <=, not <, since
    # the envelope underflows to 0 in the far corridor and u goes with it.
    assert (u >= 0.0).all() and (u <= bound_of(muhat, tauhat, etahat) + 1e-6).all()

    v = DimensionlessValueFunction(premium)(muhat, tauhat, etahat)

    assert torch.allclose(v, torch.relu(muhat) + u)

    # The stitch: at etahat = 0 the net sees exactly two_arm's four inputs and the
    # envelope collapses onto nu, so the whole premium is bitwise.
    two_arm = TwoArmPremium([32, 16])
    stitched = ExplorationPremium([32, 16])
    stitched.stitch(two_arm.state_dict())

    zero = torch.zeros_like(muhat)

    scaled = stitched._features(muhat, tauhat, zero) / stitched.feature_scale
    two_arm_scaled = two_arm._features(muhat, tauhat) / two_arm.feature_scale

    assert torch.equal(
        stitched.net(scaled).squeeze(-1), two_arm.net(two_arm_scaled).squeeze(-1)
    )
    assert torch.equal(stitched.feature_scale[:4], two_arm.feature_scale)

    # The kink graft is a no-op at step 0: zero-init kink_out, so a smooth checkpoint
    # grafted with kinks is bitwise the same function.
    smooth = ExplorationPremium([16, 8])
    grafted = ExplorationPremium([16, 8], kinks=8)
    grafted.stitch(smooth.state_dict())

    assert grafted.kinks == 8
    assert torch.equal(grafted(muhat, tauhat, etahat), smooth(muhat, tauhat, etahat))

    # And the branch is trainable, not decoration.
    assert grafted.kink_out.weight.requires_grad
    assert (grafted.kink_in.bias == 0.5).all()

    # Function-preserving: a wider graft computes the same function at step 0, though
    # not bitwise, since the matmuls accumulate over more (zero) columns.
    narrow = ExplorationPremium([16, 8])
    wider = ExplorationPremium([48, 24])
    wider.stitch(narrow.state_dict())

    assert torch.allclose(
        wider(muhat, tauhat, etahat),
        narrow(muhat, tauhat, etahat),
        rtol=1e-5,
        atol=1e-7,
    )

    # The headroom is trainable: new units are live, simply unheard until their output
    # weights leave zero.
    assert wider.net[0].weight.shape == (48, FEATURE_COUNT)
    assert (wider.net[2].weight[:, 16:] == 0.0).all()
    assert (wider.net[4].weight[:, 8:] == 0.0).all()
    assert wider.net[2].weight.requires_grad

    # Composes with the kink graft: both are the same leading-slice rule.
    both = ExplorationPremium([48, 24], kinks=8)
    both.stitch(narrow.state_dict())

    assert torch.allclose(
        both(muhat, tauhat, etahat),
        narrow(muhat, tauhat, etahat),
        rtol=1e-5,
        atol=1e-7,
    )

    # Bitwise, not merely close: the exact bootstrap
    # `pinn init --problem two_arm_drift --from data/two_arm.pt` rests on it.
    with torch.no_grad():
        assert torch.equal(stitched(muhat, tauhat, zero), two_arm(muhat, tauhat))

    # The deployment wrapper: at rho = sigma = 1, eta = 0 on muhat >= 0 it equals the
    # dimensionless form at etahat = 0; across the ridge V(mu) - V(-mu) is exactly the
    # commit-value gap; units scale by the readout dictionary.
    dimensionless = DimensionlessValueFunction(premium)
    wrapper = ValueFunction(dimensionless, rho=1.0, sigma=1.0, eta=0.0)

    assert torch.allclose(
        wrapper(muhat, tauhat), torch.relu(muhat) + premium(muhat, tauhat, zero)
    )
    assert torch.allclose(
        wrapper(muhat, tauhat) - wrapper(-muhat, tauhat), muhat, atol=1e-5
    )

    rho, sigma, eta = 0.04, 2.5, 0.3
    real = ValueFunction(dimensionless, rho=rho, sigma=sigma, eta=eta)
    etahat_of = torch.full_like(muhat, eta / (rho * sigma))
    reference = torch.relu(muhat) + premium(muhat, tauhat, etahat_of)

    assert torch.allclose(
        real(muhat * sigma * rho**0.5, tauhat / (rho * sigma**2)),
        (sigma / rho**0.5) * reference,
        rtol=1e-5,
    )

    # The policy: a valid allocation, antisymmetric across the ridge.
    alpha = wrapper.policy(muhat, tauhat)

    assert (alpha >= 0).all() and (alpha <= 1).all()
    assert torch.allclose(
        alpha + wrapper.policy(-muhat, tauhat), torch.ones_like(alpha)
    )
    print("ok")
