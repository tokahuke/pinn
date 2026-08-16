"""
Models for the drift problem: the premium net, the trainer-facing value, and
the deployment adapter. Structured in deliberate parallel to two_arm/model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pathlib import Path
from torch import Tensor
from typing import Self

from ...net import DeclaresTopology, GainedTanh, parse_topology
from ...net import read_features, read_topology
from ..two_arm.simplex import Maximum, maximize_quadratic
from .envelope import envelope
from .sample import sample_sobol

# Width of the feature stack in ExplorationPremium.forward: two_arm's four,
# plus the drift coordinate.
FEATURE_COUNT = 5


class ExplorationPremium(DeclaresTopology):
    """
    two_arm's premium net with one extra feature and a wider envelope:

        u = exp(log_scale) * u_env(muhat, tauhat, etahat) * y / (1 + y),
        y = relu(response)**2

    Read two_arm's docstring first; everything it says about the response map,
    the feature choices and the rational saturation still applies. The envelope
    is kb/two_arm_drift.md section 7, the fifth feature is section 8.

    RESPONSE: two_arm's, unchanged. It maps into [0, 1), so 0 <= u < envelope
    is ARCHITECTURAL, and it reaches exactly 0 with a curvature jump. Two
    intermediate maps were tried and both gave the bound away for nothing --
    sigmoid cannot represent 0, exp is unbounded above (0.53% of points went
    over the envelope, up to 1.16x, at the tauhat floor).

    Drift has no contact set -- committing is not absorbing, so u > 0 strictly
    -- but u = 0 also solves the interior equation exactly, and the loss pays
    the net to spread it: measured 2026-08-07, the dead region grew 52% -> 65%
    over training until a 16% live sliver carried 41% of the squared residual.
    That is a training pathology to fix in the loss, not a reason to surrender
    an exact upper bound the free-information argument proves.

    Only trained on muhat >= 0; the true premium is even, so evaluate at
    |muhat| yourself if you must go left.
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

            # Head bias 1, two_arm's: relu**2 has an all-dead absorbing state
            # (u = 0 zeroes every gradient), so every unit must start alive.
            if head is True:
                nn.init.constant_(linear.bias, 1.0)
            layers.append(linear)
            layers.append(GainedTanh(sizes[i + 1]))

        self.net = nn.Sequential(*layers[:-1])

        # Kink units: curvature-jump primitives for the commit/explore surface,
        # which relu**2's own zero set no longer sits on under drift. Zero-init
        # out (graft is a no-op at step 0), bias +0.5 (start alive), y/(1+y)
        # (cannot colonise the bulk). Stitch, do not co-train.
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
        narrower), or a drift one with no kink branch where this net has one.

        Pad the drift column of the first layer with zeros and append its
        calibrated scale, keeping two_arm's four. At etahat = 0 the fifth
        feature is exactly 0 and the envelope collapses onto nu, so the net
        sees exactly two_arm's inputs there.

        Bitwise at etahat = 0: same response map as two_arm, and the envelope
        collapses onto nu there, so the stitched net IS the source. That exact
        bootstrap is what the shared map buys, and the __main__ check holds it.

        WIDER targets graft the same way, function-preserving: the source block
        lands in the leading slice, new OUTPUT units keep this net's fresh init,
        and every new INPUT column is ZEROED so nothing new feeds the old path.
        The one-feature pad is the same rule on the input dimension.
        """
        state = dict(source)

        if read_features(state, prefix="") == FEATURE_COUNT - 1:
            state["feature_scale"] = torch.cat(
                [state["feature_scale"], self.feature_scale[-1:]]
            )

        grafted = {}

        for name, want in self.state_dict().items():
            have = state.get(name)

            # The shape buffers describe THIS net, never the source: a graft
            # changes the feature width and can change the kink count, so
            # copying the source's declaration would make the net save a shape
            # its own weights disprove.
            if name in ("topology", "kink_count"):
                grafted[name] = want
                continue

            # A source without kinks keeps this net's zero-init branch, so the
            # graft is bit-exact at step 0. Only the kink BRANCH is defaulted --
            # by name, since a "kink_" prefix also catches kink_count -- and
            # anything else missing is a real mismatch that should fail loudly.
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
        The raw (uncalibrated) feature stack: two_arm's four, then the drift
        coordinate. Order matters -- stitch pads on the right.
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

        # Associated as two_arm does it, envelope LAST. The old grouping
        # (log_scale * envelope) * gate is the same function to 1e-7 and broke
        # the bitwise etahat = 0 bootstrap the moment two_arm's forward was
        # rewritten -- float multiplication does not associate, and the
        # self-check below is the one that catches it.
        return envelope(muhat, tauhat, etahat) * gated


class DimensionlessValueFunction(nn.Module):
    """
    Dimensionless value on top of the premium: v = max(muhat, 0) + u, exactly
    two_arm's split. The commit-value term rescales to max(muhat, 0), so the
    value costs one relu.
    """

    def __init__(self, premium: nn.Module) -> None:
        super().__init__()

        self.premium = premium

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        A trained checkpoint as a model, architecture inferred from the state
        dict (hidden widths from the premium net's weight shapes).
        """
        state = torch.load(path)
        hidden, kinks = read_topology(state)
        value = cls(ExplorationPremium(hidden, kinks=kinks))
        value.load_state_dict(state)

        return value

    def forward(self, muhat: Tensor, tauhat: Tensor, etahat: Tensor) -> Tensor:
        return torch.relu(muhat) + self.premium(muhat, tauhat, etahat)

    def hamiltonian(
        self, muhat: Tensor, tauhat: Tensor, etahat: Tensor
    ) -> tuple[Tensor, Maximum, Tensor]:
        """
        The HJB's two sides on the similarity chart (z, s) of
        kb/two_arm_drift.md section 6, where with

            L_ab[g]      = g_s + (1/2) g_zz + (z/2) g_z - (1/2) g
            tauhat_slope = L_ab[g] - (1/2) g_zz

        the equation reads

            e^s (z + g) + etahat^2 e^(2s) tauhat_slope
                = max over alpha of { alpha e^s z + alpha(1-alpha) L_ab[g] }

        etahat = 0 is two_arm's equation exactly. The drift term is
        tauhat^2 v_tauhat transcribed, and it lands derivative-free-weighted
        (carrying e^(2s)) like the source, so the O(1) conditioning of the
        similarity chart survives: no derivative term is multiplied by a large
        number. Returns the left side and the maximization, both
        graph-connected to the premium's parameters, so subsolution_loss grades
        their gap and policy reads the argmax off the same derivation.
        muhat >= 0 only, like forward.

        Returns L_ab as well. It is the learning operator -- the value of being
        able to change your mind -- and it is what the POLICY hangs on, while
        the residual only ever sees it through alpha(1-alpha) <= 1/4. It is
        provably >= 0 for the answer (learning is a mean-preserving spread on a
        belief the value is convex in), so the loss can grade its negative part.
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
        The argmax allocation (treatment share). muhat >= 0 only, like forward;
        ValueFunction.policy handles the arm swap.
        """
        _, best, _ = self.hamiltonian(muhat, tauhat, etahat)

        return best.x.detach()


class ValueFunction(nn.Module):
    """
    Deployment-facing value: real units in and out, either sign of the mean.

    Wraps a trained DimensionlessValueFunction with the readout dictionary
    (muhat = mu/(sigma sqrt(rho)), tauhat = rho sigma^2 tau, etahat =
    eta/(rho sigma), V = (sigma/sqrt(rho)) vhat) and evaluates the premium at
    |muhat| -- the true premium is even, and the net is only trained on
    muhat >= 0 -- while the commit term keeps the sign.
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
        The argmax allocation (treatment share): the dimensionless policy
        evaluated at |muhat| and reflected back by the arm swap.
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
    existing checkpoint's `state`. Exactly one of the two.

    The CLI reads the file; this takes the state dict. A problem module has no
    business knowing where checkpoints live, and passing the dict keeps
    `pinn init --from` the only place that decides what a source file means.
    """
    if state is None and topology is None:
        raise ValueError("pass at least one of state, topology")

    if topology is not None:
        hidden, kinks = parse_topology(topology)
        value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))

        # Both: topology is the TARGET shape, state the source to adapt into
        # it. This is how a kink branch is grafted onto a trained smooth net.
        if state is not None:
            value.premium.stitch(
                {k.removeprefix("premium."): v for k, v in state.items()}
            )

        return value

    hidden, kinks = read_topology(state)
    value = DimensionlessValueFunction(ExplorationPremium(hidden, kinks=kinks))
    features = read_features(state)

    # A two_arm checkpoint is one feature narrower: stitch it as the etahat = 0
    # slice.
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

    # 0 <= u < envelope, architectural. The reason the response map is not
    # negotiable: the free-information argument PROVES the upper bound, and
    # no loss term can be as tight as a construction.
    # <=, not <: the envelope underflows to 0 in the far corridor and u goes
    # with it. two_arm's check, verbatim.
    assert (u >= 0.0).all() and (u <= bound_of(muhat, tauhat, etahat) + 1e-6).all()

    v = DimensionlessValueFunction(premium)(muhat, tauhat, etahat)

    assert torch.allclose(v, torch.relu(muhat) + u)

    # The stitch: at etahat = 0 the net sees exactly two_arm's four inputs and
    # the envelope collapses onto nu, so the whole premium is bitwise.
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

    # The kink graft is a no-op at step 0: zero-init kink_out, so a smooth
    # checkpoint grafted with kinks is bitwise the same function.
    smooth = ExplorationPremium([16, 8])
    grafted = ExplorationPremium([16, 8], kinks=8)
    grafted.stitch(smooth.state_dict())

    assert grafted.kinks == 8
    assert torch.equal(grafted(muhat, tauhat, etahat), smooth(muhat, tauhat, etahat))

    # And the branch is trainable, not decoration.
    assert grafted.kink_out.weight.requires_grad
    assert (grafted.kink_in.bias == 0.5).all()

    # Function-preserving: a wider graft computes the same function at step 0.
    # Not bitwise -- the matmuls accumulate over more (zero) columns.
    narrow = ExplorationPremium([16, 8])
    wider = ExplorationPremium([48, 24])
    wider.stitch(narrow.state_dict())

    assert torch.allclose(
        wider(muhat, tauhat, etahat),
        narrow(muhat, tauhat, etahat),
        rtol=1e-5,
        atol=1e-7,
    )

    # The headroom is trainable: new units are live, simply unheard until their
    # output weights leave zero.
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

    # The deployment wrapper: at rho = sigma = 1, eta = 0 on muhat >= 0 it
    # equals the dimensionless form at etahat = 0; across the ridge
    # V(mu) - V(-mu) is exactly the commit-value gap; units scale by the
    # readout dictionary.
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
