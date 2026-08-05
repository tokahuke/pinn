# Two-Armed Bayesian Allocation — HJB Free-Boundary Problem

Briefing for a working session. Everything below is exact unless flagged. Goal at the end.

## 0. Notation

- Uppercase = dimensional quantities: `V` the value, `U` the exploration premium.
- Lowercase = dimensionless quantities: `u`.
- Hats = dimensionless coordinates: `muhat`, `tauhat`.
- Free boundary: `G(tauhat)` (dimensional location `mu_Gamma`). Similarity profile: `F(xi)`.
- Earlier drafts used `U` for the even part `V - mu/(2 rho)`; statements from those sessions
  transcribe via `u_even = u + |muhat|/2`.

## 1. Problem statement

Observations from the uncertain arm: `x_t ~ N(mu, sigma^2 / (alpha(1-alpha)))` per unit time,
with `alpha in [0,1]` the traffic split between the two arms. Unknown mean `mu` with conjugate
Gaussian prior; known noise `sigma^2`. Expected reward flow `alpha * mu`, discount rate `rho`.

State: conjugate posterior `(mu, tau)` — posterior mean and precision.

- Precision advances deterministically: `dtau/dt = alpha(1-alpha)/sigma^2`.
  Self-limiting: learning dies at both `alpha = 0` and `alpha = 1`.
- The posterior mean is a martingale with innovation variance `(dtau/dt)/tau^2 dt`.

## 2. Fixed-policy PDE

For a given policy `alpha(mu, tau)`:

    rho V = alpha*mu + (alpha(1-alpha)/sigma^2) * L[V]

    L[V] := dV/dtau + (1/(2 tau^2)) d^2V/dmu^2

The HJB is the same equation with `max over alpha in [0,1]` on the right side.
The inner problem is a pointwise quadratic in `alpha`; interior FOC:

    alpha* = 1/2 + sigma^2 * mu / (2 L[V])

## 3. Exploration-premium substitution

Arm-swap antisymmetry pins the odd part of V exactly (martingale property). The value of
committing to the better arm with no further learning is `max(mu, 0)/rho`. Define the
exploration premium — the value of learning over committing:

    U := V - max(mu, 0)/rho

`U` is even in `mu` (antisymmetry), `U >= 0`, and `U = 0` exactly where exploration has
stopped. WLOG work on `mu >= 0`, where `V = mu/rho + U` and `L[V] = L[U]` (`L` annihilates
the linear part). Substituting `alpha*` into the maximized equation gives a quadratic in
`L := L[V]`:

    L^2 - (4 rho sigma^2 U + 2 sigma^2 mu) L + sigma^4 mu^2 = 0

Branch selection: continuity of `alpha*` across `mu = 0` forces the double root at the ridge,
which selects the `+` root globally on the corridor. It is a perfect square:

    L[U] = sigma^2 * ( sqrt(rho U) + sqrt(rho U + mu) )^2        (mu >= 0)

    (expanded: sigma^2 * ( 2 rho U + mu + 2 sqrt(rho U (rho U + mu)) ))

valid where the discriminant is non-negative: `U >= 0`.

## 4. Dimensionless PDE (exact rescaling, no ansatz)

Two parameters, `rho` (1/time) and `sigma^2` (mu^2 * time), fix every unit:

    muhat  = mu / (sigma sqrt(rho))       # dimensionless mean
    tauhat = rho sigma^2 tau              # dimensionless precision
    u      = (sqrt(rho)/sigma) U          # dimensionless exploration premium

Every parameter cancels identically:

    du/dtauhat + (1/(2 tauhat^2)) d^2u/dmuhat^2 = ( sqrt(u) + sqrt(u + muhat) )^2

on the corridor `{ u >= 0 }`, `muhat >= 0` (even extension: replace `muhat` by `|muhat|`).
Policy, in closed form (no operator left — use the PDE right side for `Lhat[u]`):

    alpha* = 1/2 + muhat / ( 2 (sqrt(u) + sqrt(u + muhat))^2 )

One PDE, one free boundary, zero parameters. The prior enters only through where on the
tauhat-axis the solution is read off: `tauhat_0 = rho sigma^2 / nu^2`.

## 5. Boundary conditions

The operator is second-order in `muhat`, first-order in `tauhat`, on the half-strip
`{ 0 <= muhat <= G(tauhat) }` with the free boundary `G(tauhat)` unknown. Complete set —
2 lateral + 1 free-boundary + 1 terminal:

    BC1 (ridge kink):        du/dmuhat = -1/2     as muhat -> 0+
    BC2 (value matching):    u = 0                on muhat = G(tauhat)
    BC3 (smooth pasting):    du/dmuhat = 0        on muhat = G(tauhat)
    BC4 (terminal/decay):    u -> 0               as tauhat -> infinity

- BC1 is equivalent to the even part of `V` being smooth (Neumann) at the ridge; the
  premium inherits the kink from `max(mu, 0)/rho`.
- BC3 pays for the unknown boundary location.
- BC4 is the datum in the tauhat-direction: the diffusion sign makes the equation well-posed
  integrating BACKWARD in tauhat. Terminal-value problem anchored at infinity; decay is a
  selection condition (kills growing modes), not a Cauchy datum.
- BC3 caveat: in obstacle problems smooth pasting is a theorem from optimality, not an
  axiom. Imposing it as boundary data is standard and correct here, but it is the one
  item derived globally rather than local data.

## 6. Derived properties — NOT boundary conditions

Use these as consistency checks on any numerical solution:

- Double-root/ridge identity: `Lhat[u]|_{0+} = 4 u|_0` — the PDE evaluated at the ridge
  (right side -> `4u` at `muhat = 0`); checks branch consistency numerically.
- Positivity: `u > 0` in the open corridor; `u = 0` outside — the premium vanishes exactly
  where you commit.
- Policy continuity: `alpha* -> 1` as `muhat -> G(tauhat)-` (immediate from the closed-form
  policy as `u -> 0`).
- Exterior solution, exact: `u = 0` for all `muhat > G(tauhat)` (bang-bang kills the
  learning term; nothing to solve there). This is why pasting is the right operation.
- Tail asymptotics: `G(tauhat) ~ 1/(2 tauhat)` as `tauhat -> infinity`. In original units:
  `|mu_Gamma| = 1/(2 sqrt(rho) sigma tau)`. The corridor narrows like 1/tauhat, not
  1/sqrt(tauhat).
- Regularity: solution is C^1 across the free boundary with a curvature jump (control
  shuts the diffusion off at the wall; `u` is identically 0 beyond it).
- Free-boundary curvature law at the edge (exact): original-units trusted form
  `d^2U/dmu^2 |_{Gamma-} = 2 sigma^2 tau^2 mu_Gamma`. The dimensionless transcription
  was first recorded WITHOUT the factor 2 (`tauhat^2 G`) — flagged unverified. Empirical
  vote (feature-net PINN, 2026-08-04): the measured ratio
  `u_muhatmuhat|_{G-} / (tauhat^2 G)` is stable at `2.2 +/- 0.15` across
  `tauhat in [0.5, 4]`, i.e. the form is right and the constant is 2:

      d^2u/dmuhat^2 |_{G-} = 2 tauhat^2 G(tauhat)     [pencil-verify; PINN-supported]

  (The law is unchanged by the premium substitution: the removed piece is linear
  in `mu` on `mu > 0`.)

## 7. Prior art / failed and partial attempts (context, do not redo)

- Parabola ansatz for the even part of V, `V - mu/(2 rho) = a(tau) mu^2 + c(tau)` (i.e.
  `U = a mu^2 + c - |mu|/(2 rho)`), glued C^1 to the exterior: qualitatively right,
  quantitatively overshoots the edge-curvature law by ~2.8x at finite tauhat. Tangency
  + ridge exhaust its degrees of freedom; imposing edge-curvature instead is infeasible
  (negative discriminant). Rejected.
- Large-tauhat similarity reduction `u_even = F(xi)/(2 tauhat)`-type (exact xi/amplitude
  scaling constants never pinned down — VERIFY before use) collapses the PDE to a
  parameter-free ODE two-point BVP: `F'' = (1/2)(F + sqrt(F^2 - xi^2))`, `F'(0) = 0`,
  `F(xi_e) = xi_e`, `F'(xi_e) = 1` at free `xi_e`. In premium variables `W := F - xi`:
  `W'' = (1/2)(W + xi + sqrt(W(W + 2 xi)))`, `W'(0) = -1`, `W(xi_e) = 0`, `W'(xi_e) = 0`.
  Stated in a prior session, never solved. This is the tail asymptotic of the full PDE,
  so its solution doubles as the far-field boundary layer / validation target.
- One-step improved policy from the parabola ansatz measured ~95.5% of optimal
  (in-session estimate, unverified).
- Approximate self-similar policy, closed form (stated in a prior session; checked
  against the trained PINN 2026-08-04). One special function, dummy argument y,
  integration dummy t:

      h(y) = 1 - y e^y E1(y) = int_0^inf t e^-t / (y + t) dt,   h(0) = 1,  h(y) ~ 1/y

  Policy, with the scaled score q := muhat sqrt(tauhat) / sqrt(2 h(8 tauhat)) and
  B := 1 / (8 tauhat h(8 tauhat)) - 1:

      alpha* = 1/2 + q / (2 (1 + B q^2))    for |q| <= 1
      alpha* = 1{muhat > 0}                 otherwise

  Quality vs the trained PINN at tauhat = 1: near-identical on the interior ramp;
  the ansatz jumps 0.91 -> 1 at |q| = 1 (its known defect: violates policy
  continuity, which the true solution satisfies), implying a wall at muhat ~ 0.46.
  Boundary bracket history at tauhat = 1: early raw-coordinate PINNs saturated at
  ~0.36 (wall region under-resolved), giving [0.36, 0.49]. The feature net
  (log tauhat, z-score, tail-coordinate inputs; 2026-08-04) saturates continuously
  at ~0.43-0.44, tightening the bracket to [0.43, 0.46]: the two methods now
  disagree only about the ansatz's jump itself, so the truth is ~0.44-0.46 with
  the PINN's continuous shape.
