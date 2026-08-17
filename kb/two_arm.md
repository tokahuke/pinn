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

    rho V = alpha*mu + (alpha(1-alpha)/sigma^2) * L_ab[V]

    L_ab[V] := dV/dtau + (1/(2 tau^2)) d^2V/dmu^2

The HJB is the same equation with `max over alpha in [0,1]` on the right side.
The inner problem is a pointwise quadratic in `alpha`; interior FOC:

    alpha* = 1/2 + sigma^2 * mu / (2 L_ab[V])

## 3. Exploration-premium substitution

Arm-swap antisymmetry pins the odd part of V exactly (martingale property). The value of
committing to the better arm with no further learning is `max(mu, 0)/rho`. Define the
exploration premium — the value of learning over committing:

    U := V - max(mu, 0)/rho

`U` is even in `mu` (antisymmetry), `U >= 0`, and `U = 0` exactly where exploration has
stopped. WLOG work on `mu >= 0`, where `V = mu/rho + U` and `L_ab[V] = L_ab[U]` (`L_ab` annihilates
the linear part). Substituting `alpha*` into the maximized equation gives a quadratic in
writing `L_ab` for `L_ab[V]`:

    L^2 - (4 rho sigma^2 U + 2 sigma^2 mu) L + sigma^4 mu^2 = 0

Branch selection: continuity of `alpha*` across `mu = 0` forces the double root at the ridge,
which selects the `+` root globally on the corridor. It is a perfect square:

    L_ab[U] = sigma^2 * ( sqrt(rho U) + sqrt(rho U + mu) )^2        (mu >= 0)

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
Policy, in closed form (no operator left — use the PDE right side for `Lhat_ab[u]`):

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

- Double-root/ridge identity: `Lhat_ab[u]|_{0+} = 4 u|_0` — the PDE evaluated at the ridge
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
- Free-boundary LOCATION, empirical (2026-08-11): the boundary is very nearly a
  level set of `muhat * tauhat**q` with `q` in [0.75, 0.8]. Measured as the
  spread of the coordinate over points straddling the learned boundary
  (IQR / |median|, smaller is tighter), on two independently trained nets:

      net                     q=0.5 (z)   q=0.75   q=0.8   q=1.0
      32:256, 9154 params        1.19       0.29      --     0.52
      16:16,   402 params         --        0.152    0.114   0.522

  Two things worth keeping. The exponent is NOT 0.5, so the similarity
  coordinate `z = muhat sqrt(tauhat)` is the wrong variable for the boundary
  even though it is the right one for the chart. And the SMALLER net has the
  crisper boundary (0.114 against 0.29), so this is a property of the
  solution, not an artifact of capacity.

  Feeding `muhat * tauhat**0.75` to the net as a fifth feature does NOT help:
  tried 2026-08-11 on two_arm, the net ignored it (weight mass 1.2e-3 against
  102 for the four it keeps) and pde was unchanged. It is a monomial in
  features already present, so there is nothing to hand over. Recorded so the
  experiment is not repeated.

  OPEN, and measured differently on 2026-08-13: locating the boundary by
  BISECTION on the response's zero set (the largest lead still worth an
  experiment, per tauhat) and fitting `muhat_b ~ tauhat**(-q)` over four
  decades gives q = 0.899 on the 16:16 champion, against the 0.75-0.80 above.
  The two are not the same measurement -- the table scores how TIGHT a given
  exponent's level set is around the boundary, the fit asks what exponent the
  boundary actually follows -- but they should agree and do not. The likely
  reason is that it is not a pure power law: the LOCAL exponent between
  consecutive bisected points runs 0.72 at the low-precision end to 1.23 at
  the high, so a single q is a summary of something curved, and the two
  methods weight the decades differently.

      tauhat        1e-2    1e-1    1e0     1e1     1e2
      muhat_b      21.13    3.783   0.485   0.0648  0.0049
      local q         --    0.775   0.947   0.854   1.231

  Worth resolving because the curvature law and the tauhat-tail anchoring both
  reference the boundary's shape. Do not quote a single q without saying which
  decade it came from.
- How the net represents the tau dependence (2026-08-11, 402-param 16:16 net,
  all 16 first-layer units live): 11 of 16 units carry OPPOSING weights on
  `muhat sqrt(tauhat)` and `muhat tauhat`, making each preactivation
  `muhat sqrt(tauhat) (a - b sqrt(tauhat))`, which changes sign at
  `tauhat = (a/b)**2`. The learned crossovers tile the information axis --
  0.34, 0.60, 0.62, 0.65, 0.85, 0.93, 1.18, 1.65, 4.72, 8.56, 49.1 -- i.e. the
  first layer is a ladder of characteristic information scales, built from the
  only feature pair that can produce a tunable crossover. Sets a floor on
  useful width: the ladder needs rungs.

  In two_arm_drift the same trick is available on the log pair: `log tauhat`
  against `log1p(2 etahat tauhat)` crosses where `2 etahat tauhat ~ 1`, the
  erosion timescale. Note that fifth feature vanishes identically at
  `etahat = 0`, so the eta = 0 slice of a drift net is a function of the first
  four columns alone whatever the fifth column learns.

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

## 8. Similarity coordinates (exact transcription; startup solution)

Measure the lead in posterior standard deviations, the premium in posterior standard
deviations, and let the clock be logarithmic:

    z = muhat * sqrt(tauhat)              # the z-score
    s = log tauhat
    u(muhat, tauhat) = tauhat^(-1/2) * g(z, s)

Chain rule (`z_muhat = sqrt(tauhat)`, `z_tauhat = z/(2 tauhat)`, `s_tauhat = 1/tauhat`):

    u_muhat = g_z                         # the sd factors cancel exactly
    u_mm    = sqrt(tauhat) * g_zz
    u_tauhat = tauhat^(-3/2) * ( g_s + (z/2) g_z - g/2 )

Substituting into the section 4 PDE, every left-side term carries the common factor
`tauhat^(-3/2)`; dividing it out:

    g_s + (1/2) g_zz + (z/2) g_z - (1/2) g = tauhat * ( sqrt(g) + sqrt(g + z) )^2,
    tauhat = e^s

Why this matters numerically: every derivative term has an O(1) coefficient at every
tauhat. The `1/(2 tauhat^2)` that multiplied curvature in raw coordinates — 9 decades
of coefficient spread over the sampled range, amplifying any curvature error of a
numerical solution — is cancelled algebraically. The only surviving scale factor,
`e^s`, multiplies the derivative-free source term. A residual formed in `(z, s)` never
multiplies differentiation error by a large number.

Boundary conditions transcribe to:

    BC1:  g_z(0, s) = -1/2                          (s-independent, no prefactor)
    BC2:  g = 0    at  z = Z(s) := G(tauhat) sqrt(tauhat)
    BC3:  g_z = 0  at  z = Z(s)
    Curvature law:  g_zz|_{Z-} = 2 e^s Z(s)

The two regimes of the "almost self-similar" structure are now explicit: as
`s -> -infinity` the source dies and the equation is linear; as `s` grows the source
dominates and squeezes the corridor (`Z ~ e^(-s/2)/2` from the section 6 tail law).
The raw-coordinate stiffness at small tauhat was rent paid on this crossover, not
physics.

### The startup solution is the free-information envelope, exactly

In the startup limit the stationary, source-free equation is

    (1/2) g_zz + (z/2) g_z - (1/2) g = 0.

Try `g(z) = nu(-z, 1)` — the free-information bound shape (docs/three_arm.md
section 13), `nu(m, 1) = m Phi(m) + phi(m)`. Its derivatives are `g_z = -Phi(-z)`
and `g_zz = phi(z)`, so

    (1/2) phi(z) - (z/2) Phi(-z) - (1/2) ( -z Phi(-z) + phi(z) ) = 0    identically,

and `g_z(0) = -Phi(0) = -1/2`: BC1 holds on the nose. Therefore

    u -> tauhat^(-1/2) * nu(-z, 1)    as  tauhat -> 0,   exactly.

Consequences:

- The proven envelope `nu(-muhat, tauhat^(-1/2)) = tauhat^(-1/2) nu(-z, 1)` is not
  just an upper bound: it is TIGHT at zero information. The architecture's response
  factor must tend to 1 as `tauhat -> 0`, and `log_scale = 0` initialization is the
  asymptotically exact start.
- The premium-net architecture (envelope times bounded response on z-score features)
  is literally the similarity substitution: the network models `g / nu`. Input side
  fully transformed; only a raw-coordinate residual pays the stiffness tax.
- Startup anchor for validation: at small tauhat, `u * sqrt(tauhat)` plotted against
  `z` must collapse onto `nu(-z, 1)`; departure measures distance from the startup
  regime, not error, once the residual is clean.

This is the low-tauhat sibling of the section 7 large-tauhat similarity ODE: the two
asymptotic anchors now bracket the crossover at `tauhat ~ 1`, which is the only region
with no closed-form structure.

## 9. Below the sampling floor: what cannot be constructed there (2026-08-14/15)

The net has no training signal below `PRIOR_FLOOR = 1e-3`, and two consumers
reach there anyway: three_arm_v3's drop-one bases (pair Schur marginals dip to
~floor/2 on 0.33% of wedge states) and any deployment state at a flatter
prior. A day was spent trying to BLEND the response toward a constructed
"exact" sub-floor value. Every target failed, each on a different axis, and
the failures share one mechanism. Do not redo them.

**The mechanism.** Raw residual = chart residual x `tauhat^(-3/2)` (section 8).
The chart equation is `g_s + L0[g] = e^s (sqrt(g) + sqrt(g+z))^2`, so a
construction is gradeable only if its chart residual decays like `e^s`.

- `nu` is the unique construction that does: `L0[nu] = 0` IDENTICALLY (it is an
  information martingale), leaving chart residual `O(e^s)` -> raw `O(u)`. But
  that same identity is why its learning numbers vanish: the Hamiltonian
  degenerates to the commit vertex.
- ANY tau-frozen shape has `g_s = 0` with `L0[g] != 0`: chart residual `O(1)`
  -> raw `O(u/tauhat)`. Policy alive, value diverging.
- The true solution has both. No pointwise construction does.

**The four targets, measured** (champion pde is 8.35e-6 unblended; all figures
are 7-draw medians at batch 4096):

| target | pde | failure |
|---|---|---|
| bare `nu` | 0.96 | `L = 0`; arena regret 99,529 vs TS 13,225, commits at epoch 1 on 1.0 epochs of evidence, 45.6% wrong |
| first-order patient `g0 + tauhat g1` | 0.42 | `g1 >= 0` exceeds the proven envelope (+36% at the floor); alpha = 0 pocket at 1.3-1.6e-3 |
| frozen floor shape | 6.90 | a frozen shape cannot track a source that DOUBLES per octave; +9.7 ridge overclaim mid-band |
| `e^s`-linear between the two | 1.16 | the champion's floor departure from `nu` is ~40x the linear-regime scale, so the manufactured `g_s` flipped the start state itself |

Blend SHAPE is an orthogonal axis and none of them mattered: `1/(1+y)` in
log-tau (power-law tail, 1% three decades up), a Gaussian blip
`exp(-relu(x)^2)`, and the C-infinity compact octave
`sigmoid((1-2x)/(x(1-x)))` all regraded the champion to ~1, because the band's
cost is the target-truth VALUE gap, not the tail. The patient-limit derivation
(one linear ODE, Chebyshev solution, ODE residual 6e-9) is kept at
kb/two_arm_patient.md: the mathematics is sound and the correction does drop
the residual `1/tauhat` where it is applied PURELY; it is the blending into
the graded law that fails.

Also measured: L-BFGS (60 strong-Wolfe evaluations, batch 131072) moved a
blended net's plateau by 0.13%. These are representability floors, not
optimization floors.

**What shipped instead: the clamp.** `forward` evaluates the response at the
z-preserving floor state (`tau_eff = max(tauhat, PRIOR_FLOOR)`, `muhat` scaled
so `z` is held) while the envelope stays at the true state; the
`sqrt(FLOOR/tauhat)` similarity amplitude cancels through `nu`'s homogeneity,
so the premium continues self-similarly as the floor's own shape. It binds
ONLY off the sampling law, so training and every trained checkpoint are
untouched BITWISE (champion pde 8.35e-6, pos_learning exactly 0), the
sub-floor policy inherits the floor's positive learning content (alpha* 0.5
down to tauhat 1e-4), and inputs never leave the trained support. Its cost is
a first-derivative kink at the floor, on a surface no loss samples.

Anchoring the clamp ABOVE the floor was measured and rejected: it overwrites
trained territory, at 13.2 (1.25x floor), 25.9 (1.5x) and 54.1 (2x) against
8.35e-6, with no policy improvement at any setting.

The clamp buys VALUE honesty, not gradeability -- nothing can buy that. For
three_arm_v3, whose sampler reaches sub-floor marginals, grading is the
SAMPLER FENCE's job: fenced 4.9e-4 against unfenced 10.8 at a fresh init, so
0.33% of states carry 99.995% of the loss.

## 10. The subsolution objective (THE two_arm objective since 2026-08-15)

V* is the MAXIMAL subsolution of the HJB, so

    maximize u   subject to   v <= max H

has the true value function as its optimum and every feasible point is a
PROVEN lower bound: the greedy policy provably earns at least v (learnings
section 9). Built as the `two_arm_v2` clone, measured, and PROMOTED into
`pinn/problems/two_arm/loss.py` on 2026-08-15; the clone is deleted and the
two-sided residual it replaced survives only in this record. Its reason to
exist was never two_arm itself: three_arm_v3's
base overclaims on 27% of wedge states because it is a trained net rather than
the true u2, and that deadlocked a v3 run (kb/three_arm.md section 17).

    loss = violation + RIDGE * ridge - CLIMB * climb
    violation = mean(relu(v - max H) / tauhat**1.5)      LINEAR
    climb     = mean(u / tauhat**1.5)                    natural units
    RIDGE 2e4, CLIMB 1e-7

There is NO overall penalty weight. Both terms are measured the same way, so
only the RATIO is a knob and the outer factor cancels; it existed as
PENALTY 1e9 while it was being swept and was folded out once the sweep showed
the answer does not depend on it. Read any historical PENALTY/CLIMB pair as the
ratio -- 1e2 against 1e9 is this 1e-7 exactly.

Four design points, each measured rather than argued:

- LINEAR violation, not squared. An L1 penalty is EXACT -- above a finite
  threshold (the largest Lagrange multiplier) the penalized optimum IS the
  constrained optimum -- while a quadratic only approaches feasibility as the
  weight grows. It is also what the house does for every other sign condition.
  A tenfold change in the climb-to-violation ratio gives indistinguishable
  curves, which reads as being above the threshold: past it the weight stops
  mattering. TRUST THAT ONLY OVER A LONG RUN. The same probe on two_arm_drift
  showed five decades indistinguishable at 3k iterations and a monotone
  collapse of the bound by 50k (kb/two_arm_drift.md section 10).
- The climb is in the VIOLATION'S UNITS. A uniform climb (mean of the gate
  u / nu) lost the floor decade: the violation carries tauhat**-1.5, 3.2e4 at
  the floor, so the penalty outbid a flat climb by four orders there. The net
  then bought slack by INFLATING THE LEARNING NUMBER (natural-units L
  21.1 -> 23.4), which raises max H and drives v - max H deeply negative at no
  cost, since undershoot is free. Result: violation 5.7e-7 and a two-sided
  residual of 1.8, 99.2% of it in the floor decade. Matching the units fixes
  the incentive. NOTE the inflation is also partly transient: at 3k iterations
  the matched-units run still read L 24.6, and by 350k it had receded to 21.8.
- pos_learning is REMOVED, by proof rather than calibration. Where L_ab < 0
  the Hamiltonian is convex in alpha, so max H = e^s z at a vertex, while
  v = e^s (z + g) with g >= 0 architectural -- the violation is then exactly
  u. A negative learning operator is infeasible wherever the premium is alive
  and harmless where it is not. Measured: violation/u in [0.94, 1.00] over
  12,030 such states, the exceptions all at u <= 1.5e-6 where the difference
  falls into float32 cancellation.
- DUAL ASCENT ON THE PENALTY WAS TRIED AND REMOVED. It ramped lambda between
  1e-2 and 1e6 because the update saturated: exp(RATE * clamp(violation /
  BUDGET - 1, -1, 1)) is bang-bang whenever the violation is far from the
  budget, which is almost always. A fixed weight needs no controller.

Result after ~350k iterations from the champion, and the point of the whole
exercise:

    net                     pde        overclaim   sup(residual+)   floor L
    old champion            8.17e-6    23.7%       2.08e-2          21.09
    subsolution (promoted)  1.47e-1     0.60%      4.27e-4          21.73

39x fewer overclaiming states and a 49x smaller shift constant, with the
premium intact (climb 2474 against 2497) and BC1 still at -0.4978. The pde is
four orders WORSE, and that is the headline finding:

**The arena does not notice.** 48,000 paired reps at production parameters,
frontier scenario: the subsolution net beats Thompson by 21.5% against the old
champion's 20.8%, and paired head to head it is -94.8 +/- 89.5 -- an interval
that clears zero by a hair, after being +142 +/- 425 at 1,500 reps and
-188 +/- 190 at 12,000. The defensible claim is NOT WORSE, plausibly ~1%
better; the sign flipped once and this was the third look at overlapping seed
blocks, so a clean confirmation wants fresh seeds. Four orders of residual
bought at most 1.8% of regret in either direction. Learnings section 9 said
policy error is second order and the residual is not the referee; this is the
measurement.

FAILURE MODE TO EXPECT: the run died at ~350k on the relu(r)**2 absorbing
state (CLAUDE.md traps), and the signature differs from the documented one --
not a frozen total, but `climb` AND `violation` printing exactly 0 while
`ridge` stays SATISFIED at ~1e-11. The premium survives in a sliver at
muhat = 0, enough to hold BC1's slope, and is dead everywhere the cloud
samples. Nothing was lost: a dead net cannot beat a live one on this objective
(the -CLIMB * climb term makes a live total ~ -2.5e-4, the champion's climb of
2476 times 1e-7, against a dead one of ~ +2e-7 -- ridge only), so
best-EMA checkpointing is structurally safe here.
