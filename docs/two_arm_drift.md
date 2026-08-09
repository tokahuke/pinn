# two_arm_drift — Two-Armed Allocation with a Drifting Mean

HJB problem for `pinn/problems/two_arm_drift/`. Extension of `two_arm.md` to a
mean that itself diffuses. Everything below is exact unless flagged
UNVERIFIED. Read `two_arm.md` first; this doc only records what changes.

## 0. Notation

Same contract as `two_arm.md`: uppercase dimensional (`V`, `U`), lowercase
dimensionless (`u`), hats dimensionless coordinates (`muhat`, `tauhat`).

One new symbol:

- `eta`     the drift volatility of the true mean, `d theta = eta dW`
- `etahat`  its dimensionless form, `etahat := eta / (rho sigma)`
- `tauhat*` the precision ceiling, `1/(2 etahat)`

`etahat = 0` recovers `two_arm.md` in every formula below.

## 1. Problem statement

Same as two_arm except the unknown mean is no longer constant:

    d theta = eta dW_theta        independent of the observation noise

Observations `x_t ~ N(theta_t, sigma^2/(alpha(1-alpha)))` per unit time,
reward flow `alpha theta`, discount `rho`.

State: still the conjugate posterior `(mu, tau)` — drift adds NO coordinate.
It is a Kalman-Bucy filter, and only the precision clock changes:

    dP/dt   = eta^2 - P^2 alpha(1-alpha)/sigma^2         (P = 1/tau)
    dtau/dt = alpha(1-alpha)/sigma^2  -  eta^2 tau^2

The posterior mean is still a martingale with innovation variance
`(alpha(1-alpha)/sigma^2)/tau^2`, unchanged from two_arm: `eta` erodes
precision, it does not add noise to the mean.

**The clock now has a ceiling.** `alpha(1-alpha) <= 1/4`, so `dtau/dt <= 0`
above

    tau*  = 1/(2 sigma eta)        tauhat* = 1/(2 etahat)

Precision above the ceiling drains; below it, learning wins. `tauhat*` is an
attractor of the tauhat-flow, approached from both sides.

## 2. Fixed-policy PDE

For a given policy `alpha(mu, tau)`:

    rho V + eta^2 tau^2 V_tau = alpha*mu + (alpha(1-alpha)/sigma^2) * L_ab[V]

    L_ab[V] := dV/dtau + (1/(2 tau^2)) d^2V/dmu^2         (unchanged operator)

The HJB is the same with `max over alpha in [0,1]` on the right. **The new
term is control-free**, so the inner problem is the same pointwise quadratic
as two_arm section 2, with the same interior FOC:

    alpha* = 1/2 + sigma^2 * mu / (2 L_ab[V])

## 3. Exploration-premium substitution

Committing forever still pays `max(mu, 0)/rho`: `theta` is a martingale, so a
frozen allocation earns its current expectation however much the mean
wanders afterwards. So define, as before,

    U := V - max(mu, 0)/rho

`max(mu,0)/rho` is annihilated by `L_ab` and by `d/dtau`, so the new term acts on
`U` alone. On `mu >= 0`, with `A := L_ab[U]/sigma^2` and
`max over alpha of [alpha mu + alpha(1-alpha) A] = (A + mu)^2/(4A)`:

    L_ab[U] = sigma^2 * ( sqrt(W) + sqrt(W + mu) )^2

    W := rho U + eta^2 tau^2 U_tau                       (mu >= 0)

Two_arm's quadratic with `rho U` replaced by `W`. This is the general pattern:
a control-free term added to the generator never touches the maximization, it
only replaces `rho U` inside the source.

The equation is now IMPLICIT in `U_tau` (it sits under a square root). For the
solver, keep the maximization explicit and hand it to `simplex.maximize_quadratic`
as two_arm already does — never substitute the FOC back (two_arm section 7:
that road leads to sqrt NaNs).

## 4. Dimensionless PDE

Three parameters now, `rho [1/T]`, `sigma^2 [mu^2 T]`, `eta [mu/T^(1/2)]`,
against two dimensions, so exactly ONE dimensionless group survives:

    etahat = eta / (rho sigma)

Same readout dictionary as two_arm, plus that group:

    muhat  = mu / (sigma sqrt(rho))
    tauhat = rho sigma^2 tau
    u      = (sqrt(rho)/sigma) U

Premium form:

    u_tauhat + (1/(2 tauhat^2)) u_muhatmuhat = ( sqrt(w) + sqrt(w + muhat) )^2

    w := u + etahat^2 tauhat^2 u_tauhat

Value form, which is what the solver grades (maximization explicit):

    v + etahat^2 tauhat^2 v_tauhat = max over alpha of
        { alpha muhat + alpha(1-alpha) Lhat_ab[v] }

    Lhat_ab[v] := v_tauhat + (1/(2 tauhat^2)) v_muhatmuhat

and the tauhat clock:

    dtauhat/dt = rho [ alpha(1-alpha) - etahat^2 tauhat^2 ]

**Cost accounting.** +0 state, +1 parameter. The state is still
`(muhat, tauhat)`; `etahat` is the third net input, so one checkpoint serves
every deployment.

## 5. Boundary conditions

The count DROPS from two_arm's four to two. Not because the free boundary is
gone -- it is not -- but because the CONTACT SET is, and BC2/BC3 lived on it.

    BC1 (ridge kink):   du/dmuhat = -1/2    as muhat -> 0+, every etahat
    BC2 (far field):    u -> 0              as muhat -> infinity

BC1 survives for two_arm's reason and is still the degeneracy breaker: the
arm-swap `theta -> -theta` maps the problem to itself (drift and noise are
both symmetric), so `V(mu) - V(-mu) = mu/rho` exactly, `U` is even, and the
premium inherits the `-1/2` slope from `max(mu,0)/rho`.

**The degeneracy it breaks is still there.** `u = 0` gives `V = max(mu,0)/rho`,
whose `V_tau` and `V_mumu` both vanish, so the HJB reduces to
`max(mu,0) = max over alpha of alpha mu` — satisfied exactly. The never-explore
solution survives the extension, so BC1 remains load-bearing.

**BC2/BC3 of two_arm are vacuous, but a free boundary remains.** CORRECTED
2026-08-07: an earlier draft said "there is no free boundary to locate". Wrong,
and the distinction is the whole architecture.

Two_arm has ONE surface doing two jobs: the contact set `{u = 0}` IS the commit
region `{alpha = 1}`, so `relu(r)**2` locates both at once -- it zeroes the
premium and supplies the curvature jump, in the same place. Drift separates
them.

- The CONTACT SET is empty. At `alpha in {0,1}` precision decays,
  `dtau/dt = -eta^2 tau^2 < 0`, so committing is not absorbing: uncertainty
  regrows, the option to switch has value, `u > 0` STRICTLY everywhere. BC2
  (value matching) and BC3 (smooth pasting) have no surface to sit on.
- The COMMIT REGION is not empty. `alpha = 1` whenever `Lhat_ab[v] <= muhat`,
  measured at ~68% of the sampled cloud. Inside it the governing equation is
  NOT the `(sqrt + sqrt)^2` relation -- that assumes an interior max -- but

      v + etahat^2 tauhat^2 v_tauhat = muhat,  so  u + etahat^2 tauhat^2 u_tauhat = 0

  giving `u = A exp(1/(etahat^2 tauhat))` along a characteristic that flows
  DOWN in tauhat until the state re-enters the exploring region. Entered and
  left: the restless cycle in one line.
- The SWITCHING SURFACE `Lhat_ab[v] = muhat` between them is a genuine free
  boundary. The maximised Hamiltonian is C1 across it by the envelope theorem
  (the interior vertex meets the endpoint, value and first derivative agree)
  and its SECOND derivative jumps. Measured 2026-08-07: residual density peaks
  at 8.2x the batch average in the band `1 - alpha` in `[1e-2, 1e-1]`, rising
  monotonically on approach.

So drift needs TWO mechanisms where two_arm needed one: a strictly positive
response, so `u > 0` on the commit region, plus a movable curvature-jump
primitive for the switching surface (section 8).

Deep in the corridor `u` is exponentially small, and the rate is exact rather
than sketched: the envelope's far field is
`(etahat/(2 sqrt 2)) exp(-sqrt 2 muhat/etahat)`, verified against the closed
form to the digit.

**BC4 of two_arm is not needed.** `tauhat* = 1/(2 etahat)` is an attractor of
the tauhat-flow, approached from both sides, so characteristics do not enter
the domain at either end of the tauhat axis and no datum is required there.
UNVERIFIED as a well-posedness claim; it is the one structural statement here
that deserves a proof or a numerical test before being relied on.

## 6. Similarity coordinates

Two_arm section 8 transcribes with one extra term. With `z = muhat sqrt(tauhat)`,
`s = log tauhat`, `u = tauhat^(-1/2) g`, and

    L_ab[g]  := g_s + (1/2) g_zz + (z/2) g_z - (1/2) g
    tauhat_slope := L_ab[g] - (1/2) g_zz = g_s + (z/2) g_z - (1/2) g

the value-form equation becomes

    e^s (z + g) + etahat^2 e^(2s) tauhat_slope
        = max over alpha of { alpha e^s z + alpha(1-alpha) L_ab[g] }

`etahat = 0` is two_arm's equation exactly. The drift term is `tauhat^2 v_tauhat`
transcribed: `v_tauhat = tauhat^(-3/2) tauhat_slope`, so the whole equation still
carries one common factor `tauhat^(3/2)`, and the same conditioning argument
holds — no derivative term is multiplied by a large number. The new term
carries `e^(2s)`, so it is derivative-free-weighted like the source.

## 7. The envelope

Two_arm's `nu(-muhat, tauhat^(-1/2))` is no longer an upper bound: knowing
`theta` now does not tell you `theta` later, and the option to relearn is
worth more than the static free-information value. The correct replacement is
the DISCOUNTED PERFECT-INFORMATION premium -- knowing `theta_t` at every
instant, which dominates any policy:

    u_env(muhat, tauhat, etahat)
        = integral_0^inf  e^(-t) * nu( -muhat, sqrt(1/tauhat + etahat^2 t) )  dt

in dimensionless time. The forecast variance is the current posterior variance
plus the drift accumulated since, `1/tauhat + etahat^2 t`, and the `-muhat`
form is the same identity two_arm uses: `nu(m, sd) - max(m, 0) = nu(-m, sd)`
for `m >= 0`, which turns "perfect-information value minus commit value" into
one call.

### Closed form (derived and verified 2026-08-07)

It integrates. With

    a = 1 / (etahat sqrt(tauhat))          b = muhat sqrt(tauhat / 2)

    u_env = nu(-muhat, tauhat^(-1/2))
          + (etahat / (4 sqrt 2)) * exp(-b^2) * [ erfcx(a + b) + erfcx(a - b) ]

**Derivation, two steps.** Integrate by parts in `t`. Since
`d nu(m, sd) / d sd = phi(m/sd)`, the `Phi` term dies and

    u_env = nu(-muhat, tauhat^(-1/2))
          + (etahat^2/2) integral_0^inf e^(-t) phi(muhat/sd) / sd  dt

The boundary term IS the `etahat -> 0` limit, so that limit is structural, not
a numerical coincidence. Then substitute `sd^2 = 1/tauhat + etahat^2 t`
(`dt = 2 sd d(sd) / etahat^2`), which lands the remainder exactly on

    integral_x^inf exp(-p s^2 - q/s^2) ds
      = (1/4) sqrt(pi/p) [ e^( 2 sqrt(pq)) erfc(sqrt p x + sqrt q / x)
                         + e^(-2 sqrt(pq)) erfc(sqrt p x - sqrt q / x) ]

with `p = 1/etahat^2`, `q = muhat^2/2`, `x = tauhat^(-1/2)`
(Gradshteyn-Ryzhik 3.472.5; the `x = 0` specialisation is GR 3.471.15). The
`s^2`-moment of that family is NOT needed -- integrating by parts first is what
removes the term that would have required it, and why the answer stays short.

**Why erfcx and not erfc.** The raw form carries `exp(1/(tauhat etahat^2))`,
whose exponent is `1e8` at `tauhat = 1e-2, etahat = 1e-3` -- it overflows on
sight. Both terms share `A - z^2 = -b^2`, so one erfcx substitution cancels
every large exponential analytically and nothing large multiplies anything
small. One hazard survives: `a < b` (that is, `etahat muhat tauhat > sqrt 2`),
where `erfcx` of a negative argument overflows while `exp(-b^2)` underflows to
zero, giving `0 * inf = nan`. There the algebraically identical
`exp(a(a - 2b)) erfc(a - b)` is used, whose exponent is negative exactly on
that branch. Use the product `a(a - 2b)`, not the equal `a^2 - b^2`: the
latter subtracts two numbers of magnitude ~3000 to land in `[-88, 0]` and
spends four digits doing it.

**The etahat = 0 anchor is bitwise.** `etahat` multiplies the entire
correction, so at `etahat = 0` the envelope returns `nu(-muhat,
tauhat^(-1/2))` to the last bit, in both dtypes -- `torch.equal`, not a
tolerance. This is what makes the two_arm champion an exact bootstrap
(section 8).

**But `etahat` must be floored inside `a` alone.** At `etahat = 0` the forward
survives (`a = inf`, `erfcx(inf) = 0`) while the BACKWARD gives `inf * 0 =
nan`. Flooring `etahat` only where it appears in the reciprocal keeps `a`
finite; leaving the prefactor untouched keeps the limit exact. Found by
`policy()` returning allocations outside `[0, 1]`, which is the cheap symptom
to watch for.

**Accuracy.** Against composite Simpson on `t in [0, 500]` with a
cancellation-free `nu`, worst relative error **2.0e-14** across six decades of
the value. For comparison, the 32-node Gauss-Laguerre rule it replaced is
2.6e-06 at `(muhat, tauhat, etahat) = (2, 30, 0.1)` and **79x too big** at
`(6, 30, 0.1)`. The closed form has no tail regime: `a` and `b` are algebraic
and there are no nodes to run out of.

### A limit inherited from `nu`, not introduced here

The leading term is `pinn/utils.py:nu`, and it OVERESTIMATES by a factor of
exactly `z^2 = muhat^2 tauhat` once `z` exceeds ~8 in float64:

    z       nu (repo)       truth           ratio    z^2
    8.50    8.166e-17       1.086e-18       75.2     72.2
    9.18    2.002e-19       2.295e-21       87.2     84.3
    33.0    1.341e-237      1.228e-240      1092     1089

Mechanism: `torch.special.ndtr` computes `0.5 (1 + erf(z/sqrt 2))`, and beyond
`|z| ~ 8` the sum cancels to exactly zero in float64, so `nu` silently drops
its `-m Phi(-z)` term and returns `sd phi(z)` where the truth is
`sd phi(z)/z^2`. In float32 the threshold is `|z| ~ 4`, and there is a band
below it where the error goes the OTHER way (measured underestimates up to
5000x by the derivation agent, not independently reproduced here) -- that
direction would break the upper-bound property.

Overestimating is safe for a bound and the drift correction above is exact
regardless, so nothing here is wrong; the far-corridor bound is simply `z^2`
looser than it should be. The fix belongs in `nu`, where two_arm and three_arm
share it:

    nu(m, sd) = exp(-z^2/2) [ sd/sqrt(2 pi) - (m/2) erfcx(z / sqrt 2) ]   (m <= 0)

which moves the cancellation from catastrophic (a term dropped entirely) to
mild (`z^2` amplification of one ulp). NOT applied -- it changes what two_arm
and three_arm compute, and their champions were trained against the current
behaviour.

## 8. Architecture consequences

Everything two_arm does carries over except the response's zero set:

- Features: two_arm's four, plus `log1p(2 etahat tauhat)` -- the ceiling ratio,
  which the sampler bounds by 1, and which is exactly 0 at `etahat = 0` so the
  two_arm graft is bitwise. NOT `log(etahat)`.
- Envelope: `u_env` above in place of `nu`, times `exp(log_scale)`.
- Response: `relu(r)^2/(1+relu(r)^2)`, two_arm's, RESOLVED 2026-08-08 after
  two detours. It maps into [0, 1), so `0 <= u < envelope` is ARCHITECTURAL,
  and the free-information argument PROVES that bound -- no loss term is as
  tight as a construction. Both intermediate maps surrendered it for nothing:
  sigmoid cannot represent 0, and exp is unbounded above (0.53% of points went
  over the envelope, up to 1.16x, at the tauhat floor).
  The cost is real and known: relu**2 pins `u = 0` on the commit region where
  the truth is merely very small, and since `u = 0` also solves the interior
  equation exactly, the loss pays to spread it -- measured 2026-08-07, the dead
  region grew 52% -> 65% over training until a 16% live sliver carried 41% of
  the squared residual. That is a training pathology to fix in the loss, not a
  reason to give up an exact upper bound. With the section 7 envelope and the
  current sampler it has not recurred: 60.8% at graft, 60.2% after a million
  iterations, flat.
- Kink units (`k<count>` in the topology): saturated `relu(.)**2` added to the
  response BEFORE the response map, so the curvature jump reaches `u`. They
  serve the switching surface of section 5, which a smooth response cannot
  represent. Zero-init output, so grafting onto a trained smooth net is
  bit-exact at step 0. three_arm's evidence says STITCH, never co-train: bare
  relu**2 units from scratch colonised the smooth bulk and cost two orders.
  Confirmed the hard way on 2026-08-08 -- grafted kinks left to train after the
  trunk had died took over the whole function, output weight 0 -> 1.48, and the
  policy came out non-monotone in muhat.
- Loss: `pde_loss` grades the section 6 equation, one extra term against
  two_arm's; `ridge_loss` is unchanged.
- Sampling: RESOLVED 2026-08-08. `tauhat` is drawn as a fraction of the ceiling
  rather than on an absolute scale, so `2 etahat tauhat <= 1` holds exactly and
  the clipped mass lands ON the ceiling, which is the attractor. `etahat` gets
  tauhat's own shape -- decade-spread scale times an exponential tail, reaching
  0 (the two_arm anchor) and capped where the ceiling would collapse onto the
  precision floor. The scale is set by deployment, not taste: a drift of 3% of
  baseline over one discount horizon is `etahat ~ 10.7`, which is why an
  earlier law centred on 0.1 covered nothing that matters.

## 9. Verification

DONE (`scratchpad/verify_drift.py`, 2026-08-07, float64):

1. VERIFIED. The filter identity `dP/dt = eta^2 - P^2 alpha(1-alpha)/sigma^2`
   against the exact discrete Bayes update
   `P' = 1/(1/(P + eta^2 dt) + alpha(1-alpha) dt/sigma^2)` in the `dt -> 0`
   limit; worst relative error 1.2e-06 at `dt = 1e-7`, across
   `alpha in {0, 0.25, 0.5, 1}`. `alpha in {0,1}` gives `dP/dt = eta^2`
   exactly — the pure-erosion case that kills the contact set.
2. VERIFIED. The similarity transcription of section 6 equals
   `tauhat^1.5` times the raw-coordinate residual, worst gap 5.7e-14 over
   `etahat in {0, 0.3, 1, 4}` on 128 states — the analogue of two_arm's
   `tauhat^1.5` self-check, drift term included.
3. VERIFIED, and superseded by the closed form of section 7:
   `nu(m, sd) - max(m,0) = nu(-m, sd)` to 4.0e-16; the envelope at
   `etahat = 0` now equals `nu(-muhat, tauhat^(-1/2))` BITWISE rather than to
   quadrature precision; the closed form matches a cancellation-free Simpson
   reference to 2.0e-14 across six decades; and the envelope increases with
   `etahat` (0.1978 -> 0.6113 as `etahat` goes 0 -> 2 at
   `muhat = 0.5, tauhat = 1`).
4. VERIFIED. `u = 0` gives residual exactly 0.0 at every `etahat` tested —
   the degeneracy is intact, so BC1 stays load-bearing, and the derivative
   chain is end-to-end correct.
6. VERIFIED as module self-checks, all four passing
   (`poetry run python -m pinn.problems.two_arm_drift.<envelope|sample|model|loss>`):
   the bitwise `etahat = 0` envelope anchor in both dtypes; the two_arm
   champion loading with a bitwise-identical response and an absolute premium
   gap of 1.9e-6 on values up to 9.3; the item-2 transcription identity
   carried into `loss.py` as a permanent self-check; and the full float32 training path -- two
   `create_graph` derivatives plus backward through the second.

OPEN:

5. Well-posedness without a tauhat datum (section 5). The cheapest probe is a
   solved instance: train, then check the solution is insensitive to what is
   imposed at large `tauhat`.
7. The float32 branch of the `nu` limit (section 7): the underestimate
   direction, which would break the upper-bound property, is reported but not
   independently reproduced.
