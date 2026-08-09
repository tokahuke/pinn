# two_arm_t — Two-Armed Allocation with Unknown Variance

HJB free-boundary problem for `pinn/problems/two_arm_t/`. Extension of
`two_arm.md` to an unknown outcome variance carried as a conjugate
inverse-gamma posterior; the `t` in the name is the Student-t posterior that
replaces the normal. Everything below is exact unless flagged UNVERIFIED.
Read `two_arm.md` first; this doc only records what changes.

## 0. Notation and relation to two_arm

Same contract: uppercase dimensional (`V` value, `U` premium), lowercase
dimensionless (`u`), hats dimensionless coordinates (`muhat`, `tauhat`).

New symbols:

- `S`     the agent's mean outcome variance, `S := b/(a-1) = E[s^2]`
          (dimensional, `mu^2`). NOT `b/a` — see section 2.
- `sc`    the posterior t scale of `theta`, `sc^2 = b/(a n) = S (df-2)/(df n)`
- `df`    degrees of freedom of the variance posterior, `df := 2a`
- `n`     effective units of contrast information (dimensionless count)
- `rho`   the PER-SAMPLE discount rate

**Time is measured in samples.** One unit of time is one observation, so the
sampling rate is 1 by definition of the clock and never appears. The problem's
parameters are the prior `(m_0, n_0, a_0, b_0)` and `rho`; nothing else.

Bridge to two_arm's noise parameter: **`sigma^2 = S`** on the sample clock.
With `S` known and `df = infinity` every formula below collapses to
`two_arm.md` exactly (section 9).

## 1. Problem statement

Two arms, contrast `theta = mu_B - mu_A` unknown, per-unit outcome variance
`S_true` common to both arms and ALSO unknown. One sample arrives per unit
time; allocation `alpha in [0,1]` to the treatment. Reward flow `alpha * theta`,
expectation `alpha * m` under the posterior. Discount `rho`.

Conjugate normal-inverse-gamma posterior:

    theta | s^2 ~ N(m, s^2/n),    s^2 ~ IG(a, b),    theta ~ t_df(m, sc)

with `sc^2 = b/(a n)`. Since `df` is deterministic (below), `sc` and `S` differ
by a known factor and either may carry the state; `S` is used here because its
dynamics are a martingale. In the `df -> infinity` limit `sc^2 -> S/n`, which
is two_arm's `1/tau`.

State: `(m, n, df, S)`.

**Two clocks, driven by different sufficient statistics.** This is the whole
structural content of the extension:

- The contrast mean is identified by the DIFFERENCE of arm means. For a batch
  of `N` units split `alpha`, `Var(mhat_B - mhat_A) = S (1/(alpha N) +
  1/((1-alpha)N)) = S/(N alpha(1-alpha))`, so

      dn/dt = alpha(1-alpha)

  Self-limiting, dying at both `alpha = 0` and `alpha = 1`, exactly as in
  two_arm.

- The variance is identified by WITHIN-arm dispersion — the pooled residual
  sum of squares — which needs no comparison. A split batch contributes
  `N - 2` degrees of freedom, a fully committed batch `N - 1`. To leading
  order in `N`:

      d(df)/dt = 1          (independent of alpha)

  The agent CANNOT trade variance learning against mean learning: `df` is a
  deterministic, uncontrolled clock, `df(t) = df_0 + t`.

## 2. State dynamics

`m` and `S` are both martingales in the agent's filtration.

**Which scale is the martingale — VERIFIED 2026-08-07, and the obvious guess
is wrong.** With `da = dt/2` and `db` the squared-surprise increment of
section 11, `E[s^2] = b/(a-1)` is a martingale exactly; `b/a` is NOT. Monte Carlo over the agent's
own predictive (draw `s^2 ~ IG(a,b)`, then `k` residuals, then update),
2e6 reps, relative drift per step:

    definition        a=20,k=1   a=20,k=5   a=50,k=10   a=200,k=20
    b/a               +1.27e-3   +5.72e-3   +1.80e-3    +2.25e-4
    b/(a-1)           -1.5e-5    -1.3e-4    -5.2e-5     -1.5e-5
    MC standard error   2.7e-5     5.7e-5     3.1e-5      1.1e-5

The `b/a` row is a 47-sigma drift, and it matches the exact bias
`a(2a-2+k)/((a-1)(2a+k)) - 1` term for term. Recorded because `b/a` is the
natural-looking plug-in and would have poisoned the dynamics silently.

    Var(dm)/dt = alpha(1-alpha) * S / n^2
    dn/dt      = alpha(1-alpha)
    d(df)/dt   = 1
    Var(dS)/dt = 2 S^2 / df^2

**The two noises are independent.** For normal samples the sample mean and the
sample variance are independent, so the `dm` and `dS` innovations are
uncorrelated. There is NO cross-derivative term in the PDE. This is the one
gift the extension hands out and it should not be taken for granted if the
outcome model is ever changed away from normal.

Consistency check on the variance diffusion: in `df`-time (`d(df) = dt`)
the log-scale has variance rate `2/df^2` per unit `df`, and
`integral from df to infinity of 2/x^2 dx = 2/df`, recovering the textbook
`Var(log s^2_hat) ~ 2/df`. Parameter-free, as it must be.

VERIFIED 2026-08-07 against the same Monte Carlo: observed/predicted step sd
1.067, 1.016, 0.981, 0.982 across the four cells above. The residual few
percent is the discrete-`k` correction — the exact finite-batch variance is

    Var(dS) = S^2 [ 2k(a-1)/(a-2) + k^2/(a-2) ] / (4 (a + k/2 - 1)^2)

which reproduces the `a=200` cell to 0.1%. The `k^2` term is `O(k/a)` relative
to the first and vanishes in the diffusion limit (`k -> 0`, `k/dt = 1`),
leaving `Var(dS)/dt = 2 S^2/df^2` as stated.

## 3. Fixed-policy PDE and the HJB

For a given policy `alpha(m, n, df, S)`:

    rho V = alpha*m + alpha(1-alpha) * L_ab[V] + V_df + (S^2/df^2)*V_SS

    L_ab[V] := V_n + (S/(2 n^2)) * V_mm

The HJB is the same with `max over alpha in [0,1]` on the first two terms
only: the `df` transport and the `S` diffusion are CONTROL-FREE, so the inner
problem is the same pointwise quadratic in `alpha` as two_arm section 2, with
interior FOC

    alpha* = 1/2 + m / (2 L_ab[V])

Check against two_arm: with `tau = n/S`, `L_ab[V] = (1/S)(V_tau + (1/(2tau^2))
V_mm)` and `L_ab[V] = L_twoarm[V]/sigma^2` with `sigma^2 = S`. Exact.

## 4. Exploration-premium substitution

Committing forever still pays `max(m,0)/rho`: `theta` is constant, `m` is a
martingale, and a frozen allocation earns its expectation. So define, as
before,

    U := V - max(m, 0)/rho

`max(m,0)/rho` is annihilated by `L_ab`, by `d/d(df)`, and by `d^2/dS^2`, so all
three new terms act on `U` alone. On `m >= 0`, writing `A := L_ab[U]` and
using `max over alpha of [alpha m + alpha(1-alpha)A] = (A+m)^2/(4A)`:

    m + rho U - U_df - (S^2/df^2) U_SS = (A + m)^2 / (4A)

which is two_arm's quadratic with `rho U` replaced by a modified `W`. The `+`
branch (same continuity-at-the-ridge selection) gives

    L_ab[U] = ( sqrt(W) + sqrt(W + m) )^2

    W := rho U - U_df - (S^2/df^2) U_SS                  (m >= 0)

**This is the general pattern for every extension of two_arm considered so
far.** Adding a control-free term to the generator does not change the
inner maximization at all; it only replaces `rho U` inside the source by
`rho U - (the new terms applied to U)`. The drift extension
(`d theta = eta dW`) obeys the same rule with `W = rho U + eta^2 tau^2 U_tau`.
Worth keeping as the reusable lemma.

Note the equation is now IMPLICIT in `U_df` and `U_SS` (they sit under a
square root). For the solver, keep the maximization explicit — evaluate and
max, as `three_arm/simplex.py` does — and never substitute the FOC back
(two_arm section 7: that road leads to sqrt NaNs).

## 5. Dimensional analysis and symmetry

Dimensions on the sample clock: `[rho] = 1/T`, `[m] = mu`, `[S] = mu^2`, `n`
and `df` pure numbers.

**The continuous symmetry is the scale invariance, and it is the only one.**
Under `theta -> c theta`, `S -> c^2 S`, `m -> c m` (with `n`, `df`, `rho`
fixed) the problem maps to itself and `V -> c V`. `V` is homogeneous of
degree 1 in `(m, sqrt(S))`, which removes exactly one coordinate. No further
continuous symmetry survives: the `df` clock breaks the two_arm similarity
group (section 8 there) because `df` advances on the traffic clock while
`tauhat` advances on the allocation-weighted clock, and their ratio is
control-dependent.

Applying the same readout dictionary as two_arm with `sigma_hat^2 := S`:

    muhat  = m / (sigma_hat sqrt(rho)) = m / sqrt(rho S)
    tauhat = rho sigma_hat^2 tau_hat   = rho n
    u      = (sqrt(rho)/sigma_hat) U   = U sqrt(rho/S)

**`S` cancels out of `tauhat` identically**: `tauhat = rho n` is elapsed
allocation-weighted time in units of the discount horizon,
`dtauhat/dt = rho alpha(1-alpha)`, and it is known exactly however wrong the
agent is about the variance. Mis-specifying `S` is a PURE HORIZONTAL
displacement in `muhat` — and since the free boundary sits at
`m_Gamma = G(tauhat) sigma_hat sqrt(rho)`, the commit threshold is exactly
proportional to `sqrt(S)`.

**Cost accounting, stated plainly.** The reduction gives

    state:      (muhat, tauhat, df)          +1 over two_arm
    parameters: none new

+1 state (`df`); no new parameters — `rho`, absorbable into the time unit in
two_arm, is no longer absorbable because `df` is an absolute counter on the
sample clock. The two clocks still run at different rates,
`d(df)/dtauhat = 1/(rho alpha(1-alpha))`, so they cannot be merged into one
coordinate; that ratio is expressed in `rho`.

## 6. Dimensionless PDE

VERIFIED 2026-08-07 (`scratchpad/verify_assembly.py`), worst relative error
6.35e-16 in float64. Writing `x := muhat`, `T := tauhat`:

    u_T + (1/(2 T^2)) u_xx = ( sqrt(w) + sqrt(w + x) )^2

    w    := u - (1/rho) * [ u_df + (1/df^2) * D[u] ]
    D[u] := -(1/4) * ( u - x u_x - x^2 u_xx )

`D` is the operator produced by pushing the `S`-diffusion through the scale
quotient. It is second order in `muhat` ALONE plus lower-order terms — no new
independent diffusion direction, because the scale direction IS the quotiented
one. It arises from `U_S = (u - x u_x)/(2 sqrt(rho S))`, differentiated once
more.

Both sides of the raw-coordinate equation carry the common factor
`sqrt(rho S)`, which divides out; the check confirms that factorisation holds
to 5.3e-15 across seven decades of `S`.

Two readings of the coefficient are worth keeping in view:

- `1/(rho df^2)` multiplies the scale diffusion. With `df = df_0 + t` and
  `s = rho t`, that coefficient is `1/(rho (df_0 + s/rho)^2)`, i.e. `O(rho)`
  away from a boundary layer of width `s ~ rho df_0` at the start.
- `1/rho` multiplies the `df` transport, so `df` runs away fast in discounted
  time whenever `rho` is small (per-sample discount negligible).

## 7. Boundary conditions

Same count as two_arm plus one anchor, on `{0 <= muhat <= G(tauhat, df)}`:

    BC1 (ridge kink):        du/dmuhat = -1/2      as muhat -> 0+, every df
    BC2 (value matching):    u = 0                 on muhat = G(tauhat, df)
    BC3 (smooth pasting):    du/dmuhat = 0         on muhat = G(tauhat, df)
    BC4 (terminal/decay):    u -> 0                as tauhat -> infinity
    BC5 (known-variance):    u -> u_twoarm         as df -> infinity

BC1 is unchanged for the two_arm reason: the premium inherits the kink from
`max(m,0)/rho`, which has no `df` or `S` dependence.

BC5 is the anchor that makes the whole extension cheap: it is an exact
identity, not an asymptotic guess, and it is what the trained two_arm champion
supplies.

**Is the contact set exact?** At finite `df` an upward revision of `S` can
dislodge a committed state (`m`, `n` frozen, but `tauhat`'s companion `muhat`
shrinks as `sqrt(S)` grows), so strictly `u > 0` everywhere and the contact
set is empty. But `df` is monotone and `S` converges, so the leak seals: deep
in the commit region the required revision has exponentially small
probability, and `u` is exponentially small rather than zero. This is
qualitatively unlike the drift extension, where `dtau/dt = -eta^2 tau^2 < 0`
makes non-absorption permanent. Practical consequence: the `relu(r)^2 /
(1 + relu(r)^2)` response remains the right primitive, with its zero set exact
only in the `df -> infinity` limit.

## 8. The envelope

The free-information value is the same object with a `t` posterior in place of
the normal. For `X ~ t_nu(m, sc)` and `nu > 1`:

    E[max(0, X)] = m * F_t(m/sc; nu)
                 + sc * ((nu + (m/sc)^2)/(nu - 1)) * f_t(m/sc; nu)

`f_t`, `F_t` the standard `t` density and CDF, and here `nu = df`,
`sc^2 = b/(a n)`. It is finite only for `nu > 1`, so `df > 1` bounds the
usable domain (`df > 2` for a finite posterior variance).

VERIFIED 2026-08-07. Against Monte Carlo (4e6 draws, `F_t` by quadrature so
the identity is not used to test itself), relative error 1e-5 to 1.6e-3 across
`dof` 3 to 200 — consistent with MC noise on a heavy-tailed sample. And at
`dof = 1e6` it agrees with `pinn/utils.py:nu` to 1e-7 relative, confirming it
tends to `m Phi(m/sc) + sc phi(m/sc)`: the same function, same role as in
two_arm — a proven upper bound on the premium and the exact startup solution.

## 9. Limits, anchors, and the size of the prize

- `df -> infinity`: recovers `two_arm.md` in every formula. Exact, and the
  bootstrap slice.
- `alpha -> 0` or `1`: `dn/dt -> 0` but `d(df)/dt = 1` continues. The agent
  keeps learning `S` while learning nothing about `theta`. This is the
  content of section 1's two clocks and it is what makes the contact set leaky.
- **The prize is a startup transient, and `rho` sizes it.** `df` advances by
  `1/rho` per unit discounted time while `tauhat` advances by at most `1/4`.
  When the per-sample discount is negligible (`rho` small) `df` is effectively
  infinite outside a boundary layer of width `s ~ rho df_0`. The extension
  earns its keep at larger `rho` — few samples per discount horizon — and at
  small `df_0`.
- Caveat worth recording: real variance estimates are far less certain than
  `df = N` suggests, because unit-level outcomes are not independent
  (day effects, clustering, seasonality). The honest `df` is the EFFECTIVE
  degrees of freedom after correlation, which can be orders of magnitude
  below the user count. That, not the raw traffic, is what should set `df_0`
  and the `rho` at which the transient matters.

## 10. To verify before implementing

Every item is cheap and each kills a specific error.

DONE (`scratchpad/verify_variance.py`, 2026-08-07):

1. VERIFIED, and it corrected the doc: the martingale is `b/(a-1)`, not `b/a`
   (section 2). The diffusion `Var(dS)/dt = 2 S^2/df^2` holds in the
   diffusion limit, with the exact finite-batch form recorded.
4. VERIFIED: the `t` envelope against Monte Carlo, and its `nu -> infinity`
   agreement with `pinn/utils.py:nu` to 1e-7 relative (section 8).

OPEN:

2. The `+` branch selection in section 4 with `W` in place of `rho U` — the
   two_arm argument (continuity of `alpha*` at the ridge) should carry, but it
   was proved for `W = rho U`.
3. VERIFIED (`scratchpad/verify_assembly.py`): the assembled dimensionless
   equation of section 6 against the raw-coordinate form, float64, worst
   relative error 6.35e-16, plus the `sqrt(rho S)` factorisation holding to
   5.3e-15 across seven decades of `S`.
5. VERIFIED by the exact sequential result of section 11: `Delta a = 1/2` per
   observation at every allocation, `alpha` exactly 0 or 1 included.

Not on the list because it is a theorem, not a claim: the independence of the
`dm` and `dS` innovations is Cochran's — for normal samples the sample mean
and the pooled residual sum of squares are independent.

## 11. The sequential update (VERIFIED 2026-08-07)

One observation per unit time. This is the exact conjugate bookkeeping, and
the continuous model of sections 1-9 is its limit.

**State.** Per-arm NIG with a shared variance:

    mu_j | s^2 ~ N(m_j, s^2/n_j),  j in {A, B}      s^2 ~ IG(a, b)

carried as `(m_A, m_B, n_A, n_B, a, b)`.

**Update on one observation `y` drawn from arm `j`:**

    n_j' = n_j + 1
    m_j' = m_j + (y - m_j) / (n_j + 1)
    a'   = a + 1/2
    b'   = b + (y - m_j)^2 / (2 (1 + 1/n_j))

The `b` increment is the surprise in `y`, scored against its own marginal
variance `s^2 (1 + 1/n_j)` — sampling spread plus prior spread on that arm's
mean.

**`Delta a = 1/2` per observation, at every allocation.** Every sample carries
exactly one datum about the scale, whichever arm produced it, so the variance
clock is the sample clock and the agent cannot influence it. `alpha = 0` and
`alpha = 1` are not special cases: the update above is unconditional in `j`.
This closes section 10 item 5 exactly, with no asymptotics.

**Contrast readout.** `theta = mu_B - mu_A`, so

    m = m_B - m_A        n = n_A n_B / (n_A + n_B)      (harmonic)

**Why `dn/dt = alpha(1-alpha)`.** Differentiating the harmonic combination
with `n_A` growing at `1-alpha` and `n_B` at `alpha`, and writing
`f := n_B/(n_A + n_B)` for the HISTORICAL fraction:

    dn/dt = f^2 (1 - alpha) + (1 - f)^2 alpha

At `f = alpha` — historical and current allocation agreeing — this collapses
to `alpha(1-alpha)`. So section 1's law is the steady-state form of the
harmonic increment, exact on any path that holds its split, and an
approximation exactly to the extent the allocation is being changed. The
general form above is what a simulator should carry.

**Readouts.**

    posterior of theta:       t_{2a} ( m,    b/(a n) )
    predictive for next y_j:  t_{2a} ( m_j,  b (1 + 1/n_j)/a )

The predictive is what an arena entrant needs for Thompson-style sampling.

**Map to the continuous chart.** With one sample as the time unit:

    tauhat     = rho n
    d(tauhat)  = rho * (harmonic increment)     per sample   (section 5)
    d(df)      = 1                              per sample

**Calibrating `df_0`.** For `IG(a, b)` the coefficient of variation of `s^2`
is `1/sqrt(a - 2)`, i.e. `sqrt(2/(df - 4))`. A 5% relative uncertainty on the
variance is `df ~ 800`; 20% is `df ~ 54`. Use the EFFECTIVE degrees of freedom
after clustering, not the raw sample count (section 9).

**Verification.** `scratchpad/verify_update.py`: the four update formulas
against brute-force quadrature of prior x likelihood on a grid over
`(mu_j, s^2)`, comparing `E[mu_j]`, `Var[mu_j]`, `E[s^2]`. The
`Delta a = 1/2` identity checked at `alpha` in `{0, 1}` and in between.

## 12. To come

Implementation follows two_arm's shape: one extra input feature carrying
`df` (`1/df` or `2/df`, which vanishes in the anchor limit), zero-initialised
so the net reproduces the two_arm champion exactly at `1/df = 0` — the same
stitch discipline as the three_arm kink branch. The `t` envelope replaces
`nu`, the response map and free-boundary machinery are unchanged.

Open before code: whether `rho` is a net input (one net for all discount
regimes) or a fixed constant per deployment (one net per `rho`), which is the
same fixed-versus-input question the drift extension will face with `etahat`.
