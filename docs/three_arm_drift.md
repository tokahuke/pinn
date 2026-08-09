# three_arm_drift — Three-Armed Allocation with Drifting Means

HJB problem for a future `pinn/problems/three_arm_drift/`. Extension of
`three_arm.md` to arm means that themselves wander. Everything below is exact
unless flagged UNVERIFIED. Read `three_arm.md` first, and `two_arm_drift.md`
for the two-armed version of the same extension; this doc only records what
changes.

Written for a reader who has not touched this in three months. Plain words
first, equations second, and nothing named after whoever proved it.

## 0. Notation

Same contract as `three_arm.md`: uppercase dimensional, lowercase
dimensionless, hats dimensionless coordinates.

New symbols:

- `eta`     how fast each arm's true mean wanders, `d mu_i = eta dW_i`
- `etahat`  its dimensionless form, `etahat := eta / (rho sigma)`
- `E`       the wander of the two contrasts, a 2x2 matrix, defined in section 1
- `T*`      the ceiling on precision, section 6

`etahat = 0` recovers `three_arm.md` in every formula below.

**One warning about `eta` across the two docs.** In `two_arm_drift.md` `eta` is
the wander of the *contrast* between the two arms. Here it is the wander of
*each arm*. Two arms wandering independently at `eta` make their contrast
wander at `sqrt2 eta`, so a two-arm formula reused here takes `sqrt2 etahat`,
not `etahat`. Section 7 depends on this.

## 1. Problem statement

Same as three_arm except the unknown means are no longer constant:

    d mu_i = eta dW_i        i in {a, b, c}, independent of each other
                             and of the observation noise

**One `eta` for all three arms, and that is not a modelling preference.** The
whole solve rests on the problem looking the same after you shuffle the arm
labels — that is what lets us solve on one sixth of the space
(`three_arm.md` section 6) and what both wall conditions are statements about
(section 12 there). A per-arm `eta_a, eta_b, eta_c` breaks that symmetry, the
sixfold saving dies, and both wall conditions have to be rewritten. Common
`eta` keeps everything.

**The contrasts inherit a wander that is not what you would guess.** The state
is the posterior on the contrasts `theta_b = mu_b - mu_a` and
`theta_c = mu_c - mu_a`, and both of them contain the control arm. So when
`mu_a` moves, *both* contrasts move together:

    d theta_b = eta (dW_b - dW_a)
    d theta_c = eta (dW_c - dW_a)

which gives a shared, non-zero off-diagonal:

    E = eta^2 [ 2  1 ]  =  eta^2 (I + 1 1^T)
              [ 1  2 ]

That off-diagonal is forced by the shared control arm — it is not a choice, and
it lands directly on `tau_bc`, the coordinate where three_arm's worst residual
already lives.

**The belief clock gains an erosion term.** The precision matrix
`T = [[tau_bb, tau_bc], [tau_bc, tau_cc]]` used to only fill up. Now it also
leaks:

    dT/dt = G(alpha) - T E T

`G(alpha)` is unchanged (`three_arm.md` section 2). Today's best guess at the
means is still the best guess at tomorrow's -- the wandering has no direction,
so it erodes what you know without pulling the estimate anywhere. Their
jitter is unchanged at `T^-1 G T^-1`. Same split as two_arm_drift.

Useful for the algebra below, since it puts the erosion back into the pairwise
coordinates the sampler already uses:

    T E T = eta^2 ( T^2 + v v^T ),    v = T 1 = (tau_bb + tau_bc, tau_cc + tau_bc)

and `v` is exactly `(I_ab, I_ac)` from `three_arm.md` section 7.

**Why the arm-shuffle symmetry survives.** Relabelling arms acts on the
contrasts by a matrix `M`, and the wander transforms to `M E M^T`. For the
swap of the two treatments, `M = [[0,1],[1,0]]`; for making arm b the control,
`M = [[-1,0],[-1,1]]`. Both leave `E` alone, exactly:

    M E M^T = E    for all six relabellings

so everything symmetry-based in `three_arm.md` carries over untouched.

## 2. Fixed-policy PDE

For a given policy, `three_arm.md` section 4 plus the erosion, which enters as
three extra terms — one per independent entry of the precision matrix:

    rho V + (TET)_bb dV/dtau_bb + (TET)_bc dV/dtau_bc + (TET)_cc dV/dtau_cc

        = alpha_b m_b + alpha_c m_c
          + (the three mean-diffusion terms of three_arm.md section 4)
          + (gb/sigma^2) dV/dtau_bb - (g/sigma^2) dV/dtau_bc
          + (gc/sigma^2) dV/dtau_cc

Coefficient **one** on each of the three, matching the convention already in
the code: `three_arm.md` writes the fill term as `sum_ij G_ij dV/dT_ij`, but
its own explicit form two lines down, and `three_arm/model.py`'s Hamiltonian,
both use plain chain rule over the three independent entries, so `dV/dtau_bc`
carries `G_bc` with a factor of one. The erosion follows the same convention.

The HJB is the same with `max over alpha in Delta` on the right.

**The new terms do not depend on the allocation.** `E` is a property of the
world, not of what you choose, so the inner maximisation is untouched: still
the same quadratic over the triangle, still the seven candidates of
`three_arm.md` section 10. `three_arm/simplex.py` is reused unchanged, for the
same reason `two_arm_drift` reuses two_arm's. This is the general pattern from
`two_arm_drift.md` section 2: a term that does not see the control never
touches the maximisation, it only joins the left side.

## 3. Exploration premium

Unchanged from `three_arm.md` section 5. The commit value
`C(m) = max(0, m_b, m_c) / rho` is still attainable — freezing the allocation
freezes what you expect the means to be, since the wandering has no direction —
and `C` does not depend on the precision at all. So the erosion terms act on the
premium `U` alone, and the substitution `V = C + U` goes through with the three
new terms simply carried along.

## 4. Dimensionless form

+0 state, +1 parameter, exactly as in two_arm_drift. The state is still
`(m_b, m_c, tau_bb, tau_bc, tau_cc)`; the wander enters as one more number the
network is told:

    etahat = eta / (rho sigma)

One trained checkpoint therefore serves every drift regime, including
`etahat = 0`, which is three_arm exactly.

Readout dictionary is `three_arm.md` section 11 plus that one line.

**What `etahat` means in practice.** Dividing by `rho` is dividing by a rate,
so the number is bigger than intuition suggests. `etahat/2` is how far the
truth wanders in one discount horizon divided by how precisely you could
measure it over that same horizon — see `two_arm_drift.md` section 0 and the
worked arena numbers. `etahat` above 1 just means the truth outruns the
experiment, which is the ordinary case when `rho` is small.

## 5. Boundary conditions

**Both wall conditions of `three_arm.md` section 12 survive word for word.**
They are statements about relabelling arms, `E` is unchanged by relabelling
arms, so nothing about them moves. The control-tie wall remains the thing that
rules out the do-nothing solution, because the do-nothing solution still
satisfies the interior equation exactly: setting `U = 0` kills every term
including the three new ones, since they are all derivatives of `U`.

**The commit region disappears.** In three_arm, picking one arm and staying
freezes the beliefs (`G = 0` at a corner), so committing is permanent and the
premium is exactly zero on a whole region. Under drift, `dT/dt = -TET` at a
corner, which is strictly negative: what you know decays, uncertainty comes
back, and the option to switch is worth something. So

    U > 0 strictly, everywhere

and there is no region where the premium is flat zero.

Carry `two_arm_drift.md` section 5's self-correction across, because the same
mistake is available here: **this does not mean the free boundary is gone.**
Three_arm currently has one surface doing two jobs — the place where the
premium hits zero is also the place where the policy commits — and the
`relu(r)^2` response locates both at once. Drift separates them. The
commit-or-explore switching surface is still there, still a genuine boundary
where the solution's curvature jumps, but it is no longer where the premium
vanishes.

## 6. What states are reachable, and the ceiling

**The sampler's geometry survives.** `three_arm.md` section 7 samples in
pairwise coordinates `I_ab = tau_bb + tau_bc`, `I_ac = tau_cc + tau_bc`,
`I_bc = -tau_bc`, because the reachable states are exactly the corner
`{I >= 0}` — in raw precision coordinates the same set is a skewed cone with
about four times the wasted volume. That choice rested on precision being the
prior plus a straight-line function of what each pair accumulated, which the
erosion breaks: `T E T` is quadratic in `T` and does not split per pair.

The *set* survives anyway, which is all the sampler ever used, because drift
pushes inward at both faces:

    at tau_bc = 0 :  d tau_bc/dt = -alpha_b alpha_c/sigma^2 - eta^2 tau_bb tau_cc  < 0
    at I_ab   = 0 :  d I_ab/dt   =  alpha_a alpha_b/sigma^2 + eta^2 |tau_bc| I_ac  >= 0

(both by hand from `TET = eta^2 (T^2 + v v^T)`; at `tau_bc = 0` the erosion's
off-diagonal is `eta^2 tau_bb tau_cc`, and at `I_ab = 0` the `v v^T` part drops
out and the `T^2` part leaves `eta^2 tau_bc I_ac` with `tau_bc <= 0`).

So `{I >= 0}` is still exactly the reachable region, relabelling arms is still
a shuffle of three coordinates, and `Sample.fold_ordered` needs no change.

**Drift adds a ceiling on how much you can ever know.** Learning stops when
what you buy each instant equals what the wandering destroys, `G = T E T`. That
equation solves exactly. Feeding it the most informative allocation there is —
ignore the control, split evenly between the two treatments,
`alpha = (0, 1/2, 1/2)`, which is where `G` is largest — caps every reachable
state:

    T  <=  T*  =  E^(-1/2) / (sigma sqrt2),     det T* = 1 / (2 sqrt3 sigma^2 eta^2)

Two things here that the two-armed intuition gets wrong.

**The ceiling is not a single number, it has a shape.** In two arms there is
one precision and one ceiling. Here the cap is tight in the b-versus-c
direction and slack by a factor of two in the other two directions, so "there
is a maximum precision" is the wrong picture; there is a maximum *ellipse*.

**The ceiling itself is not symmetric under relabelling arms**, even though the
problem is. That is not a contradiction — it comes from feeding in one specific
allocation. Since the problem is symmetric, each of the six relabelled copies
of `T*` is equally valid as a cap, so use the tightest of the six. Measurably
better than any one of them alone.

**Sampling recipe**, following two_arm_drift's `_tauhat` pattern:

1. draw `etahat` first, spanning decades and reaching zero (the three_arm
   anchor), capped where the ceiling would collapse onto the precision floor;
2. draw the *shape* of `T` in the existing corner, unchanged;
3. compute how far along that ray the ceiling sits — for each of the six
   relabellings, `det(T_shape - mu C) = 0` is a scalar quadratic, so this is
   closed form and sampler-only, no derivatives needed;
4. scale by a fraction of that distance, drawn on the existing decade-spread
   law and clipped at 1, so the clipped mass lands **on** the ceiling, which is
   where trajectories converge;
5. means last, unchanged;
6. the two wall samplers each gain `etahat`, since both conditions hold at
   every `etahat`.

## 7. The envelope

### What it is and why the current one breaks

The envelope is the value of being handed the right answer for free. It caps
the premium, because paying for information cannot beat being given it. The
architecture multiplies it by a response between 0 and 1, so the cap holds by
construction rather than by a penalty term.

Three_arm's envelope, `nu2`, prices *one* free answer, *now*. Under drift that
is worth less than it used to be, because the answer goes stale: knowing the
truth today tells you nothing about next month. The honest price is being told
the answer over and over, forever, discounted:

    u_env = integral_0^inf  e^(-t)  nu2( m_b, m_c, sd_b(t), sd_c(t), corr(t) )  dt

        sd_b(t) = sqrt( Sigma_bb + 2 eta^2 t )
        sd_c(t) = sqrt( Sigma_cc + 2 eta^2 t )
        corr(t) = ( Sigma_bc + eta^2 t ) / ( sd_b(t) sd_c(t) )

with `Sigma = T^-1` the posterior covariance, widening as `Sigma_0 + E t`. Note
that both the spreads *and* the correlation move with `t`. At `t = 0` this is
three_arm's current envelope.

This is not a refinement. Three_arm's bare `nu2` is about ten times too small
at the drift steady state, so under drift it stops being a cap at all.

### The structure that makes it cheap

Ask how fast that price changes as the spread widens. The answer splits into
three pieces, one per pair of arms, and each piece is a product of two things:

    d/dt nu2  =  eta^2 * sum over the three pairs of

                 ( how likely that pair is neck and neck )
               x ( how likely that pair is the one that decides it )

The three pairs are a-versus-b (tie at `theta_b = 0`), a-versus-c (tie at
`theta_c = 0`), and b-versus-c (tie at `theta_b = theta_c`). Their spreads all
widen at exactly the same rate, `2 eta^2`, so no pair is special.

The first factor is a plain normal density at the tie. The second is a
probability that the third option is below the tied pair — a normal
distribution function, but with a mean and spread that both change with `t`,
and that is what has no closed form.

**The first factor is the two-armed problem we already solved.** Integrated
against the discount, it is exactly `two_arm_drift/envelope.py`'s formula, at
that pair's mean and spread, with `sqrt2 eta` for the reason in section 0.

### What we ship: set the second factor to 1

Integrating by parts turns the integral into its value at `t = 0` plus the
discounted rate of change. Bounding each of the three "decides it"
probabilities by 1 — its largest possible value — gives a closed form:

    u_env = nu2(m_b, m_c, Sigma_0)  +  C_ab + C_ac + C_bc

    C_p   = two_arm_drift correction at ( m_p , 1/V_p , sqrt2 etahat )

      pair a-b :  m_ab = m_b,        V_ab = Sigma_bb
      pair a-c :  m_ac = m_c,        V_ac = Sigma_cc
      pair b-c :  m_bc = m_b - m_c,  V_bc = Sigma_bb + Sigma_cc - 2 Sigma_bc

Replacing a probability by 1 only makes the answer bigger, so this is still a
genuine cap — which is the only thing the architecture needs. No new numerical
integration beyond the `nu2` already there.

Three properties worth stating because they are what make it safe:

- **Exact at zero drift, to the last bit.** `sqrt2 etahat` multiplies each
  correction throughout, so at `etahat = 0` all three vanish and the formula
  returns three_arm's current envelope identically. Grafting the three_arm
  champion is therefore exact, not approximate.
- **Symmetric under relabelling by inspection.** Relabelling shuffles the three
  pairs among themselves, and each term looks only at its own pair. Both wall
  conditions need that symmetry to hold exactly, so being able to *see* it
  beats having to prove it.
- **Never below the truth**, checked directly rather than assumed.

### Why the looseness is acceptable

How much bigger than the truth, measured against a fine numerical integral,
with the means scaled by the local spread so the states are comparable:

    etahat        at the ceiling      bare nu2, for contrast
      0.1        1.010 - 1.074           0.911 - 0.990
      1.0        1.087 - 1.402           0.547 - 0.916
      7.5        1.333 - 1.760           0.201 - 0.674
     50.0        1.638 - 1.913           0.069 - 0.367

Worst case just under 1.92. That lands in the same band as the envelope that
failed once before (1.17 to 1.87, `three_arm.md` section 13), so the doc has to
say why this is not the same mistake.

**A loose cap has hurt in exactly one way, ever, and it is not general
looseness.** Before any data, the cap is not merely a cap — it *is* the answer,
because with no information the best you can do is exactly the value of being
told. The network has no freedom there: its output must sit at the very top of
its range. That single pinned point is what holds the whole solution away from
the do-nothing answer, which otherwise satisfies the equation perfectly and is
the standing trap.

The old envelope was 17 to 87 per cent too big *at that pinned point*, so the
network sat at half its range where the true answer was the top of it, the pin
dissolved, and training slid into never exploring.

This one is 1 to 3 per cent off there. The pin holds. Its looseness is at the
other end, the drift steady state, where nothing analogous is at stake: the
true premium there is large and positive — that is the regime where you never
stop exploring — and the network's output sits comfortably mid-range rather
than pressed against either edge.

### The tight version: derived, recorded, not built

Keeping the "decides it" probabilities instead of dropping them needs a
sixteen-point rule in `t`, and gives 1e-4 accuracy. Worth recording so the
upgrade is a swap rather than a re-derivation, and worth noting *why* it is
cheap when the obvious nesting would not be: the numerical rule never touches
the steep part, which stays closed form, only a bounded ratio between 0 and 1
where node error largely cancels. That distinction is exactly what made
two_arm's early thirty-two-point attempt wrong by a factor of 79.

It is provably no bigger than what we ship, so swapping it in can only tighten.

**Trigger for building it**, stated now so it is not a judgement call later:
the do-nothing region growing across training (the recorded failure went from
52 to 65 per cent), or the network's output pressing against the cap at the
steady state.

### REJECTED, with the measurement: one call at the average spread

Tempting, and wrong. The integral averages over time with mean 1, so *if* the
thing being averaged were concave in time, the average would be at most its
value at the mean, and the whole envelope would collapse to a single
`nu2(m_b, m_c, Sigma_0 + E)` — one call, no pairs, exact at both anchors.

It is not concave. The value of a free answer is **convex** in the spread, so
along time it is an S-curve: convex while `eta^2 t < m^2`, concave only after.
Averaging at the mean therefore undershoots, and undershoots worst in the deep
wedge, which is where three_arm is already hardest:

    mean, in local spreads      one call / truth
              0                     1.039
              1                     1.052
              2                     0.919
              3                     0.649
              6                     0.046

Below 1 is not a loose cap, it is not a cap. A twentyfold undershoot removes
the architectural bound entirely.

Note the trap that hid this: draw the means proportional to the local spread
and the deep wedge is never visited, and every ratio comes back above 1.

Also worth recording, since it will be asked: averaging inside the max rather
than outside gives a floor, not a cap — the wrong direction entirely.

## 8. Grading and conditioning — nothing changes

Two_arm and two_arm_drift grade the equation in a stretched chart where the
drift term is guaranteed to stay a sensible size. Three_arm does not have that
chart in code: it grades in raw coordinates with a single number multiplying
the residual, `w = (det T)^(3/4) / (tau_bb + tau_cc + tau_bc)`
(`three_arm.md` section 14). The obvious worry is that the erosion term, whose
size goes like `etahat^2 det T`, wrecks that.

It does not, because the ceiling bounds it:

    etahat^2 det T  <=  etahat^2 det T*  =  1 / (2 sqrt3)  =  0.2887

exactly, and independent of `etahat`. Graded in premium units the erosion's
coefficient is at most `0.537 etahat`, which is the same shape as
two_arm_drift's `etahat/2` — a factor of a few at deployment, which that
problem lives with.

The failure-mode enumeration of `three_arm.md` section 14.4 is unchanged. The
commit value does not depend on precision, so the erosion acts on the premium
alone; the do-nothing solution still zeroes the residual exactly; and no new
failure mode with a different scaling appears.

**So: same weight, no chart in code, no new loss term.** Stated explicitly
because it is the section most likely to be second-guessed later.

The one thing to *measure* rather than assume: after the first training run,
bucket the residual by `etahat` decade. If the high decades dominate, the fix
is sampling density, not a new weight — those are one mechanism, per
`learnings.md` section 7.

## 9. Architecture consequences

- **Feature.** One more, `log1p(2 sqrt3 etahat^2 det T)`: exactly zero at
  `etahat = 0` so the champion graft is exact, unchanged by relabelling arms
  since `det T` is the symmetric invariant, bounded on the reachable set, and
  aligned with the erosion's own coefficient. Appended last, since grafting
  pads on the right.
- **Envelope.** Section 7, replacing `nu2`.
- **Response.** `relu(r)^2 / (1 + relu(r)^2)`, unchanged. Note this puts the
  premium at exactly zero on a region where section 5 says it should merely be
  very small. That is a known approximation, inherited from two_arm_drift,
  where it trains well; it is a training question, not a reason to give up a
  cap that is true by construction.
- **Maximisation.** `three_arm/simplex.py` reused unchanged.
- **Sampler.** Section 6.
- **Loss.** `etahat` threaded through; weight, power and wall weight untouched.
- **Grafting.** Copy `stitch` from two_arm_drift: pad the first layer with a
  zero column, append the calibrated scale, default the kink tensors.

## 10. Verification

DONE (float64, in-session 2026-08-08):

1. VERIFIED. `M E M^T = E` for all six relabellings, exactly 0.0 — the fact
   section 1 and section 5 both rest on.
2. VERIFIED. `T E T = eta^2 (T^2 + v v^T)` with `v = T 1`, worst 1.4e-14 over
   2000 random precision matrices.
3. VERIFIED. `G(alpha) <= I/(2 sigma^2)` over the whole triangle, with equality
   at `alpha = (0, 1/2, 1/2)`: the most information any allocation buys in any
   one direction is exactly 0.5, and the gap never goes negative in any
   direction. This is what makes the section 6 ceiling a genuine cap.
4. VERIFIED. The ceiling identity `T* E T* = I/(2 sigma^2)` to 3e-16, and
   `det T* = 1/(2 sqrt3 sigma^2 eta^2)` to machine precision, across
   `eta in {0.2, 1, 3}`. Hence `etahat^2 det T <= 1/(2 sqrt3) = 0.288675`
   exactly, the section 8 bound.
5. VERIFIED. The section 7 rate decomposition against a finite difference of
   `nu2`: worst relative error 2.1e-09 over 27 states spanning
   `eta in {0.3, 1, 5}` and precision scales `{0.05, 1, 20}`, including the
   deep wedge. Dropping the three probabilities never fell below the true
   rate, which is the cap.
6. VERIFIED. The shipped envelope is at or above a fine numerical integral of
   the true one everywhere probed, with the ratios of section 7.
7. VERIFIED as a rejection. The one-call form of section 7 falls below the
   truth in the deep wedge, down to 0.046, so it is not a cap.
8. VERIFIED by hand. The two face derivatives of section 6, from
   `T E T = eta^2 (T^2 + v v^T)`.

OPEN:

9. **Does anything escape the ceiling?** The containment argument is the
   standard one for this kind of equation but was not proved here, only
   checked by simulating trajectories under random allocations. A tighter
   ceiling that is already symmetric under relabelling can be written down, but
   I could not show states cannot escape it, so section 6 uses the
   six-copies version instead. If the tighter one holds, the sampler improves
   and one decade of waste disappears.
10. **Well-posedness without a datum at large precision.** Inherited from
    `two_arm_drift.md` item 5 and unverified there too. Cheapest probe is the
    same: solve, then check the answer is insensitive to what is imposed at the
    far end.
11. **Does one drift feature suffice?** The erosion depends on the *shape* of
    the precision matrix in a way no single number sees. Measure before adding
    a second.
12. **Deployment `etahat`.** Per arm it is the contrast number divided by
    `sqrt2`, so around 7.5 where two_arm's is 10.7 for the same underlying
    drift. The sampler's scale is a guess until the real number is settled.
13. **The arena is the only referee, and it is not built.** A drift-aware
    three-arm zoo must wander all three arms; it cannot be copied from
    `arena/two_arm_drift.py`, which wanders only the treatment. That is a legal
    shortcut with two arms, where only the contrast matters, and an illegal one
    with three, where it breaks the arm-shuffle symmetry this whole doc rests
    on. Worth its own plan, after the two-arm drift arena has actually been
    run — that payoff is still unmeasured.

## To come

Nothing mathematical is missing for an implementation. The order that follows
`two_arm_drift`'s: envelope module first with its self-checks, then the
sampler, then thread `etahat` through model and loss, then graft the three_arm
champion and confirm step zero is bit-exact.
