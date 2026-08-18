# Arena results (2026-08-06)

Policy shoot-out in `pinn/arena`: discrete-epoch simulation against the true
effect, discounted regret vs the always-pick-the-winner oracle, 4,000 paired
experiments per problem (same effect draw and noise stream across policies
within a rep). Effects drawn iid from a zero-centered normal; parameters
chosen to mirror a realistic deployment (values intentionally not recorded
here). All numbers are relative to Thompson sampling = 1.00 within each
problem; absolute regret is parameter-scale-dependent and meaningless across
configs.

Policies play prior-blind (priors are policy parameters, never the
environment's effect distribution). The PINN entrants run the champion
models of this date — two_arm `value_2a_32:512.pt` from
`data/frontier4k.pkl`, three_arm `value_3a_64:64:64.pt` from
`data/frontier3a.pkl` (both nets on the `checkpoints-2026-08-06` release) —
with the near-flat default prior, evaluated directly (no policy table),
commits landing on exact simplex vertices. Those two filenames are historical
and no longer resolve; see the staleness note.

**SUPERSEDED 2026-08-13.** Every row below is a faithful record of what the
2026-08-06 models did at size 4,000, and nothing else: both nets have
since been replaced several times over, the residual grading changed under
them (natural units, learnings section 3), and the arena's
`_FLATTEST_TAUHAT` guard moved 1e-2 -> 1e-3. Current numbers, 2026-08-13, at
production parameters with the paired estimator:

    two_arm    6,000 reps   PINN 0.77 vs Thompson, on 0.45 of the evidence
    three_arm  4,000 reps   PINN 0.83 vs Thompson, on 0.35 of the evidence

Two things those replace. The old tables have the three-arm margin BEATING
the two-arm one (0.77 against 0.80), which supported a claim that the margin
grows with the number of arms; it now runs the other way (0.83 against 0.77)
and that claim is withdrawn. And in the harsh-drift world the old record had
the drift net losing badly to Thompson -- at the loosened guard it wins,
23,661 against 27,771, +4,110 +/- 778 paired.

## Two arms

| policy | regret (TS = 1.00) | wrong-commit | commits | median commit (fraction of horizon) | evidence (TS = 1.00) |
|---|---|---|---|---|---|
| PINN | **0.80** | 6.6% | 99.2% | 0.06 | **0.46** |
| Thompson sampling | 1.00 | 0.0% | never | — | 1.00 |
| explore-then-commit | 1.65 | 13.0% | 100% | 0.07 | 0.33 |
| z-test at 5% | 1.81 | 10.7% | 93.7% | 0.02 | 0.67 |

Baseline values are stable across independent sweeps at different sizes, so
the harness is measuring on a settled scale. CIs (95%, 4k runs): about
+/-0.04 on the PINN and TS rows in these relative units; the PINN-TS gap is
~7 combined standard errors before crediting the pairing.

## Three arms

| policy | regret (TS = 1.00) | wrong-commit | commits | median commit (fraction of horizon) | evidence (TS = 1.00) |
|---|---|---|---|---|---|
| PINN | **0.77** | 10.4% | 99.8% | 0.09 | **0.38** |
| Thompson sampling | 1.00 | 0.0% | never | — | 1.00 |
| elimination at 5% (generalized z-test) | 1.53 | 11.8% | 90.5% | 0.04 | 0.71 |
| explore-then-commit | 1.81 | 20.9% | 100% | 0.07 | 0.29 |

(Original ratios vs best: TS 1.29, elimination 1.97, ETC 2.34; renormalized
to TS = 1.00 above for comparability with the two-arm table.)

## Readings

- **The PINN margin grows with arms**: 20% less regret than Thompson at two
  arms, 22.5% at three. Thompson splits exploration by win-probability; the
  HJB prices each contrast's information against the discount, and that
  advantage compounds as contrasts multiply.
- **It wins while buying less**: 46% of Thompson's information spend at two
  arms, 38% at three — the PINN's spend went DOWN with an extra arm while
  Thompson's went up. The mechanism is not measuring better; it is knowing
  when measurement stops being worth its discounted price, and committing
  (99%+ of runs) through the one-way door Thompson never takes.
- **Classical methods degrade with arms exactly as theory warns**:
  explore-then-commit's wrong-commit rate jumps 13% -> 21% (one fixed
  deadline cannot serve two unknown gaps), and peeking-style testing stays
  ~2x despite buying more information than anyone but Thompson.
- **Soft commit time** (`N (1 - sum a^2) / (N - 1)` summed over epochs, the
  uniform-equivalent epochs of evidence) is what makes the information
  column comparable across arms counts; explore-then-commit's value equals
  its deadline exactly with zero variance, the metric's built-in canary.
  Commit timings are reported as fractions of the horizon so the tables
  carry no absolute parameter information.

Reproduce: `poetry run arena simulate <out.pkl> --problem two_arm|three_arm
--rho ... --size 4000 --workers 8`, then `poetry run arena analyze <out.pkl>`.
Raw studies of this date: `data/frontier4k.pkl` (two arms),
`data/frontier3a.pkl` (three arms) — gitignored, parameter-bearing.

## The prior floor: `_FLATTEST_TAUHAT` = 1e-3 (2026-08-13)

The weakest prior a champion is trusted at, in dimensionless precision, and the
same value in all four zoos. It is the sampler's own `PRIOR_FLOOR`, so the guard
does not clamp inside the training support.

Measured against the drift champion of that date, 3000 paired reps at production
parameters: at 1e-3 rather than 1e-2, harsh-drift regret falls 62%
(-38,753 +/- 3,591) on 3.6x the evidence, while the deployment world does not
move (+361 +/- 601, a CI covering zero).

The corner a tighter guard protected is gone with it: the champion of that date
satisfies pairwise positivity on 100.0% of states and full concavity on 99.9%.
An earlier measurement (2026-08-10) said the opposite on both counts, taken
against a model since replaced by one 39x better on the residual. That is
the lesson: a guard justified against a broken net must be re-measured every
time the net is replaced, or it silently becomes the thing being measured.

## The explore-then-commit deadline

`optimal_deadline` reads the horizon and the discount, and nothing else.

**Design restriction.** It may not read `effect`, `effect_std` or `sigma`. The
true regret-minimising deadline does depend on `effect_std / sigma` (and only on
that ratio), but a policy does not know the distribution it is being tested
against, so tuning to it would make the comparison against the other policies a
lie.

Blind to the effect size, the RATE `T**(2/3)` is the standard
explore-then-commit result for an unknown gap: the known-gap optimum is
`(4/d**2) log(T d**2 / 4)`, and the worst case over `d` sits at `d ~ T**(-1/3)`.
`T` is the effective horizon, the shorter of the discount's `1/gamma` and the
hard horizon.

The leading constant of 1 is CALIBRATED, not derived: swept against the exact
optimum, `c = 1` costs at most 1.13x the oracle deadline's regret over
`effect_std / sigma` in 0.03 to 0.30.

## Harness invariants

**A rep's numbers do not depend on the batch around it.** Each rep's noise
stream is a function of its seed alone, and a masked `normal` advances only the
consuming reps' cursors, so a rep at a vertex (which buys no draw) keeps its
stream aligned with the same seed in any other batch. `harness.demo` asserts it
for every zoo by replaying a permuted sub-batch.

That equality is bitwise for the closed-form policies at any horizon. The Pinn
nets are float32, whose matmuls round batch-size-dependently, and a chaotic
trajectory amplifies that wobble: measured ~3e-7 relative in the static worlds
and ~1e-3 under drift, and over thousands of epochs it can decorrelate
entirely, leaving the same policy on the same noise in a different
micro-realization. The trajectory tolerances sit above the wobble and far below
the O(1) that any cross-rep stream leakage would produce; `delta` and the commit
fields stay exact, which is what a misaligned cursor breaks first.

A NET-CARRYING policy is exempt from the trajectory comparisons entirely,
because no tolerance can be set for it: the wobble depends on which model
is loaded, not on any arena code, so a passing `rtol` is a property of today's
champion rather than of the harness.

**Noise appetite.** `Runner.draws_per_epoch` is 2 by default, the static zoos'
appetite. The row is drawn in one `randn(capacity)` call and a longer call does
not share its prefix, so raising the default for everyone would silently move
every arena number ever recorded. A zoo that needs more declares
`DRAWS_PER_EPOCH` and only its own streams change; the drift zoos do, which is
why their runs cannot be compared draw for draw against their static siblings.

## The drift zoos

**One zoo, not two.** Every policy carries its own `sigma` and `eta` as POLICY
parameters, so drift-blind is simply `eta = 0` and there is no separate
drift-unaware class to keep in step. `init` ties them to the environment's,
which is the correctly-specified case, and direct construction unties them,
which is how the misspecification grid is swept (the same pattern as
`ExploreThenCommit`'s `deadline`):

    eta_policy    = 0 < eta        the cost of ignoring real drift
    eta_policy    > eta = 0        the premium for insuring against none
    sigma_policy != sigma          the cost of misjudging the noise

With `eta = eta_policy = 0` and `sigma_policy = sigma`, two_arm_drift reproduces
the two_arm arena exactly, Pinn included, because the filter folds the prior in
exactly as the two_arm conjugate posterior does. Its demo asserts that.

**three_arm_drift is assembled, not written.** The effect draw, the observation
model and every policy's `propose` are three_arm's unchanged, because drift
changes neither what an epoch buys nor how a policy reads a posterior. Two
things are new, both the two_arm_drift move carried to a 2x2 posterior:

- `advance` walks the ARMS, not the contrasts. Three independent walks at
  volatility eta give the contrasts covariance `eta^2 (I + 11')`, the shared
  `-theta_a` putting `eta^2` in the off-diagonal, which is the same erosion
  matrix the trained net is graded against (three_arm_drift/loss.py pins it).
- `Filter` forecasts before it updates, so precision decays where the two-arm
  zoo's scalar recursion decays. Written as `T (I + E T)^-1` rather than
  `(T^-1 + E)^-1` for the reason two_arm_drift writes `tau / (1 + eta^2 tau)`: a
  flat start has `T = 0` and needs no special case.

At `eta = 0` every number in it is three_arm's, which its demo asserts on the
FILTER rather than on a run: the zoo eats three extra variates an epoch, so two
runs cannot be compared draw for draw. Both posteriors are driven with the same
observations instead, at `rtol` 1e-4 and no tighter, because three_arm
accumulates the evidence `q` in float32 and inverts at read time while the
filter carries the mean in float64 and folds each observation in. The gap is
that float32 accumulation, the filter being the more precise side, and it lands
at ~1e-7 relative.

## Reading the report

**`wrong%` under drift** scores the committed arm against `delta`, which is the
effect at epoch 0. When `eta > 0` the truth moves afterwards, so the column
reads "committed against the arm that was best when the run started", not
"against the arm that was best while committed". The honest drift metric is the
regret column, measured per epoch against the moving oracle.

**The median commit epoch** comes from `committed_at`, not `epochs`: the runner
plays the full horizon, so `epochs` is the horizon for every run and says
nothing about commitment. Studies predating either field read as `None` or 0 and
drop out of the median.

**The paired comparison** is the one to read. Every policy plays the same drawn
effects and the same noise, so the difference in regret on one rep cancels the
environment, which is nearly all of the variance. Comparing unpaired means
throws that away and reports a confidence interval dominated by how hard the
draws were rather than by how the policies differ. `analyze` prints what the
pairing buys as the reps needed for a 2-sigma read on a 2% effect, which is how
the NEXT sweep should be sized: sizing from the unpaired spread is how a 50k
sweep gets run to resolve what a few thousand paired reps would have settled.
