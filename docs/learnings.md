# PINN learnings — the reusable method

Distilled from solving the two_arm and three_arm HJB free-boundary problems
(2026-08). Nothing here is specific to bandits; this is the playbook we would
carry to any new PDE/control problem. Each lesson is stated as a rule, then
the evidence that bought it.

## 0. Before any net exists: shrink the problem

These moves came before this project's first line of torch and paid more than
everything after; they are listed first because they must happen first.

- **Dimensional analysis collapses a CLASS of PDEs to one function.** Every
  parameter that units can scale out, scale out (Buckingham): the
  dimensionless problem here is parameter-FREE, so one trained net serves
  every (rho, sigma, prior) instance forever. Parameters re-enter only
  through the readout dictionary — where on the solved manifold a given
  real experiment boots (tauhat_0 = rho sigma^2 / nu^2) — not through
  retraining. Corollary: benchmarks, anchors, and brackets computed once
  are universal. If the dimensionless problem still has a parameter, hunt
  again before training a family of nets.
- **Subtract what is known exactly; learn only the remainder.** The value
  splits as known-kinky-part plus smooth-unknown-part
  (v = commit + premium): the closed-form part carries the kinks, the net
  stays smooth, and ridge/wall conditions get imposed on the smooth factor
  where differentiation is legal. Never make a net learn something you can
  write down.
- **Continuous symmetries delete state dimensions.** Each one found is a
  whole coordinate gone. Worth a dedicated hunt at the start (agent-grade
  investigation); here dimensional analysis had already spent them — but
  the NEGATIVE result ("the problem is truly 5-D") is itself load-bearing:
  it licenses the 5-D sampler instead of leaving a nagging doubt.
- **Discrete symmetries pay four times.** (a) The domain quotients to a
  fundamental wedge (|G|-fold smaller: 6x here) and every sample lands
  where it counts (the fold is a sort-and-permute, cheap). (b) The quotient
  boundary GENERATES boundary conditions — the tie/mirror wall conditions
  are pure symmetry content, and such conditions are precisely what break
  the interior PDE's degeneracies (the never-explore solution dies by a
  symmetry-boundary condition, not by the PDE). (c) Invariants pick the
  right coordinates (det T, the exact S3 invariant, became the scale
  variable; the mirrors became transpositions). (d) Free diagnostics
  forever: quantities the symmetry says are equal (l_ab = l_ac at
  symmetric states) are standing probes of training health. One duty in
  return: the LOSS must inherit the invariance (an asymmetric residual
  scale silently breaks a symmetry the math guarantees) — and the fold's
  discarded permutation must be recoverable wherever readout returns to
  physical labels.

## 1. Architecture: make theorems invariants

Anything PROVEN about the solution goes into the function class, never into
the loss. A loss term negotiates; an architecture guarantees.

- **Bounds become envelopes.** `u = envelope * bounded_response` makes
  positivity and the upper bound structural. Corollary discovered late and
  painfully: the envelope's QUALITY matters beyond validity — an envelope
  that is EXACT in some limit makes that entire limit free (the response
  tends to 1 and the net learns nothing there). A merely-valid bound that is
  1.2-1.9x loose in a regime forces the net to learn the gap, and if the
  training signal there is weak, it simply won't (we watched a never-explore
  attractor eat two training runs until the exact bivariate envelope
  replaced the loose sum-of-univariates one).
- **Kinks are placed, not learned.** Smooth nets cannot fold derivative
  discontinuities; residual concentrates along them forever. Put the kink in
  the architecture with the RIGHT regularity: `relu(r)**2` touches down with
  value and slope zero and a curvature jump — the free-boundary (smooth
  pasting) class. A plain clamp kinks the first derivative and breaks the
  physics. This turned "irreducible smoothing error along the boundary" into
  "is the boundary curve in the right place", and made the commit region
  EXACTLY solved (zero residual by construction).
- **Saturations need polynomial tails.** float32 tanh(y) is exactly 1 above
  y ~ 9 with underflowed gradient: a one-way cliff. Training transients
  overshoot (we measured r ~ 14 from a wall-condition slam); under tanh that
  is death, under y/(1+y) it walks home. Rule: any squashing function on a
  trainable quantity must keep nonzero float gradients at every reachable
  magnitude.
- **Start alive.** If the architecture can represent a state with
  identically-zero loss gradient (our all-dead relu**2 net), some seed will
  find it and freeze forever (signature: loss constant at exactly the
  weight of the unsatisfiable term). Init so that every constraint has
  gradient everywhere (head bias 1 = response alive everywhere) and let the
  PDE carve dead regions from above.
- **Init theories have preconditions — audit them after every sampling
  change.** Xavier assumes unit-variance inputs; a sampling-law change
  quietly pushed raw feature stds to 30x that, railing half the first tanh
  layer (49% saturated). Wide-shallow nets survive by redundancy and hide
  the bug; deep profiles die of it, and "depth doesn't work here" was the
  misdiagnosis. Fix: calibrate a per-feature scale FROM THE SAMPLING LAW
  once at init, stored as a buffer so checkpoints carry it.
- **Feature alignment beats capacity.** Inputs in the coordinates where the
  solution's structures are axis-aligned (z-scores, log scales, tail
  coordinates) moved orders of magnitude; width and depth moved little.
  Deep nets kept losing to shallow ones BECAUSE the architecture kept
  absorbing the jobs depth would have learned; that is success, not failure.

## 2. Sampling: reachability from coordinates, coverage from laws

- **Sample in coordinates where the constraints are trivial.** Draw in the
  nonnegative box of accumulated information and map affinely: every draw
  reachable, no rejection, symmetry acts by permutation. Sampling raw
  matrix entries would spend budget on states no trajectory visits.
- **Composition murders tails.** Coordinates built as SUMS of independent
  draws make jointly-small states exponentially rare (three exponentials
  conspiring: the startup corner had 1e-5 mass and the policy there was
  garbage). Fix: one COMMON log-spread scale multiplying all components —
  jointly-low becomes a first-class event with per-decade mass. Check
  coverage by plotting marginals and joints in log space, always.
- **Floors are numerics, not priors.** A sampling floor that encodes a
  modeling assumption (a specific prior) silently specializes the net.
  Push floors down to numerical-stability territory and let the use case,
  not the sampler, decide where to read the solution. Concede decades only
  on use-case grounds (nobody starts 100 sd agnostic).
- **Measure before adding samplers — in both directions.** Every "stubborn
  region" we hit was EITHER unvisited (fix: sampling law) OR structurally
  unconstrained (no law helps). Discriminate by measurement: bucket the
  relevant residual by distance-to-region on the current checkpoint. We
  found one of each: the startup corner was starvation (1e-5 mass); the
  wall-seam cross-derivative error was NOT (wall residuals flat in seam
  distance — the conditions underdetermine those derivatives, and a
  sampling tweak would have been cargo cult).

## 3. Grading: residual units are a law, derive them

- **A relative-residual scale is a guessed scaling law.** `1 + |H|`-style
  per-point normalizers estimate reactively what a similarity analysis
  supplies exactly. Do the units analysis (self-similar coordinates): the
  transformed equation has O(1) coefficients, the whole scale dependence
  collects into known powers of the information scale, and the "variable
  transformation" enters the code as ONE multiplicative weight (identity:
  residual_transformed = S^p * residual_raw). The horrendous transformed
  operators stay in the doc; code only reweights.
- **Choose WHICH units knowingly — failure modes have their own S-powers.**
  Grading in the equation's own units treats shape error fairly but let a
  degenerate branch (residual one power of S smaller than the equation
  scale) become invisible at small S: a dead-Hamiltonian net scored WELL.
  Grading one power lower (value/premium units) keeps degeneracies loud at
  every scale. Rule: enumerate the failure modes' residual scalings before
  choosing the weight's exponent; the right units are the ones where the
  WORST failure is O(1) everywhere.
- **Keep the parameter-free seatbelt.** log-cosh costs nothing, has no
  knobs, and caps each point's gradient during the violent transients that
  demonstrably happen (wall slams). Parsimony bites knobs, not seatbelts.

## 4. Optimization: usually innocent

- **Fix conditioning in the equation, not the optimizer.** After the
  similarity grading, L-BFGS improved nothing (it had been worth 2.5x
  before): the preconditioning it used to provide was baked into the
  units. Plain Adam + lr decay sufficed everywhere after that.
- **Training prints lie two ways.** At hot lr the printed loss is the Adam
  orbit radius, not the valley floor (a checkpoint evaluated 5-20x better
  than its own training prints). And a mean loss flatters localized error
  (loss 3e-5 with 40% relative error in a thin corner). Judge checkpoints
  by probes, never by the training curve.
- **Loss values do not compare across grading or sampling changes.** Every
  such change re-denominates the objective. Keep invariant yardsticks:
  raw-residual-per-decade buckets, structural probes, and an end-to-end
  benchmark (see 6).

## 5. Diagnosis: probes, signatures, discriminating experiments

Standing battery per checkpoint: 2-D slices in similarity-scaled windows
(mean axes in units of posterior sd), line probes across the scale decades,
residual bucketed per decade, and structure probes (symmetry of quantities
that must be equal, argmax fields at known-symmetric points).

Signature dictionary (all observed, all diagnostic):
- Residual field IDENTICAL to premium field => the operator side is dead
  (all derivative terms ~ 0): a degenerate branch, not undertraining.
- Loss frozen at exactly the weight of one term => absorbing state.
- Two independent runs stall at the SAME frontier => wall, not patience;
  a frontier that MOVES BACKWARD under training => attractor (act now).
- Red/blue dipole straddling a curve => boundary misplaced, not region
  unlearned.
- "Alive" indicators from strict positivity tests => check for float
  subnormal dust before believing them (envelopes underflow).
- Exactly-zero anything => structural, never luck; verify which structure.

Discriminating experiments are cheap — design for one bit: same checkpoint
continued under (a) vs (b) (lr, tie weight, batch); float32 vs float64
evaluation of the SAME weights (separates arithmetic noise from function
error — it exonerated float32 when we were sure it was guilty); residuals
bucketed by the suspected cause.

## 6. Process

- **Keep a cheap sibling problem as the laboratory.** Every risky change
  (envelope form, response map, grading, law reshapes) was validated on the
  2-D problem in ten-minute retrains before the 5-D problem inherited it.
  The lab also carries the anchors: exact startup solutions, boundary
  brackets, closed-form policies.
- **The benchmark is the referee of "does it matter".** Residual metrics
  cannot answer whether an error costs anything; the Monte Carlo policy
  shoot-out can (a 40% residual corner turned out to cost < 4.5% of value).
  Report DIFFERENCES in value, not ratios: ratios divide out the scale that
  pays (the same policy edge was 10x more money at startup while the ratio
  looked 4x smaller).
- **Baselines must be strengthened before comparison.** We auto-optimize
  the rival's one parameter (commit time) per regime; beating a mistuned
  baseline is self-deception.
- **Exact constants over measured ones.** A limit measured as "0.854 by
  20M draws" became (2+sqrt(2))/4 exactly — then an acceptance test to
  1e-6. Chase closed forms for anchors; they upgrade every later check.
- **Agent grinds: verified, fenced, and interruptible.** Long derivations
  go to a subagent with numerical verification MANDATORY per identity, no
  repo writes (report first), and explicit effort caps — and look up known
  results before deriving them. Kept results: only what survived
  independent re-verification.
- **Record refutations with the measurement attached.** The graveyard
  (Fourier features, clamp-not-add, corner wall sampling, tanh saturation,
  equation-units grading) prevents relearning; each entry carries the
  experiment that killed it, so a future doubt can be re-run, not
  re-argued.

## 7. The fat tail: attention is a loss-space instrument (2026-08-06)

- **Histogram the residual population before touching anything.** A stalled
  loss has a shape: point mass at zero (architecturally-exact regions) +
  Gaussian bulk + a flat one-sided shelf at 5-6 sd (kurtosis ~10 vs the
  normal's 3). The shelf is a coherent subpopulation the mean objective has
  priced in as acceptable outliers — no amount of patience fixes that. A
  normal-ish histogram means keep grinding; a shelf means reweight or
  restructure.
- **The power mean is the knob-light attention.** Replacing mean(g) with
  (mean g^P)^(1/P) weights each point by (g/M_P)^(P-1) — size relative to
  the batch's own population, annealing to the plain mean as the tail thins,
  magnitude preserved (it is still a mean), P = 1 recovers mean-of-squares
  exactly. Costs: the gradient's effective sample size collapses (measured
  13% of batch at P = 2), so batch must grow with P or the loss oscillates;
  and amplifying one loss term silently deflates every other term's
  effective weight — check the auxiliary losses after, not during.
- **Sampling density and loss weights are one mechanism.** Identical in
  expectation (both enter as density-times-weight); extra samples buy only
  variance reduction. Do not build both; and recognize "clever sampling
  corridor" as attention in disguise before arguing about which is cleaner.
- **Attention doubles as a discriminator.** Crank tail pressure for one
  short run: if the region yields (ours did — mean AND tail improved
  together), it was under-served; if the error waterbeds, the limit is
  structural and no reweighting will move it. Cheapest one-bit experiment
  in the book.
- **L-BFGS refutes or convicts the objective, not just the optimizer.**
  Full-strength L-BFGS on the plain-mean loss moved the whole residual
  distribution DOWN UNIFORMLY (~13%, shape and kurtosis preserved, worst
  point worse): the shelf is part of the mean-objective's minimum, so
  attention changes WHICH function you converge to, not how fast. Corollary
  yardstick: the size of an L-BFGS jump measures formulation quality — 100x
  means the units left conditioning on the table; 10% means the hand-work
  is done and the residual gap is approximation or objective design.

## 8. Kink primitives: graft matched-regularity units (2026-08-06)

- **When tanh stalls on a curvature manifold, hand it primitives of exactly
  the solution's singularity class.** A parallel branch of relu(w.x+b)**2
  units = movable curvature jumps (value and slope zero at the crease). A
  handful (8 units, 137 parameters) beat every objective-side intervention
  on the free-boundary junction, and the units self-oriented onto the tie
  coordinates. Plain relu is one derivative too violent — its crease is a
  first-derivative jump the true solution does not have, and the residual
  (which reads second derivatives) punishes every crease inside the cloud
  until the unit dies. The parabolic-splined relu is the difference of two
  shifted relu**2s, so the relu**2 basis already spans it: inspect trained
  pairs (near-parallel directions, opposite-sign heads) before promoting
  the spline to a primitive.
- **Stitch, don't retrain: zero-init the branch head.** The grafted net is
  bit-identical to the checkpoint at step 0 (asserted, not assumed), and
  gradient wakes the branch. The zero head is a benign saddle — Adam's
  per-parameter normalization walks out of it in a few thousand steps when
  the residual signal is sign-consistent. Alive-start applies one level
  down: a relu**2 unit that is never active gets no gradient forever; open
  every unit's bias onto a healthy slice of the cloud (default init left 3
  of 8 dead).
- **Co-training lets the fast learner colonize.** From scratch, bare relu**2
  units outrun tanh early and take over the smooth bulk — the job they are
  structurally worst at (2 orders worse, branch at half the field). Two
  cures, one architectural: saturate each unit (y/(1+y), same crease
  regularity, bounded output) so bulk-painting is expensive by construction
  — co-training then works (within 4x, closing) but organizes the branch as
  communal direction-clusters, not specialists. The stitch (smooth basis
  first, primitives after) still produces the cleaner anatomy: the stitched
  branch's defining unit is the junction specialist the co-trained branch
  never grew. Sequencing is load-bearing.

## 9. Policy is second order; the referee is a regret arena (2026-08-06)

- **Value error is first order, policy error is second order — except at
  ties.** With ~2% relative residual, the greedy policy's suboptimality is
  basis points (the Hamiltonian is flat at interior maxima); the fitted
  value's CLAIM meanwhile inflates linearly (measured: claims 9% above the
  policy's simulated value, all of it overshoot in the flux region).
  Exceptions where residual converts to policy at full strength: (a)
  degenerate argmax — a small operator-sign error flipped the Hamiltonian
  convex in an untrained decade and the policy vertex-committed on zero
  evidence; (b) commit-boundary placement (one-way doors cost linearly);
  (c) any use of v as a NUMBER (pricing, certificates, basis functions for
  the next problem). Spend accuracy there, not on the bulk mean.
- **The subsolution program: the residual's SIGN is the certificate.** If
  v <= max H everywhere (plus obstacle/growth conditions), the greedy
  policy provably achieves at least v. The Hamiltonian reads only
  derivatives, so v - c has the same policy for any constant c and residual
  lowered by c: an instant a-posteriori bound V_policy >= v - sup(residual+)
  with no retraining. One-sided (asymmetric) grading trains toward the
  right sign; interval-bound verification of a 2-D net makes it a proof.
- **Benchmark policies in a discrete-epoch regret arena against the true
  effect, with the environment's parameters, not the trainer's.** Pieces
  that mattered: PAIR the seeds (same effect draw and noise stream across
  policies within a rep — most regret variance is draw luck, and it cancels
  in comparisons); allocations are simplex vectors with exact-vertex
  commits (interpolated policies never trigger absorbing borders — tables
  lie here, evaluate the net directly); priors are POLICY PARAMETERS, never
  environment knowledge; soft commit time N(1 - sum a^2)/(N - 1) is the
  uniform-equivalent information spend, comparable across arms counts. The
  payoff here: the HJB policy beat Thompson sampling — the previously
  unbeaten champion — by ~20% of discounted regret at realistic parameters
  while buying less than half the information, BECAUSE it prices
  when to stop buying; and it did so from the fitted value function this
  playbook spent a week complaining about.

## 10. Postscript: the phantom artifact (added after being burned)

For a week, a "corner artifact" was hunted: the policy at symmetric-mean
triple points parked on alpha = (0, 1/2, 1/2) where "symmetry says 1/3 each",
and the cross learning number l_bc sat "inflated" at ~2.5x its siblings.
Sampling was reshaped, envelopes upgraded, depth hypothesized — and every
architecture, era, and grading CONVERGED to the same "wrong" answer.

It was right. The probe states (tau_bc = 0) are invariant under the b<->c
relabel only — the a<->b relabel maps them to DIFFERENT states, so full-S3
conclusions (equal thirds) never applied there. At tau_bc = 0 the b-c pair
carries ZERO accumulated information, so the b-c contrast is the most
valuable thing to observe, and (0, 1/2, 1/2) is the physically correct play.
At the genuinely S3-fixed states (equicorrelated, tau_bc = -t/2) every
trained net produces the mandated near-thirds and always had.

Rules purchased:
- Before declaring a symmetry violation, verify the probe state is a FIXED
  POINT of the specific group element being invoked. Partial symmetry gives
  partial conclusions (here: alpha_b = alpha_c only).
- When independent function classes converge to the same "wrong" structure
  under a well-conditioned loss, the expectation is the prime suspect, not
  the nets. Re-derive it before re-engineering them.
- Transient agreement with a wrong expectation is doubly treacherous: the
  half-trained net LOOKED like it confirmed the hypothesis (three-way splits
  everywhere) and then "regressed" — the regression was convergence.
