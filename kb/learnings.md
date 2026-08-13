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

- **NEVER scale the PDE residual. Grade it in the equation's own units, in
  any shape or form, ever.** (2026-08-10, reversing this section's earlier
  advice — which is preserved below as the mistake it was, because it is
  seductive and was believed here for months.) The equation as written,
  `rho V = max{...}`, is the only form whose residual has a physical
  meaning: it is an error in VALUE, and it is the quantity the comparison
  principle bounds, `||V - V*|| <= ||residual|| / rho`. Multiply it by
  anything state-dependent and you are weighting the domain — deciding
  where the net is allowed to be wrong — while believing you are choosing
  units.
- **A similarity chart is a derivation tool, not a grading tool.** It earns
  its keep by conditioning the autograd chain (cancelling the coefficient
  spread algebraically so no derivative term is multiplied by a large
  number). Keep it for that. But the identity it produces,
  `residual_transformed = S^p * residual_raw`, means grading the transformed
  residual SQUARED applies a silent `S^(2p)` weight. Keep that identity as a
  CHECK (a self-check asserting it), and divide it back out before grading.
- **The measurement, so nobody re-derives this the hard way.** two_arm_drift,
  2026-08-10: the chart weight was `tauhat^3`, ~15 decades across the
  sampled range. It sent 70% of the p-mean's gradient to the near-static
  corner — where the natural error is ~1e-6 relative and 78% of the
  collocation points are dead — and 5% to every drifting regime combined.
  The absolute residual looked beautifully FLAT across decades, which is
  what a chart weight cancelling the true error's growth looks like. A loss
  that appears balanced under a weight is balanced in the weight's
  coordinates, not in the problem.
- **THE MISTAKE (kept deliberately): "choose WHICH units knowingly".** The
  old reasoning ran: grading in the equation's own units lets a degenerate
  branch whose residual is one power of `S` smaller become invisible at
  small `S` (a dead-Hamiltonian net scored WELL, 2026-08-05), so grade one
  power lower, in premium units, to keep the degeneracy loud. It is a real
  failure mode and the fix is wrong. A degeneracy is broken by a term that
  provably breaks it — the ridge condition, the tie losses — not by a thumb
  on the residual that also reweights every non-degenerate point in the
  domain. If the dead branch scores well, strengthen the breaker.
- **Removing a residual weight detonates every other weight.** The pde term
  changes magnitude by orders (measured: x8e7 two_arm, x1.6e8 two_arm_drift,
  x1.2e3 three_arm, x2.7e4 three_arm_drift), so every constant calibrated
  against it — boundary conditions, degeneracy breakers, sign penalties — is
  instantly negligible. Rescale them by the measured pde factor to hold the
  balance the record was tuned at, and expect to recheck once the retrained
  net's error distribution settles. Auxiliary weights carry the factor; the
  residual never does.
- **A PARAMETER is not a STATE, and pooling lets family members compete.**
  When a net solves a family of problems (a parameter as a net input, here
  `etahat`), one pooled residual loss lets the member with the largest raw
  numbers take the gradient. Natural units fixes the SCALE half of that.
  It does not fix the POPULATION half: a family member that is 3% of the
  batch with a light residual tail stays starved no matter the weighting,
  and that needs stratified sampling of the parameter. Diagnose the two
  separately — measure attention share against population share per slice.
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
  equation-units grading, the complementarity term) prevents relearning;
  each entry carries the experiment that killed it, so a future doubt can be
  re-run, not re-argued.
- **A term that reweights what the metric discounts cannot improve the
  metric.** If a defect survives training because the residual weight is
  small there, then the region's share of the loss is small too, and that
  share is the ceiling on any repair (measured: 9.8%, three_arm's
  complementarity band, docs/three_arm.md section 15). Worse, such a term
  competes for capacity with the 90% and can erode the guard terms that
  break a degeneracy. Before adding one, compute the region's share of the
  objective — it is the entire prize — and decide whether the real
  scoreboard is the loss at all.

## 7. The fat tail: attention is a loss-space instrument (2026-08-06)

- **AMENDED 2026-08-10: attention and units are the same knob, so tune one
  at a time.** The p-mean below was calibrated against a chart-WEIGHTED
  residual, which suppressed the tail; the exponent was compensation. Grade
  in natural units (section 3) and the compensation becomes over-correction:
  measured on the same batch of 4096, `POWER = 2` left an effective sample
  size of 2.1 points on three_arm and 4.9 on three_arm_drift — a gradient
  decided by a handful of points, which also broke the S3-invariance
  self-check by making it precision-bound. `POWER = 1` (plain
  mean-of-squares) restores 54 and 36. Whenever the residual's units change,
  RE-MEASURE the effective sample size `1 / sum(w_i^2)` before trusting any
  attention exponent; and prefer relative tolerances in invariance checks,
  since an absolute one is calibrated to a loss magnitude that just moved.
- **A tail-dominated loss is the CORRECT state once units are natural.** The
  error really is orders larger at low information; that is the signal, not
  a pathology to be normalized away. The cost is that batch size now buys
  gradient quality directly — matching the old absolute effective sample
  size wanted ~20k points, 5x the step cost.
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

## 11. Warm starts: a converged parent is a local minimum (2026-08-12)

Inheriting weights from a solved sibling problem buys the first few thousand
iterations and then costs everything after. Measured on two_arm_drift, which
is two_arm plus one coordinate: the drift net can be initialized from a
two_arm checkpoint and reproduces it BITWISE on the `etahat = 0` slice, because
the drift feature `log1p(2 etahat tauhat)` vanishes identically there. The
bootstrap is exact, cheap, and structurally protected -- and still the wrong
move.

Same architecture (823 params, 24:24), same learning rate, matched iterations:

    pde            @10k        @20k        @29k
    scratch        7.323e-01   1.230e-02   6.150e-03
    bootstrapped   5.418e-02   1.863e-02   7.720e-03

    pos_learning   @10k        @20k        @29k
    scratch        1.75e-01    4.69e-03    4.18e-04   (exactly 0 by 97k)
    bootstrapped   5.54e-02    4.50e-02    1.33e-02

The bootstrap leads 13x at 10k, is overtaken before 20k, and by 29k is behind
on both. The pde gap is modest; the CONSTRAINT gap is the real damage --
the scratch net sheds three orders of learning-operator violation while the
bootstrapped one sheds a factor of four and parks.

Mechanism: the parent arrives with every first-layer unit committed to a
converged 1-D structure (in two_arm, a ladder of `tauhat` crossovers, section
in kb/two_arm.md). The child's problem needs that structure to become 2-D. The
net must dismantle what it inherited, and while dismantling, the cheapest local
move is to let the constrained quantity go negative.

The damage is SLOW, not permanent -- corrected 2026-08-12 after the run
continued. The bootstrapped net did reach `pos_learning` exactly 0, at ~145k
iterations, having been at 1.3e-2 at 29k and 2.1e-5 at 106k. So the honest
cost is roughly 100k iterations spent undoing the inheritance, not a floor the
net never leaves. Judge a warm start on the ITERATIONS IT WASTES, and note
that at 29k every measurement said the penalty looked permanent.

Rules:
- Warm-start ACROSS problems only when the parent's structure is a SUBSET of
  what the child needs, not when it must be reorganized. "Exact on a slice" is
  not the same as "useful off it", and the exactness is what makes the trap
  convincing.
- Judge a warm start at MATCHED ITERATIONS against scratch, and judge it on
  the constraint terms, not the residual. At 10k every number said the
  bootstrap was winning.
- Warm starts want a HOT rate, not a gentle one, exactly opposite to the
  instinct that a converged parent must be protected from smearing. At 3e-4
  the bootstrapped net was 4.7x worse on pde and 2.6x worse on pos_learning
  than the same net at 1e-3: a gentle rate does not preserve the inherited
  solution, it strands the net in the parent's basin.
- This indicts WEIGHT inheritance, not feature reuse. Handing the child the
  parent's OUTPUT as an input feature is a different mechanism -- the child
  keeps its own weights free and only queries a function. Do not let this
  result talk you out of the pairwise-basis route (kb/three_arm.md).


## 12. Artifacts declare their shape; loaders never infer it (2026-08-13)

A checkpoint is a state dict, and it is tempting to recover the architecture
from it by measuring: hidden widths from weight shapes, kink count from
whether `kink_in.weight` exists, feature count from the first layer's input
width. It works, so it survives — and every one of those is a load-bearing
fact reconstructed from a coincidence of geometry.

What it cost here, in one session:

- A checkpoint could not be loaded without a second file on disk being the
  right shape. A net that embedded a frozen basis sized it by reading the
  champion symlink, so an unrelated experiment repointing that link made
  every such checkpoint unloadable — even though the basis weights were IN
  the file.
- Two grafts silently corrupted what they wrote. Stitching a smooth
  checkpoint into a kinked net copied the source's `kink_count = 0` over the
  target's 8, saving a declaration the file's own weights disprove; the same
  bug, independently, copied a source's hidden widths into a wider target.
- The `kink_` PREFIX matched more than the kink branch. Two sibling modules
  filtered branch tensors with `name.startswith("kink_")`, which also caught
  the `kink_count` metadata and deleted it. Both were written by someone who
  knew exactly what the prefix meant on the day they wrote it.

The fix is to say it instead of measuring it. A base class registers
`(features, hidden, kinks)` as BUFFERS, so `state_dict()` carries them at no
cost to the save path and the loader reads a declaration. Two rules make it
hold:

- The buffers describe the MODULE, not the file. A loaded state dict cannot
  contradict them: the declaration decides what to BUILD, and after that the
  module is the authority. Without this, every graft — which legitimately
  changes shape — writes a lie.
- Migrate the artifacts, then DELETE the inference. Keeping the old path "for
  compatibility" means the fragile code is still what runs on anything old,
  which is exactly the case nobody tests. Rewriting every checkpoint took one
  idempotent script and one pass; the backfill belongs in that script, which
  may guess because it is told which problem each file is, never in the
  library, which must not.

Generalizes past checkpoints: whenever a consumer reconstructs a producer's
intent from the shape of what arrived, the reconstruction is a second source
of truth that drifts. Prefix-matching a serialized namespace is the same
mistake wearing a string.

## 13. Size a sweep from the PAIRED spread (2026-08-13)

The arena gives every policy the same drawn effects and the same noise, so a
per-rep DIFFERENCE cancels the environment — which is nearly all of the
variance. Measured on a 800-rep two-arm sweep: the leader's margin over
Thompson has a 95% CI of 0.3 paired against 0.8 unpaired, and the reps needed
to resolve a 2% effect at 2 sigma fall from tens of thousands to ~1,900.

Two failures follow from ignoring it:

- Sizing from the unpaired spread. A 50k-rep sweep was launched to compare two
  checkpoints whose difference a few thousand paired reps would have settled.
  The rep count was not the waste; the ESTIMATOR was, and the rep count was
  merely how the waste got paid for.
- Running the comparison as two SEPARATE processes. Both entrants belong in
  one run, where the pairing is exact by construction. Splitting them halves
  the statistics, doubles the wall-clock, and makes the pairing something to
  verify afterwards rather than something guaranteed. It also forbids changing
  `--size` between the two, because batch width reorders float32 reductions and
  a policy carrying a net is sensitive to that over a long horizon.

Also: `--workers` is not a throughput dial. On a 128-core box already running
trainers, 48 threads gave 68 runs/s at load average 177 — oversubscribed and
thrashing. Time a `--size 500` probe at two or three thread counts on the box
you will actually use, then launch. It costs ~15 seconds against sweeps
measured in hours.
