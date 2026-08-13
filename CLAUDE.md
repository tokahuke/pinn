# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An exploratory PINN (physics-informed neural network) solving the HJB free-boundary
problem specified in `kb/two_arm.md`. That document is the source of truth for all
math: the PDE, boundary conditions, notation, and the graveyard of already-rejected
approaches (do not redo them). Read it before touching any equation code.
`kb/learnings.md` is the problem-agnostic playbook distilled from this project
(architecture/sampling/grading/diagnosis method) — read it before designing for a
NEW problem.

Two documentation trees: `kb/` is the agent-facing knowledge base (derivations,
graveyards, the source of truth for math — dense, no audience polish); `docs/`
is human-facing — showcase pages and the images the README embeds. New
derivation records go to `kb/`, never `docs/`.

Notation contract (Section 0 of the doc): uppercase = dimensional (`V` value,
`U` exploration premium), lowercase = dimensionless (`u`, `v`), hats = dimensionless
coordinates (`muhat`, `tauhat`).

LEARNING OPERATORS are `L_xy`, one per PAIR of arms -- `L_ab` in two_arm and
two_arm_drift, `L_ab`/`L_ac`/`L_bc` in the three-arm pair. N arms give
N(N-1)/2 of them. Each is (mean-diffusion along that pair's direction) +
(precision gained in that pair's coordinate), it multiplies `alpha(1-alpha)`
in the Hamiltonian, and it is what the POLICY depends on. `L_xy >= 0` is
provable (buying information is a mean-preserving spread on a belief the value
is convex in) and is ASSERTED in two_arm_drift's loss (`pos_learning`,
`relu(-L_ab).mean()`, linear not squared). The three-arm modules grade the
FULL concavity statement instead (see the concavity frontier below): the
sampled-direction term `relu(-L[f]).mean()` in both losses, one fresh
contrast direction per point per step (`directional_learning` in
three_arm/loss.py, reused by the drift sibling), which subsumes all three
pairwise signs. Earlier drafts used `U` for the even part
`V - mu/(2 rho)`; transcribe old statements via `u_even = u + |muhat|/2`.

## Commands

- The `pinn` CLI (mounted via pyproject `[project.scripts]`, one module per
  command in `pinn/cli/`):
  - `poetry run pinn init --problem P --topology 64:64:64k16 --out ckpt.pt`
    creates an untrained net; `--from other.pt` adapts one instead (at least
    one of the two; given both, `--topology` is the target shape and `--from`
    the source adapted into it — the kink-graft path, supported by three_arm
    and both drift problems). Topology is widths colon-separated with an optional
    `k<count>` of kink units, parsed by `pinn/net.py:parse_topology`.
  - `poetry run pinn train --problem P --in ckpt.pt` trains, saving back to
    --in unless --out says otherwise. `train` cannot create a net, only
    continue one.
  - `poetry run pinn plot --in` (two_arm fields), `poetry run pinn validate`
    (two_arm section-6 identities).
- Diagnostics still at the repo root: `poetry run python probes.py --in
  data/<ckpt>.pt` (three_arm wedge slices, tau lines, symmetric-point table).
- The arena (policy shoot-out vs TS/ETC/elimination on discrete epochs,
  regret vs true effects): `poetry run arena simulate <out.pkl> --problem
  two_arm|two_arm_drift|three_arm --rho ... --size N --workers K --eta E`,
  then `poetry run arena analyze <out.pkl>`. The runner is always
  drift-aware; `--eta 0` (the default) is a static world and no branch
  anywhere depends on it. Realistic parameter values live outside the repo
  (project memory), never committed.
- Each module in `pinn/` carries an `assert`-based self-check:
  `poetry run python -m pinn.problems.two_arm.loss`, `poetry run python -m
  pinn.arena.three_arm`, `poetry run python -m pinn.net`.
- Final step of any Python change: `poetry run black <files>`.
- Torch runs on CPU deliberately (tiny net; MPS overhead loses). Revisit only when
  batch/width grows.

## Architecture

Organized one-module-per-problem; the separation should be kept:

- `pinn/problems/two_arm/` — the A/B-test problem, structured in deliberate
  parallel to three_arm (`sample.py`, `simplex.py`, `model.py`, `loss.py`).
  `ExplorationPremium`: GainedTanh MLP with Xavier-tanh init, four aligned
  features `(muhat, log tauhat, muhat sqrt(tauhat), muhat tauhat)`, times the
  proven free-information envelope `nu(-muhat, tauhat**(-1/2))` (smooth form,
  NOT `-|muhat|`: the abs would kink the premium at the ridge where BC1
  differentiates) with
  learnable `log_scale`, linear-head response mapped through
  `relu(r)**2 / (1 + relu(r)**2)` (exact zero in the commit region,
  curvature-jump regularity on the learned free boundary; a plain clip kinks
  the first derivative and breaks smooth pasting) so `0 <= u < envelope` is
  architectural (the old `tauhat**(-1/2)` envelope is its ridge slice; the
  pre-envelope linear-head class lives at the initial commit and is what loads
  the champion checkpoint). `ValueFunction` wraps it as `v = relu(muhat) + u`.
  Samplers: scrambled Sobol pushed through a long-tailed law spread over
  decades by a common log-scale (floor `1e-3` ≈ a 30-sd-wide prior — numerics
  only, no prior baked in; below that is 100x/decade stiffer territory nobody
  visits; same principle as three_arm's sampling), strictly `muhat > 0`. The
  loss grades the HJB in similarity coordinates (kb/two_arm.md section 8:
  leaves are `(z, s)`, autograd performs the chain rule, and a self-check
  asserts the `tauhat**1.5` identity against the raw form), maximization kept
  explicit and handed to `simplex.maximize_quadratic` (evaluate-and-max over
  vertex + endpoints, gather selection), graded IN NATURAL UNITS — the chart's
  `tauhat**1.5` is divided back out, so what is graded is the residual of
  `v = max{...}` itself, the only error with a physical meaning and the one
  the comparison principle bounds. NEVER put a scale on the residual, in any
  shape or form, ever (learnings section 3 holds the rule and the measurement
  behind it: the chart weight was an undeclared `tauhat**3`, ~15 decades,
  spending 70% of the gradient on a corner already exact to ~1e-6 relative).
  `POWER = 1.0` since 2026-08-10 — plain mean-of-squares; the p-mean at 2 was
  compensating for the suppressed tail and became over-correction once the
  units were fixed (learnings section 7). Plus the
  BC1 ridge term and the learning-operator positivity term (`pos_learning`,
  `relu(-L_ab).mean()` on the natural-units `L_ab`, since `L_ab` carries the
  same `tauhat**1.5`; linear — same term and reasons as two_arm_drift).
  Do not substitute the interior FOC back
  in (that road leads to `sqrt` NaNs). `objective(batch)` packages sampling +
  loss for the trainer. New problems get their own sibling package.
  Class layout (both problems, deliberately identical):
  `DimensionlessValueFunction` is the trainer-facing class (dimensionless,
  restricted domain) carrying `load(path)` (architecture inferred from the
  state dict), `hamiltonian(state) -> (lhs, Maximum)` (the ONE derivative
  chain, shared by `pde_loss` and policy readout), and `policy`;
  `ValueFunction(dimensionless, rho, sigma)` is the deployment adapter
  (doc section 11 readout dictionary, wedge fold / arm-swap, `policy`
  un-permuted to physical labels).
- `pinn/problems/two_arm_drift/` — two_arm with a mean that itself diffuses,
  `d theta = eta dW` (kb/two_arm_drift.md). +0 state, +1 parameter: the
  state is still `(muhat, tauhat)` and `etahat = eta/(rho sigma)` is a third
  NET INPUT, so one checkpoint serves every drift regime. Structured exactly
  like two_arm (`sample.py`, `envelope.py`, `model.py`, `loss.py`) and reusing
  `two_arm/simplex.py` unchanged — the drift term is control-free, so it never
  touches the maximization, it only replaces `rho U` inside the source with
  `rho U + eta^2 tau^2 U_tau`. Two structural differences from two_arm: there
  is NO contact set (committing is not absorbing, precision decays at
  `eta^2 tau^2`), so BC2/BC3 are vacuous and `relu(r)**2` models a region that
  is merely exponentially small; and the envelope is the discounted
  perfect-information premium in closed form (`envelope.py`), which collapses
  BITWISE onto two_arm's `nu` at `etahat = 0` — which is what makes
  `pinn init --problem two_arm_drift --from data/two_arm.pt` an exact
  bootstrap.
- `pinn/problems/three_arm/` — the ABC-test problem, one module per concern
  (`sample.py` with the wedge fold, `simplex.py` plain quadratic-max calculus,
  `model.py`, `loss.py`), fully derived (kb/three_arm.md, "nothing
  mathematical" left) and training-ready. Key structure: solve only on the
  fundamental wedge {m_c <= m_b <= 0} (6-fold S3 quotient, `Sample.fold()`),
  pairwise-precision coordinates, seven-candidate simplex max, two
  mirror-paired wall losses, and the proven free-information envelope
  `nu2(m_b, m_c, sd_b, sd_c, k)` — the EXACT free-information value
  `E[max(0, theta_b, theta_c)]`, correlation included, which is also the
  exact startup solution (response -> 1 at zero information; the earlier
  nu-sum envelope was 1.17-1.87x loose there and bred a never-explore
  attractor at low tau) — times a `relu(r)**2 / (1 + relu(r)**2)` response
  (same free-boundary trick as two_arm: exact zero in the deep-wedge commit
  region, curvature-jump regularity on the learned boundary) — positivity
  and the upper bound are architectural invariants, not loss terms. The
  premium optionally carries a KINK BRANCH (`kinks` arg): parallel saturated
  `relu(.)**2` units (`y/(1+y)`, bounded so from-scratch co-training cannot
  colonize the bulk) added to the response — movable curvature-jump
  primitives for the free-boundary junction, zero-init head so stitching
  onto a trained checkpoint is bit-exact at step 0, alive-start bias +0.5.
  `Sample.fold_ordered()` returns the applied relabel for policy readout
  (the old discarded-permutation TODO is resolved). Grading: natural units
  like the two-arm pair — the similarity PREMIUM weight
  `det**0.75 / (tau_bb + tau_cc + tau_bc)` was DELETED 2026-08-10 as the same
  bug in another dress; the never-explore mode it was defending against is
  the tie losses' job, and they are the terms that provably break it. Self-
  checks: `poetry run python -m pinn.problems.three_arm.<sample|simplex|model|loss>`;
  the loss one includes the S3-invariance identity, now on a RELATIVE
  tolerance (an absolute one is pinned to a loss magnitude that moves).
- `pinn/net.py` — generic building blocks only (`GainedTanh`: trainable per-unit
  activation gains).
- `pinn/utils.py` — shared math (`nu`: Gaussian expected positive part;
  `nu2`: bivariate version `E[max(0, X, Y)]` via a 24-node Gauss-Legendre
  bivariate normal CDF, smooth and double-backward-safe; both clamped at 0
  against float32 tail cancellation).
- `pinn/train.py` — generic trainer: endless-generator `train(model, objective,
  lr)` (consumer stops via `itertools.islice`; mutates the model in place; `lr`
  is a float or a rate generator, see `decay`; a finite generator ends training).
- `pinn/arena/` — the policy arena (a discrete-epoch policy-benchmark
  harness): `harness.py` the generic
  N-arm core (simplex-vector allocations, discounted regret vs the oracle,
  paired seeds per rep, soft commit time `N(1 - sum a^2)/(N - 1)`),
  `two_arm.py`/`three_arm.py` per-problem zoos (ETC, Thompson via exact
  normal/bivariate CDFs, ZTest/elimination, and the `Pinn` entrant whose
  PRIOR IS A POLICY PARAMETER — `prior_std` in arena units, `None` = the
  flattest the checkpoint supports), `main.py` the CLI (mounted as `arena`
  via pyproject `[project.scripts]`; reflection discovers the chosen
  problem's zoo).
- `probes.py` — a click CLI at the repo root; everything else is a `pinn`
  subcommand in `pinn/cli/`. benchmark3.py lived here too and was deleted
  2026-08-12: a posterior-space Monte Carlo shoot-out written in the same
  commit that introduced the arena, never developed after it, and half
  duplicating the arena's three_arm zoo. Its one unique reading -- the net's
  value CLAIM against its own policy simulated, which is how the ~9% junction
  inflation was found -- has no replacement, but it is ~50 lines against the
  vectorized arena if it is wanted again.
- Module self-checks run as `poetry run python -m pinn.problems.two_arm.loss`
  (relative imports; both problem packages check per-module).

`data/` is gitignored and holds checkpoints (`*.pt`) and diagnostic plots. Plots
belong there (visible in the IDE), not in temp dirs.

CHECKPOINT NAMING (re-cut 2026-08-12): every real checkpoint file is
`<problem>.<topology>.pt` with the topology spelled in `x` —
`two_arm.32x256.pt`, `two_arm.16x16.pt`, `three_arm.64x64x64k8.pt`. The
champion is a SYMLINK at the bare `<problem>.pt` pointing at whichever of them
currently wins, so `ls -l data/` reads as a leaderboard and promotion is one
`ln -sf` with no bytes copied and no code edit — `pinn/arena/*.py`,
`three_arm_v2`'s `BASIS`, the CLI defaults and the README all name the bare
link. Topology is the archive key because it is the axis nets are compared on
and it is fixed for the file's whole life; a date is the TIEBREAKER only, for
two nets of the same shape (`two_arm.16x16.2026-08-04.pt`), and it earns its
place mainly across the 2026-08-10 natural-units break, where the two are not
comparable at all. Still banned: hyperparameter tags (they expire — `_pos10`
outlived its weight in a day), loss values (metric-dependent, and one already
caused a false comparison), colons (scp and rsync read `a:b` as host:path).

CHECKPOINTS DECLARE THEIR ARCHITECTURE (2026-08-13). Every premium subclasses
`net.DeclaresTopology` and passes `(FEATURE_COUNT, hidden, kinks)` up; those
become buffers, so `state_dict()` carries the shape and a loader READS it
(`read_topology`, `read_features`) instead of reverse-engineering it from
weight geometry. `hidden_widths` and the per-problem `_kinks` scans are
DELETED and all 19 checkpoints were rewritten that day — a `KeyError` on
`premium.topology` means a file older than that, which is a file to migrate,
not a case to handle. The buffers always describe the module holding them and
a loaded state dict cannot contradict them; that invariant is what stops a
graft (smooth checkpoint into a kinked net) from saving a declaration its own
weights disprove, and it is why `three_arm_v2` can load without consulting
`BASIS` at all. Do not reintroduce shape inference: prefix-matching a state
dict for `kink_in.weight` or `net.0.weight` was the bug class twice in one
session.

Two consequences of the link. Training through it writes the TARGET and leaves
the link intact, so `pinn train --in data/two_arm.pt` still ages the champion in
place — the tag stays honest because training cannot change a topology, but the
previous weights are gone, so branch by `cp`-ing to a new tag first when they
matter. And rsync sends links as links: `jobq cp` passes `-L` for this, anything
else moving `data/` needs it too.

## State of play (2026-08-10) and open frontiers

EVERY PRE-2026-08-10 LOSS FIGURE IS ON A DEAD SCALE. The residual grading was
chart-weighted until the natural-units fix of that date, so the old "pde
7.7e-6", "full objective 1.15e-5", "loss ~8e-8" numbers are scores on a
functional that no longer exists — off by `tauhat**3` and a p-mean exponent,
and NOT comparable to anything the code prints now. They are deleted here
rather than converted; the ordering they encoded was dominated by the
near-static corner where every net was already exact to ~5 significant
figures, so little is lost. What survives translation is the qualitative
record (which move helped, and how it was diagnosed).

Champions, one net per problem, all in gitignored `data/`. The GitHub release
`checkpoints-2026-08-09` predates both the renames and the fix, so its asset
names and figures follow the old conventions.

Natural-units pde at `POWER = 1`, batch 4096, measured 2026-08-10 as the
STARTING points of the retrains now in flight — these are baselines to beat,
not results:

    two_arm 5.704   two_arm_drift 98.9   three_arm 1.42e-3   three_arm_drift 3.02

- two_arm: `two_arm.pt` — retrained 2026-08-09 through an escalating
  pos_learning weight. Floor-decade `L_ab < 0` went 16.0% -> 6.8% and the
  commit-on-no-evidence shelf is gone at every muhat.
- two_arm_drift: `two_arm_drift.pt` — the 10x-positivity net, bootstrapped
  `--from data/two_arm.pt` (the exact `etahat = 0` slice) with the raised
  weight on from step 0. It beat its predecessor on every training-side
  metric: the targeted
  middle decade `etahat` in [1, 10) went 10.5% -> 6.5% violating, and the
  top decades cleaned up too. Bootstrapping rather than resuming was
  deliberate: every 2ad net that acquired the term partway through kept a
  commit-on-no-evidence needle near the ridge. No kink branch.
- three_arm: `three_arm.pt` — 3 hidden layers with 8 stitched saturated kink
  units, trained through the concavity term overnight 2026-08-10. The kink
  stitch was the decisive move (tail -19%, worst point -25% in one 30k run,
  junction specialist unit self-oriented onto `z_bc`). A from-scratch
  16-kink co-training run gave communal branch anatomy and no junction
  specialist -- sequencing is load-bearing, see learnings section 8.
- three_arm_drift: `three_arm_drift.pt` — concavity-trained 2026-08-10. Still the
  furthest from converged of the four (its starting pde is ~3 orders above
  three_arm's) and it was descending when stopped. Grafted from the
  pre-2026-08-07 three_arm champion, so its bit-exact-at-`etahat = 0`
  ancestry can no longer be re-derived from a filename.

The headline (2026-08-06): in the arena, at realistic parameters (values in
project memory, not committed), the two_arm PINN policy beat
Thompson sampling — the strongest practical baseline — by ~20% of
discounted regret while buying less than half the information, with baseline
values stable across independent sweeps. CAVEAT: that sweep and the tables in
kb/arena_results.md were run on checkpoints two or three generations back
(the names they cite no longer exist), and predate the natural-units fix
entirely. Re-run both once the retrains land.

Open frontiers:
- The low-tauhat floor decade: below `tauhat ~ 1e-2` `L_ab` goes NEGATIVE at
  the ridge, flipping the Hamiltonian convex and vertex-committing on zero
  evidence — a policy pathology, not just a residual. SOLVED for
  two_arm_drift by the `pos_learning` term (floor decade 54.2% -> ~10%
  violating, commit-on-no-evidence gone) and for two_arm (retrofit through
  an escalating weight, 2026-08-09: shelf gone at every muhat, floor decade
  16.0% -> 6.8%); still open for both three-arm modules, where nothing
  asserts it and the arena's `_FLATTEST_TAUHAT` guards it meanwhile.
  That guard is LOAD-BEARING, measured 2026-08-10: lowering it 1e-2 -> 1e-3
  (the sampler's own `PRIOR_FLOOR`, so still inside the training support)
  raised harsh-drift regret by ~50% and CUT the evidence bought to a quarter.
  Mechanism — at a vertex the mean is frozen (no design, no update) and the
  clamped tau input freezes too, so the policy's inputs stop moving and
  commitment becomes absorbing; the guard accidentally prevents that by
  parking the net just off the exact vertex. Do not "fix" the guard without
  fixing the low-tau policy first.
- Concavity for N >= 3, derived 2026-08-08, implemented 2026-08-09 as the
  sampled-direction loss, TRAINED THROUGH overnight 2026-08-10 (both
  three-arm champions carry it; the concavity term is now small on
  `three_arm.pt`, so the original "start ~100x above pde" calibration cannot
  be reproduced against it — `CONCAVITY_WEIGHT` is provisional). The
  Hamiltonian is a quadratic on the simplex and must be concave. Writing the
  tangent direction as `f`, `d' M d = -2 Phi(f f')`, so concavity is
  `L[f] >= 0` for EVERY contrast direction, not only the N(N-1)/2 pair ones.
  It is provable by the same mean-preserving-spread argument (a signal about
  `f'theta` is still a spread on a belief `V` is convex in), so it holds at
  the true `V*`. At N = 2 contrast space is 1-D and it collapses to
  `L_ab >= 0` — the term already shipped. At N = 3 the shipped term is
  `relu(-L[f]).mean()` over ONE random direction per point per step
  (`directional_learning`, three_arm/loss.py, reused by the drift sibling):
  linear in the learning numbers, kink-free, exactly silent on the dead
  region, and exact in expectation. The `lambda_min` closed form of
  `[[L_ab, h], [h, L_ac]]`, `h = (L_ab + L_ac - L_bc)/2`, is the
  EVAL-ONLY diagnostic: as a loss it needs a `clamp_min` inside the sqrt
  (unfloored it nans wherever the eigenvalues coincide, which includes the
  entire contact set), and the naive clamp makes relu's full-strength
  gradient push the dead region alive — three pieces of epsilon-carpentry
  the sampled form does not need. Measured on the three_arm champion
  (8k-batch, 2026-08-09): 92.6% of points satisfy pairwise positivity but
  only 78.1% are concave, so the all-directions test finds 3x more.
- Boundary placement under drift, the frontier the arena opened 2026-08-10
  and the one no current loss term can see. In a harsh-drift world (winner
  flipping ~every horizon) the drift net loses badly to plain Thompson: it
  owns the best MEDIAN of any entrant and is destroyed in the TAIL, because a
  commitment buys no information and only erosion can undo it — replayed tail
  runs sit ~78% of the horizon on the arm that is losing at that moment, and
  switch on the erosion clock (~an order of magnitude slower than evidence
  would). The value function is not innocent there (its natural-units error
  is ~7x worse in that corner than at deployment), but the residual, the
  positivity term and the concavity term are all blind to WHERE the free
  boundary sits. This is the case for the decision-side program: the
  claim-vs-simulation check for 2ad first (a batched rollout evaluator
  vectorizes over states and is ~90% of what policy iteration needs), then
  policy iteration. Build it on the vectorized arena, not on the deleted
  benchmark3. Do it AFTER the natural-units
  retrains: grading decisions before the equation is solved in that regime
  is fixing the second problem first.
- The b/c-junction blob (three_arm) at ~±0.07 relative, best ever, still
  the dominant error; systematically POSITIVE (v > max H), which also
  inflates the net's value claims ~9% above its policy's simulated value.
  The subsolution program (one-sided grading + the constant-shift
  a-posteriori bound; see learnings section 9) would turn this into a
  certificate.
- The curvature law constant and the tauhat-tail anchoring flags of
  kb/two_arm.md remain open.

Planned next (intent recorded 2026-08-06):
- `three_arm_v2`, the pairwise ansatz: `u = f(u2_ab, u2_ac, u2_bc) +
  correction`, a frozen two_arm net as basis, evaluated once per PAIR. It is
  the N-arm scaling route: the basis is ONE net whatever N is, and only the
  call count grows (3 pairs at N=3, 6 at N=4).

  THE `_v2` SUFFIX IS A PROMISE, NOT A NAME. It either supersedes `three_arm`
  and takes that name, or it is deleted. There is never a `three_arm` and a
  `three_arm_v2` in the tree at rest -- one module per problem, and the
  suffix exists only for the length of the bake-off.

  Evidence for it, measured 2026-08-12: three two_arm premia explain 98.0% of
  three_arm's premium by LINEAR regression (coefficients 0.921 leader-vs-
  runner-up, 0.465 leader-vs-last, 0.093 the losers' own contest), and the
  leading pair's dead set matches three_arm's commit region at Jaccard 0.93.
  Both numbers are IDENTICAL for a 406-parameter basis and the 9,154-parameter
  one, so the basis contributes shape, not precision, and can be as small as
  two_arm's own winner. Inside three_arm's feature stack the pair premia carry
  the LARGEST `d/dtau` of all fifteen features (91.9 against 27.1 for the
  next), which is the quantity `l_ab = mean_diffusion + v_tbb` is built from.

  The case against the monolith is the scaling: three_arm's best pde is ~2e-3
  against two_arm's 2.8e-5, two orders worse for 24x the parameters, and in
  the 2026-08-12 architecture sweep the 9,754-parameter net still led at 92k
  iterations while every small net trailed 2-4x -- the opposite of two_arm,
  where 406 parameters beat 9,154 outright.

  The combiner, SETTLED 2026-08-13: a learned linear term in the three pair
  premia plus a correction net over all eighteen features, summed in RESPONSE
  space. Response space is what makes it legal -- those coefficients sum to
  1.501, so adding them in PREMIUM space overcounts by half and the result is
  not a value function, while behind the envelope-times-saturated-response gate
  `0 <= u < nu2` stays architectural whatever they do.

  The correction is not a perturbation, which the regression settles (20k wedge
  points, 2026-08-13): the pair premia explain R2 0.977 of the champion's
  premium VALUE but only 0.761 of `d/d tau_bb` and 0.872 of `d/d tau_cc` -- and
  the HJB grades derivatives, not values. Worse for a seeded combiner, the
  coefficients differ by quantity: +0.923/+0.476/+0.102 on the value against
  +0.724/+0.703/+0.697 on the slope, where all three pairs enter about equally.
  So the linear head is LEARNED and zero-init, never seeded from the value fit.
  Still open: frozen vs fine-tuned basis, and the fact that the 2% the basis
  does not explain is exactly the b/c-junction blob, i.e. the correction net
  inherits the hard part rather than the easy one.

  Cost, measured on a crowded 4090 at batch 4096: a v2 step is 1.30x a v1 step,
  and step cost is FLAT in width for both (619 ms at 2,754 parameters against
  670 at 9,739; 894 at 914 against 872 at 9,794). The step is dispatch-bound on
  the analytic machinery, so the basis toll is fixed rather than scaling, a
  SMALLER v2 is relatively more expensive, and v2's payoff at N=3 cannot be
  training time -- it has to be accuracy-per-parameter and the N-arm scaling.

  NOT indicted by learnings section 11: that result is about WEIGHT
  inheritance (a warm start is a local minimum). This is feature reuse -- the
  child keeps its own weights and only queries a function.
- Policy iteration (v3-sized): the arena evaluates policies; fit v to the
  EVALUATED policy value and re-extract.
- Arena analyze: exploit the seed-pairing (per-rep differences vs the best)
  for ~2x tighter comparison CIs.

## Numerical traps already hit — don't reintroduce

- `torch.where` evaluates (and backprops) both branches: guard divisions in dead
  branches with `masked_fill` dummies; the stronger fix is gather-based
  selection (both problems' `simplex.py`), whose backward zeroes unselected rows.
- Test stubs that a loss will differentiate must be graph-connected: return
  `0.0 * m_b`, never `zeros_like(m_b)` (autograd refuses to start from a
  constant tensor, and `allow_unused` cannot save it).
- Bulk text edits lie: BSD `sed` treats `\b` as a literal and exits 0 having
  matched nothing, and replace-scripts print success whether or not the target
  matched (black may have rewrapped it). Verify replaces with grep, always.
- Never place collocation or derivative evaluation exactly at `muhat = 0` on the
  value (`relu` kink); ridge conditions are imposed on the premium, which is smooth
  there.
- Derivatives via `torch.autograd.grad(out.sum(), inputs, create_graph=True)`; the
  `.sum()` trick is valid because batches are diagonal.
- Never saturate a trainable response with `tanh` of an unbounded argument:
  float32 `tanh(y)` is exactly 1 for `y > ~9` and its gradient underflows — a
  one-way cliff (three_arm's first relu-squared start died with `r ~ 14`
  everywhere and a machine-zero wall loss; only `log_scale` was left alive).
  Saturate rationally (`y/(1+y)`): polynomial tail, gradients survive overshoot.
- The relu-squared response has an all-dead absorbing state: premium
  identically 0 zeroes the pde residual exactly AND every loss gradient, so a
  run that falls in freezes forever (signature: total loss stuck at exactly
  `TIE_WEIGHT`). Guard is the head-bias-1 init (start alive everywhere); if a
  frozen run shows that signature, it's dead, restart it.
- Bare (unsaturated) relu**2 kink units co-trained from scratch colonize the
  smooth bulk (they outrun tanh early): 2 orders worse. The saturated form
  `y/(1+y)` is the co-training-safe primitive; the stitch (smooth net first,
  kinks grafted after) remains the better cast either way.
- DEFUSED 2026-08-10, kept because it explains old records: training used to
  decay the lr with a 100k half-life, and every resume restarted that schedule
  at its hot end, so short resumed runs first SMEARED a polished checkpoint
  (~1k iterations of bounce) before improving it. Read the first ~1k prints of
  any historical resume with that in mind. The lr is now constant, no schedule.
  Adam's moments are still not checkpointed, so a resume does pay one
  `sign(g)` step of size `lr` on every parameter.
- NEVER put a scale on the PDE residual, and audit any coordinate change for
  one you did not intend. A similarity chart is a DERIVATION tool (it
  conditions the autograd chain); grading its residual squared silently
  applies the square of its Jacobian factor as a domain weight. Here that was
  `tauhat**3` over ~15 decades, undetected for months, and the tell was
  seductive: the absolute residual looked beautifully FLAT across decades,
  which is exactly what a weight cancelling the true error's growth produces.
  Diagnostic when you suspect one: compute each region's share of the loss's
  ATTENTION (the p-mean's normalized per-point weights) against its share of
  the POPULATION. 70% of the gradient sat on a corner holding 29% of the
  points and already exact to ~1e-6 relative. Standing rule: learnings
  section 3.
- Attention exponent and residual units are the SAME knob — never move one
  without re-measuring the other. The p-mean at P = 2 was compensating for a
  grading that suppressed the tail; in natural units it became
  over-correction, dropping the effective sample size (`1 / sum(w_i^2)`) to
  ~2 points out of 4096 on three_arm. A gradient decided by two points is
  also precision-bound, which is how it surfaced: the S3-invariance
  self-check began flaking one run in three. `POWER = 1.0` (plain
  mean-of-squares) restores ESS to 36-177 across the four problems, and
  `BATCH` is 4096 (matching the old ABSOLUTE ESS would want ~20k, 5x the step
  cost). Re-measure ESS after ANY change to units or exponent.
- Invariance and identity self-checks want RELATIVE tolerances. An `atol`
  is calibrated against a loss magnitude, and loss magnitudes move by orders
  when the functional changes; the S3 checks now use `rtol=1e-3`, which still
  catches a transposed erosion entry (that moves them by O(1)).
- Changing the pde term's magnitude detonates every weight calibrated against
  it — recheck ridge, ties, positivity and concavity after any
  loss-functional change. Set degeneracy breakers from the DEAD SOLUTION
  instead of from a checkpoint's error level: `u = 0` scores pde exactly 0,
  ridge exactly 0.25 and the control tie exactly 1.0, so `W * dead_value`
  must beat the live pde or the dead branch is the better minimum. That floor
  is analytic and net-independent in form; the enforcement level above it is
  a tuning question — watch the printed ridge/tie on the first run, falling is
  fine, climbing means raise 10x.

## Working style for this repo

Exploratory project, deliberately incremental: small bites, one design decision at
a time, discussed before coded. Ponytail (lazy-minimal) mode applies: shortest
working code, no speculative scaffolding, modules stay small. Python follows
Pedro's house style (full type-hinted signatures, real-bool conditionals, ASCII
only, black as the final pass).

AUXILIARY WEIGHTS LAND IN 1-10% OF pde, calibrated on the MEDIAN over several
draws. Both halves are load-bearing: a single draw has burned this twice (a
1.9e4 tie weight from a pde reading of 187 whose median was 2.4; a concavity
weight at 730%), because these losses are heavy-tailed enough that one batch
misleads by two orders. Re-derive after any large pde move -- a FIXED weight
against a falling pde silently inflates, which is how a ridge weight ended up
7000x its own criterion. The floor is separate and lower: on the never-explore
solution pde is exactly 0 while ridge is 0.25 and the control tie is 1.0, so
the weight must beat the live pde or the dead branch wins.

THE BORE TEST, applied to every comment and docstring paragraph. Load-bearing
for someone editing this code? It stays. Not load-bearing but true, useful and
written down NOWHERE else? MOVE it to kb/ -- derivations and rejected
approaches to the problem doc, method lessons to kb/learnings.md. Neither?
Delete it. A paragraph restating what kb/ already says is the most common
failure, and it rots independently of the copy in kb/.

COMMENTS ARE NOT A NOTEBOOK. Any comment block of four lines or more, and any
docstring paragraph, must be audited before it lands: keep the measured numbers
and the reason a constant has its value, cut the narration, the deliberation,
and the account of what was tried. Short is not automatically clean either --
a two-line comment restating the code earns nothing. Stale is worse than long:
when a constant is re-derived or a term rewritten, REPLACE its comment rather
than appending to it, or the next reader gets three stacked paragraphs
contradicting each other on a dead scale.
