# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An exploratory PINN (physics-informed neural network) solving the HJB free-boundary
problem specified in `docs/two_arm.md`. That document is the source of truth for all
math: the PDE, boundary conditions, notation, and the graveyard of already-rejected
approaches (do not redo them). Read it before touching any equation code.
`docs/learnings.md` is the problem-agnostic playbook distilled from this project
(architecture/sampling/grading/diagnosis method) — read it before designing for a
NEW problem.

Notation contract (Section 0 of the doc): uppercase = dimensional (`V` value,
`U` exploration premium), lowercase = dimensionless (`u`, `v`), hats = dimensionless
coordinates (`muhat`, `tauhat`). Earlier drafts used `U` for the even part
`V - mu/(2 rho)`; transcribe old statements via `u_even = u + |muhat|/2`.

## Commands

- Run everything through poetry: `poetry run python train.py --problem
  two_arm|three_arm` (trains, saves `data/<problem>.pt`; `HIDDEN`/`KINKS`
  constants at the top size fresh nets and first-time stitches).
- Diagnostics: `poetry run python probes.py --in data/<ckpt>.pt` (three_arm
  wedge slices, tau lines, symmetric-point table); `plot.py --in` (two_arm
  fields); `benchmark3.py --in` (three_arm posterior-space self-consistency:
  the net's claim vs its policy simulated); `validate.py` (two_arm section-6
  identities, hardcodes `data/two_arm.pt`).
- The arena (policy shoot-out vs TS/ETC/elimination on discrete epochs,
  regret vs true effects): `poetry run arena simulate <out.pkl> --problem
  two_arm|three_arm --rho ... --size N --workers K`, then `poetry run arena
  analyze <out.pkl>`. Realistic parameter values live outside the repo
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
  loss grades the HJB in similarity coordinates (docs/two_arm.md section 8:
  leaves are `(z, s)`, autograd performs the chain rule, and a self-check
  asserts the `tauhat**1.5` identity against the raw form), maximization kept
  explicit and handed to `simplex.maximize_quadratic` (evaluate-and-max over
  vertex + endpoints, gather selection), graded by the POWER-MEAN of squared
  residuals with NO relative scale (the similarity equation self-normalizes;
  the p-mean is population-relative attention for the fat-tail shelf — see
  learnings section 7; `POWER = 1` recovers plain mean-of-squares), plus the
  BC1 ridge term. Do not substitute the interior FOC back
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
- `pinn/problems/three_arm/` — the ABC-test problem, one module per concern
  (`sample.py` with the wedge fold, `simplex.py` plain quadratic-max calculus,
  `model.py`, `loss.py`), fully derived (docs/three_arm.md, "nothing
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
  (the old discarded-permutation TODO is resolved). Self-checks:
  `poetry run python -m pinn.problems.three_arm.<sample|simplex|model|loss>`.
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
- `train.py`, `plot.py`, `probes.py`, `benchmark3.py` — click CLIs;
  `validate.py` — two_arm consistency checks.
- Module self-checks run as `poetry run python -m pinn.problems.two_arm.loss`
  (relative imports; both problem packages check per-module).

`data/` is gitignored and holds checkpoints (`*.pt`) and diagnostic plots. Plots
belong there (visible in the IDE), not in temp dirs.

## State of play (2026-08-06 evening) and open frontiers

Champions (all in gitignored `data/`, backed up to the GitHub release
`checkpoints-2026-08-06`):
- two_arm: `value_2a_32:512.pt` (p-mean era; also promoted to the canonical
  `two_arm.pt`). The old-architecture champion `value_champion_8e-8.pt`
  (loss ~8e-8, loads only at the initial commit's class) remains the
  historical reference.
- three_arm: `value_3a_64:64:64.pt` — pde ~2.7e-6 (log-cosh yardstick),
  3 hidden layers with 8 stitched saturated kink units; the kink stitch was
  the decisive move (tail -19%, worst point -25% in one 30k run, junction
  specialist unit self-oriented onto `z_bc`). `value_3a_64:64:64k16.pt` is
  the from-scratch 16-kink co-training experiment (~4x champion loss,
  closing; communal branch anatomy, no junction specialist — sequencing is
  load-bearing, see learnings section 8).

The headline (2026-08-06): in the arena, at realistic parameters (values in
project memory, not committed), the two_arm PINN policy beat
Thompson sampling — the strongest practical baseline — by ~20% of
discounted regret while buying less than half the information (precision
time 332 vs 717), with the other baselines reproduced against an earlier
private study to within a fraction of a CI. Three-arm arena results
pending.

Open frontiers:
- The low-tauhat floor decade: below `tauhat ~ 1e-2` the two_arm champion's
  `M[g]` goes NEGATIVE at the ridge, flipping the Hamiltonian convex and
  vertex-committing on zero evidence — a policy pathology, not just a
  residual. The startup/similarity anchor is now commercially load-bearing
  (the arena's `_FLATTEST_TAUHAT` guards it meanwhile).
- The b/c-junction blob (three_arm) at ~±0.07 relative, best ever, still
  the dominant error; systematically POSITIVE (v > max H), which also
  inflates the net's value claims ~9% above its policy's simulated value.
  The subsolution program (one-sided grading + the constant-shift
  a-posteriori bound; see learnings section 9) would turn this into a
  certificate.
- The curvature law constant and the tauhat-tail anchoring flags of
  docs/two_arm.md remain open.

Planned next (intent recorded 2026-08-06):
- three_arm v2, the pairwise ansatz: `u = f(u2_ab, u2_ac, u2_bc) + correction`
  with the two_arm champion as basis — the kink-unit result is its proof of
  concept, and it is the N-arm scaling route (N^2 pairs). Open: the combiner
  (sum overcounts; must keep `0 <= u < nu2`), frozen vs fine-tuned basis.
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
- Every training resume restarts the lr schedule at the hot end
  (`decay(3e-4, ...)` in train.py): short resumed runs first SMEAR a
  polished checkpoint (~1k iterations of bounce) before improving it. Never
  judge a resume by its first prints; L-BFGS-polished minima are the most
  fragile to this.
- At P = 2 the p-mean's gradient rides ~13% of the batch (effective sample
  size); batch 1024 oscillates, 2048 is the defended floor, 4096 is
  comfortable. And amplifying the pde term silently deflates `TIE_WEIGHT` —
  recheck tie losses after any loss-functional change.

## Working style for this repo

Exploratory project, deliberately incremental: small bites, one design decision at
a time, discussed before coded. Ponytail (lazy-minimal) mode applies: shortest
working code, no speculative scaffolding, modules stay small. Python follows
Pedro's house style (full type-hinted signatures, real-bool conditionals, ASCII
only, black as the final pass).
