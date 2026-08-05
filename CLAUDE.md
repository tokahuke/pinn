# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

An exploratory PINN (physics-informed neural network) solving the HJB free-boundary
problem specified in `docs/two_arm.md`. That document is the source of truth for all
math: the PDE, boundary conditions, notation, and the graveyard of already-rejected
approaches (do not redo them). Read it before touching any equation code.

Notation contract (Section 0 of the doc): uppercase = dimensional (`V` value,
`U` exploration premium), lowercase = dimensionless (`u`, `v`), hats = dimensionless
coordinates (`muhat`, `tauhat`). Earlier drafts used `U` for the even part
`V - mu/(2 rho)`; transcribe old statements via `u_even = u + |muhat|/2`.

## Commands

- Run everything through poetry: `poetry run python train.py --problem
  two_arm|three_arm` (trains, saves `data/<problem>.pt`).
- Each module in `pinn/` carries an `assert`-based self-check:
  `poetry run python -m pinn.problems.two_arm.loss`, `poetry run python -m pinn.net`.
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
  vertex + endpoints, gather selection), plain log-cosh with NO relative
  scale (the similarity equation self-normalizes: bounded `g`, O(1)
  coefficients; the old `1 + |H - commit|` scale was a raw-coordinate
  vestige), plus the BC1 ridge term. Do not substitute the interior FOC back
  in (that road leads to `sqrt` NaNs). `objective(batch)` packages sampling +
  loss for the trainer. New problems get their own sibling package.
- `pinn/problems/three_arm/` — the ABC-test problem, one module per concern
  (`sample.py` with the wedge fold, `simplex.py` plain quadratic-max calculus,
  `model.py`, `loss.py`), fully derived (docs/three_arm.md, "nothing
  mathematical" left) and training-ready. Key structure: solve only on the
  fundamental wedge {m_c <= m_b <= 0} (6-fold S3 quotient, `Sample.fold()`),
  pairwise-precision coordinates, seven-candidate simplex max, two
  mirror-paired wall losses, and the proven free-information envelope
  `nu(m_b) + nu(m_c)` times a `relu(r)**2 / (1 + relu(r)**2)` response (same
  free-boundary trick as two_arm: exact zero in the deep-wedge commit region,
  curvature-jump regularity on the learned boundary) — positivity and the
  upper bound are architectural invariants, not loss terms. Self-checks:
  `poetry run python -m pinn.problems.three_arm.<sample|simplex|model|loss>`.
- `pinn/net.py` — generic building blocks only (`GainedTanh`: trainable per-unit
  activation gains).
- `pinn/utils.py` — shared math (`nu`: Gaussian expected positive part, the
  envelope building block; clamped at 0 against float32 tail cancellation).
- `pinn/train.py` — generic trainer: endless-generator `train(model, objective,
  lr)` (consumer stops via `itertools.islice`; mutates the model in place; `lr`
  is a float or a rate generator, see `decay`; a finite generator ends training).
- `train.py`, `plot.py` — click CLIs (`--in`/`--out`, and `--in` respectively);
  `validate.py`, `benchmark.py` — consistency checks and the Monte Carlo policy
  shoot-out vs explore-then-commit.
- Module self-checks run as `poetry run python -m pinn.problems.two_arm.loss`
  (relative imports; both problem packages check per-module).

`data/` is gitignored and holds checkpoints (`*.pt`) and diagnostic plots. Plots
belong there (visible in the IDE), not in temp dirs.

## State of play (2026-08-04) and open frontiers

Solved to loss ~8e-8 (worst-case residual ~1e-2, confined to `tauhat < 0.5`).
Best checkpoint preserved as `data/value_champion_8e-8.pt`. The policy beats
optimally-timed explore-then-commit by ~41% at `(m, tauhat) = (0, 1)` and its
simulated value matches the net's own `v(0,1)` to three decimals (benchmark.py).

Open frontiers:
- The stiff corner `tauhat < 0.5` holds nearly all remaining error (wobble
  ~2% relative at `tauhat = 0.1`).
- The curvature law constant: PINN measures `u_mm|_{G-} / (tauhat^2 G) ~ 2.2`,
  supporting the original-units factor 2 — needs the pencil derivation
  (docs/two_arm.md section 6 flag).
- Beyond `tauhat ~ 15` the net extrapolates wrongly (corridor re-widens);
  honest fix is anchoring the tail with the similarity ODE of section 7.

Hard-won lessons (do not relearn): BC1 in the loss is what kills the trivial
`u ≡ 0` solution; fixed Fourier features imprint themselves (reverted, see
two_arm's ExplorationPremium docstring); jitter/low-discrepancy beats both
fixed grids and iid
draws; Xavier-tanh init moved the floor when width and gains could not.

Planned experiments (intent recorded 2026-08-05):
- DONE 2026-08-05: two_arm rebuilt in parallel to three_arm (package split,
  nu envelope, (0,1)-mapped response, gather simplex max, regret-form scale)
  and `train.py` grew `--problem`; the from-scratch nu-envelope retrain is
  running. Old checkpoints (incl. the champion) load only at the initial
  commit's class.
- three_arm first long training run, then 5-D diagnostics (probes and the
  benchmark simulation, not field plots) with special attention to the
  triple-point region (no known solution exists there).
- three_arm v2: warm-start pairwise structure from the two_arm champion.

Known TODO with a bite: `Sample.fold()` discards WHICH relabel it applied.
Training never needs it (the premium is invariant), but policy readout does —
the argmax alphas must be un-permuted back to physical arms. Add a
permutation return before building three_arm policy extraction.

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

## Working style for this repo

Exploratory project, deliberately incremental: small bites, one design decision at
a time, discussed before coded. Ponytail (lazy-minimal) mode applies: shortest
working code, no speculative scaffolding, modules stay small. Python follows
Pedro's house style (full type-hinted signatures, real-bool conditionals, ASCII
only, black as the final pass).
