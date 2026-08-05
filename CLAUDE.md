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

- Run everything through poetry: `poetry run python train.py` (trains, saves
  `data/value.pt`).
- Each module in `pinn/` carries an `assert`-based self-check:
  `poetry run python -m pinn.problems.two_arm`, `poetry run python -m pinn.net`.
- Final step of any Python change: `poetry run black <files>`.
- Torch runs on CPU deliberately (tiny net; MPS overhead loses). Revisit only when
  batch/width grows.

## Architecture

Organized one-module-per-problem; the separation should be kept:

- `pinn/problems/two_arm.py` — THE problem module (models + loss + samplers for the
  docs/two_arm.md problem). `ExplorationPremium`: tanh MLP with Xavier-tanh init,
  four aligned features `(muhat, log tauhat, muhat sqrt(tauhat), muhat tauhat)`,
  times a `tauhat**(-1/2)` envelope with learnable `log_scale`. `ValueFunction`
  wraps it as `v = relu(muhat) + u`. Samplers: scrambled Sobol pushed through a
  long-tailed law (floored at `tauhat = 0.1`), strictly `muhat > 0`. The loss
  keeps the HJB maximization explicit: `v = max_{alpha in [0,1]} alpha*muhat +
  alpha*(1-alpha)*Lhat[v]`, closed form per point (clamped vertex when concave,
  else best endpoint), relative residual under log-cosh, plus the BC1 ridge term.
  Do not substitute the interior FOC back in (that road leads to `sqrt` NaNs).
  `objective(batch)` packages sampling + loss for the trainer. New problems get
  their own sibling module.
- `pinn/problems/three_arm/` — the ABC-test problem, one module per concern
  (`sample.py` with the wedge fold, `simplex.py` plain quadratic-max calculus,
  `model.py`, `loss.py`), fully derived (docs/three_arm.md, "nothing
  mathematical" left) and training-ready. Key structure: solve only on the
  fundamental wedge {m_c <= m_b <= 0} (6-fold S3 quotient, `Sample.fold()`),
  pairwise-precision coordinates, seven-candidate simplex max, two
  mirror-paired wall losses, and the proven free-information envelope
  `nu(m_b) + nu(m_c)` times a (0,1)-mapped tanh response — positivity and the
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
- Module self-checks run as `poetry run python -m pinn.problems.two_arm` (relative imports).

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
two_arm.py docstring); jitter/low-discrepancy beats both fixed grids and iid
draws; Xavier-tanh init moved the floor when width and gains could not.

Planned experiments (intent recorded 2026-08-05, not yet started):
- Retrain two_arm with the nu envelope (`nu(-muhat.abs(), tauhat.rsqrt())`,
  docs/three_arm.md section 13): the derived bound retroactively upgrades the
  guessed `tauhat**-1/2` envelope (adds muhat-decay the old one lacked).
- three_arm first long training run, then 5-D diagnostics (probes and the
  benchmark simulation, not field plots) with special attention to the
  triple-point region (no known solution exists there).
- three_arm v2: warm-start pairwise structure from the two_arm champion.
- Root `train.py`/`plot.py` CLIs currently hardcode two_arm; a problem
  selector is needed before three_arm trains via the CLI.

Known TODO with a bite: `Sample.fold()` discards WHICH relabel it applied.
Training never needs it (the premium is invariant), but policy readout does —
the argmax alphas must be un-permuted back to physical arms. Add a
permutation return before building three_arm policy extraction.

## Numerical traps already hit — don't reintroduce

- `torch.where` evaluates (and backprops) both branches: guard divisions in dead
  branches (`safe_lhat` pattern in `pinn/problems/two_arm.py`); the stronger fix
  is gather-based selection (`pinn/problems/three_arm/simplex.py`), whose
  backward zeroes unselected rows.
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

## Working style for this repo

Exploratory project, deliberately incremental: small bites, one design decision at
a time, discussed before coded. Ponytail (lazy-minimal) mode applies: shortest
working code, no speculative scaffolding, modules stay small. Python follows
Pedro's house style (full type-hinted signatures, real-bool conditionals, ASCII
only, black as the final pass).
