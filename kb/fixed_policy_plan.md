# Fixed-policy HJB: refactor plan

STATUS: PLAN ONLY, NOT VOUCHED. Discussion pending; nothing here is
implemented and no design below is settled. Written 2026-08-19 from the seam
survey of that date.

## Why

Policy iteration needs both halves: the arena (or the planned batched rollout
evaluator) EVALUATES a policy, and a fixed-policy solve FITS v to that
policy's value so a better policy can be re-extracted. The optimal-control
HJB the four problems solve today is `v = max_alpha H(alpha)`; the
fixed-policy equation is the same PDE with the max replaced by evaluation at
a given `alpha(s)` -- linear in v, no free boundary from optimization, no
subsolution structure.

## Seams found (2026-08-19 survey)

- S1. `DimensionlessValueFunction.hamiltonian` (all four problems) assembles
  the Hamiltonian's quadratic coefficients from the one derivative chain and
  then resolves the control in a single call to `maximize_quadratic`. That
  call is the only place optimization enters the problem. The max-vs-evaluate
  choice is one line deep.
- S2. `simplex.maximize_quadratic`'s inner `f(x, y)` IS the fixed-policy
  Hamiltonian; it is just not exported.
- S3. The loss family forks, it does not refactor: subsolution machinery
  (violation / climb / slack price) is meaningful only because the max makes
  the HJB an inequality with a maximal solution. The linear equation wants a
  plain two-sided residual in natural units.
- S4. The `Problem` ABC serves one objective per package; a fixed-policy
  problem is a family indexed by a policy callable.
- S5. The premium architecture bakes in `u >= 0` (response times envelope),
  a theorem for V* and only a hope for arbitrary policies.

## Plan

1. Split assembly from resolution (S1, S2).
   - `simplex.py`: hoist the inner quadratic into an exported
     `evaluate_quadratic(c_xx, c_yy, c_xy, c_x, c_y, x, y)`;
     `maximize_quadratic` calls it. two_arm's 1-D sibling gets the same.
   - `model.py`: `hamiltonian_coefficients(state) -> (v, coefficients,
     learning)`; `hamiltonian` becomes coefficients + maximize and must stay
     BIT-IDENTICAL (the float-association discipline of the two_arm_drift
     anchor: keep the final products grouped as today, do not re-associate).
   - Self-check: `hamiltonian` output equality against the pre-split
     implementation on a fixed draw, `torch.equal`, both dtypes.
   - Mechanical; can land before any fixed-policy work, zero behavior change.

2. Fixed-policy left side (S1).
   - `fixed_hamiltonian(state, alpha) -> (v, h_alpha)`: coefficients +
     `evaluate_quadratic` at `alpha` (wedge roles; deployment fold and
     un-permutation reuse `ValueFunction` machinery unchanged).
   - The policy is FROZEN: evaluate `alpha(s)` under no_grad per draw;
     residual gradients flow through v only. Any batched state -> allocation
     callable qualifies (arena entrants, a frozen net's readout).

3. Fixed-policy loss (S3). New module per problem (`policy_loss.py` or
   sibling name, undecided), NOT a modification of `loss.py`:
   - Two-sided residual `(v - h_alpha)`, natural units, POWER = 1, never
     scaled (learnings section 3 applies verbatim).
   - Value ties carry over IFF the policy is S3-equivariant (fold-based net
     policies are; assert or document as precondition). Concavity and
     learning ties DO NOT transfer: their proofs are statements about V*.
   - No degeneracy breaker needed: the linear equation with the ties has no
     dead branch.
   - Weights: the tie weight re-derived against this loss's own pde median
     (the standing 1-10% / median-over-draws rule); nothing inherited.

4. Problem plumbing (S4). Leave the ABC and `PROBLEMS` alone. Add a
   `FixedPolicyProblem` wrapper constructed from (base problem, policy
   callable), exposing the `objective / draw / loss` surface the generic
   trainer already consumes. Not registered; whoever runs policy iteration
   constructs it. CLI support deferred until a real workflow exists.

5. Architecture precondition (S5). Document in the new module: the net class
   represents `u >= 0` only, so the policy being evaluated must dominate
   commit-now (near-greedy policies do). A policy that does not will produce
   a silent floor at the commit envelope, not an error. Revisit only if
   someone actually needs to evaluate bad policies.

## Non-goals

- No change to samplers, envelopes, folds, trained checkpoints, or the
  optimal-control losses. Steps 2-5 are pure additions.
- No policy-improvement loop here: extraction, iteration scheduling and the
  rollout evaluator are separate work (CLAUDE.md "Planned next").

## Open questions for the discussion

- Where the evaluated policy comes from first: a frozen champion's readout
  (self-iteration) or an arena entrant (evaluating TS/drop-one for contrast)?
- Fixed-policy value smoothness: an argmax policy is discontinuous across
  ties, and its value inherits ridges there; whether the smooth premium class
  fits it well enough, or the kink branch is needed from the start, is a
  measurement to make, not a decision.
- Whether the two-arm problems get the fixed-policy treatment at all, or
  only the three-arm pair where policy iteration is actually planned.
- Naming: `policy_loss.py` vs a `fixed/` subpackage; ONE-NAME-WIDE rule
  interaction with the wrapper.
