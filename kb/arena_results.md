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
checkpoints of this date — two_arm `value_2a_32:512.pt` from
`data/frontier4k.pkl`, three_arm `value_3a_64:64:64.pt` from
`data/frontier3a.pkl` (both nets on the `checkpoints-2026-08-06` release) —
with the near-flat default prior, evaluated directly (no policy table),
commits landing on exact simplex vertices. Those two filenames are historical
and no longer resolve; see the staleness note.

**Stale as of 2026-08-10 — the numbers are right, the nets are not.** Every
row below was re-derived from the two raw studies on 2026-08-10 and
reproduces exactly, so the tables stand as a faithful record of what those
checkpoints did at size 4,000. Both nets have since been replaced: the
two_arm one was deleted (it was byte-identical to what is now
`data/two_arm.2026-08-09.pt`), and the three_arm path was repointed
2026-08-07, retrained through the concavity term on 2026-08-10, and retrained
again under the natural-units grading fix (learnings section 3, which also
moved every loss figure onto a new scale). Re-run both sweeps once those
retrains land; until then read the ratios as two-to-three generations old.

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
