# pinn

Physics-informed neural networks for the optimal allocation of experiment
traffic: given noisy A/B(/C) test observations, a discount rate, and the
option to commit to an arm forever, how should traffic be split *right now*?
That question is a Hamilton-Jacobi-Bellman free-boundary problem; this repo
solves it by training a network on the PDE itself — no simulation data — and
reads the allocation policy off the trained value function's derivatives.

![The learned three-arm allocation policy](docs/policy_atlas.png)

The picture is the trained three-arm policy: each point is a belief state
(posterior means of challengers B and C relative to control), each color is
the traffic split played there. With little information, the policy blends —
paying for evidence on every arm at once. As information accumulates, three
commit basins crystallize, separated by narrowing exploration corridors that
meet at the triple point where all three arms are still in play. Nobody drew
those boundaries; they are free boundaries of the HJB equation, found by the
network.

## Does it matter?

![Arena results](docs/arena_two_arm.png)

In a discrete-epoch simulation harness (`pinn/arena`) against the true
effect, with paired seeds and realistic economics, the PINN
policy pays **20% less discounted regret than Thompson sampling** — the
strongest practical baseline, which itself beats explore-then-commit and
peeking z-tests by wide margins — while buying **less than half the
information**. The mechanism: Thompson sampling prices uncertainty but not
patience; the HJB value function prices both, so it exploits earlier and
knows when further evidence stops being worth its discounted cost.

## What's inside

- `pinn/problems/two_arm`, `pinn/problems/three_arm` — the trainable models
  (dimensionless, wedge-quotiented by the problems' exact symmetries),
  samplers, and PDE losses. The math lives in `docs/two_arm.md` and
  `docs/three_arm.md`; the transferable method in `docs/learnings.md`.
- `pinn/arena` — the policy shoot-out: N-arm regret harness, baseline zoos
  (Thompson, explore-then-commit, z-test/elimination), and the PINN entrant.
  `poetry run arena simulate ... && poetry run arena analyze ...`.
- `train.py`, `probes.py`, `plot.py`, `benchmark3.py`, `validate.py` — CLIs
  for training and diagnostics.

Trained checkpoints are published as GitHub release assets
(`checkpoints-2026-08-06`); drop them in `data/` and every CLI finds them.

## Quickstart

```sh
poetry install
poetry run python train.py --problem three_arm     # train (Ctrl-C saves)
poetry run python probes.py --in data/value_3a_64:64:64.pt   # diagnostics
poetry run arena simulate data/study.pkl --problem three_arm \
    --rho 0.999 --horizon 500 --sigma 1 --effect 0 --effect-std 0.3 --size 1000
poetry run arena analyze data/study.pkl
```
