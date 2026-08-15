# jobq — pod conventions

## Default cloud is COMMUNITY

`jobq up` defaults to `--cloud COMMUNITY`. Same card, roughly half the
price (RTX 4090 measured 2026-08-13: $0.34/hr community against $0.74
secure), and the 4090 is already the work-per-dollar pick (5x an A6000 on
secure; the measured table lives in `jobq/up.py`).

Community is other people's machines, so interruption is possible — and the
extra code for that is already built, which is why the discount is free:

- `jobq up` passes `--public-ip` on community creates (those hosts only
  publish an ssh port when they have a public ip; without the flag the
  create waits forever on a port mapping that never appears — this was the
  code community needed).
- The backup daemon mirrors `/workspace` into `data/pod` continuously, so a
  killed pod loses at most one sync interval.
- The trainer saves the best-EMA checkpoint at print cadence; an
  interruption loses minutes of descent, not the run.
- Logs APPEND (`jobq run --log`), so a resume on a fresh pod keeps one
  history: `jobq cp` the checkpoint and log up, relaunch with the same
  `--log` path.

Use SECURE only when an interruption mid-run would cost more than the ~2x
price gap — nothing in this repo's training loop qualifies, since resumes
are cheap (constant lr, one `sign(g)` step of Adam-moment loss).

Verified in practice 2026-08-13: a three_arm_drift run was moved
secure -> community mid-training (SIGINT saves, `down` fetches, `up`,
`cp` back, relaunch); the resume continued from the checkpoint's loss level
with no smearing.

## Card choice is work per dollar, and small Ada wins

Same benchmark (three_arm, graphed, batch 16384), community prices:

    RTX 4000 Ada  20.5 ms/step   $0.20   1.00 (reference)
    RTX 4090      14.5 ms/step   $0.34   0.83
    RTX A6000     46.7 ms/step   $0.33   0.27

The RTX 4000 Ada is 1.41x SLOWER than the 4090 and still the better buy,
because the step is dispatch-bound: SM count barely matters, and the small
card carries the same 3105 MHz max SM clock and the same Ada scheduler on a
third of the silicon. Ampere is on the wrong side of this whatever it costs.

`runpodctl get cloud` lists TYPES WITH PRICES, NOT LIVE STOCK: on 2026-08-15
every one of the four cards then in `jobq up`'s walk was listed on community
and every create failed. The walk is nine candidates now for that reason.
