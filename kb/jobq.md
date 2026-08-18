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
- The trainer saves the best-EMA model at print cadence; an
  interruption loses minutes of descent, not the run.
- Logs APPEND (`jobq run --log`), so a resume on a fresh pod keeps one
  history: `jobq cp` the model file and log up, relaunch with the same
  `--log` path.

Use SECURE only when an interruption mid-run would cost more than the ~2x
price gap — nothing in this repo's training loop qualifies, since resumes
are cheap (constant lr, one `sign(g)` step of Adam-moment loss).

Verified in practice 2026-08-13: a three_arm_drift run was moved
secure -> community mid-training (SIGINT saves, `down` fetches, `up`,
`cp` back, relaunch); the resume continued from the model's loss level
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

## Detaching a run: the parentheses are load-bearing

`jobq run --log FILE` wraps the remote command as

    cd ... && export ... && (setsid nohup CMD < /dev/null >> LOG 2>&1 & echo $!)

Unparenthesized, `&` binds the WHOLE and-list, so bash backgrounds a wrapper
subshell that keeps the ssh pipe open as its stdout. sshd then waits for the
JOB to exit: a "detached" launch measured 2026-08-13 returned after 40 minutes,
at the exact moment its trainer was killed. The wrapper is also what `$!`
names, so the printed pid was one off the real job.

Parenthesized, `&` backgrounds the fully-redirected simple command alone: the
launch returns in ssh round-trip time and `$!` is the job. `setsid` does not
fork under a non-interactive shell; it is there so the job escapes the session
group, which is what the seppuku daemon counts.

Three more things that are not optional. `PYTHONUNBUFFERED=1`, because a
detached job's stdout is a FILE and python block-buffers those: without it a
log stays empty for hours while the job runs fine, indistinguishable from a
hang. No `-t`, because a tty and a backgrounded nohup fight over the terminal
and the pid never comes back. And the pid is printed because it is the only
safe way to stop the job later: a pattern match on "pinn train" also matches
the command doing the matching, and once cost a running job.

## Backups accumulate, they do not mirror

`jobq backup` is content-blind: it copies whatever changed rather than what
jobq thinks is interesting, skipping only the venv and the repo, which jobq
put there itself. There is NO `--delete`, so a file removed on the pod
survives locally. The pod is disposable and its files only grow, so "the
latest copy of everything that ever existed there" is the useful shape. It
keeps no history, though: one version per file.

The event stream is `inotifywait` running ON THE POD, so the kernel there
decides when something changed and there is no polling interval to get wrong.
Both `modify` and `close_write` are watched and MODIFY is the load-bearing
one: close_write fires only when a writer CLOSES the file, so a log held open
by a running process never emits it. Watching close_write alone produced zero
events in a minute against eight live jobs. `inotifywait` takes ONE
`--exclude` and it is a regex, not a repeatable flag: passing several makes it
silently honour the last only. rsync's is repeatable, which is why `fetch()`
takes a list.

A backup does NOT keep the pod alive. The seppuku timer counts training
processes and tty sessions; rsync is neither.

## Where the pid file lives

`STATE` is `XDG_RUNTIME_DIR/jobq`, falling back to the per-user tempdir. A pid
file is RUNTIME state, so it belongs where the OS clears it. On Linux that is
`/run/user/$UID`: mode 0700, local filesystem, removed at logout by
specification, which makes a stale pid impossible. macOS sets no
XDG_RUNTIME_DIR but TMPDIR is per-user and 0700 (`/var/folders/../T`, NOT the
shared `/tmp`), cleaned on a schedule rather than at logout, so there the
command-line check in `Daemon.pid` is the real guarantee.

Not `~/.jobq`: that persists across reboots, which is the one thing a live pid
must not do. Not the repo: `data/` gets rsynced to the pod on push.
