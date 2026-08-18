"""
Released models, loaded straight from GitHub.

No clone, no `data/` folder, no curl: `load("two_arm")` fetches the asset a release
ships for that problem, and `bind` turns it into a model that speaks your experiment's
units. The asset *name* is read off the release rather than hardcoded, since it carries
the topology and the topology changes with every new champion.
"""

import json
import os
import time
import torch
import torch.nn as nn

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.request import Request, urlopen

from .net import DimensionlessValue
from .problems import Problem


REPO = "tokahuke/pinn"
"""The GitHub repository whose releases carry the trained nets."""

LATEST_TTL = 4.0 * 3600.0
"""
How long a "latest" resolution answers from disk before the API is asked
again, in seconds. Four hours keeps a process storm to one call per problem
per window against the anonymous 60/hr limit, and a re-pointed release still
reaches everyone same-day.
"""


@dataclass
class LoadedModule:
    """
    A released net, with no way to read a policy off it until it is bound.

    The net inside speaks its own chart, where a mean of 1.0 means whatever
    `sigma sqrt(rho)` says it means. Wrapping it keeps `policy` out of reach of anyone
    who has not yet said which experiment they are running; the trainer, the arena and
    the probes take `dimensionless` and speak the chart.
    """

    dimensionless: DimensionlessValue
    """The net as trained, reading and returning numbers on its own chart."""

    def bind(self, **kwargs: float) -> nn.Module:
        """
        The model tied to one experiment: `rho` discounts a single observation, `sigma`
        scales its noise, the drift problems add `eta`. Each problem's `bind` names
        what it needs, so a missing or foreign parameter raises here.
        """
        return self.dimensionless.bind(**kwargs)


@cache
def asset_url(problem: str, tag: str = "latest") -> str:
    """
    Where a release keeps its model for one problem.

    `tag` is a release codename (kb/releases.md) or "latest".
    """
    reference = "latest" if tag == "latest" else f"tags/{tag}"
    request = Request(f"https://api.github.com/repos/{REPO}/releases/{reference}")

    # A token lifts the API limit from 60/hr per shared IP to 5,000/hr; without
    # one the call goes out anonymous, same as before.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if token is not None and len(token) > 0:
        request.add_header("Authorization", f"Bearer {token}")

    with urlopen(request) as reply:
        release = json.load(reply)

    for asset in release["assets"]:
        if asset["name"].startswith(f"{problem}."):
            return asset["browser_download_url"]

    raise LookupError(f"release {release['tag_name']} ships no {problem} model")


@cache
def load(problem: str, tag: str = "latest") -> LoadedModule:
    """
    The released net for a problem, downloaded once and cached by torch.hub. It
    arrives *dimensionless*, which is to say inert, so bind it to an experiment:
    `load("two_arm").bind(rho=0.001, sigma=50.0)`, `eta=` too for the drift pair. The
    trainer, arena and probes want the net itself, at `load(problem).dimensionless`.
    """
    # Resolved before the network call, so a misspelled problem fails on the spot
    # rather than as a missing asset.
    net = Problem.named(problem).net
    hub = Path(torch.hub.get_dir()) / "pinn"

    # A pinned codename names an immutable release, so a cached asset answers
    # without the API round trip (rate-limited at 60/hr unauthenticated, and the
    # one thing that breaks offline).
    if tag != "latest":
        cached = sorted((hub / tag).glob(f"{problem}.*"))

        if len(cached) > 0:
            return LoadedModule(net.load(cached[0]))

    # "latest" must ask, since re-pointing it is how a new champion reaches
    # users, but a fresh stamp answers instead: the stamp holds the tag the API
    # last resolved, and its own age is the resolution's age.
    stamp = hub / "latest-tag"

    if tag == "latest" and stamp.exists() is True:
        resolved = stamp.read_text().strip()
        cached = sorted((hub / resolved).glob(f"{problem}.*"))

        if len(cached) > 0 and time.time() - stamp.stat().st_mtime < LATEST_TTL:
            return LoadedModule(net.load(cached[0]))

    try:
        url = asset_url(problem, tag)
    except OSError as unreachable:
        # Rate-limited or offline. Serve the staleness we have, loudly: the
        # stamped tag first, then the newest cached release holding the asset.
        stamped = [stamp.read_text().strip()] if stamp.exists() is True else []
        by_age = sorted(
            (d for d in hub.glob("*") if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        for candidate in [*stamped, *(d.name for d in by_age)]:
            cached = sorted((hub / candidate).glob(f"{problem}.*"))

            if len(cached) > 0:
                print(
                    f"pinn.release: GitHub unreachable ({unreachable}); "
                    f"serving cached {cached[0].name} from release {candidate}"
                )

                return LoadedModule(net.load(cached[0]))

        raise

    *_, release_tag, name = url.split("/")
    path = hub / release_tag / name

    if path.exists() is False:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path)

    if tag == "latest":
        hub.mkdir(parents=True, exist_ok=True)
        stamp.write_text(release_tag)

    return LoadedModule(net.load(path))


if __name__ == "__main__":
    value = load("two_arm").bind(rho=0.001, sigma=50.0)

    # A lead worth chasing splits the traffic; a settled one takes all of it.
    testing = value.policy(torch.tensor([1.0]), torch.tensor([0.1]))
    settled = value.policy(torch.tensor([3.0]), torch.tensor([0.25]))

    assert 0.0 < testing.item() < 1.0, testing
    assert settled.item() == 1.0, settled

    drifting = load("two_arm_drift").bind(rho=0.001, sigma=50.0, eta=1e-3)

    assert drifting.policy(torch.tensor([1.0]), torch.tensor([0.1])).item() > 0.0

    try:
        load("two_arm").bind(rho=0.001, sigma=50.0, eta=1e-3)
    except TypeError as refused:
        print(f"as expected: a static problem takes no drift rate ({refused})")

    try:
        load("three_arm_drift")
    except LookupError as absent:
        print(f"as expected: {absent}")

    # Offline paths, with the network monkeypatched away. A fresh stamp must
    # answer without any call; an expired stamp with the API down must fall
    # back to the cache, loudly, instead of raising.
    import sys
    from urllib.error import URLError

    def refuse(*args: object, **kwargs: object) -> object:
        raise URLError("stubbed out")

    module = sys.modules[__name__]
    module.urlopen = refuse
    torch.hub.download_url_to_file = refuse
    load.cache_clear()
    asset_url.cache_clear()

    assert load("two_arm").bind(rho=0.001, sigma=50.0) is not None

    stamp = Path(torch.hub.get_dir()) / "pinn" / "latest-tag"
    os.utime(stamp, (0.0, 0.0))
    load.cache_clear()
    asset_url.cache_clear()

    assert load("two_arm").bind(rho=0.001, sigma=50.0) is not None
    print(f"ok: two_arm splits {testing.item():.2f} of traffic to treatment")
