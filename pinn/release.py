"""
Released models, loaded straight from GitHub.

No clone, no `data/` folder, no curl: `load("two_arm")` fetches the asset a release
ships for that problem, and `bind` turns it into a model that speaks your experiment's
units. The asset *name* is read off the release rather than hardcoded, since it carries
the topology and the topology changes with every new champion.
"""

import json
import torch
import torch.nn as nn

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.request import urlopen

from .net import DimensionlessValue
from .problems import Problem


REPO = "tokahuke/pinn"
"""The GitHub repository whose releases carry the trained nets."""


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

    with urlopen(f"https://api.github.com/repos/{REPO}/releases/{reference}") as reply:
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

    url = asset_url(problem, tag)
    *_, release_tag, name = url.split("/")
    path = Path(torch.hub.get_dir()) / "pinn" / release_tag / name

    if path.exists() is False:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path)

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

    print(f"ok: two_arm splits {testing.item():.2f} of traffic to treatment")
