"""
What a problem is, to everything that is not the problem itself.

The CLI, the loader and the trainer each start from a problem's *name*, and
`Problem.named` turns it into the one instance that answers for that package.
Holding a `Problem` types and completes, where a module looked up by name gives
back `Any` and hides that the four packages answer the same questions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from torch import Tensor
from typing import ClassVar

from ..net import DimensionlessValue
from ..train import Objective


class Problem(ABC):
    """
    One problem: the net it trains, how to make one, and how to score it.

    Subclasses live in each `pinn/problems/<name>/__init__.py`, one instance each,
    and forward to their package's module functions. The forwarding is written out
    rather than bound with `staticmethod`, so the signature a caller gets is the
    one this class promises.
    """

    _instances: ClassVar[dict[str, Problem]] = {}
    """Every problem by name, filled in as each subclass is defined."""

    name: str
    """As `--problem` takes it, and as a release asset prefixes it."""

    net: type[DimensionlessValue]
    """The class of net this problem trains, which is what loads a state dict."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        Problem._instances[cls.name] = cls()

    @classmethod
    def named(cls, name: str) -> Problem:
        """
        The problem that answers to `name`, which is what `--problem` carries and
        what a release asset is prefixed with.
        """
        return Problem._instances[name]

    @classmethod
    def names(cls) -> list[str]:
        """
        Every problem's name, sorted. Registration order is whatever the imports
        happened to be, and one package imports another to graft off it, so
        sorting is what keeps `--problem` listing the same way twice.
        """
        return sorted(Problem._instances)

    @abstractmethod
    def init_model(
        self, state: dict | None = None, topology: str | None = None
    ) -> DimensionlessValue:
        """
        A net to start training from: fresh at `topology`, adapted from an existing
        `state`, or both, which adapts the source into the target shape.
        """

    @abstractmethod
    def objective(self, batch: int = 1024, device: str = "cpu") -> Objective:
        """This problem packaged for the trainer: its own sampling, scored by its loss."""

    @abstractmethod
    def draw(self, batch: int, device: str = "cpu") -> tuple:
        """
        One step's sample points, in `loss`'s argument order. Split out of
        `objective` so a captured cuda graph can hold them as fixed buffers.
        """

    @abstractmethod
    def loss(self, value: DimensionlessValue, *args: object) -> Tensor:
        """
        The loss, given a net, one `draw()`'s tensors, and the iteration number
        last (`None` keeps it silent). What the tensors are differs per problem,
        so they stay variadic here; `train_graphed` passes them straight through.
        """
