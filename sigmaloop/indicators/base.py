"""Custom indicator framework (Functional requirement 2).

Dual evaluation model
---------------------
Every indicator implements BOTH:

* :meth:`Indicator.update` — O(1) incremental update from one new bar. Used
  inside the engine loop, where an O(window) recompute per bar per symbol would
  dominate runtime.
* :meth:`Indicator.compute` — vectorised evaluation over a whole
  :class:`~sigmaloop.domain.bar.BarSeries`. Used for warm-up, screeners and
  research, where numpy beats any Python loop by orders of magnitude.

Requiring both is a deliberate cost imposed on indicator authors:
:class:`RollingIndicator` supplies the incremental half for the common
fixed-window case, and :meth:`Indicator.compute` has a correct (if slow)
default that replays :meth:`update`. Authors override what they need.

Indicators are stateful and single-instrument. The engine holds one instance
per (indicator, instrument) pair inside an :class:`IndicatorSet`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from sigmaloop.domain.bar import Bar, BarSeries
from sigmaloop.types import InstrumentId, ParamDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "IndicatorSpec",
    "Indicator",
    "RollingIndicator",
    "CompositeIndicator",
    "IndicatorSet",
]

TValue = TypeVar("TValue")


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    """Declarative request for an indicator instance.

    Strategies return these from ``declare_indicators()``; the engine
    instantiates and wires them, so a strategy never manages indicator state or
    subscription order itself.
    """

    name: str
    params: ParamDict
    #: ``None`` means "the strategy's primary instrument"; portfolio mode
    #: instantiates one copy per universe member.
    instrument_id: InstrumentId | None = None
    #: Optional key for lookup; defaults to ``f"{name}({params})"``.
    alias: str | None = None

    @property
    def key(self) -> str:
        raise NotImplementedError


class Indicator(ABC, Generic[TValue]):
    """Base class for all indicators.

    Subclasses declare their parameters as constructor keywords and report a
    :attr:`warmup_period`; the engine refuses to trade until every declared
    indicator :attr:`is_ready`, which prevents the classic bug of acting on a
    half-filled moving average.
    """

    #: Registry key under the ``sigmaloop.indicators`` entry-point group.
    name: ClassVar[str] = "abstract"

    def __init__(self, **params: object) -> None:
        raise NotImplementedError

    # ---- contract ----------------------------------------------------------- #

    @property
    @abstractmethod
    def warmup_period(self) -> int:
        """Bars required before :attr:`value` is meaningful."""
        raise NotImplementedError

    @abstractmethod
    def update(self, bar: Bar) -> TValue | None:
        """Fold one new bar in and return the new value (``None`` if warming up).

        Must be O(1) in the window length and must not retain the ``Bar``.
        """
        raise NotImplementedError

    def compute(self, series: BarSeries) -> npt.NDArray[np.float64]:
        """Vectorised evaluation over a full series.

        Returns an array the same length as ``series``, NaN-padded through the
        warm-up region. The default implementation replays :meth:`update`;
        override it whenever a numpy formulation exists.
        """
        raise NotImplementedError

    # ---- state -------------------------------------------------------------- #

    @property
    @abstractmethod
    def value(self) -> TValue | None:
        """Most recent value, or ``None`` while warming up."""
        raise NotImplementedError

    @property
    def is_ready(self) -> bool:
        """True once :attr:`warmup_period` bars have been consumed."""
        raise NotImplementedError

    @property
    def dependencies(self) -> Sequence[IndicatorSpec]:
        """Other indicators this one consumes; resolved topologically."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called between walk-forward folds and sweep runs."""
        raise NotImplementedError

    @property
    def params(self) -> ParamDict:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class RollingIndicator(Indicator[float]):
    """Convenience base for fixed-window indicators over a single price field.

    Maintains the window ``deque`` and warm-up bookkeeping so subclasses only
    implement the aggregation. Where an O(1) incremental update exists (running
    sum for SMA, Welford for standard deviation), subclasses should maintain it
    rather than re-scanning the window.
    """

    def __init__(self, period: int, field: str = "close", **params: object) -> None:
        raise NotImplementedError

    @property
    def period(self) -> int:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    @property
    def window(self) -> deque[float]:
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, window: deque[float], new_value: float, evicted: float | None) -> float:
        """Produce the new value given the window and the just-shifted values."""
        raise NotImplementedError

    def update(self, bar: Bar) -> float | None:
        raise NotImplementedError

    @property
    def value(self) -> float | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class CompositeIndicator(Indicator[TValue]):
    """An indicator built from other indicators (e.g. MACD = EMA - EMA).

    Children are updated before the parent on each bar; :attr:`warmup_period` is
    the max over children plus any additional smoothing this level applies.
    """

    def __init__(self, children: Sequence[Indicator[object]], **params: object) -> None:
        raise NotImplementedError

    @property
    def children(self) -> Sequence[Indicator[object]]:
        raise NotImplementedError

    @property
    def warmup_period(self) -> int:
        raise NotImplementedError

    def update(self, bar: Bar) -> TValue | None:
        raise NotImplementedError

    @property
    def value(self) -> TValue | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class IndicatorSet:
    """All indicator instances for one run, keyed by (instrument, alias).

    Owns update ordering (dependencies first) and exposes the strategy-facing
    lookup ``ctx.indicator("sma_20")``. Updating through the set — rather than
    letting strategies call ``update`` themselves — guarantees each indicator
    sees each bar exactly once, which is what makes results reproducible.
    """

    __slots__ = ("_by_instrument", "_order", "_warmup")

    def __init__(self) -> None:
        raise NotImplementedError

    def add(self, spec: IndicatorSpec, instance: Indicator[object]) -> None:
        raise NotImplementedError

    def get(self, alias: str, instrument_id: InstrumentId | None = None) -> Indicator[object]:
        raise NotImplementedError

    def update_all(self, bar: Bar) -> None:
        """Update every indicator subscribed to ``bar.instrument_id``."""
        raise NotImplementedError

    def warmup_bars(self) -> int:
        """Max warm-up across the set — how much history the feed must prepend."""
        raise NotImplementedError

    def all_ready(self, instrument_id: InstrumentId) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
