"""Simulation clock — the single source of truth for "now".

Every component that needs the current time asks the clock. Nothing calls
``datetime.now()``; a backtest has no wall-clock present, and mixing the two is
how "it worked in backtest" bugs are born.

The clock also owns the lookahead guard: :meth:`SimulationClock.assert_visible`
is what :class:`~sigmaloop.engine.context.RunContext` calls before serving any
timestamped data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sigmaloop.data.calendar import TradingCalendar
from sigmaloop.types import Timeframe, UtcDatetime

__all__ = ["Clock", "SimulationClock", "ClockState"]


@dataclass(frozen=True, slots=True)
class ClockState:
    """Immutable snapshot of the clock, safe to hand to observers."""

    now: UtcDatetime
    bar_index: int
    is_warmup: bool
    is_session_close: bool
    is_last_bar: bool


class Clock(ABC):
    """Read-only time source."""

    @property
    @abstractmethod
    def now(self) -> UtcDatetime:
        """End timestamp of the bar being processed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bar_index(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_warmup(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assert_visible(self, timestamp: UtcDatetime, what: str = "data") -> None:
        """Raise :class:`~sigmaloop.errors.LookaheadViolationError` if
        ``timestamp`` is in the future relative to :attr:`now`."""
        raise NotImplementedError


class SimulationClock(Clock):
    """Advanced by the engine once per snapshot.

    Also converts bar counts into year fractions via the trading calendar,
    which is what annualised metrics (CAGR, Sharpe) depend on. Using 252 as a
    hardcoded constant breaks the moment someone runs an hourly backtest, so
    the conversion lives here.
    """

    __slots__ = ("_now", "_bar_index", "_warmup_bars", "_timeframe", "_calendar", "_total_bars")

    def __init__(
        self,
        timeframe: Timeframe,
        calendar: TradingCalendar,
        warmup_bars: int = 0,
        total_bars: int | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def now(self) -> UtcDatetime:
        raise NotImplementedError

    @property
    def bar_index(self) -> int:
        raise NotImplementedError

    @property
    def is_warmup(self) -> bool:
        raise NotImplementedError

    @property
    def is_last_bar(self) -> bool:
        raise NotImplementedError

    def advance(self, timestamp: UtcDatetime) -> None:
        """Move to the next bar. Rejects non-monotonic timestamps, which would
        indicate a mis-sorted feed and silently corrupt the whole run."""
        raise NotImplementedError

    def assert_visible(self, timestamp: UtcDatetime, what: str = "data") -> None:
        raise NotImplementedError

    def state(self) -> ClockState:
        raise NotImplementedError

    def bars_per_year(self) -> float:
        """Annualisation factor for the configured timeframe and calendar."""
        raise NotImplementedError

    def year_fraction_since(self, start: UtcDatetime) -> float:
        raise NotImplementedError
