"""Trading calendars — session boundaries, holidays and bar scheduling.

The engine needs to know when a session opens and closes to decide whether a
bar is the last of the day (for MOC orders, EOD liquidation and option expiry)
and to convert bar counts into year fractions for annualised metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, time

from sigmaloop.types import Timeframe, UtcDatetime

__all__ = ["Session", "TradingCalendar", "NyseCalendar", "ContinuousCalendar"]


@dataclass(frozen=True, slots=True)
class Session:
    """One trading day's boundaries, in UTC."""

    session_date: date
    open_at: UtcDatetime
    close_at: UtcDatetime
    is_half_day: bool = False


class TradingCalendar(ABC):
    """Maps wall-clock time onto market sessions."""

    #: Sessions per year, used to annualise Sharpe/volatility from bar returns.
    @property
    @abstractmethod
    def sessions_per_year(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def regular_hours(self) -> tuple[time, time]:
        raise NotImplementedError

    @abstractmethod
    def is_session(self, day: date) -> bool:
        raise NotImplementedError

    @abstractmethod
    def session_for(self, timestamp: UtcDatetime) -> Session | None:
        raise NotImplementedError

    @abstractmethod
    def sessions_between(self, start: date, end: date) -> Sequence[Session]:
        raise NotImplementedError

    def is_session_close(self, timestamp: UtcDatetime, timeframe: Timeframe) -> bool:
        """True if a bar ending here is the final bar of its session."""
        raise NotImplementedError

    def next_session(self, day: date) -> Session | None:
        raise NotImplementedError

    def bar_times(self, day: date, timeframe: Timeframe) -> Iterator[UtcDatetime]:
        """Expected bar-close timestamps for one session.

        Used to detect gaps: a symbol missing a bar the calendar expects is a
        data problem, not a holiday, and is surfaced as a warning.
        """
        raise NotImplementedError

    def year_fraction(self, start: UtcDatetime, end: UtcDatetime) -> float:
        """Elapsed time in years — the denominator of CAGR and borrow accrual."""
        raise NotImplementedError


class NyseCalendar(TradingCalendar):
    """US equity/options calendar: 09:30-16:00 ET, NYSE holidays, half days."""

    __slots__ = ("_holidays", "_half_days")

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def sessions_per_year(self) -> float:
        raise NotImplementedError

    @property
    def regular_hours(self) -> tuple[time, time]:
        raise NotImplementedError

    def is_session(self, day: date) -> bool:
        raise NotImplementedError

    def session_for(self, timestamp: UtcDatetime) -> Session | None:
        raise NotImplementedError

    def sessions_between(self, start: date, end: date) -> Sequence[Session]:
        raise NotImplementedError


class ContinuousCalendar(TradingCalendar):
    """24/7 calendar — every day is a session. Crypto, FX, and unit tests."""

    __slots__ = ()

    @property
    def sessions_per_year(self) -> float:
        raise NotImplementedError

    @property
    def regular_hours(self) -> tuple[time, time]:
        raise NotImplementedError

    def is_session(self, day: date) -> bool:
        raise NotImplementedError

    def session_for(self, timestamp: UtcDatetime) -> Session | None:
        raise NotImplementedError

    def sessions_between(self, start: date, end: date) -> Sequence[Session]:
        raise NotImplementedError
