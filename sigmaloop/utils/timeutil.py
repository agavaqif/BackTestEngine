"""Timezone and epoch conversion helpers.

Single rule, enforced at every boundary: **all datetimes are timezone-aware
UTC**. Naive datetimes are rejected rather than assumed, because the assumption
is wrong roughly half the time and the resulting off-by-hours errors are nearly
invisible in aggregate results.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sigmaloop.types import EpochNanos, Timeframe, UtcDatetime

__all__ = [
    "ensure_utc",
    "to_epoch_ns",
    "from_epoch_ns",
    "parse_timestamp",
    "localize",
    "bar_duration",
    "floor_to_timeframe",
    "date_to_utc",
]


def ensure_utc(value: datetime) -> UtcDatetime:
    """Return ``value`` as tz-aware UTC. Raises ``ValidationError`` if naive."""
    raise NotImplementedError


def to_epoch_ns(value: UtcDatetime) -> EpochNanos:
    raise NotImplementedError


def from_epoch_ns(value: EpochNanos) -> UtcDatetime:
    raise NotImplementedError


def parse_timestamp(value: str, fmt: str | None = None, tz: str = "UTC") -> UtcDatetime:
    """Parse a timestamp string from a data file and convert to UTC."""
    raise NotImplementedError


def localize(value: datetime, tz: str) -> UtcDatetime:
    """Interpret a naive datetime as being in ``tz``, then convert to UTC.

    The CSV provider's entry point: exchange files are usually naive local time.
    """
    raise NotImplementedError


def bar_duration(timeframe: Timeframe) -> timedelta:
    raise NotImplementedError


def floor_to_timeframe(value: UtcDatetime, timeframe: Timeframe) -> UtcDatetime:
    """Truncate to the start of the containing bar."""
    raise NotImplementedError


def date_to_utc(value: date, hour: int = 0, minute: int = 0) -> UtcDatetime:
    raise NotImplementedError
