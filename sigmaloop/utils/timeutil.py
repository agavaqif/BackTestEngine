"""Timezone and epoch conversion helpers.

Single rule, enforced at every boundary: **all datetimes are timezone-aware
UTC**. Naive datetimes are rejected rather than assumed, because the assumption
is wrong roughly half the time and the resulting off-by-hours errors are nearly
invisible in aggregate results.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sigmaloop.errors import ValidationError
from sigmaloop.types import EpochNanos, Timeframe, UtcDatetime

__all__ = [
    "bar_duration",
    "date_to_utc",
    "ensure_utc",
    "floor_to_timeframe",
    "from_epoch_ns",
    "localize",
    "parse_timestamp",
    "to_epoch_ns",
    "zone",
]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NS_PER_US = 1_000
_NS_PER_S = 1_000_000_000

#: ``ZoneInfo`` construction reads and parses a tzdata file; the CSV provider
#: calls it once per file, so a tiny memo keeps that off the hot path.
_ZONE_CACHE: dict[str, ZoneInfo] = {}


def zone(name: str) -> ZoneInfo:
    """Return a cached :class:`ZoneInfo`, with a readable error for typos."""
    cached = _ZONE_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        tz = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValidationError(
            f"Unknown timezone {name!r}. Use an IANA name such as 'America/New_York' or 'UTC'.",
            timezone=name,
        ) from exc
    _ZONE_CACHE[name] = tz
    return tz


def ensure_utc(value: datetime) -> UtcDatetime:
    """Return ``value`` as tz-aware UTC. Raises ``ValidationError`` if naive."""
    if not isinstance(value, datetime):
        raise ValidationError(
            f"Expected a datetime, got {type(value).__name__}.", value=repr(value)
        )
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValidationError(
            "Naive datetime crossed a SigmaLoop boundary. Attach a timezone "
            "(e.g. datetime(..., tzinfo=timezone.utc)) or use localize().",
            value=value.isoformat(),
        )
    return value if value.tzinfo is UTC else value.astimezone(UTC)


def to_epoch_ns(value: UtcDatetime) -> EpochNanos:
    """Exact UTC-nanosecond conversion (no float round-trip)."""
    delta = ensure_utc(value) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * _NS_PER_S + delta.microseconds * _NS_PER_US


def from_epoch_ns(value: EpochNanos) -> UtcDatetime:
    """Inverse of :func:`to_epoch_ns`.

    ``datetime`` resolves to microseconds, so sub-microsecond precision present
    in tick data is truncated here. Columnar paths keep the raw int64.
    """
    return _EPOCH + timedelta(microseconds=value // _NS_PER_US)


def parse_timestamp(value: str, fmt: str | None = None, tz: str = "UTC") -> UtcDatetime:
    """Parse a timestamp string from a data file and convert to UTC.

    ``tz`` applies only when the parsed value is naive; an explicit offset in
    the string always wins.
    """
    text = value.strip()
    if fmt is not None:
        # Naive by design: an offset-free format is localised below via ``tz``.
        parsed = datetime.strptime(text, fmt)  # noqa: DTZ007
    else:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValidationError(
                f"Could not parse {value!r} as a timestamp. Supply "
                "CsvProviderConfig.timestamp_format for non-ISO layouts.",
                value=value,
            ) from exc
    return localize(parsed, tz)


def localize(value: datetime, tz: str) -> UtcDatetime:
    """Interpret a naive datetime as being in ``tz``, then convert to UTC.

    The CSV provider's entry point: exchange files are usually naive local time.
    Already-aware values pass through unchanged (converted to UTC), so callers
    need not branch on whether their source carries an offset.
    """
    if value.tzinfo is not None:
        return ensure_utc(value)
    return value.replace(tzinfo=zone(tz)).astimezone(UTC)


def bar_duration(timeframe: Timeframe) -> timedelta:
    return timeframe.duration


def floor_to_timeframe(value: UtcDatetime, timeframe: Timeframe) -> UtcDatetime:
    """Truncate to the start of the containing bar.

    Fixed-width frames floor against the UTC epoch. ``W1`` floors to Monday
    00:00 UTC and ``MO1`` to the first of the month, because those periods are
    calendar-defined rather than a fixed number of nanoseconds.
    """
    moment = ensure_utc(value)
    if timeframe is Timeframe.MO1:
        return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if timeframe is Timeframe.W1:
        midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())
    width_ns = int(timeframe.duration.total_seconds()) * _NS_PER_S
    ns = to_epoch_ns(moment)
    return from_epoch_ns(ns - ns % width_ns)


def date_to_utc(value: date, hour: int = 0, minute: int = 0) -> UtcDatetime:
    return datetime(value.year, value.month, value.day, hour, minute, tzinfo=UTC)
