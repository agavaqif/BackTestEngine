"""Trading calendars — session boundaries, holidays and bar scheduling.

The engine needs to know when a session opens and closes to decide whether a
bar is the last of the day (for MOC orders, EOD liquidation and option expiry)
and to convert bar counts into year fractions for annualised metrics.

Division of labour
------------------
A subclass supplies three things — :attr:`TradingCalendar.sessions_per_year`,
:attr:`TradingCalendar.regular_hours` and :meth:`TradingCalendar.is_session` —
plus, optionally, an early-close rule. Everything else (session construction,
containment, bar scheduling, year fractions) is derived from those in the base
class, so adding a venue is a holiday table rather than a reimplementation of
the time arithmetic. That is deliberate: DST-correct local-to-UTC conversion and
right-labelled bar containment are exactly the details that get re-derived
subtly differently every time they are copied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import ClassVar

from sigmaloop.types import Timeframe, UtcDatetime
from sigmaloop.utils.timeutil import ensure_utc, zone

__all__ = ["ContinuousCalendar", "NyseCalendar", "Session", "TradingCalendar"]

#: How far :meth:`TradingCalendar.next_session` will walk before giving up.
#: The longest run of consecutive NYSE closures on record is four days
#: (September 2001), so ten covers any real holiday cluster with room to spare.
_MAX_CLOSURE_DAYS = 10


@dataclass(frozen=True, slots=True)
class Session:
    """One trading day's boundaries, in UTC."""

    session_date: date
    open_at: UtcDatetime
    close_at: UtcDatetime
    is_half_day: bool = False


class TradingCalendar(ABC):
    """Maps wall-clock time onto market sessions."""

    #: Registry key for the ``sigmaloop.calendars`` entry-point group.
    name: ClassVar[str] = "abstract"
    #: IANA zone in which :attr:`regular_hours` is expressed.
    timezone: ClassVar[str] = "UTC"

    __slots__ = ("_sessions",)

    def __init__(self) -> None:
        # One entry per distinct day asked about — a few hundred per simulated
        # year. Building a Session allocates two DST-aware conversions, and
        # `is_session_close` asks for one on every intraday bar.
        self._sessions: dict[date, Session | None] = {}

    # ---- the subclass contract --------------------------------------------- #

    #: Sessions per year, used to annualise Sharpe/volatility from bar returns.
    @property
    @abstractmethod
    def sessions_per_year(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def regular_hours(self) -> tuple[time, time]:
        """Local open and close. Equal values mean a 24-hour session."""
        raise NotImplementedError

    @abstractmethod
    def is_session(self, day: date) -> bool:
        raise NotImplementedError

    def is_half_day(self, day: date) -> bool:
        """True when ``day`` closes early. Default: never."""
        return False

    def close_time(self, day: date) -> time:
        """Local closing time on ``day``; override alongside :meth:`is_half_day`."""
        return self.regular_hours[1]

    # ---- sessions ----------------------------------------------------------- #

    def session_on(self, day: date) -> Session | None:
        """The session held on ``day``, or ``None`` when the market is shut."""
        cached = self._sessions.get(day, _UNKNOWN)
        if cached is not _UNKNOWN:
            return cached  # type: ignore[return-value]
        session = self._build_session(day) if self.is_session(day) else None
        self._sessions[day] = session
        return session

    def session_for(self, timestamp: UtcDatetime) -> Session | None:
        """The session this instant falls inside, or ``None`` outside hours."""
        moment = ensure_utc(timestamp)
        session = self.session_on(self.local_date(moment))
        if session is None or not (session.open_at <= moment <= session.close_at):
            return None
        return session

    def sessions_between(self, start: date, end: date) -> Sequence[Session]:
        """Every session in the inclusive date range ``[start, end]``."""
        out: list[Session] = []
        day = start
        while day <= end:
            session = self.session_on(day)
            if session is not None:
                out.append(session)
            day += timedelta(days=1)
        return tuple(out)

    def next_session(self, day: date) -> Session | None:
        """The first session strictly after ``day``."""
        for offset in range(1, _MAX_CLOSURE_DAYS + 1):
            session = self.session_on(day + timedelta(days=offset))
            if session is not None:
                return session
        return None

    # ---- bar-level queries --------------------------------------------------- #

    def is_session_close(self, timestamp: UtcDatetime, timeframe: Timeframe) -> bool:
        """True if a bar ending here is the final bar of its session.

        A daily-or-longer bar *is* a session, so it always closes one. Consulting
        the holiday table for those would let a disagreement between the data and
        the calendar — a venue the calendar does not model, a synthetic fixture —
        silently switch off MOC handling and end-of-day liquidation.
        """
        if not timeframe.is_intraday:
            return True
        moment = ensure_utc(timestamp)
        session = self._session_for_bar_close(moment)
        return session is not None and moment >= session.close_at

    def is_session_open(self, timestamp: UtcDatetime, timeframe: Timeframe) -> bool:
        """True if a bar ending here is the first bar of its session."""
        if not timeframe.is_intraday:
            return True
        moment = ensure_utc(timestamp)
        session = self._session_for_bar_close(moment)
        return session is not None and moment <= session.open_at + timeframe.duration

    def bar_times(self, day: date, timeframe: Timeframe) -> Iterator[UtcDatetime]:
        """Expected bar-close timestamps for one session.

        Used to detect gaps: a symbol missing a bar the calendar expects is a
        data problem, not a holiday, and is surfaced as a warning.
        """
        session = self.session_on(day)
        if session is None:
            return
        if not timeframe.is_intraday:
            yield session.close_at
            return
        width = timeframe.duration
        moment = session.open_at + width
        while moment < session.close_at:
            yield moment
            moment += width
        # The last bar is truncated whenever the session is not a whole number
        # of timeframes long: NYSE's 6.5 hours against H1 leaves a half-hour
        # stub, and against H4 leaves a two-and-a-half-hour one. The feed still
        # emits it, and it is the bar the closing auction prints into, so a gap
        # check that did not expect it would ignore a missing close — or flag
        # the real one as a surprise.
        yield session.close_at

    def year_fraction(self, start: UtcDatetime, end: UtcDatetime) -> float:
        """Elapsed time in years — the denominator of CAGR and borrow accrual.

        Measured in *trading* time, not wall-clock: a fully contained session
        counts as one, a partly covered one as its fraction, and the overnight
        gap as nothing. That is what keeps an hourly backtest from being
        annualised as if the market never closed, and it degenerates to
        ``days / 365`` on a 24-hour calendar, which is the right answer there.
        """
        lo, hi = ensure_utc(start), ensure_utc(end)
        if hi <= lo:
            return 0.0
        elapsed = 0.0
        for session in self.sessions_between(self.local_date(lo), self.local_date(hi)):
            covered_from = max(session.open_at, lo)
            covered_to = min(session.close_at, hi)
            if covered_to <= covered_from:
                continue
            length = (session.close_at - session.open_at).total_seconds()
            elapsed += (covered_to - covered_from).total_seconds() / length if length else 1.0
        return elapsed / self.sessions_per_year

    # ---- helpers -------------------------------------------------------------- #

    def local_date(self, timestamp: UtcDatetime) -> date:
        """The exchange-local calendar date of a UTC instant."""
        return ensure_utc(timestamp).astimezone(zone(self.timezone)).date()

    def _session_for_bar_close(self, moment: UtcDatetime) -> Session | None:
        """Session whose ``(open, close]`` window contains a right-labelled close.

        Half-open at the start and closed at the end because bars are labelled
        with the instant they closed: a bar stamped at the opening bell belongs
        to the *previous* session, and one stamped at the closing bell belongs to
        this one. Checking yesterday too is what makes a session that runs past
        local midnight (or a 24-hour calendar) resolve correctly.
        """
        day = self.local_date(moment)
        for candidate in (day, day - timedelta(days=1)):
            session = self.session_on(candidate)
            if session is not None and session.open_at < moment <= session.close_at:
                return session
        return None

    def _build_session(self, day: date) -> Session:
        open_time, _ = self.regular_hours
        close_time = self.close_time(day)
        tz = zone(self.timezone)
        # Combining in the local zone and converting afterwards is what makes
        # 09:30 ET mean 13:30Z in summer and 14:30Z in winter, rather than a
        # fixed offset that is wrong for half the year.
        open_at = datetime.combine(day, open_time, tzinfo=tz).astimezone(UTC)
        close_day = day + timedelta(days=1) if close_time <= open_time else day
        close_at = datetime.combine(close_day, close_time, tzinfo=tz).astimezone(UTC)
        return Session(
            session_date=day,
            open_at=open_at,
            close_at=close_at,
            is_half_day=self.is_half_day(day),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}()"


#: Distinguishes "not cached" from a cached ``None`` (a non-session day).
_UNKNOWN: object = object()


# --------------------------------------------------------------------------- #
# US equities
# --------------------------------------------------------------------------- #

_NYSE_OPEN = time(9, 30)
_NYSE_CLOSE = time(16, 0)
_NYSE_EARLY_CLOSE = time(13, 0)

#: Closures with no rule behind them: the September 11 attacks, hurricane Sandy,
#: and national days of mourning. They cannot be derived, only listed.
_UNSCHEDULED_CLOSURES: frozenset[date] = frozenset(
    {
        date(2001, 9, 11),
        date(2001, 9, 12),
        date(2001, 9, 13),
        date(2001, 9, 14),
        date(2004, 6, 11),  # Ronald Reagan
        date(2007, 1, 2),  # Gerald Ford
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),
        date(2018, 12, 5),  # George H. W. Bush
        date(2025, 1, 9),  # Jimmy Carter
    }
)

#: Federal holidays the NYSE adopted later than the rest of its table.
_MLK_FROM_YEAR = 1998
_JUNETEENTH_FROM_YEAR = 2022


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (Mon == 0) of a month, 1-based."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The final ``weekday`` of a month."""
    following = date(year + month // 12, month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(day: date) -> date:
    """Weekday a fixed-date holiday is observed on.

    A Sunday holiday rolls forward to Monday and a Saturday one back to Friday.
    New Year's Day is the exception and is handled by its caller: a Saturday
    January 1 closes nothing, because the Friday before it belongs to the
    previous year.
    """
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nyse_holidays(year: int) -> frozenset[date]:
    """Full-day NYSE closures in one year."""
    days: set[date] = set()

    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:  # a Saturday New Year is simply not observed
        days.add(_observed(new_year))
    if year >= _MLK_FROM_YEAR:
        days.add(_nth_weekday(year, 1, 0, 3))
    days.add(_nth_weekday(year, 2, 0, 3))  # Washington's Birthday
    days.add(_easter(year) - timedelta(days=2))  # Good Friday
    days.add(_last_weekday(year, 5, 0))  # Memorial Day
    if year >= _JUNETEENTH_FROM_YEAR:
        days.add(_observed(date(year, 6, 19)))
    days.add(_observed(date(year, 7, 4)))
    days.add(_nth_weekday(year, 9, 0, 1))  # Labor Day
    days.add(_nth_weekday(year, 11, 3, 4))  # Thanksgiving
    days.add(_observed(date(year, 12, 25)))

    days.update(d for d in _UNSCHEDULED_CLOSURES if d.year == year)
    return frozenset(d for d in days if d.weekday() < 5)


def _nyse_half_days(year: int) -> frozenset[date]:
    """Days the NYSE closes at 13:00 ET."""
    days: set[date] = set()
    # The day after Thanksgiving.
    days.add(_nth_weekday(year, 11, 3, 4) + timedelta(days=1))
    # July 3, but only when both it and Independence Day are weekdays: if July 4
    # falls at a weekend, July 3 is either the observed holiday or a Saturday.
    third, fourth = date(year, 7, 3), date(year, 7, 4)
    if third.weekday() < 5 and fourth.weekday() < 5:
        days.add(third)
    # Christmas Eve, when it falls Monday to Thursday. A Friday December 24 is
    # itself the observed holiday, because Christmas is then a Saturday.
    eve = date(year, 12, 24)
    if eve.weekday() < 4:
        days.add(eve)
    holidays = _nyse_holidays(year)
    return frozenset(d for d in days if d.weekday() < 5 and d not in holidays)


class NyseCalendar(TradingCalendar):
    """US equity/options calendar: 09:30-16:00 ET, NYSE holidays, half days.

    Holidays are computed per year from their rules rather than shipped as a
    table, so the calendar keeps working past whatever date a table would have
    stopped at. The handful of closures that follow no rule — September 2001,
    hurricane Sandy, days of mourning — are listed explicitly.
    """

    name: ClassVar[str] = "nyse"
    timezone: ClassVar[str] = "America/New_York"

    __slots__ = ("_half_days", "_holidays")

    def __init__(self) -> None:
        super().__init__()
        self._holidays: dict[int, frozenset[date]] = {}
        self._half_days: dict[int, frozenset[date]] = {}

    @property
    def sessions_per_year(self) -> float:
        return 252.0

    @property
    def regular_hours(self) -> tuple[time, time]:
        return (_NYSE_OPEN, _NYSE_CLOSE)

    def is_session(self, day: date) -> bool:
        return day.weekday() < 5 and day not in self._holidays_in(day.year)

    def is_half_day(self, day: date) -> bool:
        return day in self._half_days_in(day.year)

    def close_time(self, day: date) -> time:
        return _NYSE_EARLY_CLOSE if self.is_half_day(day) else _NYSE_CLOSE

    def holidays(self, year: int) -> frozenset[date]:
        """Full-day closures in ``year`` — exposed for inspection and tests."""
        return self._holidays_in(year)

    def _holidays_in(self, year: int) -> frozenset[date]:
        cached = self._holidays.get(year)
        if cached is None:
            cached = _nyse_holidays(year)
            self._holidays[year] = cached
        return cached

    def _half_days_in(self, year: int) -> frozenset[date]:
        cached = self._half_days.get(year)
        if cached is None:
            cached = _nyse_half_days(year)
            self._half_days[year] = cached
        return cached


class ContinuousCalendar(TradingCalendar):
    """24/7 calendar — every day is a session. Crypto, FX, and unit tests."""

    name: ClassVar[str] = "continuous"
    timezone: ClassVar[str] = "UTC"

    __slots__ = ()

    @property
    def sessions_per_year(self) -> float:
        return 365.0

    @property
    def regular_hours(self) -> tuple[time, time]:
        # Equal bounds: midnight to the following midnight, i.e. no closed time.
        return (time(0, 0), time(0, 0))

    def is_session(self, day: date) -> bool:
        return True
