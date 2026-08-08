"""Trading calendars: holidays, half days, DST, session boundaries, year fractions."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pytest

from sigmaloop.data.calendar import ContinuousCalendar, NyseCalendar
from sigmaloop.types import Timeframe


@pytest.fixture
def nyse() -> NyseCalendar:
    return NyseCalendar()


@pytest.fixture
def always_on() -> ContinuousCalendar:
    return ContinuousCalendar()


# --------------------------------------------------------------------------- #
# Holidays
# --------------------------------------------------------------------------- #


def test_holidays_2024(nyse: NyseCalendar) -> None:
    assert sorted(nyse.holidays(2024)) == [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # MLK
        date(2024, 2, 19),  # Washington's Birthday
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving
        date(2024, 12, 25),  # Christmas
    ]


def test_weekend_holidays_roll_to_the_nearest_weekday(nyse: NyseCalendar) -> None:
    """July 4 2021 was a Sunday and Christmas 2021 a Saturday."""
    holidays = nyse.holidays(2021)
    assert date(2021, 7, 5) in holidays, "Sunday rolls forward"
    assert date(2021, 12, 24) in holidays, "Saturday rolls back"
    assert date(2021, 7, 4) not in holidays


def test_saturday_new_year_closes_nothing(nyse: NyseCalendar) -> None:
    """January 1 2022 fell on a Saturday; the NYSE did not close December 31."""
    assert nyse.is_session(date(2021, 12, 31)) is True
    assert date(2021, 12, 31) not in nyse.holidays(2021)


def test_juneteenth_only_from_2022(nyse: NyseCalendar) -> None:
    assert nyse.is_session(date(2021, 6, 18)) is True
    assert date(2022, 6, 20) in nyse.holidays(2022), "Sunday June 19 rolls to Monday"


def test_mlk_only_from_1998(nyse: NyseCalendar) -> None:
    assert nyse.is_session(date(1997, 1, 20)) is True
    assert nyse.is_session(date(1998, 1, 19)) is False


def test_unscheduled_closures_are_known(nyse: NyseCalendar) -> None:
    """No rule generates these; September 2001 and Sandy have to be listed."""
    assert nyse.is_session(date(2001, 9, 12)) is False
    assert nyse.is_session(date(2012, 10, 30)) is False


@pytest.mark.parametrize(("year", "count"), [(2023, 250), (2024, 252), (2025, 250)])
def test_session_count_per_year(nyse: NyseCalendar, year: int, count: int) -> None:
    sessions = nyse.sessions_between(date(year, 1, 1), date(year, 12, 31))
    assert len(sessions) == count


# --------------------------------------------------------------------------- #
# Session boundaries
# --------------------------------------------------------------------------- #


def test_open_and_close_follow_daylight_saving(nyse: NyseCalendar) -> None:
    """09:30 ET is 14:30Z in winter and 13:30Z in summer; a fixed offset is wrong
    for half the year."""
    winter = nyse.session_on(date(2024, 1, 2))
    summer = nyse.session_on(date(2024, 7, 1))
    assert winter is not None and summer is not None
    assert winter.open_at == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert winter.close_at == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert summer.open_at == datetime(2024, 7, 1, 13, 30, tzinfo=UTC)
    assert summer.close_at == datetime(2024, 7, 1, 20, 0, tzinfo=UTC)


def test_half_days_close_at_one_pm(nyse: NyseCalendar) -> None:
    for day in (date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24)):
        session = nyse.session_on(day)
        assert session is not None, day
        assert session.is_half_day is True, day
    early = nyse.session_on(date(2024, 11, 29))
    assert early is not None
    assert early.close_at == datetime(2024, 11, 29, 18, 0, tzinfo=UTC), "13:00 EST"


def test_july_3_is_a_full_day_when_july_4_is_a_weekend(nyse: NyseCalendar) -> None:
    """July 4 2026 is a Saturday, so July 3 is the holiday, not a half day."""
    assert nyse.is_session(date(2026, 7, 3)) is False
    assert nyse.is_half_day(date(2026, 7, 3)) is False


def test_session_for_only_covers_regular_hours(nyse: NyseCalendar) -> None:
    assert nyse.session_for(datetime(2024, 1, 2, 16, 0, tzinfo=UTC)) is not None
    assert nyse.session_for(datetime(2024, 1, 2, 23, 0, tzinfo=UTC)) is None
    assert nyse.session_for(datetime(2024, 1, 1, 16, 0, tzinfo=UTC)) is None


def test_next_session_skips_the_holiday(nyse: NyseCalendar) -> None:
    following = nyse.next_session(date(2024, 11, 27))
    assert following is not None
    assert following.session_date == date(2024, 11, 29), "Thanksgiving is skipped"
    assert following.is_half_day is True


# --------------------------------------------------------------------------- #
# Bar-level queries
# --------------------------------------------------------------------------- #


def test_intraday_session_close_is_the_closing_bell(nyse: NyseCalendar) -> None:
    assert nyse.is_session_close(datetime(2024, 1, 2, 21, 0, tzinfo=UTC), Timeframe.M1)
    assert not nyse.is_session_close(datetime(2024, 1, 2, 20, 59, tzinfo=UTC), Timeframe.M1)
    # Right-labelled: the 14:30Z stamp closes the *previous* session's last
    # minute of nothing, so it is not this session's close either.
    assert not nyse.is_session_close(datetime(2024, 1, 2, 14, 30, tzinfo=UTC), Timeframe.M1)


def test_intraday_session_open_is_the_first_bar(nyse: NyseCalendar) -> None:
    assert nyse.is_session_open(datetime(2024, 1, 2, 14, 31, tzinfo=UTC), Timeframe.M1)
    assert not nyse.is_session_open(datetime(2024, 1, 2, 14, 32, tzinfo=UTC), Timeframe.M1)


def test_daily_bars_always_close_a_session(nyse: NyseCalendar) -> None:
    """A daily bar IS a session. Consulting the holiday table here would let a
    calendar/data disagreement silently disable EOD liquidation."""
    for stamp in (
        datetime(2024, 1, 2, 21, 0, tzinfo=UTC),
        datetime(2024, 1, 6, 0, 0, tzinfo=UTC),  # a Saturday-stamped daily bar
    ):
        assert nyse.is_session_close(stamp, Timeframe.D1) is True
        assert nyse.is_session_open(stamp, Timeframe.D1) is True


def test_bar_times_walks_the_session(nyse: NyseCalendar) -> None:
    stamps = list(nyse.bar_times(date(2024, 1, 2), Timeframe.M30))
    assert stamps[0] == datetime(2024, 1, 2, 15, 0, tzinfo=UTC)
    assert stamps[-1] == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert len(stamps) == 13, "6.5 hours of half-hour bars"

    assert list(nyse.bar_times(date(2024, 1, 2), Timeframe.D1)) == [
        datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    ]
    assert list(nyse.bar_times(date(2024, 1, 1), Timeframe.M30)) == [], "holiday"


# --------------------------------------------------------------------------- #
# Annualisation
# --------------------------------------------------------------------------- #


def test_a_full_year_is_one_year(nyse: NyseCalendar) -> None:
    fraction = nyse.year_fraction(
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert fraction == pytest.approx(252 / 252)


def test_year_fraction_counts_trading_time_only(nyse: NyseCalendar) -> None:
    """A weekend adds nothing, and half a session counts as half."""
    friday_close = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
    monday_close = datetime(2024, 1, 8, 21, 0, tzinfo=UTC)
    assert nyse.year_fraction(friday_close, monday_close) == pytest.approx(1 / 252)

    midday = datetime(2024, 1, 2, 17, 45, tzinfo=UTC)  # 3h15m into a 6h30m session
    close = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert nyse.year_fraction(midday, close) == pytest.approx(0.5 / 252)


def test_year_fraction_is_zero_for_an_empty_or_reversed_span(nyse: NyseCalendar) -> None:
    moment = datetime(2024, 1, 2, 16, 0, tzinfo=UTC)
    assert nyse.year_fraction(moment, moment) == 0.0
    assert nyse.year_fraction(moment, moment.replace(hour=15)) == 0.0


# --------------------------------------------------------------------------- #
# 24/7
# --------------------------------------------------------------------------- #


def test_continuous_calendar_never_closes(always_on: ContinuousCalendar) -> None:
    assert always_on.is_session(date(2024, 1, 1)) is True
    assert always_on.is_session(date(2024, 1, 6)) is True  # a Saturday
    assert always_on.regular_hours == (time(0, 0), time(0, 0))

    session = always_on.session_on(date(2024, 1, 6))
    assert session is not None
    assert session.open_at == datetime(2024, 1, 6, tzinfo=UTC)
    assert session.close_at == datetime(2024, 1, 7, tzinfo=UTC)


def test_continuous_midnight_closes_the_day(always_on: ContinuousCalendar) -> None:
    assert always_on.is_session_close(datetime(2024, 1, 7, tzinfo=UTC), Timeframe.H1)
    assert not always_on.is_session_close(datetime(2024, 1, 6, 23, 0, tzinfo=UTC), Timeframe.H1)


def test_continuous_year_fraction_is_calendar_time(always_on: ContinuousCalendar) -> None:
    year = always_on.year_fraction(
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
    )
    assert year == pytest.approx(365 / 365)

    half_day = always_on.year_fraction(
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    )
    assert half_day == pytest.approx(0.5 / 365)


def test_the_closing_bar_is_expected_even_when_it_is_a_stub(nyse: NyseCalendar) -> None:
    """NYSE runs 6.5 hours, so hourly bars leave a half-hour stub at the close.
    Gap detection that skipped it would ignore a missing closing auction — the
    one bar MOC orders and the daily mark both price against."""
    close = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)

    for timeframe in (Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1):
        stamps = list(nyse.bar_times(date(2024, 1, 2), timeframe))
        assert stamps[-1] == close, timeframe
        assert stamps.count(close) == 1, f"{timeframe} listed the close twice"
        assert stamps == sorted(stamps), timeframe


def test_a_half_day_close_is_expected_too(nyse: NyseCalendar) -> None:
    half_day = date(2024, 7, 3)
    assert nyse.is_half_day(half_day)
    session = nyse.session_on(half_day)
    assert session is not None

    stamps = list(nyse.bar_times(half_day, Timeframe.H1))

    assert stamps[-1] == session.close_at
