"""A dataset of nothing but bid/ask must still be able to drive a backtest.

Real datasets are one shape or the other: a directory of daily or minute
aggregates, or a directory of raw top-of-book. The engine only consumes Bars,
so when the second shape is all there is, the provider folds quotes into bars
rather than serving nothing.
"""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from conftest import MARCH, SAMPLES, daily_frame, request_for, utc

from sigmaloop.data.providers.csv_provider import (
    CsvDataProvider,
    CsvFileKind,
    CsvProviderConfig,
    QuotePriceBasis,
)
from sigmaloop.errors import DataIntegrityError
from sigmaloop.types import PriceSelection, Symbol, Timeframe


def provider(path: Path, cache: Path, **kwargs: object) -> CsvDataProvider:
    return CsvDataProvider(CsvProviderConfig(path=path, cache_dir=cache, **kwargs))  # type: ignore[arg-type]


def write_quotes(path: Path, day: date, entries: list[tuple[int, float, float]]) -> None:
    """``entries`` is ``(seconds past 09:30, bid, ask)``."""
    base = int(datetime(day.year, day.month, day.day, 9, 30, tzinfo=UTC).timestamp())
    pd.DataFrame(
        {
            "Ticker": "MSFT",
            "sip_timestamp": [(base + s) * 10**9 for s, _, _ in entries],
            "bid_price": [b for _, b, _ in entries],
            "ask_price": [a for _, _, a in entries],
            "bid_size": [1] * len(entries),
            "ask_size": [2] * len(entries),
        }
    ).to_csv(path, index=False)


@pytest.fixture
def quote_only(tmp_path: Path) -> Path:
    """One session of top-of-book, no OHLC anywhere."""
    root = tmp_path / "quotes_only"
    root.mkdir()
    write_quotes(
        root / "2023-03-08.csv",
        date(2023, 3, 8),
        [
            # bar (09:30, 09:31] -> stamped 09:31
            (1, 99.0, 101.0),    # mid 100  <- open
            (10, 101.0, 103.0),  # mid 102  <- high
            (59, 97.0, 99.0),    # mid 98   <- low, and the close
            # bar (09:31, 09:32] -> stamped 09:32
            (61, 103.0, 105.0),   # mid 104
            (119, 105.0, 107.0),  # mid 106
        ],
    )
    return root


# --------------------------------------------------------------------------- #
# The core case
# --------------------------------------------------------------------------- #


def test_quote_only_dataset_produces_bars(quote_only: Path, tmp_path: Path) -> None:
    p = provider(quote_only, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))

    assert len(bars) == 2, "two minutes of quotes -> two minute bars"
    first, second = bars

    # Right-labelled: the bar covering (09:30, 09:31] is stamped 09:31.
    assert first.timestamp == datetime(2023, 3, 8, 9, 31, tzinfo=UTC)
    assert second.timestamp == datetime(2023, 3, 8, 9, 32, tzinfo=UTC)

    # OHLC of the mid, hand-computed from the fixture.
    assert first.open == pytest.approx(100.0)
    assert first.high == pytest.approx(102.0)
    assert first.low == pytest.approx(98.0)
    assert first.close == pytest.approx(98.0)
    assert second.open == pytest.approx(104.0)
    assert second.close == pytest.approx(106.0)


def test_derived_bars_carry_zero_volume(quote_only: Path, tmp_path: Path) -> None:
    """Quotes are not trades. Inventing volume would feed the participation cap."""
    p = provider(quote_only, tmp_path / "c")
    assert all(b.volume == 0.0 for b in p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))


def test_derived_bars_carry_the_real_closing_book(quote_only: Path, tmp_path: Path) -> None:
    """OHLC is mid-based, but execution must still see the true bid/ask."""
    p = provider(quote_only, tmp_path / "c")
    first = next(iter(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH))))

    assert first.quote is not None
    assert first.quote.bid == pytest.approx(97.0)
    assert first.quote.ask == pytest.approx(99.0)
    assert first.price_for(PriceSelection.WORST, is_buy=True) == pytest.approx(99.0)
    assert first.price_for(PriceSelection.WORST, is_buy=False) == pytest.approx(97.0)
    assert first.price_for(PriceSelection.MID, is_buy=True) == pytest.approx(98.0)


@pytest.mark.parametrize(
    ("basis", "expected_open", "expected_close"),
    [
        (QuotePriceBasis.MID, 100.0, 98.0),
        (QuotePriceBasis.BID, 99.0, 97.0),
        (QuotePriceBasis.ASK, 101.0, 99.0),
    ],
)
def test_price_basis_selects_the_side(
    quote_only: Path, tmp_path: Path, basis: QuotePriceBasis, expected_open: float, expected_close: float
) -> None:
    p = provider(quote_only, tmp_path / f"c{basis.value}", quote_bar_price=basis)
    first = next(iter(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH))))
    assert first.open == pytest.approx(expected_open)
    assert first.close == pytest.approx(expected_close)


def test_timeframe_comes_from_the_request(quote_only: Path, tmp_path: Path) -> None:
    """Tick data has no natural spacing, so the caller names the bar width."""
    minutes = list(provider(quote_only, tmp_path / "m").stream_bars(
        request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))
    seconds = list(provider(quote_only, tmp_path / "s").stream_bars(
        request_for("MSFT", timeframe=Timeframe.S1, **MARCH)))
    assert len(minutes) == 2
    assert len(seconds) == 5, "five distinct seconds carry a quote"
    assert all(b.timeframe is Timeframe.S1 for b in seconds)


def test_columnar_path_also_works(quote_only: Path, tmp_path: Path) -> None:
    p = provider(quote_only, tmp_path / "c")
    series = p.load_series(Symbol("MSFT"), request_for("MSFT", timeframe=Timeframe.M1, **MARCH))
    assert len(series) == 2
    assert list(series.close) == pytest.approx([98.0, 106.0])
    assert list(series.volume) == [0.0, 0.0]


def test_opting_out_is_an_explicit_error(quote_only: Path, tmp_path: Path) -> None:
    p = provider(quote_only, tmp_path / "c", derive_bars_from_quotes=False)
    with pytest.raises(DataIntegrityError, match="quotes but no OHLC aggregates"):
        list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))


# --------------------------------------------------------------------------- #
# Folding must survive chunked streaming
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 2, 3, 10, 1_000_000])
def test_slicing_never_splits_a_bar(tmp_path: Path, chunk: int) -> None:
    """A slice boundary cutting through a minute would emit two half-bars."""
    root = tmp_path / "many"
    root.mkdir()
    # 10 minutes of quotes, three per minute, so a split is easy to spot.
    entries = [(m * 60 + s, 100.0 + m, 102.0 + m) for m in range(10) for s in (5, 25, 55)]
    write_quotes(root / "2023-03-08.csv", date(2023, 3, 8), entries)

    reference = provider(root, tmp_path / "ref", stream_chunk_rows=1_000_000)
    sliced = provider(root, tmp_path / f"c{chunk}", stream_chunk_rows=chunk)
    req = request_for("MSFT", timeframe=Timeframe.M1, **MARCH)

    expected = [(b.timestamp, b.open, b.high, b.low, b.close) for b in reference.stream_bars(req)]
    actual = [(b.timestamp, b.open, b.high, b.low, b.close) for b in sliced.stream_bars(req)]

    assert len(expected) == 10
    assert actual == expected
    stamps = [t for t, *_ in actual]
    assert stamps == sorted(stamps)
    assert len(stamps) == len(set(stamps)), "one bar per minute, never two halves"


# --------------------------------------------------------------------------- #
# The realistic layout: a directory of per-session files
# --------------------------------------------------------------------------- #


def test_quote_only_directory_of_session_files(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    days = [date(2023, 3, d) for d in (6, 7, 8, 9, 10)]
    for day in days:
        write_quotes(root / f"{day.isoformat()}.csv", day,
                     [(1, 99.0, 101.0), (30, 100.0, 102.0), (90, 101.0, 103.0)])

    p = provider(root, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))

    assert len(bars) == 2 * len(days), "two minute bars per session"
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
    assert {b.timestamp.date() for b in bars} == set(days)

    # Date pruning still applies to a quote-only dataset.
    one_day = request_for("MSFT", timeframe=Timeframe.M1,
                          start=utc(2023, 3, 8), end=utc(2023, 3, 8, 23, 59))
    assert [f.stem for f in p._prune_files(one_day, kind=CsvFileKind.QUOTE)] == ["2023-03-08"]
    assert {b.timestamp.date() for b in p.stream_bars(one_day)} == {date(2023, 3, 8)}


def test_warmup_works_on_a_quote_only_dataset(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    for day in (date(2023, 3, d) for d in (6, 7, 8, 9, 10)):
        write_quotes(root / f"{day.isoformat()}.csv", day, [(1, 99.0, 101.0), (90, 101.0, 103.0)])

    p = provider(root, tmp_path / "c")
    plain = request_for("MSFT", timeframe=Timeframe.M1, start=utc(2023, 3, 9), end=utc(2023, 3, 31))
    warm = request_for("MSFT", timeframe=Timeframe.M1, start=utc(2023, 3, 9), end=utc(2023, 3, 31), warmup_bars=3)

    assert len(p.load_series(Symbol("MSFT"), warm)) == len(p.load_series(Symbol("MSFT"), plain)) + 3


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def test_aggregates_win_when_a_directory_holds_both(tmp_path: Path, days: list[date]) -> None:
    """The shipped DataSamples folder is this shape; real ones are usually not."""
    root = tmp_path / "both"
    root.mkdir()
    daily_frame("MSFT", days, 100).to_csv(root / "bars.csv", index=False)
    write_quotes(root / "quotes.csv", days[0], [(0, 99.0, 101.0), (90, 101.0, 103.0)])

    p = provider(root, tmp_path / "c", source_timezone="UTC")
    bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert len(bars) == len(days), "real OHLC is used, not folded quotes"
    assert bars[0].volume > 0, "aggregate volume survives"


def test_shipped_quote_sample_alone_drives_bars(tmp_path: Path) -> None:
    """The provided quote CSV, on its own, with nothing else in the directory."""
    root = tmp_path / "shipped"
    root.mkdir()
    shutil.copy(SAMPLES / "stock_quotes_sample.csv", root / "2023-03-28.csv")

    p = provider(root, tmp_path / "c")
    req = request_for("MSFT", timeframe=Timeframe.M1,
                      start=utc(2023, 3, 28), end=utc(2023, 3, 29))
    bars = list(p.stream_bars(req))

    assert len(bars) == 2, "the sample spans 08:00:00 to 08:01:07 -> two minute bars"
    assert bars[0].timestamp == datetime(2023, 3, 28, 8, 1, tzinfo=UTC)
    for bar in bars:
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.quote is not None and bar.quote.bid <= bar.quote.ask


def test_a_quote_on_the_boundary_belongs_to_the_closing_bar(tmp_path: Path) -> None:
    """Bars are right-labelled over the half-open interval ``(T - w, T]``.

    So a quote stamped exactly 09:31:00 is part of the bar that CLOSES at
    09:31, not the one that opens there. Getting this backwards would shift
    every derived bar by one period.
    """
    root = tmp_path / "edge"
    root.mkdir()
    write_quotes(root / "2023-03-08.csv", date(2023, 3, 8),
                 [(60, 99.0, 101.0), (61, 105.0, 107.0)])

    p = provider(root, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))

    assert [b.timestamp for b in bars] == [
        datetime(2023, 3, 8, 9, 31, tzinfo=UTC),
        datetime(2023, 3, 8, 9, 32, tzinfo=UTC),
    ]
    assert bars[0].close == pytest.approx(100.0), "the 09:31:00 quote closed the 09:31 bar"
    assert bars[1].close == pytest.approx(106.0)
