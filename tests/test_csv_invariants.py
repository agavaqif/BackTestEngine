"""Adversarial checks on the invariants the engine actually depends on.

These are deliberately property-shaped rather than example-shaped: each one
compares the provider against an independently computed answer, so a change
that quietly drops, duplicates or reorders rows fails here even if every
hand-written example still passes.
"""

from __future__ import annotations

import itertools
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import MARCH, business_days, daily_frame, request_for, utc

from sigmaloop.data.providers.csv_provider import (
    CsvDataProvider,
    CsvProviderConfig,
    TimestampPolicy,
    _decimal_shift,
    _TimestampNormaliser,
)
from sigmaloop.domain.bar import Bar
from sigmaloop.types import Symbol, Timeframe


def provider(path: Path, cache: Path, **kwargs: object) -> CsvDataProvider:
    kwargs.setdefault("source_timezone", "UTC")
    return CsvDataProvider(CsvProviderConfig(path=path, cache_dir=cache, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pruning must never change the answer, only the work
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("first_day", [1, 3, 8, 14, 20])
@pytest.mark.parametrize("length", [1, 2, 7])
def test_every_window_returns_exactly_the_bars_in_it(
    per_day_layout: Path, tmp_path: Path, days: list[date], first_day: int, length: int
) -> None:
    """Sweep windows across a per-day directory and compare to the fixture itself."""
    start = date(2023, 3, first_day)
    end = start + timedelta(days=length - 1)
    expected = sorted(d for d in days if start <= d <= end)

    p = provider(per_day_layout, tmp_path / f"c{first_day}-{length}")
    req = request_for(
        "MSFT",
        start=utc(start.year, start.month, start.day),
        end=utc(end.year, end.month, end.day, 23, 59),
    )
    got = [b.timestamp.date() for b in p.stream_bars(req)]

    assert got == expected, "pruning must not drop or add a session"


def test_pruning_matches_the_unpruned_answer(per_day_layout: Path, tmp_path: Path) -> None:
    """Filename hints on vs. off must agree, bar for bar."""
    req = request_for("MSFT", "AAPL", start=utc(2023, 3, 6), end=utc(2023, 3, 16))
    hinted = provider(per_day_layout, tmp_path / "c1", trust_filename_hints=True)
    blind = provider(per_day_layout, tmp_path / "c2", trust_filename_hints=False)

    def key(b: Bar) -> tuple[datetime, str, float, float, float, float, float]:
        return (b.timestamp, b.instrument_id, b.open, b.high, b.low, b.close, b.volume)

    assert [key(b) for b in hinted.stream_bars(req)] == [key(b) for b in blind.stream_bars(req)]


# --------------------------------------------------------------------------- #
# Slicing must not change the answer either
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 13, 1_000_000])
def test_any_chunk_size_yields_the_same_stream(
    per_day_layout: Path, tmp_path: Path, chunk: int
) -> None:
    req = request_for("MSFT", "AAPL", **MARCH)
    reference = provider(per_day_layout, tmp_path / "ref", stream_chunk_rows=1_000_000)
    sliced = provider(per_day_layout, tmp_path / f"c{chunk}", stream_chunk_rows=chunk)

    def key(b: Bar) -> tuple[datetime, str, float]:
        return (b.timestamp, b.instrument_id, b.close)

    expected = [key(b) for b in reference.stream_bars(req)]
    actual = [key(b) for b in sliced.stream_bars(req)]

    assert actual == expected
    assert len(actual) == len(set(actual)), "no row emitted twice at a slice boundary"
    assert [t for t, _, _ in actual] == sorted(t for t, _, _ in actual)


def test_overlapping_files_merge_in_order(tmp_path: Path, days: list[date]) -> None:
    """Two files covering the same dates for different tickers, interleaved."""
    root = tmp_path / "overlap"
    root.mkdir()
    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)
    daily_frame("AAPL", days, 200).to_csv(root / "AAPL.csv", index=False)

    p = provider(root, tmp_path / "c", stream_chunk_rows=2)
    bars = list(p.stream_bars(request_for("MSFT", "AAPL", **MARCH)))

    assert len(bars) == 2 * len(days)
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
    for _stamp, group in itertools.groupby(bars, key=lambda b: b.timestamp):
        assert sorted(b.instrument_id for b in group) == ["EQ:AAPL", "EQ:MSFT"]


# --------------------------------------------------------------------------- #
# Quotes: strictly backward-looking, including across slice boundaries
# --------------------------------------------------------------------------- #


def _minute_bars(root: Path, day: date, count: int) -> None:
    base = int(datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC).timestamp()) * 10**9
    pd.DataFrame(
        {
            "ticker": "MSFT",
            "window_start": [base + i * 60 * 10**9 for i in range(count)],
            "open": [100.0 + i for i in range(count)],
            "high": [101.0 + i for i in range(count)],
            "low": [99.0 + i for i in range(count)],
            "close": [100.5 + i for i in range(count)],
            "volume": [10] * count,
        }
    ).to_csv(root / "bars.csv", index=False)


def _quotes(root: Path, day: date, offsets_seconds: list[int]) -> list[int]:
    base = int(datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC).timestamp()) * 10**9
    stamps = [base + s * 10**9 for s in offsets_seconds]
    pd.DataFrame(
        {
            "Ticker": "MSFT",
            "sip_timestamp": stamps,
            "bid_price": [10.0 + i for i in range(len(stamps))],
            "ask_price": [11.0 + i for i in range(len(stamps))],
            "bid_size": [1] * len(stamps),
            "ask_size": [2] * len(stamps),
        }
    ).to_csv(root / "quotes.csv", index=False)
    return stamps


@pytest.mark.parametrize("chunk", [1, 2, 1_000_000])
def test_attached_quote_is_the_last_one_at_or_before_the_bar_close(
    tmp_path: Path, chunk: int
) -> None:
    """The as-of join must never reach forward, and must carry across slices."""
    root = tmp_path / f"q{chunk}"
    root.mkdir()
    day = date(2023, 3, 8)
    _minute_bars(root, day, 10)
    # Quotes at 0s, 90s, 95s, 400s: some minutes have several, some have none.
    quote_ns = _quotes(root, day, [0, 90, 95, 400])
    bids = [10.0 + i for i in range(len(quote_ns))]

    p = provider(root, tmp_path / f"c{chunk}", stream_chunk_rows=chunk, timeframe=Timeframe.M1)
    req = request_for("MSFT", timeframe=Timeframe.M1, include_quotes=True, **MARCH)
    bars = list(p.stream_bars(req))
    assert len(bars) == 10

    for bar in bars:
        bar_ns = int(bar.timestamp.timestamp()) * 10**9
        eligible = [i for i, q in enumerate(quote_ns) if q <= bar_ns]
        if not eligible:
            assert bar.quote is None, f"{bar.timestamp} saw a quote from the future"
        else:
            assert bar.quote is not None
            assert bar.quote.bid == pytest.approx(bids[eligible[-1]])


def test_quote_never_precedes_a_bar_it_should_not_see(tmp_path: Path) -> None:
    """A quote timestamped one nanosecond after the close is not visible."""
    root = tmp_path / "edge"
    root.mkdir()
    day = date(2023, 3, 8)
    _minute_bars(root, day, 2)
    # First bar closes at 14:31:00. Place quotes at exactly the close and 1ns after.
    base = int(datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC).timestamp()) * 10**9
    pd.DataFrame(
        {
            "Ticker": "MSFT",
            "sip_timestamp": [base + 60 * 10**9, base + 60 * 10**9 + 1],
            "bid_price": [10.0, 999.0],
            "ask_price": [11.0, 1000.0],
            "bid_size": [1, 1],
            "ask_size": [1, 1],
        }
    ).to_csv(root / "quotes.csv", index=False)

    p = provider(root, tmp_path / "c", timeframe=Timeframe.M1)
    bars = list(
        p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, include_quotes=True, **MARCH))
    )
    assert bars[0].quote is not None
    assert bars[0].quote.bid == pytest.approx(10.0), "the +1ns quote must not be visible"


# --------------------------------------------------------------------------- #
# Cache invalidation
# --------------------------------------------------------------------------- #


def test_same_size_different_content_invalidates(tmp_path: Path, days: list[date]) -> None:
    """Byte-identical length is not enough; mtime must catch an in-place edit."""
    root = tmp_path / "edit"
    root.mkdir()
    cache = tmp_path / "c"
    frame = daily_frame("MSFT", days, 100)
    frame.to_csv(root / "MSFT.csv", index=False)

    first = provider(root, cache)
    before = [b.close for b in first.stream_bars(request_for("MSFT", **MARCH))]
    first.close()

    edited = frame.copy()
    # +0.1 keeps close inside [low, high] and keeps every cell the same width,
    # so the file size is byte-for-byte identical and only mtime can catch it.
    edited["close"] = edited["close"] + 0.1
    edited.to_csv(root / "MSFT.csv", index=False)
    assert (root / "MSFT.csv").stat().st_size == len(frame.to_csv(index=False))

    second = provider(root, cache)
    after = [b.close for b in second.stream_bars(request_for("MSFT", **MARCH))]
    second.close()
    assert after == [pytest.approx(c + 0.1) for c in before]


def test_column_map_change_does_not_reuse_the_cache(tmp_path: Path, days: list[date]) -> None:
    from sigmaloop.data.providers.csv_provider import CsvColumnMap

    root = tmp_path / "cm"
    root.mkdir()
    cache = tmp_path / "c"
    frame = daily_frame("MSFT", days, 100)
    frame["alt_close"] = frame["high"]  # a different, still in-range, close
    frame.to_csv(root / "MSFT.csv", index=False)

    default = provider(root, cache)
    remapped = provider(
        root,
        cache,
        columns=CsvColumnMap(
            timestamp="date",
            symbol="ticker",
            open="open",
            high="high",
            low="low",
            close="alt_close",
            volume="volume",
        ),
    )
    a = next(iter(default.stream_bars(request_for("MSFT", **MARCH))))
    b = next(iter(remapped.stream_bars(request_for("MSFT", **MARCH))))
    assert a.close != pytest.approx(a.high)
    assert b.close == pytest.approx(a.high), "the remapped provider must not reuse the first cache"


# --------------------------------------------------------------------------- #
# Timestamp scaling
# --------------------------------------------------------------------------- #


def test_decimal_shift_rejects_the_unscalable() -> None:
    assert _decimal_shift(0) is None
    assert _decimal_shift(-5) is None
    assert _decimal_shift(1_679_990_400) == 9  # seconds
    assert _decimal_shift(1_679_990_400_000) == 6  # millis
    assert _decimal_shift(1_679_990_400_000_000_000) == 0  # nanos


def test_huge_values_do_not_wrap_around(tmp_path: Path) -> None:
    """An int64 multiply that would overflow must be rejected, not wrapped."""
    config = CsvProviderConfig(path=tmp_path, on_bad_timestamp=TimestampPolicy.DROP)
    normaliser = _TimestampNormaliser(config, tmp_path / "x.csv")
    import pyarrow as pa

    # A plausible seconds column plus one absurd value.
    values = [1_679_990_400] * 10 + [9_223_372_036_854_775_000]
    scaled, valid = normaliser.normalise(pa.array(values, type=pa.int64()))

    assert valid[:10].all()
    assert not valid[10], "the overflowing value is dropped, not silently wrapped"
    assert (scaled[:10] == 1_679_990_400_000_000_000).all()
    assert scaled[valid].min() > 0


def test_all_zero_timestamp_column_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "zeros"
    root.mkdir()
    pd.DataFrame(
        {
            "ticker": "MSFT",
            "window_start": [0, 0, 0],
            "open": [1.0] * 3,
            "high": [1.0] * 3,
            "low": [1.0] * 3,
            "close": [1.0] * 3,
            "volume": [1] * 3,
        }
    ).to_csv(root / "MSFT.csv", index=False)
    p = provider(root, tmp_path / "c")
    assert list(p.stream_bars(request_for("MSFT", **MARCH))) == []


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("workers", [1, 8])
def test_parallel_conversion_is_deterministic(tmp_path: Path, workers: int) -> None:
    root = tmp_path / f"many{workers}"
    root.mkdir()
    all_days = business_days(date(2023, 3, 1), 30)
    for day in all_days:
        pd.concat(
            [daily_frame(s, [day], 100 + i) for i, s in enumerate(("MSFT", "AAPL", "NVDA"))]
        ).to_csv(root / f"{day.isoformat()}.csv", index=False)

    p = provider(root, tmp_path / f"c{workers}", max_workers=workers)
    assert sorted(p.available_symbols()) == ["AAPL", "MSFT", "NVDA"]

    req = request_for("MSFT", "AAPL", "NVDA", start=utc(2023, 3, 1), end=utc(2023, 4, 30))
    bars = list(p.stream_bars(req))
    assert len(bars) == 3 * len(all_days)
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
    assert len(p._index) == len(all_days)
    p.close()


def test_repeated_runs_over_a_shared_cache_agree(tmp_path: Path, days: list[date]) -> None:
    """Three providers over one cache directory, as a parameter sweep would."""
    root = tmp_path / "sweep"
    root.mkdir()
    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)
    cache = tmp_path / "shared"

    req = request_for("MSFT", **MARCH)
    runs = []
    for _ in range(3):
        p = provider(root, cache)
        runs.append([(b.timestamp, b.close) for b in p.stream_bars(req)])
        p.close()
    assert runs[0] == runs[1] == runs[2]
    assert len(runs[0]) == len(days)


# --------------------------------------------------------------------------- #
# Columnar and streaming paths must agree
# --------------------------------------------------------------------------- #


def test_stream_and_load_series_return_the_same_data(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c")
    req = request_for("MSFT", timeframe=Timeframe.M1, **MARCH)
    bars = list(p.stream_bars(req))
    series = p.load_series(Symbol("MSFT"), req)

    assert len(bars) == len(series)
    assert np.array_equal(series.close, np.array([b.close for b in bars]))
    assert np.array_equal(series.open, np.array([b.open for b in bars]))
    assert np.array_equal(series.volume, np.array([b.volume for b in bars]))
    assert [b.timestamp for b in bars] == [series.bar_at(i).timestamp for i in range(len(series))]


def test_load_many_matches_per_symbol_load_series(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    req = request_for("MSFT", "AAPL", **MARCH)
    batched = p.load_many(req)
    for symbol in (Symbol("MSFT"), Symbol("AAPL")):
        one = p.load_series(symbol, req)
        assert np.array_equal(batched[symbol].close, one.close)
        assert np.array_equal(batched[symbol].timestamps, one.timestamps)


def test_bars_never_violate_their_own_invariant(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c")
    for bar in p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)):
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.volume >= 0
        assert bar.timestamp.tzinfo is not None


def test_two_parallel_conversions_in_one_process_do_not_crash(tmp_path: Path) -> None:
    """Regression: a per-call ThreadPoolExecutor segfaults the *next* one.

    Letting a pool's threads exit leaves pyarrow per-thread state dangling, and
    the following pool dies inside a timestamp cast. It only shows up when a
    second provider runs in the same process, which is what a sweep does.

    Run out-of-process: the failure mode is SIGSEGV, which would take the whole
    test session down rather than failing one test.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import csv, sys, tempfile
        from datetime import UTC, date, datetime, timedelta
        from pathlib import Path

        from sigmaloop.data.provider import DataRequest
        from sigmaloop.data.providers.csv_provider import CsvDataProvider, CsvProviderConfig
        from sigmaloop.types import Symbol, Timeframe

        root = Path(tempfile.mkdtemp())

        def build(name, count):
            folder = root / name
            folder.mkdir()
            day, written = date(2024, 1, 1), 0
            while written < count:
                if day.weekday() < 5:
                    with (folder / f"{day.isoformat()}.csv").open("w", newline="") as fh:
                        w = csv.writer(fh)
                        w.writerow(["ticker", "date", "open", "high", "low", "close", "volume"])
                        for i, sym in enumerate(("MSFT", "AAPL", "NVDA")):
                            b = 100 + i * 10 + written * 0.25
                            w.writerow([sym, day.isoformat(), b, b + 1.5, b - 1.2, b + 0.4, 1000])
                    written += 1
                day += timedelta(days=1)
            return folder

        req = DataRequest(
            symbols=(Symbol("MSFT"), Symbol("AAPL"), Symbol("NVDA")),
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 6, 1, tzinfo=UTC),
            timeframe=Timeframe.D1,
        )
        # Two providers, each converting many files in parallel, in one process.
        for run in ("first", "second", "third"):
            folder = build(run, 30)
            p = CsvDataProvider(CsvProviderConfig(
                path=folder, cache_dir=root / f"cache-{run}", source_timezone="UTC", max_workers=8))
            assert sum(1 for _ in p.stream_bars(req)) == 90, run
            p.close()
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, (
        f"exit {result.returncode} (139 == SIGSEGV): {result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout
