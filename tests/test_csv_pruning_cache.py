"""Date pruning, the Parquet conversion cache, warm-up and bounded streaming."""

from __future__ import annotations

import types
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from conftest import MARCH, daily_frame, request_for, utc

from sigmaloop.data.cache import CacheKey, MemoryDataCache, ParquetDataCache, TieredDataCache
from sigmaloop.data.provider import DataRequest
from sigmaloop.data.providers.csv_provider import CsvDataProvider, CsvProviderConfig
from sigmaloop.domain.bar import BarSeries
from sigmaloop.domain.instrument import Equity
from sigmaloop.types import AssetClass, Symbol, Timeframe


def provider(path: Path, cache: Path, **kwargs: object) -> CsvDataProvider:
    kwargs.setdefault("source_timezone", "UTC")
    return CsvDataProvider(CsvProviderConfig(path=path, cache_dir=cache, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pruning — requirement 3
# --------------------------------------------------------------------------- #


def test_pruning_wide_layout_skips_other_symbols(wide_layout: Path, tmp_path: Path) -> None:
    """A request for MSFT must not open AAPL.csv, even on a cold cache."""
    p = provider(wide_layout, tmp_path / "c")
    assert len(p._discover_files()) == 2
    assert [f.name for f in p._prune_files(request_for("MSFT", **MARCH))] == ["MSFT.csv"]
    assert sorted(f.name for f in p._prune_files(request_for("MSFT", "AAPL", **MARCH))) == [
        "AAPL.csv",
        "MSFT.csv",
    ]


def test_cold_pruning_avoids_converting_skipped_files(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    p = provider(wide_layout, cache)
    list(p.stream_bars(request_for("MSFT", **MARCH)))
    p.close()
    converted = {f.name.split("-")[0] for f in (cache / "files").glob("*.parquet")}
    assert converted == {"MSFT"}, "AAPL.csv was never parsed"


def test_pruning_per_day_files(per_day_layout: Path, tmp_path: Path) -> None:
    p = provider(per_day_layout, tmp_path / "c")
    one_day = request_for("MSFT", start=utc(2023, 3, 8), end=utc(2023, 3, 8, 23, 59))

    cold = p._prune_files(one_day)
    assert len(cold) < len(p._discover_files()), "filename dates prune before anything is read"

    bars = list(p.stream_bars(one_day))
    assert [b.timestamp.date() for b in bars] == [date(2023, 3, 8)]

    warm = p._prune_files(one_day)
    assert [f.stem for f in warm] == ["2023-03-08"], "the index prunes exactly once it exists"


def test_pruning_does_not_drop_boundary_data(per_day_layout: Path, tmp_path: Path) -> None:
    """Padding around a filename date must never lose a real bar."""
    p = provider(per_day_layout, tmp_path / "c")
    everything = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert len(everything) == 20
    for day in (date(2023, 3, 1), date(2023, 3, 15), date(2023, 3, 28)):
        req = request_for("MSFT", start=utc(day.year, day.month, day.day), end=utc(day.year, day.month, day.day, 23, 59))
        assert [b.timestamp.date() for b in p.stream_bars(req)] == [day]


def test_filename_hints_can_be_disabled(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c", trust_filename_hints=False)
    assert len(p._prune_files(request_for("MSFT", **MARCH))) == 2, "no hints, so nothing is skipped"
    bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert {b.instrument_id for b in bars} == {"EQ:MSFT"}, "results are identical either way"


def test_misleading_filename_warns(tmp_path: Path, days: list[date], caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "liar"
    root.mkdir()
    daily_frame("AAPL", days, 200).to_csv(root / "MSFT.csv", index=False)
    p = provider(root, tmp_path / "c")
    with caplog.at_level("WARNING", logger="sigmaloop.data.providers.csv"):
        p.available_symbols()
    assert any("named for MSFT but contains AAPL" in r.getMessage() for r in caplog.records)


def test_out_of_range_request_returns_nothing(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    req = request_for("MSFT", start=utc(2019, 1, 1), end=utc(2019, 12, 31))
    assert list(p.stream_bars(req)) == []


# --------------------------------------------------------------------------- #
# Parquet conversion cache
# --------------------------------------------------------------------------- #


def test_parquet_cache_is_written_and_reused(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    first = provider(wide_layout, cache)
    baseline = [b.close for b in first.stream_bars(request_for("MSFT", "AAPL", **MARCH))]
    first.close()

    assert list(cache.rglob("*.parquet")), "conversion cache written"
    assert (cache / "index-v1.json").exists(), "file index persisted"

    # A brand-new provider stands in for a new process.
    second = provider(wide_layout, cache)
    assert [b.close for b in second.stream_bars(request_for("MSFT", "AAPL", **MARCH))] == baseline
    second.close()


def test_cache_invalidated_when_source_changes(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "mut"
    root.mkdir()
    cache = tmp_path / "c"
    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)

    p = provider(root, cache)
    assert len(list(p.stream_bars(request_for("MSFT", **MARCH)))) == 20
    p.close()

    extended = days + [date(2023, 3, 30), date(2023, 3, 31)]
    daily_frame("MSFT", extended, 100).to_csv(root / "MSFT.csv", index=False)

    p2 = provider(root, cache)
    assert len(list(p2.stream_bars(request_for("MSFT", **MARCH)))) == 22, "stale Parquet must not win"
    p2.close()


def test_different_interpretations_do_not_share_a_cache(tmp_path: Path, days: list[date]) -> None:
    """Timezone is part of the fingerprint: same bytes, different timestamps."""
    root = tmp_path / "tz"
    root.mkdir()
    cache = tmp_path / "c"
    frame = daily_frame("MSFT", days, 100)
    frame["date"] = [f"{d.isoformat()} 16:00:00" for d in days]
    frame.to_csv(root / "MSFT.csv", index=False)

    utc_bar = next(iter(provider(root, cache, source_timezone="UTC").stream_bars(request_for("MSFT", **MARCH))))
    ny_bar = next(
        iter(provider(root, cache, source_timezone="America/New_York").stream_bars(request_for("MSFT", **MARCH)))
    )
    assert utc_bar.timestamp.hour == 16
    assert ny_bar.timestamp.hour == 21


def test_provider_works_without_parquet_cache(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    p = provider(wide_layout, cache, use_parquet_cache=False)
    bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert len(bars) == 20
    assert not list(cache.rglob("*.parquet"))
    p.close()


def test_readonly_cache_directory_is_not_fatal(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "ro"
    cache.mkdir()
    p = provider(wide_layout, cache)
    assert len(list(p.stream_bars(request_for("MSFT", **MARCH)))) == 20
    cache.chmod(0o500)
    try:
        p2 = provider(wide_layout, cache)
        assert len(list(p2.stream_bars(request_for("MSFT", **MARCH)))) == 20
        p2.close()
    finally:
        cache.chmod(0o700)


# --------------------------------------------------------------------------- #
# Series cache
# --------------------------------------------------------------------------- #


def test_load_series_is_cached(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    req = request_for("MSFT", **MARCH)
    first = p.load_series(Symbol("MSFT"), req)
    second = p.load_series(Symbol("MSFT"), req)
    assert first is second, "identical requests share one series object"


def test_cache_key_distinguishes_warmup() -> None:
    base = {"start": utc(2023, 3, 1), "end": utc(2023, 3, 31)}
    cold = DataRequest(symbols=(Symbol("MSFT"),), **base)  # type: ignore[arg-type]
    warm = DataRequest(symbols=(Symbol("MSFT"),), warmup_bars=5, **base)  # type: ignore[arg-type]
    k1 = CacheKey.from_request("csv", Symbol("MSFT"), cold)
    k2 = CacheKey.from_request("csv", Symbol("MSFT"), warm)
    assert k1 != k2 and k1.digest() != k2.digest()


def test_memory_cache_lru_eviction() -> None:
    def series(symbol: str, n: int) -> BarSeries:
        import numpy as np

        return BarSeries.from_arrays(
            Equity.make_id(Symbol(symbol)),
            Timeframe.D1,
            np.arange(n, dtype="int64"),
            *(np.ones(n, dtype="float64") for _ in range(5)),
        )

    one = series("A", 1_000)
    per_entry = one.timestamps.nbytes + 5 * one.open.nbytes
    cache = MemoryDataCache(max_bytes=per_entry * 2)

    keys = [CacheKey("csv", Symbol(s), "equity", "1d", 0, 1, True) for s in ("A", "B", "C")]
    for key, symbol in zip(keys, ("A", "B", "C"), strict=True):
        cache.put(key, series(symbol, 1_000))

    assert cache.stats.evictions >= 1
    assert not cache.contains(keys[0]), "least recently used is evicted first"
    assert cache.contains(keys[2])


def test_memory_cache_refuses_oversized_entries() -> None:
    import numpy as np

    big = BarSeries.from_arrays(
        Equity.make_id(Symbol("A")),
        Timeframe.D1,
        np.arange(10_000, dtype="int64"),
        *(np.ones(10_000, dtype="float64") for _ in range(5)),
    )
    cache = MemoryDataCache(max_bytes=1_024)
    key = CacheKey("csv", Symbol("A"), "equity", "1d", 0, 1, True)
    cache.put(key, big)
    assert not cache.contains(key), "one huge series must not evict the whole cache"


def test_parquet_cache_roundtrip_and_atomicity(tmp_path: Path) -> None:
    import numpy as np

    cache = ParquetDataCache(tmp_path / "pq")
    key = CacheKey("csv", Symbol("MSFT"), "equity", "1d", 0, 1, True)
    original = BarSeries.from_arrays(
        Equity.make_id(Symbol("MSFT")),
        Timeframe.D1,
        np.array([1, 2, 3], dtype="int64"),
        *(np.array([1.0, 2.0, 3.0]) for _ in range(5)),
    )
    assert cache.get(key) is None
    cache.put(key, original)

    assert cache.contains(key)
    assert len(list((tmp_path / "pq").rglob("*.parquet"))) == 1
    assert not list((tmp_path / "pq").rglob("*.tmp")), "no temp file survives a completed write"

    restored = cache.get(key)
    assert restored is not None
    assert restored.instrument_id == "EQ:MSFT"
    assert restored.timeframe is Timeframe.D1
    assert list(restored.close) == [1.0, 2.0, 3.0]

    cache.invalidate(key)
    assert not cache.contains(key)


def test_tiered_cache_promotes_from_disk(tmp_path: Path) -> None:
    import numpy as np

    fast = MemoryDataCache(max_bytes=1 << 20)
    slow = ParquetDataCache(tmp_path / "pq")
    tiered = TieredDataCache(fast, slow)
    key = CacheKey("csv", Symbol("MSFT"), "equity", "1d", 0, 1, True)
    series = BarSeries.from_arrays(
        Equity.make_id(Symbol("MSFT")),
        Timeframe.D1,
        np.array([1, 2], dtype="int64"),
        *(np.array([1.0, 2.0]) for _ in range(5)),
    )
    tiered.put(key, series)
    fast.clear()

    assert tiered.get(key) is not None
    assert fast.contains(key), "a disk hit is promoted into memory"


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #


def test_warmup_expands_the_window_backwards(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    plain = request_for("MSFT", start=utc(2023, 3, 15), end=utc(2023, 3, 31))
    warm = request_for("MSFT", start=utc(2023, 3, 15), end=utc(2023, 3, 31), warmup_bars=4)

    without = p.load_series(Symbol("MSFT"), plain)
    with_warmup = p.load_series(Symbol("MSFT"), warm)

    assert len(with_warmup) == len(without) + 4
    assert with_warmup.bar_at(-1).timestamp == without.bar_at(-1).timestamp
    assert with_warmup.bar_at(4).timestamp == without.bar_at(0).timestamp


def test_warmup_is_capped_by_available_history(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    warm = request_for("MSFT", start=utc(2023, 3, 2), end=utc(2023, 3, 31), warmup_bars=500)
    series = p.load_series(Symbol("MSFT"), warm)
    assert len(series) == 20, "asking for more warm-up than exists is not an error"


def test_warmup_applies_to_streaming(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    plain = list(p.stream_bars(request_for("MSFT", start=utc(2023, 3, 15), end=utc(2023, 3, 31))))
    warm = list(p.stream_bars(request_for("MSFT", start=utc(2023, 3, 15), end=utc(2023, 3, 31), warmup_bars=4)))
    assert len(warm) == len(plain) + 4
    assert warm[0].timestamp < plain[0].timestamp


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def test_stream_bars_is_lazy(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c")
    stream = p.stream_bars(request_for("MSFT", **MARCH))
    assert isinstance(stream, types.GeneratorType)
    assert next(stream).instrument_id == "EQ:MSFT"


def test_chunked_streaming_matches_one_shot(per_day_layout: Path, tmp_path: Path) -> None:
    """Slicing the window must not change what comes out, or its order."""
    req = request_for("MSFT", "AAPL", **MARCH)
    whole = list(provider(per_day_layout, tmp_path / "c1").stream_bars(req))
    chunked = list(provider(per_day_layout, tmp_path / "c2", stream_chunk_rows=3).stream_bars(req))

    assert [(b.instrument_id, b.timestamp, b.close) for b in whole] == [
        (b.instrument_id, b.timestamp, b.close) for b in chunked
    ]
    assert [b.timestamp for b in chunked] == sorted(b.timestamp for b in chunked)


def test_stream_is_ordered_across_symbols_and_files(per_day_layout: Path, tmp_path: Path) -> None:
    p = provider(per_day_layout, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", "AAPL", **MARCH)))
    assert len(bars) == 40
    stamps = [b.timestamp for b in bars]
    assert stamps == sorted(stamps), "the k-way merge contract the feed relies on"


def test_parallel_conversion(per_day_layout: Path, tmp_path: Path) -> None:
    p = provider(per_day_layout, tmp_path / "c", max_workers=4)
    assert sorted(p.available_symbols()) == ["AAPL", "MSFT"]
    assert len(list((tmp_path / "c" / "files").glob("*.parquet"))) == 20


def test_context_manager_flushes_index(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    with provider(wide_layout, cache) as p:
        list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert (cache / "index-v1.json").exists()


def test_small_series_stay_out_of_the_disk_tier(per_day_layout: Path, tmp_path: Path) -> None:
    """One tiny Parquet file per ticker costs more to write than it ever saves."""
    cache = tmp_path / "c"
    p = provider(per_day_layout, cache)
    p.load_many(request_for("MSFT", "AAPL", **MARCH))
    p.close()

    assert list((cache / "files").glob("*.parquet")), "the per-file conversion cache is still written"
    assert not list((cache / "series").rglob("*.parquet")), "20-row series are memory-only"


def test_large_series_do_reach_the_disk_tier(wide_layout: Path, tmp_path: Path) -> None:
    cache = tmp_path / "c"
    p = provider(wide_layout, cache, min_disk_cache_rows=1)
    p.load_series(Symbol("MSFT"), request_for("MSFT", **MARCH))
    p.close()
    assert list((cache / "series").rglob("*.parquet"))


def test_tiered_cache_threshold_is_honoured(tmp_path: Path) -> None:
    import numpy as np

    def series(n: int) -> BarSeries:
        return BarSeries.from_arrays(
            Equity.make_id(Symbol("A")),
            Timeframe.D1,
            np.arange(n, dtype="int64"),
            *(np.ones(n, dtype="float64") for _ in range(5)),
        )

    fast, slow = MemoryDataCache(max_bytes=1 << 20), ParquetDataCache(tmp_path / "pq")
    tiered = TieredDataCache(fast, slow, min_slow_rows=100)

    small = CacheKey("csv", Symbol("A"), "equity", "1d", 0, 1, True)
    large = CacheKey("csv", Symbol("A"), "equity", "1d", 0, 2, True)
    tiered.put(small, series(10))
    tiered.put(large, series(500))

    assert fast.contains(small) and not slow.contains(small)
    assert fast.contains(large) and slow.contains(large)


def test_left_labelled_without_inferable_width_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A per-session file has one bar per symbol, so there is no gap to measure."""
    root = tmp_path / "oneperday"
    root.mkdir()
    for day in (date(2023, 3, 1), date(2023, 3, 2)):
        frame = daily_frame("MSFT", [day], 100).rename(columns={"date": "window_start"})
        frame["window_start"] = [int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())]
        frame.to_csv(root / f"{day.isoformat()}.csv", index=False)

    p = provider(root, tmp_path / "c")
    with caplog.at_level("WARNING", logger="sigmaloop.data.providers.csv"):
        bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert len(bars) == 2
    assert any("too few bars per symbol to infer" in r.getMessage() for r in caplog.records)

    # Naming the timeframe removes the ambiguity and the shift happens.
    explicit = provider(root, tmp_path / "c2", timeframe=Timeframe.D1)
    shifted = list(explicit.stream_bars(request_for("MSFT", **MARCH)))
    assert shifted[0].timestamp - bars[0].timestamp == Timeframe.D1.duration


# --------------------------------------------------------------------------- #
# Warm-up across a per-session directory — the realistic layout
# --------------------------------------------------------------------------- #


def _session_files(root: Path, days: list[date], symbol: str = "MSFT") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, day in enumerate(days):
        pd.DataFrame(
            {
                "ticker": symbol,
                "date": [day.isoformat()],
                "open": [100 + i],
                "high": [101 + i],
                "low": [99 + i],
                "close": [100.5 + i],
                "volume": [1_000],
            }
        ).to_csv(root / f"{day.isoformat()}.csv", index=False)


def test_warmup_reaches_into_earlier_session_files(tmp_path: Path) -> None:
    """Regression: warm-up bars live in files *outside* the request window.

    Pruning on the request window alone silently returned zero warm-up bars,
    so every indicator would have primed on a truncated history and the
    strategy would have traded on it. Nothing raised — that is what made it bad.
    """
    root = tmp_path / "sessions"
    days = [date(2023, 3, d) for d in (1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
    _session_files(root, days)

    p = provider(root, tmp_path / "c")
    in_window = request_for("MSFT", start=utc(2023, 3, 14), end=utc(2023, 3, 31))
    warmed = request_for("MSFT", start=utc(2023, 3, 14), end=utc(2023, 3, 31), warmup_bars=5)

    plain = p.load_series(Symbol("MSFT"), in_window)
    warm = p.load_series(Symbol("MSFT"), warmed)

    assert len(plain) == 4, "2023-03-14 onwards"
    assert len(warm) == 9, "plus five sessions of priming from earlier files"
    assert warm.bar_at(0).timestamp.date() == date(2023, 3, 7)
    assert warm.bar_at(-1).timestamp == plain.bar_at(-1).timestamp


def test_warmup_larger_than_history_is_capped_not_broken(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    days = [date(2023, 3, d) for d in (1, 2, 3, 6, 7, 8, 9, 10, 13, 14)]
    _session_files(root, days)

    p = provider(root, tmp_path / "c")
    series = p.load_series(
        Symbol("MSFT"), request_for("MSFT", start=utc(2023, 3, 13), end=utc(2023, 3, 31), warmup_bars=500)
    )
    assert len(series) == len(days), "everything available, and no error"


def test_no_warmup_still_prunes_tightly(tmp_path: Path) -> None:
    """The look-back must not quietly disable pruning for ordinary requests."""
    root = tmp_path / "sessions"
    days = [date(2023, 3, d) for d in (1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17)]
    _session_files(root, days)

    p = provider(root, tmp_path / "c")
    window = request_for("MSFT", start=utc(2023, 3, 14), end=utc(2023, 3, 31))
    assert len(p._bar_source(window)[0]) == 4, "only the four sessions in the window"


# --------------------------------------------------------------------------- #
# Cache key identity — what may and may not share an entry
# --------------------------------------------------------------------------- #


def test_the_key_separates_two_asset_classes_on_one_ticker() -> None:
    """A ticker is not unique across asset classes, and the class routes to a
    different provider and a different instrument_id."""
    def request(asset_class: AssetClass) -> DataRequest:
        return DataRequest(
            symbols=(Symbol("SPY"),),
            start=utc(2023, 3, 1),
            end=utc(2023, 3, 31),
            timeframe=Timeframe.D1,
            asset_class=asset_class,
        )

    etf = CacheKey.from_request("csv", Symbol("SPY"), request(AssetClass.ETF))
    equity = CacheKey.from_request("csv", Symbol("SPY"), request(AssetClass.EQUITY))

    assert etf != equity
    assert etf.digest() != equity.digest(), "the on-disk path would collide too"


def test_the_key_separates_two_revisions_of_the_same_source() -> None:
    """Without this the disk tier outlives the data it describes."""
    request = DataRequest(
        symbols=(Symbol("MSFT"),), start=utc(2023, 3, 1), end=utc(2023, 3, 31), timeframe=Timeframe.D1
    )
    before = CacheKey.from_request("csv", Symbol("MSFT"), request, "digest-of-revision-1")
    after = CacheKey.from_request("csv", Symbol("MSFT"), request, "digest-of-revision-2")

    assert before != after
    assert before.digest() != after.digest()


def test_an_edited_csv_is_not_served_from_the_previous_run(tmp_path: Path, days: list[date]) -> None:
    """load_series' disk tier survives the process, so a second run over an
    edited file was answered with the first run's bars."""
    root, cache = tmp_path / "mut", tmp_path / "c"
    root.mkdir()

    def read() -> float:
        p = provider(root, cache)
        series = p.load_series(Symbol("MSFT"), request_for("MSFT", **MARCH))
        p.close()
        return float(series.close[0])

    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)
    before = read()

    daily_frame("MSFT", days, 222).to_csv(root / "MSFT.csv", index=False)
    after = read()

    assert after != pytest.approx(before), "stale series cache served the old close"
    assert after == pytest.approx(before + 122.0), "and it served the new one"


def test_reinterpreting_the_same_bytes_does_not_reuse_the_series(
    tmp_path: Path, days: list[date]
) -> None:
    """Same file, different source_timezone — different instants, so a shared
    entry would silently shift every bar."""
    root, cache = tmp_path / "tz2", tmp_path / "c"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100)
    frame["date"] = [f"{d.isoformat()} 16:00:00" for d in days]
    frame.to_csv(root / "MSFT.csv", index=False)

    def first_ts(tz: str) -> int:
        p = provider(root, cache, source_timezone=tz)
        series = p.load_series(Symbol("MSFT"), request_for("MSFT", **MARCH))
        p.close()
        return int(series.timestamps[0])

    assert first_ts("UTC") != first_ts("America/New_York")
