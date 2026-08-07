"""CsvDataProvider: file layouts, schema detection, timestamps and quotes."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from conftest import MARCH, daily_frame, request_for

from sigmaloop.data.providers.csv_provider import (
    CsvColumnMap,
    CsvDataProvider,
    CsvFileKind,
    CsvProviderConfig,
    DuplicatePolicy,
    EpochUnit,
    TimestampPolicy,
)
from sigmaloop.errors import DataIntegrityError, DataNotAvailableError
from sigmaloop.types import AssetClass, Symbol, Timeframe


def provider(path: Path, cache: Path, **kwargs: object) -> CsvDataProvider:
    return CsvDataProvider(CsvProviderConfig(path=path, cache_dir=cache, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Capabilities and instruments
# --------------------------------------------------------------------------- #


def test_capabilities(minute_sample: Path, tmp_path: Path) -> None:
    caps = provider(minute_sample, tmp_path / "c").capabilities
    assert caps.name == "csv"
    assert AssetClass.EQUITY in caps.asset_classes
    assert {Timeframe.D1, Timeframe.M1} <= caps.timeframes
    assert caps.supports_streaming is True
    assert caps.supports_quotes is True
    assert caps.requires_credentials is False


def test_resolve_instrument(minute_sample: Path, tmp_path: Path) -> None:
    instrument = provider(minute_sample, tmp_path / "c").resolve_instrument(Symbol("msft"), AssetClass.EQUITY)
    assert instrument.instrument_id == "EQ:MSFT"
    assert instrument.symbol == "MSFT"
    assert instrument.multiplier == 1.0


def test_missing_path_is_reported_at_construction(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="No such file or directory"):
        CsvDataProvider(CsvProviderConfig(path=tmp_path / "nope"))


# --------------------------------------------------------------------------- #
# Layout 1 — a single file holding one symbol (the shipped sample)
# --------------------------------------------------------------------------- #


def test_single_file_single_symbol_sample(minute_sample: Path, tmp_path: Path) -> None:
    """The provided Polygon minute file: auto-detected end to end."""
    p = provider(minute_sample, tmp_path / "c")
    assert p.available_symbols() == (Symbol("MSFT"),)

    req = request_for("MSFT", timeframe=Timeframe.M1, **MARCH)
    bars = list(p.stream_bars(req))

    assert len(bars) == 99, "every row of the sample should survive"
    assert all(b.instrument_id == "EQ:MSFT" for b in bars)
    assert all(b.timeframe is Timeframe.M1 for b in bars)
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)
    # window_start 08:00Z labels the bar OPEN, so the first close is 08:01Z.
    assert bars[0].timestamp == datetime(2023, 3, 28, 8, 1, tzinfo=UTC)
    assert bars[0].open == pytest.approx(276.75)
    assert bars[0].close == pytest.approx(275.52)
    assert bars[0].volume == pytest.approx(1975.0)


def test_timeframe_inferred_from_spacing(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.D1, **MARCH)))
    assert bars[0].timeframe is Timeframe.M1, "spacing wins over the request's nominal frame"


def test_load_series_is_columnar(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c")
    series = p.load_series(Symbol("MSFT"), request_for("MSFT", timeframe=Timeframe.M1, **MARCH))

    assert len(series) == 99
    assert series.close.dtype.name == "float64"
    assert series.timestamps.dtype.name == "int64"
    assert len(series.tail(5)) == 5
    assert series.tail(5).close.base is not None, "tail must be a view, not a copy"
    assert series.bar_at(-1).close == pytest.approx(276.0)
    assert series.bar_at(0).timestamp == datetime(2023, 3, 28, 8, 1, tzinfo=UTC)


def test_no_data_for_unknown_symbol(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c")
    with pytest.raises(DataNotAvailableError):
        p.load_series(Symbol("TSLA"), request_for("TSLA", timeframe=Timeframe.M1, **MARCH))


# --------------------------------------------------------------------------- #
# Layout 2 — a single long-format file holding the whole universe
# --------------------------------------------------------------------------- #


def test_single_file_universe_long_format(long_layout: Path, tmp_path: Path) -> None:
    p = provider(long_layout, tmp_path / "c", source_timezone="UTC")
    assert sorted(p.available_symbols()) == ["AAPL", "MSFT"]

    bars = list(p.stream_bars(request_for("MSFT", "AAPL", **MARCH)))
    assert {b.instrument_id for b in bars} == {"EQ:MSFT", "EQ:AAPL"}
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)

    per_symbol = p.load_many(request_for("MSFT", "AAPL", **MARCH))
    assert set(per_symbol) == {Symbol("MSFT"), Symbol("AAPL")}
    assert len(per_symbol[Symbol("MSFT")]) == len(per_symbol[Symbol("AAPL")]) == 20


def test_narrowing_to_one_symbol_excludes_the_other(long_layout: Path, tmp_path: Path) -> None:
    p = provider(long_layout, tmp_path / "c", source_timezone="UTC")
    bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
    assert {b.instrument_id for b in bars} == {"EQ:MSFT"}


# --------------------------------------------------------------------------- #
# Layout 3 — one file per ticker
# --------------------------------------------------------------------------- #


def test_multiple_files_per_symbol_wide(wide_layout: Path, tmp_path: Path) -> None:
    p = provider(wide_layout, tmp_path / "c", source_timezone="UTC")
    assert sorted(p.available_symbols()) == ["AAPL", "MSFT"], "notes.txt must be ignored"

    bars = list(p.stream_bars(request_for("MSFT", "AAPL", **MARCH)))
    assert len(bars) == 40
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)


def test_wide_layout_without_symbol_column_uses_filename(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "nosym"
    root.mkdir()
    daily_frame("MSFT", days, 100).drop(columns=["ticker"]).to_csv(root / "MSFT.csv", index=False)
    p = provider(root, tmp_path / "c", source_timezone="UTC")
    assert p.available_symbols() == (Symbol("MSFT"),)


def test_symbol_column_absent_and_filename_unusable(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "anon"
    root.mkdir()
    daily_frame("MSFT", days, 100).drop(columns=["ticker"]).to_csv(root / "prices export.csv", index=False)
    p = provider(root, tmp_path / "c", source_timezone="UTC")
    with pytest.raises(DataIntegrityError, match="no symbol column"):
        p.available_symbols()


def test_default_symbol_config(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "anon2"
    root.mkdir()
    daily_frame("MSFT", days, 100).drop(columns=["ticker"]).to_csv(root / "prices export.csv", index=False)
    p = provider(root, tmp_path / "c", source_timezone="UTC", default_symbol=Symbol("MSFT"))
    assert p.available_symbols() == (Symbol("MSFT"),)


# --------------------------------------------------------------------------- #
# Layout 4 — one file per session, holding the whole universe
# --------------------------------------------------------------------------- #


def test_multiple_files_per_day_universe(per_day_layout: Path, tmp_path: Path) -> None:
    p = provider(per_day_layout, tmp_path / "c", source_timezone="UTC")
    assert sorted(p.available_symbols()) == ["AAPL", "MSFT"]

    bars = list(p.stream_bars(request_for("MSFT", "AAPL", **MARCH)))
    assert len(bars) == 40
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)


def test_multiple_files_per_day_single_symbol(per_day_layout: Path, tmp_path: Path) -> None:
    p = provider(per_day_layout, tmp_path / "c", source_timezone="UTC")
    bars = list(p.stream_bars(request_for("AAPL", **MARCH)))
    assert len(bars) == 20
    assert {b.instrument_id for b in bars} == {"EQ:AAPL"}


def test_file_glob_and_string_path(per_day_layout: Path, tmp_path: Path) -> None:
    p = CsvDataProvider(
        CsvProviderConfig(
            # Declared as Path; __post_init__ coerces, and that coercion is what
            # this test is here to pin down.
            path=str(per_day_layout),  # type: ignore[arg-type]
            cache_dir=tmp_path / "c",
            file_glob="2023-03-0*.csv",
            source_timezone="UTC",
        )
    )
    matched = {f.stem for f in p._discover_files()}
    assert matched == {f"2023-03-0{d}" for d in (1, 2, 3, 6, 7, 8, 9)}


# --------------------------------------------------------------------------- #
# Mixed directory: aggregates and quotes side by side
# --------------------------------------------------------------------------- #


def test_mixed_directory_classifies_each_file(samples_dir: Path, tmp_path: Path) -> None:
    p = provider(samples_dir, tmp_path / "c")
    p.available_symbols()
    kinds = {Path(path).name: entry.kind for path, entry in p._index.items()}
    assert kinds["stock_quotes_sample.csv"] is CsvFileKind.QUOTE
    assert kinds["stocks_minute_candlesticks_example.csv"] is CsvFileKind.AGGREGATE


def test_duplicate_files_are_collapsed(samples_dir: Path, tmp_path: Path) -> None:
    """DataSamples ships the minute file twice; bars must not double up."""
    p = provider(samples_dir, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))
    assert len(bars) == 99


def test_duplicate_policy_error(samples_dir: Path, tmp_path: Path) -> None:
    p = provider(samples_dir, tmp_path / "c", on_duplicate=DuplicatePolicy.ERROR)
    with pytest.raises(DataIntegrityError, match="duplicate"):
        list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #


def test_quote_file_is_read(quote_sample: Path, tmp_path: Path) -> None:
    p = provider(quote_sample, tmp_path / "c")
    quotes = list(p.stream_quotes(request_for("MSFT", timeframe=Timeframe.TICK, **MARCH)))

    assert len(quotes) == 94, "the five zero-priced placeholder rows are dropped"
    assert all(q.bid > 0 and q.ask >= q.bid for _, _, q in quotes)
    assert [t for _, t, _ in quotes] == sorted(t for _, t, _ in quotes)
    _, first_ts, first = quotes[0]
    assert first_ts.date() == date(2023, 3, 28)
    assert first.mid == pytest.approx((first.bid + first.ask) / 2)


def test_invalid_quotes_kept_when_configured(quote_sample: Path, tmp_path: Path) -> None:
    p = provider(quote_sample, tmp_path / "c", drop_invalid_quotes=False)
    quotes = list(p.stream_quotes(request_for("MSFT", timeframe=Timeframe.TICK, **MARCH)))
    assert len(quotes) == 99, "every row of the file, placeholders included"


def test_quotes_attach_to_bars(samples_dir: Path, tmp_path: Path) -> None:
    p = provider(samples_dir, tmp_path / "c")
    req = request_for("MSFT", timeframe=Timeframe.M1, include_quotes=True, **MARCH)
    bars = list(p.stream_bars(req))

    assert all(b.quote is not None for b in bars)
    for bar in bars[:5]:
        assert bar.quote is not None
        assert bar.quote.bid <= bar.quote.ask
    # WORST pricing now means something: buy at the ask, sell at the bid.
    from sigmaloop.types import PriceSelection

    first = bars[0]
    assert first.quote is not None
    assert first.price_for(PriceSelection.WORST, is_buy=True) == first.quote.ask
    assert first.price_for(PriceSelection.WORST, is_buy=False) == first.quote.bid


def test_quotes_absent_when_not_requested(samples_dir: Path, tmp_path: Path) -> None:
    p = provider(samples_dir, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))
    assert all(b.quote is None for b in bars)


def test_stale_quotes_are_not_attached(samples_dir: Path, tmp_path: Path) -> None:
    """The sample quotes stop at 08:01Z; later bars must not inherit them."""
    from datetime import timedelta

    p = provider(samples_dir, tmp_path / "c", max_quote_age=timedelta(minutes=5))
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, include_quotes=True, **MARCH)))
    attached = [b for b in bars if b.quote is not None]
    assert 0 < len(attached) < len(bars)
    assert all(b.timestamp <= datetime(2023, 3, 28, 8, 6, tzinfo=UTC) for b in attached)


# --------------------------------------------------------------------------- #
# Schema detection and column mapping
# --------------------------------------------------------------------------- #


def test_autodetect_without_column_map(minute_sample: Path, tmp_path: Path) -> None:
    """No CsvColumnMap: ticker/window_start/ohlcv found from the header alone."""
    p = provider(minute_sample, tmp_path / "c")
    assert len(list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))) == 99


def test_explicit_column_map(minute_sample: Path, tmp_path: Path) -> None:
    columns = CsvColumnMap(
        timestamp="window_start", symbol="ticker", open="open", high="high", low="low", close="close", volume="volume"
    )
    p = provider(minute_sample, tmp_path / "c", columns=columns)
    assert len(list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))) == 99


def test_missing_columns_raises(minute_sample: Path, tmp_path: Path) -> None:
    columns = CsvColumnMap(timestamp="window_start", symbol="ticker", close="closing_price")
    p = provider(minute_sample, tmp_path / "c", columns=columns)
    with pytest.raises(DataIntegrityError, match="closing_price"):
        list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))


def test_unrecognisable_header_raises(tmp_path: Path) -> None:
    root = tmp_path / "weird"
    root.mkdir()
    (root / "x.csv").write_text("alpha,beta,gamma\n1,2,3\n")
    p = provider(root, tmp_path / "c")
    with pytest.raises(DataIntegrityError, match="Could not identify the columns"):
        p.available_symbols()


def test_empty_file_raises(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    (root / "x.csv").write_text("")
    p = provider(root, tmp_path / "c")
    with pytest.raises(DataIntegrityError, match="no header"):
        p.available_symbols()


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #


def test_corrupted_timestamp_is_repaired(minute_sample: Path, tmp_path: Path) -> None:
    """One sample row lost four trailing zeros; the heuristic restores it.

    Its raw ``window_start`` is 1680008040000 where its neighbours are 17 digits.
    Repaired, it lands at 12:54Z — between the 12:49Z and 13:00Z rows around it,
    which is the evidence that the rescale is right rather than merely plausible.
    """
    p = provider(minute_sample, tmp_path / "c")
    bars = list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))

    healed = [b for b in bars if b.timestamp == datetime(2023, 3, 28, 12, 55, tzinfo=UTC)]
    assert len(healed) == 1
    assert healed[0].close == pytest.approx(275.71)
    assert healed[0].volume == pytest.approx(901.0)
    assert [b.timestamp for b in bars] == sorted(b.timestamp for b in bars)

    index = bars.index(healed[0])
    assert bars[index - 1].timestamp == datetime(2023, 3, 28, 12, 50, tzinfo=UTC)
    assert bars[index + 1].timestamp == datetime(2023, 3, 28, 13, 1, tzinfo=UTC)


def test_strict_timestamp_policy_rejects(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c", on_bad_timestamp=TimestampPolicy.STRICT)
    with pytest.raises(DataIntegrityError, match="1990 and 2100"):
        list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))


def test_drop_timestamp_policy(minute_sample: Path, tmp_path: Path) -> None:
    p = provider(minute_sample, tmp_path / "c", on_bad_timestamp=TimestampPolicy.DROP)
    assert len(list(p.stream_bars(request_for("MSFT", timeframe=Timeframe.M1, **MARCH)))) == 98


@pytest.mark.parametrize(
    ("scale", "unit"),
    [(1, EpochUnit.SECONDS), (10**3, EpochUnit.MILLIS), (10**6, EpochUnit.MICROS), (10**9, EpochUnit.NANOS)],
)
def test_epoch_units_autodetected(tmp_path: Path, days: list[date], scale: int, unit: EpochUnit) -> None:
    root = tmp_path / f"epoch{scale}"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100)
    epochs = [int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()) * scale for d in days]
    frame = frame.drop(columns=["date"]).assign(timestamp=epochs)
    frame.to_csv(root / "MSFT.csv", index=False)

    auto = provider(root, tmp_path / f"ca{scale}")
    explicit = provider(root, tmp_path / f"ce{scale}", epoch_unit=unit)
    for p in (auto, explicit):
        bars = list(p.stream_bars(request_for("MSFT", **MARCH)))
        assert len(bars) == 20
        assert bars[0].timestamp == datetime(days[0].year, days[0].month, days[0].day, tzinfo=UTC)


def test_timezone_conversion(tmp_path: Path, days: list[date]) -> None:
    """Naive exchange-local timestamps are lifted to UTC, not assumed to be UTC."""
    root = tmp_path / "tz"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100)
    frame["date"] = [f"{d.isoformat()} 16:00:00" for d in days]
    frame.to_csv(root / "MSFT.csv", index=False)

    ny = provider(root, tmp_path / "cny", source_timezone="America/New_York")
    as_utc = provider(root, tmp_path / "cutc", source_timezone="UTC")

    ny_bar = next(iter(ny.stream_bars(request_for("MSFT", **MARCH))))
    utc_bar = next(iter(as_utc.stream_bars(request_for("MSFT", **MARCH))))
    assert ny_bar.timestamp.hour == 21, "16:00 EST is 21:00 UTC"
    assert utc_bar.timestamp.hour == 16
    assert ny_bar.timestamp.tzinfo is not None


def test_left_labelled_shift(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "left"
    root.mkdir()
    daily_frame("MSFT", days, 100).to_csv(root / "MSFT.csv", index=False)

    right = provider(root, tmp_path / "cr", source_timezone="UTC", left_labelled=False, timeframe=Timeframe.D1)
    left = provider(root, tmp_path / "cl", source_timezone="UTC", left_labelled=True, timeframe=Timeframe.D1)

    right_first = next(iter(right.stream_bars(request_for("MSFT", **MARCH))))
    left_first = next(iter(left.stream_bars(request_for("MSFT", **MARCH))))
    assert left_first.timestamp - right_first.timestamp == Timeframe.D1.duration


def test_adjusted_close_back_adjusts_ohlc(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "adj"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100)
    frame["adj_close"] = frame["close"] / 2.0
    frame.to_csv(root / "MSFT.csv", index=False)

    p = provider(root, tmp_path / "c", source_timezone="UTC")
    raw = next(iter(p.stream_bars(request_for("MSFT", adjusted=False, **MARCH))))
    adj = next(iter(p.stream_bars(request_for("MSFT", adjusted=True, **MARCH))))

    assert adj.close == pytest.approx(raw.close / 2)
    assert adj.open == pytest.approx(raw.open / 2)
    assert adj.volume == pytest.approx(raw.volume * 2)
    assert adj.is_adjusted is True


def test_rows_violating_ohlc_invariant_are_dropped(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100)
    frame.loc[3, "high"] = 0.0  # high below low/open/close
    frame.to_csv(root / "MSFT.csv", index=False)

    p = provider(root, tmp_path / "c", source_timezone="UTC")
    assert len(list(p.stream_bars(request_for("MSFT", **MARCH)))) == len(days) - 1


def test_bom_and_padded_headers(tmp_path: Path, days: list[date]) -> None:
    root = tmp_path / "bom"
    root.mkdir()
    frame = daily_frame("MSFT", days, 100).rename(columns={"ticker": "Ticker ", "close": " Close"})
    (root / "MSFT.csv").write_text("﻿" + frame.to_csv(index=False), encoding="utf-8")
    p = provider(root, tmp_path / "c", source_timezone="UTC")
    assert len(list(p.stream_bars(request_for("MSFT", **MARCH)))) == len(days)


def test_adjustment_survives_files_with_mixed_schemas(tmp_path: Path, days: list[date]) -> None:
    """One file has adj_close, the other does not — each is handled on its own terms.

    AAPL.csv sorts first and has no adjusted close. Deciding from the first file
    alone would silently leave MSFT unadjusted; asking AAPL.csv for a column it
    lacks would raise.
    """
    root = tmp_path / "mixed"
    root.mkdir()
    with_adj = daily_frame("MSFT", days, 100)
    with_adj["adj_close"] = with_adj["close"] / 2.0
    with_adj.to_csv(root / "MSFT.csv", index=False)
    daily_frame("AAPL", days, 200).to_csv(root / "AAPL.csv", index=False)

    p = provider(root, tmp_path / "c", source_timezone="UTC")
    raw = {(b.instrument_id, b.timestamp): b.close for b in p.stream_bars(
        request_for("MSFT", "AAPL", adjusted=False, **MARCH))}
    adjusted = {(b.instrument_id, b.timestamp): b.close for b in p.stream_bars(
        request_for("MSFT", "AAPL", adjusted=True, **MARCH))}

    assert raw and len(raw) == len(adjusted)
    for (instrument, stamp), close in raw.items():
        if instrument == "EQ:MSFT":
            assert adjusted[(instrument, stamp)] == pytest.approx(close / 2)
        else:
            assert adjusted[(instrument, stamp)] == pytest.approx(close)
