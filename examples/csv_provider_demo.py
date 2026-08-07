#!/usr/bin/env python3
"""Run me from anywhere, with the project's virtualenv:


Who consumes CsvDataProvider, and how it hands data over.

    BacktestEngine.prepare()
        reads the strategy's declarations and builds a FeedPlan — a tuple of
        DataRequest objects saying which symbols, which window, which timeframe.

    MergedDataFeed(plan, providers)          <-- THE DIRECT CONSUMER
        calls provider.stream_bars(request) once per provider and k-way merges
        the resulting iterators on (epoch_ns, instrument_id), grouping every bar
        that shares a timestamp into one MarketSnapshot.

    BacktestEngine.run()
        for snapshot in feed:  ->  13 fixed phases per bar (DESIGN.md 9.1)

    Indicator warm-up and screeners take the other path: provider.load_series()
    / load_many(), which return columnar BarSeries for vectorised work.

So the provider is never touched by strategy code. It answers exactly two
questions, and the feed decides which to ask:

    stream_bars(request) -> Iterator[Bar]        lazy, row form, memory-bounded
    load_series(sym, req) -> BarSeries           eager, columnar, numpy views

A real dataset is ONE shape, not a mixture. The two that matter are covered
separately below: a directory of per-session OHLC aggregates (parts 2-5), and a
directory of nothing but bid/ask (part 6).

BacktestEngine is still a NotImplementedError stub; everything else here — the
provider and the MergedDataFeed in part 7 — is the real thing.
"""

from __future__ import annotations

import csv as csv_module
import itertools
import logging
import shutil
import sys
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from sigmaloop.data.calendar import NyseCalendar
from sigmaloop.data.feed import FeedPlan, MergedDataFeed
from sigmaloop.data.provider import DataRequest
from sigmaloop.data.providers.csv_provider import CsvDataProvider, CsvProviderConfig
from sigmaloop.domain.bar import Bar, Quote
from sigmaloop.types import AssetClass, PriceSelection, Symbol, Timeframe

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "DataSamples"

SAMPLE_DAY = datetime(2023, 3, 28, tzinfo=UTC)
SAMPLE_WINDOW = {"start": SAMPLE_DAY, "end": SAMPLE_DAY + timedelta(days=1)}


def banner(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * 78)


def book(bar: Bar) -> Quote:
    """The bar's quote, for the parts of the demo that only run on quoted data."""
    if bar.quote is None:
        raise AssertionError(f"expected a book on the {bar.timestamp:%H:%M:%S} bar")
    return bar.quote


def minute_request(**overrides: object) -> DataRequest:
    fields: dict[str, object] = {
        "symbols": (Symbol("MSFT"),),
        "timeframe": Timeframe.M1,
        **SAMPLE_WINDOW,
        **overrides,
    }
    return DataRequest(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. What the engine asks for
# --------------------------------------------------------------------------- #


def part_1_the_request() -> DataRequest:
    banner("1. The ask — BacktestEngine builds this, the feed passes it down")

    request = minute_request(warmup_bars=0)
    plan = FeedPlan(bar_requests=(request,), timeframe=Timeframe.M1)

    print(f"  DataRequest  symbols={request.symbols} {request.timeframe.value}")
    print(
        f"               {request.start.isoformat()} .. {request.end.isoformat()}  (closed interval)"
    )
    print(f"               adjusted={request.adjusted} include_quotes={request.include_quotes}")
    print(f"  FeedPlan     {len(plan.bar_requests)} request(s) — one per provider/asset class")
    print("\n  DataRequest is frozen and hashable, so it doubles as a cache key. It also")
    print("  normalises on construction: tickers upper-cased, bounds coerced to UTC.")
    return request


# --------------------------------------------------------------------------- #
# 2-5. Dataset shape A: OHLC aggregates
# --------------------------------------------------------------------------- #


def part_2_discovery(provider: CsvDataProvider) -> None:
    banner("2. Discovery — checked at second zero, not on bar 40,000")

    caps = provider.capabilities
    print(
        f"  capabilities   name={caps.name} streaming={caps.supports_streaming} quotes={caps.supports_quotes}"
    )
    print(f"                 asset classes: {sorted(a.value for a in caps.asset_classes)}")
    print(f"  symbols        {provider.available_symbols()}")

    span = provider.coverage(Symbol("MSFT"), Timeframe.M1)
    if span is not None:
        print(f"  coverage MSFT  {span[0].isoformat()} .. {span[1].isoformat()}")

    instrument = provider.resolve_instrument(Symbol("msft"), AssetClass.EQUITY)
    print(
        f"  instrument     {instrument.instrument_id}  multiplier={instrument.multiplier} tick={instrument.tick_size}"
    )


def part_3_streaming(provider: CsvDataProvider) -> None:
    banner("3. stream_bars() on an aggregate dataset — the engine's main input")

    stream = provider.stream_bars(minute_request())
    print(f"  returns {type(stream).__name__}: nothing is read until you iterate.\n")

    print("  ts (UTC, bar CLOSE)        open     high      low    close   volume   quote")
    for bar in itertools.islice(stream, 5):
        quote = f"{bar.quote.bid:.2f}/{bar.quote.ask:.2f}" if bar.quote else "None"
        print(
            f"  {bar.timestamp:%Y-%m-%d %H:%M:%S}  {bar.open:8.2f} {bar.high:8.2f} "
            f"{bar.low:8.2f} {bar.close:8.2f} {bar.volume:8.0f}   {quote}"
        )

    bars = list(provider.stream_bars(minute_request()))
    first = bars[0]
    print(f"\n  {len(bars)} bars, non-decreasing in timestamp — MergedDataFeed k-way merges")
    print("  several of these streams and depends on that ordering.")

    print("\n  OHLC carries no book, so every price selection collapses to close:")
    print(f"    quote                           {first.quote}")
    print(f"    price_for(WORST, buy)           {first.price_for(PriceSelection.WORST, True):.2f}")
    print(
        f"    price_for(WORST, sell)          {first.price_for(PriceSelection.WORST, False):.2f}   <- same number"
    )
    print("  The provider will not invent a spread. That is the execution layer's job:")
    print("  ExecutionConfig.spread_model ('fixed_bps') synthesises one and flags it as")
    print("  Quote.is_synthetic so the fills that relied on a guess stay countable.")


def part_4_columnar(provider: CsvDataProvider) -> None:
    banner("4. load_series() — eager columnar form, for vectorised work")

    series = provider.load_series(Symbol("MSFT"), minute_request())
    print(
        f"  BarSeries  instrument={series.instrument_id} timeframe={series.timeframe.value} len={len(series)}"
    )
    print(
        f"  columns    close dtype={series.close.dtype}  timestamps dtype={series.timestamps.dtype}"
    )
    print(f"  close[:6]  {series.close[:6]}")

    window = series.tail(20)
    print(f"\n  tail(20) is a VIEW, not a copy: shares_memory={window.close.base is not None}")
    print(f"    20-bar mean close  {window.close.mean():.4f}")
    print("\n  This is the path indicator warm-up uses: one numpy buffer per field, so a")
    print("  moving average is a slice operation, not a Python loop over objects.")


def part_5_quotes_beside_aggregates(samples: Path, cache: Path) -> None:
    banner("5. If a quote file sits beside the aggregates, bars can carry the real book")

    provider = CsvDataProvider(CsvProviderConfig(path=samples, cache_dir=cache))
    with provider:
        bars = list(provider.stream_bars(minute_request(include_quotes=True)))
        quotes = list(provider.stream_quotes(minute_request()))
        first = bars[0]
        print(
            f"  quote file spans   {quotes[0][1]:%H:%M:%S} .. {quotes[-1][1]:%H:%M:%S}  ({len(quotes)} quotes)"
        )
        print(
            f"  bars span          {bars[0].timestamp:%H:%M:%S} .. {bars[-1].timestamp:%H:%M:%S}  ({len(bars)} bars)"
        )
        print("\n  each bar gets the last quote at or before its close (never after):")
        print(
            f"    bar {first.timestamp:%H:%M}  close={first.close:.2f}  book={book(first).bid:.2f}/{book(first).ask:.2f}"
        )
        print(
            f"    price_for(WORST, buy)  {first.price_for(PriceSelection.WORST, True):.2f}  (pay the ask)"
        )
        print(
            f"    price_for(WORST, sell) {first.price_for(PriceSelection.WORST, False):.2f}  (hit the bid)"
        )

        distinct = {(b.quote.bid, b.quote.ask) for b in bars if b.quote}
        print(
            f"\n  Watch the staleness: only {len(distinct)} distinct quotes across {len(bars)} bars. A standing"
        )
        print("  NBBO is carried forward until replaced, so the 13:10 bar quotes an 08:01 book.")
        print("  Bound it with CsvProviderConfig(max_quote_age=...) when quote coverage is")
        print("  thinner than bar coverage, as it is in this sample.")


# --------------------------------------------------------------------------- #
# 6. Dataset shape B: nothing but bid/ask
# --------------------------------------------------------------------------- #


def part_6_quote_only(root: Path, cache: Path) -> None:
    banner("6. A quote-only dataset — no open/high/low/close anywhere")

    with CsvDataProvider(CsvProviderConfig(path=root, cache_dir=cache)) as provider:
        print(
            f"  {len(list(root.glob('*.csv')))} session file(s) of raw top-of-book, no aggregates."
        )
        print("  The engine consumes Bars, so the provider folds quotes into bars rather")
        print("  than serving nothing. OHLC tracks the mid; the bar still carries the real")
        print("  closing book, so execution prices off an actual spread.\n")

        for frame in (Timeframe.M1, Timeframe.S1):
            bars = list(provider.stream_bars(minute_request(timeframe=frame)))
            print(
                f"  timeframe={frame.value:3s} -> {len(bars):2d} bars   (the caller names the width;"
            )
            print("                          tick data has no spacing to infer)")
            for bar in bars[:2]:
                print(
                    f"      {bar.timestamp:%H:%M:%S}  O={bar.open:8.4f} H={bar.high:8.4f} "
                    f"L={bar.low:8.4f} C={bar.close:8.4f}  vol={bar.volume:.0f}"
                    f"  book={book(bar).bid:.2f}/{book(bar).ask:.2f}"
                )

        bar = next(iter(provider.stream_bars(minute_request())))
        print("\n  volume is 0 and stays 0 — quotes are not trades, and a made-up number")
        print("  would feed a volume-participation cap that has no basis in anything traded.")
        print(f"    price_for(WORST, buy)  {bar.price_for(PriceSelection.WORST, True):.2f}")
        print(
            f"    price_for(WORST, sell) {bar.price_for(PriceSelection.WORST, False):.2f}  <- a real spread, not a guess"
        )
        print("\n  Set derive_bars_from_quotes=False to make this an explicit error instead.")


# --------------------------------------------------------------------------- #
# 7. What MergedDataFeed will do with the stream
# --------------------------------------------------------------------------- #


def part_7_the_feed(root: Path, cache: Path) -> None:
    banner("7. MergedDataFeed — bars become one MarketSnapshot per timestamp")

    provider = CsvDataProvider(
        CsvProviderConfig(path=root, cache_dir=cache, source_timezone="UTC")
    )
    request = DataRequest(
        symbols=(Symbol("MSFT"), Symbol("AAPL"), Symbol("NVDA")),
        start=datetime(2024, 1, 2, tzinfo=UTC),
        end=datetime(2024, 1, 10, tzinfo=UTC),
        timeframe=Timeframe.D1,
    )
    plan = FeedPlan(bar_requests=(request,), timeframe=Timeframe.D1)
    feed = MergedDataFeed(plan, [provider], calendar=NyseCalendar())
    try:
        print(
            f"  universe {request.symbols} over {len(list(root.glob('*.csv')))} per-session files\n"
        )
        print("  timestamp            instruments in snapshot                 closes")
        for snapshot in itertools.islice(feed, 5):
            names = ", ".join(sorted(snapshot.bars))
            closes = " ".join(f"{b.close:7.2f}" for _, b in sorted(snapshot.bars.items()))
            print(f"  {snapshot.timestamp:%Y-%m-%d %H:%M}    {names:38s} {closes}")

        print(f"\n  instruments resolved up front: {[i.instrument_id for i in feed.instruments()]}")
        print("\n  A snapshot is a CLOSED WORLD: an instrument absent from `bars` did not")
        print("  trade at this timestamp, and an order against it is rejected with")
        print("  NO_MARKET_DATA rather than filled at a stale price.")
        print("\n  Residency is O(sources), not O(history): the heap holds one pending bar")
        print("  per stream and nothing else, so a ten-year run costs what a ten-day one does.")
        print("\n  The engine then runs each snapshot through its 13 phases. Everything")
        print("  downstream sees exactly one timestamp at a time — that is the structural")
        print("  guarantee against lookahead, not a convention anyone has to remember.")
    finally:
        feed.close()


# --------------------------------------------------------------------------- #
# 8. Pruning, warm-up and caching over a directory of sessions
# --------------------------------------------------------------------------- #


def part_8_pruning_and_cache(root: Path, cache: Path) -> None:
    banner("8. Date pruning, warm-up and the Parquet cache")

    def timed(label: str, fn: object) -> None:
        started = time.perf_counter()
        result = fn()  # type: ignore[operator]
        print(f"  {label:<46s} {(time.perf_counter() - started) * 1000:7.1f} ms   {result}")

    config = CsvProviderConfig(path=root, cache_dir=cache, source_timezone="UTC")
    universe = (Symbol("MSFT"), Symbol("AAPL"), Symbol("NVDA"))
    everything = DataRequest(
        symbols=universe,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
        timeframe=Timeframe.D1,
    )
    one_day = DataRequest(
        symbols=(Symbol("MSFT"),),
        start=datetime(2024, 1, 31, tzinfo=UTC),
        end=datetime(2024, 1, 31, 23, 59, tzinfo=UTC),
        timeframe=Timeframe.D1,
    )
    warmed = DataRequest(
        symbols=(Symbol("MSFT"),),
        start=datetime(2024, 1, 31, tzinfo=UTC),
        end=datetime(2024, 1, 31, 23, 59, tzinfo=UTC),
        timeframe=Timeframe.D1,
        warmup_bars=10,
    )

    provider = CsvDataProvider(config)
    total = len(provider._discover_files())
    print(f"  {total} session files on disk, one per trading day, whole universe in each\n")

    timed(
        "cold: parse CSV + write Parquet cache",
        lambda: f"{sum(1 for _ in provider.stream_bars(everything))} bars",
    )
    provider.close()

    warm = CsvDataProvider(config)
    timed(
        "warm: same request, fresh provider",
        lambda: f"{sum(1 for _ in warm.stream_bars(everything))} bars",
    )

    print(f"\n  one session out of {total}:")
    print(f"    files considered   {len(warm._prune_files(one_day))} of {total}")
    timed("    read it", lambda: f"{sum(1 for _ in warm.stream_bars(one_day))} bar")

    print("\n  the same session, plus 10 bars of indicator warm-up:")
    print(
        f"    files considered   {len(warm._bar_source(warmed)[0])} of {total}  <- reaches into EARLIER files"
    )
    timed(
        "    read it", lambda: f"{sum(1 for _ in warm.stream_bars(warmed))} bars (1 + 10 priming)"
    )
    warm.close()

    parquet = sorted(cache.rglob("*.parquet"))
    print(f"\n  cache layout under {cache.name}/")
    print("    index-v1.json        symbol set + timestamp range per file, for pruning")
    print(
        f"    files/*.parquet      {len(parquet)} converted, sorted by (ts, symbol), atomically written"
    )
    print(f"    all {total} were converted here only because the first request above asked")
    print("    for the whole history; a run that never widens past one month converts")
    print("    only that month's files and leaves the rest untouched on disk.")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def build_session_directory(root: Path, count: int = 60) -> None:
    """One file per trading day, each holding every ticker — shape A at scale."""
    root.mkdir(parents=True, exist_ok=True)
    day, written = date(2024, 1, 1), 0
    while written < count:
        if day.weekday() < 5:
            with (root / f"{day.isoformat()}.csv").open("w", newline="") as handle:
                writer = csv_module.writer(handle)
                writer.writerow(["ticker", "date", "open", "high", "low", "close", "volume"])
                for offset, symbol in enumerate(("MSFT", "AAPL", "NVDA")):
                    base = 100 + offset * 50 + written * 0.25
                    writer.writerow(
                        [
                            symbol,
                            day.isoformat(),
                            f"{base:.2f}",
                            f"{base + 1.5:.2f}",
                            f"{base - 1.2:.2f}",
                            f"{base + 0.4:.2f}",
                            1_000_000 + written,
                        ]
                    )
            written += 1
        day += timedelta(days=1)


# --------------------------------------------------------------------------- #


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING, format="  \033[33m[%(levelname)s]\033[0m %(message)s"
    )

    samples = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLES
    if not samples.exists():
        print(f"No such path: {samples}", file=sys.stderr)
        return 1

    workdir = Path(tempfile.mkdtemp(prefix="sigmaloop-demo-"))
    try:
        print(f"\033[1mCsvDataProvider demo\033[0m   data: {samples}   scratch: {workdir}")
        part_1_the_request()

        # Shape A on its own: the shipped minute CSV with no quote file beside it.
        aggregates = workdir / "aggregates"
        aggregates.mkdir()
        for name in ("stocks_minute_candlesticks_example.csv",):
            if (samples / name).exists():
                shutil.copy(samples / name, aggregates / "2023-03-28.csv")
        if not any(aggregates.iterdir()):
            for found in sorted(samples.glob("*.csv"))[:1]:
                shutil.copy(found, aggregates / found.name)

        # Bound before the `with` so the concrete type survives: DataProvider.__enter__
        # is annotated to return the base class, and these parts want the CSV provider.
        agg_provider = CsvDataProvider(
            CsvProviderConfig(path=aggregates, cache_dir=workdir / "agg")
        )
        with agg_provider:
            part_2_discovery(agg_provider)
            part_3_streaming(agg_provider)
            part_4_columnar(agg_provider)

        # The shipped folder happens to hold both shapes, which is unusual.
        part_5_quotes_beside_aggregates(samples, workdir / "mixed")

        # Shape B on its own.
        quotes_only = workdir / "quotes_only"
        quotes_only.mkdir()
        quote_file = samples / "stock_quotes_sample.csv"
        if quote_file.exists():
            shutil.copy(quote_file, quotes_only / "2023-03-28.csv")
            part_6_quote_only(quotes_only, workdir / "qonly")

        sessions = workdir / "sessions"
        build_session_directory(sessions)
        part_7_the_feed(sessions, workdir / "feed")
        part_8_pruning_and_cache(sessions, workdir / "pruning")

        banner("Summary")
        print("  MergedDataFeed  consumes stream_bars() and emits MarketSnapshots  [implemented]")
        print("  BacktestEngine  consumes those snapshots, 13 phases per bar       [still a stub]")
        print("  Indicators      consume load_series() / load_many() columnar      [still a stub]")
        print("\n  The provider and the feed are complete; the engine is what consumes them.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
