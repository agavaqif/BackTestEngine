"""MergedDataFeed and ReplayDataFeed: ordering, grouping and residency.

The merge is the engine's only input, so two properties are worth more than any
amount of feature coverage and are pinned down here:

* every bar reaches exactly one snapshot, and snapshots come out strictly
  increasing in time;
* memory stays proportional to the number of *sources*, not to the length of
  history — the streaming NFR.
"""

from __future__ import annotations

import gc
import itertools
import shutil
import tracemalloc
from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from conftest import AAPL_ABSENT, MARCH, request_for

from sigmaloop.data.calendar import ContinuousCalendar, NyseCalendar
from sigmaloop.data.feed import DataFeed, FeedPlan, MergedDataFeed, ReplayDataFeed
from sigmaloop.data.provider import (
    DataProvider,
    DataRequest,
    OptionChainRequest,
    OptionsDataProvider,
    ProviderCapabilities,
)
from sigmaloop.data.providers.csv_provider import CsvDataProvider, CsvProviderConfig
from sigmaloop.domain.account import CorporateAction
from sigmaloop.domain.bar import Bar, BarSeries, MarketSnapshot, OptionChain
from sigmaloop.domain.instrument import Equity, Instrument, OptionContract
from sigmaloop.errors import ConfigurationError, DataIntegrityError
from sigmaloop.types import (
    AssetClass,
    CorporateActionType,
    InstrumentId,
    OptionRight,
    Price,
    Symbol,
    Timeframe,
    UtcDatetime,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def csv_provider(root: Path, cache: Path, **kwargs: object) -> CsvDataProvider:
    return CsvDataProvider(
        CsvProviderConfig(path=root, cache_dir=cache, source_timezone="UTC", **kwargs)  # type: ignore[arg-type]
    )


def plan_for(*symbols: str, timeframe: Timeframe = Timeframe.D1) -> FeedPlan:
    return FeedPlan(
        bar_requests=(request_for(*symbols, timeframe=timeframe, **MARCH),),
        timeframe=timeframe,
    )


def feed_over(root: Path, cache: Path, *symbols: str) -> MergedDataFeed:
    return MergedDataFeed(plan_for(*symbols), [csv_provider(root, cache)])


def timestamps(snapshots: Sequence[MarketSnapshot]) -> list[UtcDatetime]:
    return [snapshot.timestamp for snapshot in snapshots]


class SyntheticProvider(DataProvider):
    """Bars straight out of a generator — no files, no parser, no allocations
    that would drown out what a memory measurement is trying to see."""

    name = "synthetic"

    def __init__(
        self,
        symbols: Sequence[str],
        bars: int,
        *,
        start: UtcDatetime = datetime(2024, 1, 1, tzinfo=UTC),
        timeframe: Timeframe = Timeframe.D1,
        actions: Sequence[CorporateAction] = (),
        descending: bool = False,
    ) -> None:
        self._symbols = tuple(Symbol(s) for s in symbols)
        self._bars = bars
        self._start = start
        self._timeframe = timeframe
        self._actions = tuple(actions)
        self._descending = descending
        self.opened = 0
        self.closed = 0

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            asset_classes=frozenset({AssetClass.EQUITY}),
            timeframes=frozenset({self._timeframe}),
            supports_corporate_actions=bool(self._actions),
        )

    def resolve_instrument(
        self, symbol: Symbol, asset_class: AssetClass = AssetClass.EQUITY
    ) -> Instrument:
        ticker = Symbol(str(symbol).upper())
        return Equity(instrument_id=Equity.make_id(ticker), symbol=ticker)

    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        return self._symbols

    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        wanted = [s for s in self._symbols if s in request.symbols]
        width = self._timeframe.duration
        steps = range(self._bars - 1, -1, -1) if self._descending else range(self._bars)
        for step in steps:
            stamp = self._start + width * step
            for symbol in wanted:
                yield Bar(
                    instrument_id=Equity.make_id(symbol),
                    timestamp=stamp,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1_000.0,
                    timeframe=self._timeframe,
                )

    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        return BarSeries.from_bars([b for b in self.stream_bars(request)])

    def corporate_actions(
        self, symbol: Symbol, start: UtcDatetime, end: UtcDatetime
    ) -> Sequence[CorporateAction]:
        return tuple(a for a in self._actions if a.symbol == symbol)

    def open(self) -> None:
        self.opened += 1

    def close(self) -> None:
        self.closed += 1


class SyntheticOptionsProvider(SyntheticProvider, OptionsDataProvider):
    """Adds one one-contract chain per timestamp, for the chain-merge path."""

    name = "synthetic-options"

    @property
    def capabilities(self) -> ProviderCapabilities:
        base = super().capabilities
        return ProviderCapabilities(
            name=self.name,
            asset_classes=base.asset_classes,
            timeframes=base.timeframes,
            supports_options=True,
        )

    def get_chain(self, underlying: Symbol, as_of: UtcDatetime) -> OptionChain:
        return self._chain(underlying, as_of)

    def stream_chains(self, request: OptionChainRequest) -> Iterator[OptionChain]:
        width = self._timeframe.duration
        for step in range(self._bars):
            yield self._chain(request.underlying, self._start + width * step)

    def resolve_contract(
        self, underlying: Symbol, expiry: UtcDatetime, right: OptionRight, strike: Price
    ) -> OptionContract:
        raise NotImplementedError

    def _chain(self, underlying: Symbol, as_of: UtcDatetime) -> OptionChain:
        return OptionChain(
            underlying_id=Equity.make_id(underlying),
            underlying_symbol=underlying,
            timestamp=as_of,
            underlying_price=100.5,
            quotes=(),
        )


def synthetic_feed(
    bars: int, symbols: Sequence[str] = ("AAA", "BBB", "CCC"), *, per_provider: bool = False
) -> MergedDataFeed:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(
        symbols=tuple(Symbol(s) for s in symbols),
        start=start,
        end=start + timedelta(days=bars + 1),
    )
    plan = FeedPlan(bar_requests=(request,))
    providers: list[DataProvider] = (
        [SyntheticProvider([symbol], bars) for symbol in symbols]
        if per_provider
        else [SyntheticProvider(symbols, bars)]
    )
    return MergedDataFeed(plan, providers)


# --------------------------------------------------------------------------- #
# FeedPlan
# --------------------------------------------------------------------------- #


def test_feed_plan_collects_symbols() -> None:
    plan = FeedPlan(
        bar_requests=(request_for("MSFT", "AAPL", **MARCH),),
        chain_requests=(
            OptionChainRequest(underlying=Symbol("SPY"), start=MARCH["start"], end=MARCH["end"]),
        ),
    )
    assert plan.symbols == frozenset({Symbol("MSFT"), Symbol("AAPL"), Symbol("SPY")})


def test_feed_plan_bar_count_is_an_upper_bound() -> None:
    plan = FeedPlan(bar_requests=(request_for("MSFT", **MARCH),), warmup_bars=5)
    estimate = plan.estimate_bar_count()
    assert estimate >= 30, "March has 30 days of span plus warm-up"
    assert estimate < 60, "an upper bound, not an unbounded guess"


# --------------------------------------------------------------------------- #
# Ordering and grouping over the real CSV provider
# --------------------------------------------------------------------------- #


def test_snapshots_are_strictly_increasing(wide_layout: Path, tmp_path: Path) -> None:
    snapshots = list(feed_over(wide_layout, tmp_path / "c", "MSFT", "AAPL"))
    stamps = timestamps(snapshots)

    assert len(stamps) == 20, "20 sessions, not 40 bars"
    assert all(later > earlier for earlier, later in itertools.pairwise(stamps))


@pytest.mark.parametrize("layout", ["wide_layout", "per_day_layout", "long_layout"])
def test_every_bar_reaches_exactly_one_snapshot(
    layout: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    """Whatever the file layout, the merge neither drops nor duplicates a bar."""
    root: Path = request.getfixturevalue(layout)
    streamed = list(
        csv_provider(root, tmp_path / "direct").stream_bars(request_for("MSFT", "AAPL", **MARCH))
    )
    snapshots = list(feed_over(root, tmp_path / "feed", "MSFT", "AAPL"))

    merged = [(s.timestamp, b) for s in snapshots for b in s.bars.values()]
    assert len(merged) == len(streamed) == 40
    assert {(b.instrument_id, b.timestamp) for _, b in merged} == {
        (b.instrument_id, b.timestamp) for b in streamed
    }
    assert all(stamp == b.timestamp for stamp, b in merged), "a bar sits under its own timestamp"


def test_bars_sharing_a_timestamp_are_grouped(per_day_layout: Path, tmp_path: Path) -> None:
    snapshots = list(feed_over(per_day_layout, tmp_path / "c", "MSFT", "AAPL"))

    assert len(snapshots) == 20
    assert all(len(s) == 2 for s in snapshots)
    assert all(set(s.instruments()) == {"EQ:MSFT", "EQ:AAPL"} for s in snapshots)


def test_a_symbol_that_did_not_trade_is_absent(ragged_layout: Path, tmp_path: Path) -> None:
    """Absence, not a stale repeat: the snapshot is a closed world."""
    snapshots = list(feed_over(ragged_layout, tmp_path / "c", "MSFT", "AAPL"))

    assert len(snapshots) == 20, "MSFT traded every session"
    missing = [index for index, s in enumerate(snapshots) if s.bar(InstrumentId("EQ:AAPL")) is None]
    assert missing == list(AAPL_ABSENT)
    assert all(s.bar(InstrumentId("EQ:MSFT")) is not None for s in snapshots)
    assert snapshots[AAPL_ABSENT[0]].price(InstrumentId("EQ:AAPL")) is None


def test_current_tracks_the_last_yielded_snapshot(wide_layout: Path, tmp_path: Path) -> None:
    feed = feed_over(wide_layout, tmp_path / "c", "MSFT", "AAPL")
    assert feed.current is None

    seen: list[MarketSnapshot] = []
    for snapshot in feed:
        assert feed.current is snapshot
        seen.append(snapshot)
    assert feed.current is seen[-1]


def test_instruments_are_resolved_before_the_first_bar(wide_layout: Path, tmp_path: Path) -> None:
    feed = feed_over(wide_layout, tmp_path / "c", "MSFT", "AAPL")
    assert [i.instrument_id for i in feed.instruments()] == ["EQ:AAPL", "EQ:MSFT"]


# --------------------------------------------------------------------------- #
# Several providers
# --------------------------------------------------------------------------- #


def test_two_providers_are_merged_into_one_stream(wide_layout: Path, tmp_path: Path) -> None:
    """One directory per source, k-way merged on (timestamp, instrument)."""
    msft_only = tmp_path / "src_msft"
    aapl_only = tmp_path / "src_aapl"
    msft_only.mkdir()
    aapl_only.mkdir()
    shutil.copy(wide_layout / "MSFT.csv", msft_only / "MSFT.csv")
    shutil.copy(wide_layout / "AAPL.csv", aapl_only / "AAPL.csv")

    feed = MergedDataFeed(
        plan_for("MSFT", "AAPL"),
        [
            csv_provider(msft_only, tmp_path / "c1"),
            csv_provider(aapl_only, tmp_path / "c2"),
        ],
    )
    snapshots = list(feed)

    assert len(snapshots) == 20
    assert all(set(s.instruments()) == {"EQ:MSFT", "EQ:AAPL"} for s in snapshots)
    assert [i.instrument_id for i in feed.instruments()] == ["EQ:AAPL", "EQ:MSFT"]


def test_the_first_provider_wins_a_duplicate(wide_layout: Path, tmp_path: Path) -> None:
    """Two sources serving the same instrument at the same instant is not a
    reason to emit it twice."""
    feed = MergedDataFeed(
        plan_for("MSFT"),
        [csv_provider(wide_layout, tmp_path / "c1"), csv_provider(wide_layout, tmp_path / "c2")],
    )
    snapshots = list(feed)

    assert len(snapshots) == 20
    assert all(len(s) == 1 for s in snapshots)
    assert feed._shadowed == 20


def test_no_providers_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="at least one data provider"):
        MergedDataFeed(plan_for("MSFT"), [])


def test_a_provider_that_cannot_serve_the_request_is_named() -> None:
    plan = plan_for("MSFT", timeframe=Timeframe.M1)
    feed = MergedDataFeed(plan, [SyntheticProvider(["MSFT"], 3, timeframe=Timeframe.D1)])
    with pytest.raises(ConfigurationError, match="No configured provider"):
        feed.prepare()


# --------------------------------------------------------------------------- #
# The provider ordering contract
# --------------------------------------------------------------------------- #


def test_out_of_order_provider_is_rejected() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=10))
    feed = MergedDataFeed(
        FeedPlan(bar_requests=(request,)), [SyntheticProvider(["AAA"], 5, descending=True)]
    )
    with pytest.raises(DataIntegrityError, match="non-decreasing timestamp order"):
        list(feed)


# --------------------------------------------------------------------------- #
# Residency — the streaming NFR
# --------------------------------------------------------------------------- #


def drain(feed: DataFeed) -> int:
    return sum(1 for _ in feed)


def peak_bytes(work: object) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        work()  # type: ignore[operator]
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_heap_holds_at_most_one_entry_per_source() -> None:
    """Residency is O(sources), not O(bars): that is the whole merge argument."""
    feed = synthetic_feed(bars=500, per_provider=True)
    for snapshot in feed:
        assert len(snapshot) == 3
        assert len(feed._heap) <= len(feed._iterators) == 3


def test_peak_memory_is_independent_of_history_length() -> None:
    drain(synthetic_feed(bars=10))  # warm the interpreter, not the measurement

    short = peak_bytes(lambda: drain(synthetic_feed(bars=250)))
    long_history = peak_bytes(lambda: drain(synthetic_feed(bars=5_000)))
    materialised = peak_bytes(lambda: list(synthetic_feed(bars=5_000)))

    assert long_history < short * 2, (
        f"20x the history cost {long_history / short:.1f}x the memory; "
        "the feed is holding on to bars it has already yielded"
    )
    assert materialised > long_history * 10, (
        "the comparison is meaningless unless keeping every snapshot is "
        f"visibly more expensive ({materialised} vs {long_history} bytes)"
    )


# --------------------------------------------------------------------------- #
# Session flags, chains, corporate actions, mid-run subscription
# --------------------------------------------------------------------------- #


def test_session_flags_come_from_the_calendar() -> None:
    """390 one-minute bars over a single NYSE session."""
    open_close = datetime(2024, 1, 2, 14, 31, tzinfo=UTC)
    request = DataRequest(
        symbols=(Symbol("AAA"),),
        start=open_close,
        end=open_close + timedelta(days=1),
        timeframe=Timeframe.M1,
    )
    plan = FeedPlan(bar_requests=(request,), timeframe=Timeframe.M1)
    provider = SyntheticProvider(["AAA"], 390, start=open_close, timeframe=Timeframe.M1)
    snapshots = list(MergedDataFeed(plan, [provider], calendar=NyseCalendar()))

    assert len(snapshots) == 390
    assert snapshots[0].is_session_open is True
    assert snapshots[0].is_session_close is False
    assert snapshots[-1].timestamp == datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    assert snapshots[-1].is_session_close is True
    assert snapshots[-1].is_session_open is False
    assert sum(1 for s in snapshots if s.is_session_close) == 1


def test_daily_bars_close_a_session_on_a_24_7_calendar() -> None:
    feed = synthetic_feed(bars=5)
    calendar_feed = MergedDataFeed(
        feed._plan, [SyntheticProvider(["AAA"], 5)], calendar=ContinuousCalendar()
    )
    assert all(s.is_session_close for s in calendar_feed)


def _premarket_feed() -> MergedDataFeed:
    """60 one-minute bars ending 09:10 ET — entirely before the NYSE open."""
    first = datetime(2024, 1, 2, 13, 11, tzinfo=UTC)  # 08:11 ET
    request = DataRequest(
        symbols=(Symbol("AAA"),),
        start=first,
        end=first + timedelta(days=1),
        timeframe=Timeframe.M1,
    )
    return MergedDataFeed(
        FeedPlan(bar_requests=(request,), timeframe=Timeframe.M1),
        [SyntheticProvider(["AAA"], 60, start=first, timeframe=Timeframe.M1)],
        calendar=NyseCalendar(),
    )


def test_a_run_with_no_session_close_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre/post-market data against NyseCalendar never closes a session, so MOC
    orders, EOD liquidation and expiry silently never fire. Say so."""
    with caplog.at_level("WARNING", logger="sigmaloop.data.feed"):
        snapshots = list(_premarket_feed())

    assert len(snapshots) == 60
    assert not any(s.is_session_close for s in snapshots), "fixture must be off-session"
    assert "no session close" in caplog.text
    assert "NyseCalendar" in caplog.text


def test_a_run_that_does_close_a_session_stays_quiet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="sigmaloop.data.feed"):
        list(
            MergedDataFeed(
                synthetic_feed(bars=5)._plan,
                [SyntheticProvider(["AAA"], 5)],
                calendar=ContinuousCalendar(),
            )
        )
    assert "no session close" not in caplog.text


def test_no_calendar_means_no_session_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Without a calendar the flags default to True and the check does not apply."""
    with caplog.at_level("WARNING", logger="sigmaloop.data.feed"):
        list(synthetic_feed(bars=5))
    assert "no session close" not in caplog.text


def test_option_chains_are_attached_to_the_snapshot() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=10)
    plan = FeedPlan(
        bar_requests=(DataRequest(symbols=(Symbol("AAA"),), start=start, end=end),),
        chain_requests=(OptionChainRequest(underlying=Symbol("AAA"), start=start, end=end),),
    )
    snapshots = list(MergedDataFeed(plan, [SyntheticOptionsProvider(["AAA"], 4)]))

    assert len(snapshots) == 4
    for snapshot in snapshots:
        chain = snapshot.chain(InstrumentId("EQ:AAA"))
        assert chain is not None
        assert chain.timestamp == snapshot.timestamp


def test_chain_requests_need_an_options_provider() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=10)
    plan = FeedPlan(
        bar_requests=(DataRequest(symbols=(Symbol("AAA"),), start=start, end=end),),
        chain_requests=(OptionChainRequest(underlying=Symbol("AAA"), start=start, end=end),),
    )
    feed = MergedDataFeed(plan, [SyntheticProvider(["AAA"], 4)])
    with pytest.raises(ConfigurationError, match="OptionsDataProvider"):
        feed.prepare()


def test_corporate_actions_are_indexed_by_ex_date() -> None:
    split = CorporateAction(
        action_type=CorporateActionType.SPLIT,
        instrument_id=InstrumentId("EQ:AAA"),
        symbol=Symbol("AAA"),
        ex_date=datetime(2024, 1, 3, tzinfo=UTC).date(),
        ratio=2.0,
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=10))
    feed = MergedDataFeed(
        FeedPlan(bar_requests=(request,)), [SyntheticProvider(["AAA"], 5, actions=[split])]
    )
    feed.prepare()

    assert feed.corporate_actions_at(datetime(2024, 1, 3, tzinfo=UTC)) == [split]
    assert feed.corporate_actions_at(datetime(2024, 1, 4, tzinfo=UTC)) == ()


def test_add_instrument_subscribes_from_the_next_bar(wide_layout: Path, tmp_path: Path) -> None:
    """A screener admitting a name mid-run must not be handed the bar it just
    made its decision without."""
    feed = feed_over(wide_layout, tmp_path / "c", "MSFT")
    seen: list[MarketSnapshot] = []
    for index, snapshot in enumerate(feed):
        if index == 4:
            feed.add_instrument(
                Equity(instrument_id=Equity.make_id(Symbol("AAPL")), symbol=Symbol("AAPL"))
            )
        seen.append(snapshot)

    assert len(seen) == 20
    with_aapl = [s for s in seen if s.bar(InstrumentId("EQ:AAPL")) is not None]
    assert len(with_aapl) == 15, "everything after the bar it was added on"
    assert seen[4].bar(InstrumentId("EQ:AAPL")) is None
    assert seen[5].bar(InstrumentId("EQ:AAPL")) is not None


def test_providers_are_opened_and_closed() -> None:
    provider = SyntheticProvider(["AAA"], 3)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=10))
    feed = MergedDataFeed(FeedPlan(bar_requests=(request,)), [provider])

    feed.prepare()
    feed.prepare()
    assert provider.opened == 1, "prepare is idempotent"
    feed.close()
    assert provider.closed == 1


def test_a_closed_feed_can_be_read_again(wide_layout: Path, tmp_path: Path) -> None:
    """Iteration is single-pass, but close() rewinds the plan — which is what a
    walk-forward fold re-reading the same window needs."""
    feed = feed_over(wide_layout, tmp_path / "c", "MSFT", "AAPL")
    first = timestamps(list(feed))
    assert list(feed) == [], "the heap is consumed as it goes"

    feed.close()
    assert timestamps(list(feed)) == first


# --------------------------------------------------------------------------- #
# ReplayDataFeed
# --------------------------------------------------------------------------- #


def replay_snapshot(day: int) -> MarketSnapshot:
    stamp = datetime(2024, 1, day, tzinfo=UTC)
    return MarketSnapshot(
        timestamp=stamp,
        bars={
            InstrumentId("EQ:AAA"): Bar(
                instrument_id=InstrumentId("EQ:AAA"),
                timestamp=stamp,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1.0,
            )
        },
    )


def test_replay_feed_yields_its_fixtures() -> None:
    instrument = Equity(instrument_id=Equity.make_id(Symbol("AAA")), symbol=Symbol("AAA"))
    feed = ReplayDataFeed([replay_snapshot(1), replay_snapshot(2)], [instrument])
    feed.prepare()

    assert feed.current is None
    assert timestamps(list(feed)) == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    assert feed.current is not None
    assert feed.current.timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert feed.instruments() == (instrument,)
    assert feed.corporate_actions_at(datetime(2024, 1, 1, tzinfo=UTC)) == ()
    feed.close()


def test_replay_feed_rejects_a_misordered_fixture() -> None:
    with pytest.raises(DataIntegrityError, match="strictly increasing"):
        ReplayDataFeed([replay_snapshot(2), replay_snapshot(1)])
    with pytest.raises(DataIntegrityError, match="strictly increasing"):
        ReplayDataFeed([replay_snapshot(1), replay_snapshot(1)])


def test_replay_feed_cannot_subscribe_mid_run() -> None:
    feed = ReplayDataFeed([replay_snapshot(1)])
    with pytest.raises(NotImplementedError, match="mid-run"):
        feed.add_instrument(
            Equity(instrument_id=Equity.make_id(Symbol("AAA")), symbol=Symbol("AAA"))
        )


# --------------------------------------------------------------------------- #
# Re-reading a closed feed — walk-forward hygiene
# --------------------------------------------------------------------------- #


class ActionProvider(SyntheticProvider):
    """Reports one 2:1 split, so a re-read can be checked for double-counting."""

    EX_DATE = date(2024, 1, 3)

    @property
    def capabilities(self) -> ProviderCapabilities:
        base = super().capabilities
        return ProviderCapabilities(
            name=base.name,
            asset_classes=base.asset_classes,
            timeframes=base.timeframes,
            supports_corporate_actions=True,
        )

    def corporate_actions(
        self, symbol: Symbol, start: UtcDatetime, end: UtcDatetime
    ) -> Sequence[CorporateAction]:
        return (
            CorporateAction(
                action_type=CorporateActionType.SPLIT,
                instrument_id=Equity.make_id(symbol),
                symbol=symbol,
                ex_date=self.EX_DATE,
                ratio=2.0,
            ),
        )


def test_corporate_actions_do_not_accumulate_across_re_reads() -> None:
    """prepare() re-collects from the providers. Left uncleared, fold 2 of a
    walk-forward applies one 2:1 split twice — a 4x adjustment, silently."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=6))
    feed = MergedDataFeed(FeedPlan(bar_requests=(request,)), [ActionProvider(["AAA"], 5)])

    for _ in range(3):
        list(feed)
        at_ex = feed.corporate_actions_at(datetime(2024, 1, 3, tzinfo=UTC))
        assert len(at_ex) == 1
        feed.close()


def test_two_providers_reporting_one_action_do_not_double_it() -> None:
    """Duplicate bars resolve first-provider-wins; duplicate actions must too.

    A 2:1 split counted twice is a 4x adjustment, and nothing downstream can
    tell that apart from a genuine 4:1.
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=6))
    feed = MergedDataFeed(
        FeedPlan(bar_requests=(request,)),
        [ActionProvider(["AAA"], 5), ActionProvider(["AAA"], 5)],
    )
    feed.prepare()

    assert len(feed.corporate_actions_at(datetime(2024, 1, 3, tzinfo=UTC))) == 1


def test_a_mid_run_subscription_can_be_made_again_on_the_next_fold() -> None:
    """add_instrument returns early for anything already registered, and the
    registry outlived close(). A name a screener admitted in fold 1 was then
    silently refused its subscription in every later fold: instruments() still
    advertised it, but no source was opened and it never traded again."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=6))
    feed = MergedDataFeed(FeedPlan(bar_requests=(request,)), [SyntheticProvider(["AAA", "BBB"], 5)])
    admitted = Equity(instrument_id=Equity.make_id(Symbol("BBB")), symbol=Symbol("BBB"))

    folds = []
    for _ in range(2):
        seen: set[str] = set()
        for index, snapshot in enumerate(feed):
            if index == 0:
                feed.add_instrument(admitted)
            seen |= set(snapshot.instruments())
        folds.append(seen)
        feed.close()

    assert "EQ:BBB" in folds[0]
    assert folds[1] == folds[0], "a re-read must resubscribe, not inherit fold 1's registry"


def test_closing_forgets_where_the_previous_run_ended() -> None:
    """add_instrument floors a new subscription at the current bar. A stale
    _current puts that floor at the end of the window, so the new symbol never
    appears — no error, it is simply absent."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    request = DataRequest(symbols=(Symbol("AAA"),), start=start, end=start + timedelta(days=6))
    feed = MergedDataFeed(FeedPlan(bar_requests=(request,)), [SyntheticProvider(["AAA", "BBB"], 5)])

    list(feed)
    feed.close()
    assert feed.current is None

    feed.prepare()
    feed.add_instrument(Equity(instrument_id=Equity.make_id(Symbol("BBB")), symbol=Symbol("BBB")))
    seen = {iid for snapshot in feed for iid in snapshot.instruments()}

    assert "EQ:BBB" in seen, "the mid-run subscription was floored out of existence"
    assert "EQ:AAA" in seen
