"""Time-ordered market data feed — the engine's single input.

The feed is the boundary that turns N independent, per-symbol provider streams
into one chronologically ordered sequence of :class:`MarketSnapshot`. Everything
downstream of it sees exactly one timestamp at a time, which is the structural
guarantee against lookahead.

Algorithm
---------
A k-way merge over per-symbol iterators using a binary heap keyed on
``(epoch_ns, instrument_id)``. Cost is O(total_bars * log n_symbols) with
O(n_symbols) resident memory — independent of history length, which is what
satisfies the streaming NFR for portfolio-mode runs over thousands of tickers.
"""

from __future__ import annotations

import heapq
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import TypeAlias

from sigmaloop.data.calendar import TradingCalendar
from sigmaloop.data.provider import (
    DataProvider,
    DataRequest,
    OptionChainRequest,
    OptionsDataProvider,
)
from sigmaloop.domain.account import CorporateAction
from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionChain
from sigmaloop.domain.instrument import Instrument
from sigmaloop.errors import ConfigurationError, DataError, DataIntegrityError
from sigmaloop.types import AssetClass, InstrumentId, Symbol, Timeframe, UtcDatetime
from sigmaloop.utils.timeutil import ensure_utc, to_epoch_ns

__all__ = [
    "DataFeed",
    "FeedPlan",
    "HistoryWindow",
    "MergedDataFeed",
    "PrefetchDataFeed",
    "ReplayDataFeed",
]

_LOG = logging.getLogger("sigmaloop.data.feed")

#: What a merged source yields. Both carry a ``timestamp``; the heap keys on it.
_Update: TypeAlias = "Bar | OptionChain"
#: ``(epoch_ns, instrument key, source index, payload)``. The first three are
#: unique across the heap — one entry per source at a time — so the payload is
#: never compared, which matters because neither ``Bar`` nor ``OptionChain`` is
#: ordered.
_HeapEntry: TypeAlias = "tuple[int, str, int, _Update]"

_NEVER_SEEN = -(2**63)


@dataclass(frozen=True, slots=True)
class FeedPlan:
    """Fully resolved description of what a run will read.

    Built once by :class:`~sigmaloop.engine.core.BacktestEngine` from the
    config + strategy declarations, then validated against provider
    capabilities. Materialising the plan before the loop starts is what lets
    the engine fail fast with an actionable message instead of dying on bar
    40,000 with a missing symbol.
    """

    bar_requests: tuple[DataRequest, ...]
    chain_requests: tuple[OptionChainRequest, ...] = ()
    timeframe: Timeframe = Timeframe.D1
    #: Bars loaded before ``start`` purely to warm indicators; not traded on.
    warmup_bars: int = 0

    @property
    def symbols(self) -> frozenset[Symbol]:
        """Every ticker the plan touches, bars and chain underlyings alike."""
        tickers: set[Symbol] = set()
        for request in self.bar_requests:
            tickers.update(request.symbols)
        tickers.update(request.underlying for request in self.chain_requests)
        return frozenset(tickers)

    def estimate_bar_count(self) -> int:
        """Rough sizing hint used to preallocate curves and series buffers.

        An upper bound on the number of *steps* — distinct timestamps — not on
        the number of bars: holidays and halts only ever make the real count
        smaller, and over-allocating a buffer costs one resize less, whereas
        under-allocating costs a copy.
        """
        if not self.bar_requests:
            return 0
        width = _frame_seconds(self.timeframe)
        longest = max(
            (request.end - request.start).total_seconds() for request in self.bar_requests
        )
        return int(longest // width) + self.warmup_bars + 1


def _frame_seconds(timeframe: Timeframe) -> float:
    """Nominal bar width in seconds; ticks are sized as if they were one second."""
    if timeframe is Timeframe.TICK:
        return 1.0
    return timeframe.duration.total_seconds()


class DataFeed(ABC):
    """Iterable source of chronologically ordered market snapshots."""

    @abstractmethod
    def __iter__(self) -> Iterator[MarketSnapshot]:
        """Yield snapshots with strictly increasing timestamps."""
        raise NotImplementedError

    @abstractmethod
    def instruments(self) -> Sequence[Instrument]:
        """Every instrument this feed may emit, known after :meth:`prepare`."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> None:
        """Open providers, resolve instruments, load warm-up history.

        Called once before the loop. May be expensive; must be idempotent.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def corporate_actions_at(self, timestamp: UtcDatetime) -> Sequence[CorporateAction]:
        """Actions with an ex-date at this step; applied before trading.

        Default: none. A feed whose providers cannot serve corporate actions
        reports nothing rather than pretending the question is unanswerable.
        """
        return ()

    @property
    @abstractmethod
    def current(self) -> MarketSnapshot | None:
        """Most recently yielded snapshot. ``None`` before the first step."""
        raise NotImplementedError

    def add_instrument(self, instrument: Instrument) -> None:
        """Subscribe mid-run — needed when a portfolio strategy's screener
        admits a new symbol, or when an options strategy selects a contract
        that was not part of the initial plan.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot subscribe to an instrument mid-run."
        )


class MergedDataFeed(DataFeed):
    """Default feed: heap-merges one or more providers into snapshots.

    Groups all bars sharing a timestamp into a single
    :class:`~sigmaloop.domain.bar.MarketSnapshot`, attaches any option chains
    for that instant, and yields once per timestamp.

    Every request goes to every provider whose capabilities cover it, and the
    first provider in the list wins when two of them return the same instrument
    at the same instant. That keeps "CSV for one directory, CSV for another"
    working without a routing table, at the cost of asking a provider for
    symbols it may not have. When sources are genuinely heterogeneous — CSV
    equities plus Polygon chains — wrap them in a
    :class:`~sigmaloop.data.provider.CompositeDataProvider` instead, so each
    request is fetched exactly once.

    Iteration is single-pass: the heap is consumed as it goes, which is the
    whole point of the streaming NFR. Call :meth:`prepare` again after
    :meth:`close` to read the same plan a second time.
    """

    __slots__ = (
        "_actions",
        "_calendar",
        "_closes",
        "_current",
        "_heap",
        "_iterators",
        "_last_emitted_ns",
        "_plan",
        "_prepared",
        "_providers",
        "_registry",
        "_shadowed",
        "_source_last",
        "_steps",
    )

    def __init__(
        self,
        plan: FeedPlan,
        providers: Sequence[DataProvider],
        *,
        calendar: TradingCalendar | None = None,
    ) -> None:
        """``calendar`` supplies the session flags on each snapshot when given."""
        sources = tuple(providers)
        if not sources:
            raise ConfigurationError(
                "MergedDataFeed needs at least one data provider.",
                symbols=sorted(plan.symbols),
            )
        self._plan = plan
        self._providers = sources
        self._calendar = calendar
        self._registry: dict[InstrumentId, Instrument] = {}
        self._iterators: list[Iterator[_Update] | None] = []
        self._source_last: list[int] = []
        self._heap: list[_HeapEntry] = []
        self._actions: dict[date, list[CorporateAction]] = {}
        self._current: MarketSnapshot | None = None
        self._prepared = False
        self._last_emitted_ns = _NEVER_SEEN
        self._shadowed = 0
        self._steps = 0
        self._closes = 0

    # ---- lifecycle ------------------------------------------------------------ #

    def prepare(self) -> None:
        if self._prepared:
            return
        for provider in self._providers:
            provider.open()
        for request in self._plan.bar_requests:
            for provider in self._providers_for(request):
                self._register_instruments(provider, request)
                self._collect_actions(provider, request)
                self._open_source(provider.stream_bars(request))
        for chain_request in self._plan.chain_requests:
            for options_provider in self._chain_providers_for(chain_request):
                self._open_source(options_provider.stream_chains(chain_request))
        self._prepared = True

    def close(self) -> None:
        self._heap.clear()
        self._iterators.clear()
        self._source_last.clear()
        for provider in self._providers:
            provider.close()
        self._prepared = False
        # A second read starts at the beginning of the plan again, so the
        # already-emitted watermark and the per-run counters go with the
        # streams that set them.
        self._last_emitted_ns = _NEVER_SEEN
        self._steps = 0
        self._closes = 0
        self._shadowed = 0
        # prepare() re-collects from the providers, and _collect_actions appends
        # blind. Left in place, a walk-forward that re-reads the same window
        # would see one 2:1 split twice on the second fold and three times on
        # the third — a 4x then 8x adjustment, applied silently.
        self._actions.clear()
        # add_instrument() floors a new subscription at _current.timestamp to
        # keep it from seeing bars the run has already passed. A _current left
        # over from the previous run puts that floor at the *end* of the window,
        # so every bar of the new symbol is dropped and it simply never appears.
        self._current = None

    # ---- iteration ------------------------------------------------------------ #

    def __iter__(self) -> Iterator[MarketSnapshot]:
        self.prepare()
        heap = self._heap
        while heap:
            epoch_ns = heap[0][0]
            timestamp = heap[0][3].timestamp
            bars: dict[InstrumentId, Bar] = {}
            chains: dict[InstrumentId, OptionChain] = {}
            # Drain every entry at this instant, refilling each source as it is
            # popped: a source that holds several symbols will immediately offer
            # its next bar at the same timestamp, and that one belongs to this
            # snapshot too.
            while heap and heap[0][0] == epoch_ns:
                _, _, index, item = heapq.heappop(heap)
                self._absorb(item, bars, chains)
                self._pull(index)
            if epoch_ns <= self._last_emitted_ns:
                # Unreachable while every source is monotone (checked in _pull),
                # which is exactly why it is worth asserting: the alternative to
                # failing here is a run whose clock silently goes backwards.
                raise DataIntegrityError(
                    "Feed produced a non-increasing snapshot timestamp.",
                    timestamp=timestamp.isoformat(),
                )
            self._last_emitted_ns = epoch_ns
            snapshot = self._snapshot(timestamp, bars, chains)
            if snapshot.is_session_close:
                self._closes += 1
            self._steps += 1
            self._current = snapshot
            yield snapshot
        self._warn_if_never_closed()

    def _warn_if_never_closed(self) -> None:
        """Flag a run in which the calendar never marked a session close.

        Everything that keys off the end of a session — MOC orders, EOD
        liquidation, option expiry — simply never fires in that case, and the
        run still reports success. The usual cause is data that lies outside the
        calendar's regular hours: a pre/post-market export against
        :class:`~sigmaloop.data.calendar.NyseCalendar` has no session close
        anywhere in it.
        """
        if self._calendar is None or not self._steps or self._closes:
            return
        _LOG.warning(
            "Feed produced %d snapshot(s) but %s marked no session close in any of them, "
            "so MOC orders, end-of-day liquidation and option expiry will never fire. "
            "The data may sit outside regular hours — use ContinuousCalendar if the "
            "session model does not apply.",
            self._steps,
            type(self._calendar).__name__,
        )

    @property
    def current(self) -> MarketSnapshot | None:
        return self._current

    def instruments(self) -> Sequence[Instrument]:
        self.prepare()
        return tuple(instrument for _, instrument in sorted(self._registry.items()))

    def corporate_actions_at(self, timestamp: UtcDatetime) -> Sequence[CorporateAction]:
        return self._actions.get(ensure_utc(timestamp).date(), ())

    def add_instrument(self, instrument: Instrument) -> None:
        """Subscribe to ``instrument`` from the next timestamp onwards.

        Bars at or before the current one are dropped rather than replayed: the
        snapshot for this instant has already been handed to the strategy, and
        re-opening it would hand a screener data its own decision was made
        without.
        """
        if instrument.instrument_id in self._registry:
            return
        self._registry[instrument.instrument_id] = instrument
        template = self._template_request(instrument)
        if template is None:
            raise ConfigurationError(
                "Cannot subscribe mid-run: the feed plan has no bar request to model "
                "the new subscription on.",
                instrument_id=instrument.instrument_id,
            )
        after_ns: int | None = None
        start = template.start
        if self._current is not None:
            start = self._current.timestamp
            after_ns = to_epoch_ns(start)
        if start >= template.end:
            return  # the run is already past this plan's window
        request = replace(
            template, symbols=(instrument.symbol,), start=start, warmup_bars=0
        )
        for provider in self._providers_for(request):
            self._open_source(provider.stream_bars(request), after_ns=after_ns)

    # ---- merge internals -------------------------------------------------------- #

    def _open_source(
        self, iterator: Iterator[_Update], *, after_ns: int | None = None
    ) -> None:
        """Register a stream and prime the heap with its first item."""
        if after_ns is not None:
            floor = after_ns
            iterator = itertools.dropwhile(
                lambda item: to_epoch_ns(item.timestamp) <= floor, iterator
            )
        index = len(self._iterators)
        self._iterators.append(iterator)
        self._source_last.append(_NEVER_SEEN)
        self._pull(index)

    def _pull(self, index: int) -> None:
        """Take the next item from one source and push it onto the heap."""
        iterator = self._iterators[index]
        if iterator is None:
            return
        item = next(iterator, None)
        if item is None:
            # Drop the reference rather than the slot: indices are heap keys.
            self._iterators[index] = None
            return
        epoch_ns = to_epoch_ns(item.timestamp)
        if epoch_ns < self._source_last[index]:
            raise DataIntegrityError(
                "A provider yielded data out of order. DataProvider.stream_bars must "
                "yield in non-decreasing timestamp order; the k-way merge depends on it.",
                instrument_id=_key_of(item),
                timestamp=item.timestamp.isoformat(),
            )
        self._source_last[index] = epoch_ns
        heapq.heappush(self._heap, (epoch_ns, _key_of(item), index, item))

    def _absorb(
        self,
        item: _Update,
        bars: dict[InstrumentId, Bar],
        chains: dict[InstrumentId, OptionChain],
    ) -> None:
        if isinstance(item, Bar):
            if item.instrument_id in bars:
                self._note_shadowed(item.instrument_id)
                return
            bars[item.instrument_id] = item
        elif item.underlying_id not in chains:
            chains[item.underlying_id] = item
        else:
            self._note_shadowed(item.underlying_id)

    def _note_shadowed(self, instrument_id: InstrumentId) -> None:
        self._shadowed += 1
        if self._shadowed == 1:
            _LOG.warning(
                "Two providers returned %s at the same timestamp; keeping the first "
                "configured one. Route sources through a CompositeDataProvider to make "
                "the choice explicit.",
                instrument_id,
            )

    def _snapshot(
        self,
        timestamp: UtcDatetime,
        bars: dict[InstrumentId, Bar],
        chains: dict[InstrumentId, OptionChain],
    ) -> MarketSnapshot:
        calendar = self._calendar
        if calendar is None:
            return MarketSnapshot(timestamp=timestamp, bars=bars, chains=chains)
        frame = self._plan.timeframe
        return MarketSnapshot(
            timestamp=timestamp,
            bars=bars,
            chains=chains,
            is_session_open=calendar.is_session_open(timestamp, frame),
            is_session_close=calendar.is_session_close(timestamp, frame),
        )

    # ---- provider routing -------------------------------------------------------- #

    def _providers_for(self, request: DataRequest) -> tuple[DataProvider, ...]:
        capable = tuple(
            provider
            for provider in self._providers
            if _can_serve(provider, request.asset_class, request.timeframe)
        )
        if not capable:
            raise ConfigurationError(
                "No configured provider can serve this request.",
                symbols=list(request.symbols),
                asset_class=request.asset_class.value,
                timeframe=request.timeframe.value,
                providers=[p.capabilities.name for p in self._providers],
            )
        return capable

    def _chain_providers_for(
        self, request: OptionChainRequest
    ) -> tuple[OptionsDataProvider, ...]:
        capable = tuple(
            provider
            for provider in self._providers
            if isinstance(provider, OptionsDataProvider)
            and provider.capabilities.supports_options
        )
        if not capable:
            raise ConfigurationError(
                "Options mode needs an OptionsDataProvider, but none of the configured "
                "providers serves chains.",
                underlying=request.underlying,
                providers=[p.capabilities.name for p in self._providers],
            )
        return capable

    def _template_request(self, instrument: Instrument) -> DataRequest | None:
        """A plan request to model a mid-run subscription on.

        Prefers one already asking for this asset class, so a new option
        contract does not inherit an equity request's window and adjustment.
        """
        for request in self._plan.bar_requests:
            if request.asset_class is instrument.asset_class:
                return request
        return self._plan.bar_requests[0] if self._plan.bar_requests else None

    def _register_instruments(self, provider: DataProvider, request: DataRequest) -> None:
        for symbol in request.symbols:
            try:
                instrument = provider.resolve_instrument(symbol, request.asset_class)
            except DataError:
                # Another provider may know it; a symbol no provider resolves is
                # simply absent from instruments(), and its orders are rejected
                # with NO_MARKET_DATA when it never produces a bar.
                continue
            self._registry.setdefault(instrument.instrument_id, instrument)

    def _collect_actions(self, provider: DataProvider, request: DataRequest) -> None:
        if not provider.capabilities.supports_corporate_actions:
            return
        for symbol in request.symbols:
            for action in provider.corporate_actions(symbol, request.start, request.end):
                self._actions.setdefault(action.ex_date, []).append(action)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MergedDataFeed(sources={len(self._iterators)}, "
            f"instruments={len(self._registry)}, prepared={self._prepared})"
        )


def _can_serve(provider: DataProvider, asset_class: AssetClass, timeframe: Timeframe) -> bool:
    capabilities = provider.capabilities
    return asset_class in capabilities.asset_classes and timeframe in capabilities.timeframes


def _key_of(item: _Update) -> str:
    """Heap tiebreaker: the instrument the update is about."""
    return item.instrument_id if isinstance(item, Bar) else item.underlying_id


class PrefetchDataFeed(DataFeed):
    """Wraps another feed with a bounded background read-ahead buffer.

    Overlaps provider I/O (file parsing, HTTP) with strategy computation. The
    buffer is bounded so that memory stays O(buffer_size), preserving the
    streaming guarantee while removing I/O from the critical path.
    """

    __slots__ = ("_buffer_size", "_inner", "_queue", "_thread")

    def __init__(self, inner: DataFeed, buffer_size: int = 512) -> None:
        raise NotImplementedError

    def __iter__(self) -> Iterator[MarketSnapshot]:
        raise NotImplementedError

    def instruments(self) -> Sequence[Instrument]:
        raise NotImplementedError

    def prepare(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    @property
    def current(self) -> MarketSnapshot | None:
        raise NotImplementedError


class ReplayDataFeed(DataFeed):
    """In-memory feed over a fixed list of snapshots. Testing and fixtures only.

    Enforces the strictly-increasing contract at construction rather than during
    iteration, so a mis-ordered fixture fails in the test that built it instead
    of somewhere inside the engine.
    """

    __slots__ = ("_index", "_instruments", "_snapshots")

    def __init__(
        self,
        snapshots: Sequence[MarketSnapshot],
        instruments: Sequence[Instrument] = (),
    ) -> None:
        ordered = tuple(snapshots)
        for previous, following in itertools.pairwise(ordered):
            if following.timestamp <= previous.timestamp:
                raise DataIntegrityError(
                    "ReplayDataFeed snapshots must be strictly increasing in time.",
                    previous=previous.timestamp.isoformat(),
                    offending=following.timestamp.isoformat(),
                )
        self._snapshots = ordered
        self._instruments = tuple(instruments)
        self._index = -1

    def __iter__(self) -> Iterator[MarketSnapshot]:
        for index, snapshot in enumerate(self._snapshots):
            self._index = index
            yield snapshot

    def instruments(self) -> Sequence[Instrument]:
        return self._instruments

    def prepare(self) -> None:
        """Nothing to open: the snapshots are already in memory."""

    def close(self) -> None:
        """Nothing to release."""

    @property
    def current(self) -> MarketSnapshot | None:
        return self._snapshots[self._index] if self._index >= 0 else None

    def __len__(self) -> int:
        return len(self._snapshots)


class HistoryWindow:
    """Read-only, lookahead-safe view of bars already seen for one instrument.

    Handed to strategies as ``ctx.history(instrument_id)``. It wraps the live
    :class:`~sigmaloop.domain.bar.BarSeries` but exposes only the region up to
    and including the current bar, so a strategy physically cannot index into
    the future.
    """

    __slots__ = ("_cursor", "_series", "instrument_id")

    def __init__(self, instrument_id: InstrumentId) -> None:
        raise NotImplementedError

    def last(self, n: int = 1) -> object:
        """Last ``n`` closes as a numpy view."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
