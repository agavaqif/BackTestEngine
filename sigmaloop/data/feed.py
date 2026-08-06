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

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from sigmaloop.data.provider import DataProvider, DataRequest, OptionChainRequest
from sigmaloop.domain.account import CorporateAction
from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import InstrumentId, Symbol, Timeframe, UtcDatetime

__all__ = [
    "FeedPlan",
    "DataFeed",
    "MergedDataFeed",
    "PrefetchDataFeed",
    "ReplayDataFeed",
    "HistoryWindow",
]


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
        raise NotImplementedError

    def estimate_bar_count(self) -> int:
        """Rough sizing hint used to preallocate curves and series buffers."""
        raise NotImplementedError


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
        """Actions with an ex-date at this step; applied before trading."""
        raise NotImplementedError

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
        raise NotImplementedError


class MergedDataFeed(DataFeed):
    """Default feed: heap-merges one or more providers into snapshots.

    Groups all bars sharing a timestamp into a single
    :class:`~sigmaloop.domain.bar.MarketSnapshot`, attaches any option chains
    for that instant, and yields once per timestamp.
    """

    __slots__ = ("_plan", "_providers", "_heap", "_registry", "_current", "_iterators")

    def __init__(
        self,
        plan: FeedPlan,
        providers: Sequence[DataProvider],
    ) -> None:
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


class PrefetchDataFeed(DataFeed):
    """Wraps another feed with a bounded background read-ahead buffer.

    Overlaps provider I/O (file parsing, HTTP) with strategy computation. The
    buffer is bounded so that memory stays O(buffer_size), preserving the
    streaming guarantee while removing I/O from the critical path.
    """

    __slots__ = ("_inner", "_buffer_size", "_thread", "_queue")

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
    """In-memory feed over a fixed list of snapshots. Testing and fixtures only."""

    __slots__ = ("_snapshots", "_index", "_instruments")

    def __init__(
        self,
        snapshots: Sequence[MarketSnapshot],
        instruments: Sequence[Instrument] = (),
    ) -> None:
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


class HistoryWindow:
    """Read-only, lookahead-safe view of bars already seen for one instrument.

    Handed to strategies as ``ctx.history(instrument_id)``. It wraps the live
    :class:`~sigmaloop.domain.bar.BarSeries` but exposes only the region up to
    and including the current bar, so a strategy physically cannot index into
    the future.
    """

    __slots__ = ("_series", "_cursor", "instrument_id")

    def __init__(self, instrument_id: InstrumentId) -> None:
        raise NotImplementedError

    def last(self, n: int = 1) -> object:
        """Last ``n`` closes as a numpy view."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
