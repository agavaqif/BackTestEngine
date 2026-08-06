"""Market-data value objects: bars, quotes, option chains and snapshots.

Two representations coexist, deliberately:

* **Row form** (:class:`Bar`, :class:`OptionQuote`) — one frozen object per
  observation. Ergonomic for strategy code and for streaming.
* **Columnar form** (:class:`BarSeries`) — struct-of-arrays ``numpy`` buffers.
  This is what indicators vectorise over and what the memory-efficiency NFR
  depends on: 8 float64 columns beat N Python objects by ~20x in RAM and allow
  zero-copy slicing.

The data layer materialises whichever the consumer asks for; the engine loop
uses row form, the indicator warm-up path uses columnar.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sigmaloop.types import (
    EpochNanos,
    InstrumentId,
    OptionRight,
    Price,
    PriceSelection,
    Symbol,
    Timeframe,
    UtcDatetime,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from sigmaloop.domain.instrument import OptionContract

__all__ = [
    "Quote",
    "Bar",
    "OptionQuote",
    "Greeks",
    "OptionChain",
    "MarketSnapshot",
    "BarSeries",
    "PricedInstrument",
]


@runtime_checkable
class PricedInstrument(Protocol):
    """Anything the execution layer can extract a transaction price from.

    Both :class:`Bar` and :class:`OptionQuote` satisfy this, which is what lets
    a single :class:`~sigmaloop.execution.pricing.FillPriceModel` handle equities
    and options without branching on asset class.
    """

    instrument_id: InstrumentId
    timestamp: UtcDatetime

    def price_for(self, selection: PriceSelection, is_buy: bool) -> Price:
        """Resolve the transaction price for one side under ``selection``."""
        ...


@dataclass(frozen=True, slots=True)
class Quote:
    """A top-of-book bid/ask snapshot.

    Providers that supply only OHLCV (CSV, Yahoo) leave this ``None`` on the
    bar; the execution layer then synthesises one via a
    :class:`~sigmaloop.execution.pricing.SpreadModel` so that
    :attr:`~sigmaloop.types.PriceSelection.WORST` remains meaningful.
    """

    bid: Price
    ask: Price
    bid_size: float = 0.0
    ask_size: float = 0.0
    is_synthetic: bool = False

    @property
    def mid(self) -> Price:
        raise NotImplementedError

    @property
    def spread(self) -> Price:
        raise NotImplementedError

    @property
    def spread_pct(self) -> float:
        """Spread as a fraction of mid."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV observation for one instrument over one :class:`Timeframe`.

    ``timestamp`` is the bar's CLOSE instant (right-labelled, UTC). A daily bar
    dated 2024-03-01 therefore becomes visible to the strategy only after the
    2024-03-01 session ends, and under the default
    :attr:`~sigmaloop.types.ExecutionTiming.NEXT_BAR_OPEN` model any signal it
    produces transacts at the 2024-03-04 open.
    """

    instrument_id: InstrumentId
    timestamp: UtcDatetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: float
    timeframe: Timeframe = Timeframe.D1
    quote: Quote | None = None
    vwap: Price | None = None
    trade_count: int | None = None
    is_adjusted: bool = False

    def __post_init__(self) -> None:
        """Validate ``low <= min(open, close) <= max(open, close) <= high``."""
        raise NotImplementedError

    @property
    def epoch_ns(self) -> EpochNanos:
        raise NotImplementedError

    @property
    def typical_price(self) -> Price:
        """``(high + low + close) / 3``."""
        raise NotImplementedError

    @property
    def range(self) -> Price:
        raise NotImplementedError

    @property
    def is_up(self) -> bool:
        raise NotImplementedError

    def price_for(self, selection: PriceSelection, is_buy: bool) -> Price:
        """Transaction price under ``selection``.

        With a :class:`Quote` present: MID -> mid, WORST -> ask when buying /
        bid when selling, BEST -> the inverse, LAST -> close. Without a quote,
        every selection degrades to ``close`` and the caller is expected to have
        applied a :class:`~sigmaloop.execution.pricing.SpreadModel` first.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order option sensitivities, as published by the feed.

    SigmaLoop does not price options itself in v1 — greeks are consumed, not
    computed. A pluggable pricer (Black-Scholes / binomial) can later fill this
    in for providers that omit it.
    """

    delta: float
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    implied_volatility: float = 0.0


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A single option contract's market state at one timestamp.

    Carries everything a strategy needs to pick a contract by delta, DTE or
    moneyness (e.g. "SPY 0DTE 20-delta put") without a second data round-trip.
    """

    instrument_id: InstrumentId
    contract: OptionContract
    timestamp: UtcDatetime
    quote: Quote
    last: Price | None = None
    volume: float = 0.0
    open_interest: float = 0.0
    greeks: Greeks | None = None
    underlying_price: Price | None = None

    @property
    def mid(self) -> Price:
        raise NotImplementedError

    @property
    def delta(self) -> float:
        """Signed delta; raises ``DataError`` if the feed supplied no greeks."""
        raise NotImplementedError

    @property
    def days_to_expiry(self) -> int:
        raise NotImplementedError

    @property
    def is_zero_dte(self) -> bool:
        raise NotImplementedError

    def price_for(self, selection: PriceSelection, is_buy: bool) -> Price:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OptionChain:
    """All quoted contracts on one underlying at one timestamp.

    Selection helpers are the strategy-facing API for options mode; they are
    pure, allocation-light filters over :attr:`quotes` and never touch the
    provider. Indexed internally by ``(expiry, right, strike)`` so delta and
    strike scans are O(log n) rather than O(n).
    """

    underlying_id: InstrumentId
    underlying_symbol: Symbol
    timestamp: UtcDatetime
    underlying_price: Price
    quotes: tuple[OptionQuote, ...]

    # ---- coarse filters -------------------------------------------------- #

    def expiries(self) -> tuple[UtcDatetime, ...]:
        """Sorted distinct expiries present in the chain."""
        raise NotImplementedError

    def strikes(self, expiry: UtcDatetime | None = None) -> tuple[Price, ...]:
        raise NotImplementedError

    def filter(
        self,
        *,
        right: OptionRight | None = None,
        expiry: UtcDatetime | None = None,
        min_dte: int | None = None,
        max_dte: int | None = None,
        min_strike: Price | None = None,
        max_strike: Price | None = None,
        min_open_interest: float | None = None,
        max_spread_pct: float | None = None,
    ) -> OptionChain:
        """Return a narrowed chain. Chainable."""
        raise NotImplementedError

    # ---- single-contract selectors --------------------------------------- #

    def nearest_expiry(self, target_dte: int) -> UtcDatetime:
        raise NotImplementedError

    def by_delta(
        self, target_delta: float, right: OptionRight, expiry: UtcDatetime | None = None
    ) -> OptionQuote:
        """Contract whose |delta| is closest to ``target_delta``.

        This is the "20-delta put" selector from the requirements.
        """
        raise NotImplementedError

    def by_strike(
        self, strike: Price, right: OptionRight, expiry: UtcDatetime | None = None
    ) -> OptionQuote:
        raise NotImplementedError

    def by_moneyness(
        self, moneyness: float, right: OptionRight, expiry: UtcDatetime | None = None
    ) -> OptionQuote:
        """``1.0`` == at-the-money, ``>1`` == ITM for calls."""
        raise NotImplementedError

    def atm(self, right: OptionRight, expiry: UtcDatetime | None = None) -> OptionQuote:
        raise NotImplementedError

    def get(self, instrument_id: InstrumentId) -> OptionQuote | None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[OptionQuote]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Everything observable at a single simulation timestamp.

    This is the unit the :class:`~sigmaloop.data.feed.DataFeed` yields and the
    only market data the engine passes downstream. It is a *closed world*: if an
    instrument is absent from :attr:`bars`, it did not trade this step (holiday,
    halt, or pre-listing) and any order against it is rejected with
    ``NO_MARKET_DATA`` rather than silently filled at a stale price.
    """

    timestamp: UtcDatetime
    bars: Mapping[InstrumentId, Bar]
    chains: Mapping[InstrumentId, OptionChain] = field(default_factory=dict)
    is_session_open: bool = True
    is_session_close: bool = True

    def bar(self, instrument_id: InstrumentId) -> Bar | None:
        raise NotImplementedError

    def require_bar(self, instrument_id: InstrumentId) -> Bar:
        """Like :meth:`bar` but raises ``DataNotAvailableError`` when missing."""
        raise NotImplementedError

    def chain(self, underlying_id: InstrumentId) -> OptionChain | None:
        raise NotImplementedError

    def price(self, instrument_id: InstrumentId) -> Price | None:
        """Mark price (close/mid) used for mark-to-market."""
        raise NotImplementedError

    def instruments(self) -> Sequence[InstrumentId]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class BarSeries:
    """Columnar OHLCV history for ONE instrument — struct of arrays.

    Backed by preallocated ``numpy`` buffers with an append cursor, so streaming
    ingestion is amortised O(1) and indicators can operate on contiguous slices
    with no Python-level iteration.

    Also serves as the rolling window handed to indicators: ``tail(n)`` returns
    zero-copy views, never copies.
    """

    __slots__ = (
        "instrument_id",
        "timeframe",
        "_ts",
        "_open",
        "_high",
        "_low",
        "_close",
        "_volume",
        "_size",
        "_capacity",
    )

    def __init__(
        self,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        capacity: int = 1024,
    ) -> None:
        raise NotImplementedError

    # ---- construction ----------------------------------------------------- #

    @classmethod
    def from_bars(cls, bars: Sequence[Bar]) -> BarSeries:
        raise NotImplementedError

    @classmethod
    def from_arrays(
        cls,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        timestamps: npt.NDArray[np.int64],
        opens: npt.NDArray[np.float64],
        highs: npt.NDArray[np.float64],
        lows: npt.NDArray[np.float64],
        closes: npt.NDArray[np.float64],
        volumes: npt.NDArray[np.float64],
    ) -> BarSeries:
        """Zero-copy adoption of already-validated columns (fast CSV/Parquet path)."""
        raise NotImplementedError

    def append(self, bar: Bar) -> None:
        """Append one bar, growing capacity geometrically when full."""
        raise NotImplementedError

    # ---- column views (zero-copy) ------------------------------------------ #

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        """Epoch-nanosecond column, view of the live region only."""
        raise NotImplementedError

    @property
    def open(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def high(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def low(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def close(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    @property
    def volume(self) -> npt.NDArray[np.float64]:
        raise NotImplementedError

    # ---- access ------------------------------------------------------------ #

    def tail(self, n: int) -> BarSeries:
        """View of the last ``n`` bars. No copy."""
        raise NotImplementedError

    def slice(self, start: UtcDatetime, end: UtcDatetime) -> BarSeries:
        raise NotImplementedError

    def bar_at(self, index: int) -> Bar:
        """Materialise one row back into a :class:`Bar` (negative index allowed)."""
        raise NotImplementedError

    def to_frame(self) -> object:
        """Export to ``pandas.DataFrame``; reporting/debugging only."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
