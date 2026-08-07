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

import numpy as np

from sigmaloop.errors import DataIntegrityError, DataNotAvailableError
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
from sigmaloop.utils.timeutil import ensure_utc, from_epoch_ns, to_epoch_ns

if TYPE_CHECKING:
    import numpy.typing as npt

    from sigmaloop.domain.instrument import OptionContract

__all__ = [
    "Bar",
    "BarSeries",
    "Greeks",
    "MarketSnapshot",
    "OptionChain",
    "OptionQuote",
    "PricedInstrument",
    "Quote",
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
        return (self.bid + self.ask) * 0.5

    @property
    def spread(self) -> Price:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as a fraction of mid."""
        mid = self.mid
        return 0.0 if mid <= 0.0 else (self.ask - self.bid) / mid


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
        low, high = self.low, self.high
        # NaN in any field makes every comparison False, so this also rejects
        # unparsed cells without a separate isnan check.
        if not (low <= self.open <= high and low <= self.close <= high and low <= high):
            raise DataIntegrityError(
                "Bar violates low <= min(open, close) <= max(open, close) <= high.",
                instrument_id=self.instrument_id,
                timestamp=self.timestamp,
                open=self.open,
                high=self.high,
                low=self.low,
                close=self.close,
            )
        if self.volume < 0.0:
            raise DataIntegrityError(
                "Bar volume is negative.",
                instrument_id=self.instrument_id,
                timestamp=self.timestamp,
                volume=self.volume,
            )

    @property
    def epoch_ns(self) -> EpochNanos:
        return to_epoch_ns(self.timestamp)

    @property
    def typical_price(self) -> Price:
        """``(high + low + close) / 3``."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def range(self) -> Price:
        return self.high - self.low

    @property
    def is_up(self) -> bool:
        return self.close > self.open

    def price_for(self, selection: PriceSelection, is_buy: bool) -> Price:
        """Transaction price under ``selection``.

        With a :class:`Quote` present: MID -> mid, WORST -> ask when buying /
        bid when selling, BEST -> the inverse, LAST -> close. Without a quote,
        every selection degrades to ``close`` and the caller is expected to have
        applied a :class:`~sigmaloop.execution.pricing.SpreadModel` first.
        """
        quote = self.quote
        if quote is None or selection is PriceSelection.LAST:
            return self.close
        if selection is PriceSelection.MID:
            return quote.mid
        if selection is PriceSelection.WORST:
            return quote.ask if is_buy else quote.bid
        return quote.bid if is_buy else quote.ask


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

    def __post_init__(self) -> None:
        """Normalise the timestamp to tz-aware UTC.

        One conversion per bar, not per instrument: a naive timestamp reaching
        the clock would make every downstream comparison — visibility guards,
        session boundaries, expiry — silently wrong by a whole timezone.
        """
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    def bar(self, instrument_id: InstrumentId) -> Bar | None:
        """The instrument's bar at this step, or ``None`` if it did not trade."""
        return self.bars.get(instrument_id)

    def require_bar(self, instrument_id: InstrumentId) -> Bar:
        """Like :meth:`bar` but raises ``DataNotAvailableError`` when missing."""
        found = self.bars.get(instrument_id)
        if found is None:
            raise DataNotAvailableError(
                instrument_id,
                timestamp=self.timestamp.isoformat(),
                hint=(
                    "The instrument is absent from this snapshot, so it did not trade at "
                    "this step (holiday, halt, or outside its listing). Orders against it "
                    "are rejected with NO_MARKET_DATA rather than filled at a stale price."
                ),
            )
        return found

    def chain(self, underlying_id: InstrumentId) -> OptionChain | None:
        return self.chains.get(underlying_id)

    def price(self, instrument_id: InstrumentId) -> Price | None:
        """Mark price used for mark-to-market, or ``None`` if unobservable here.

        Equity bars mark at the **close**, never at the attached quote's mid.
        The mid is the better estimate of fair value *when the book is current*,
        but :class:`Quote` carries no timestamp: a provider attaches the last
        quote at or before the bar close, which on thin quote coverage can be
        hours stale and is then carried forward unchanged. The close is
        contemporaneous with the bar by construction, so it is never the older
        of the two. Marking to a frozen book produces a flat equity curve while
        the market moves — the failure is silent, and it would be toggled on by
        ``DataRequest.include_quotes``, a flag that otherwise only *attaches*
        information.

        This costs nothing on quote-only datasets, where the derived OHLC
        already tracks the mid. Execution is unaffected: fills price through
        :meth:`Bar.price_for`, which uses the real book and is where spread
        selection belongs. Should ``Quote`` ever grow an as-of timestamp, a
        freshness test here could prefer a current mid again.
        """
        found = self.bars.get(instrument_id)
        if found is not None:
            return found.close
        # Options are carried in chains, not bars. The scan is linear, but it
        # runs only for instruments absent from `bars`, and `chains` is empty in
        # every non-options run.
        for chain in self.chains.values():
            for option in chain.quotes:
                if option.instrument_id == instrument_id:
                    return option.quote.mid
        return None

    def instruments(self) -> Sequence[InstrumentId]:
        """Every instrument that traded at this step."""
        return tuple(self.bars)

    def __len__(self) -> int:
        return len(self.bars)


class BarSeries:
    """Columnar OHLCV history for ONE instrument — struct of arrays.

    Backed by preallocated ``numpy`` buffers with an append cursor, so streaming
    ingestion is amortised O(1) and indicators can operate on contiguous slices
    with no Python-level iteration.

    Also serves as the rolling window handed to indicators: ``tail(n)`` returns
    zero-copy views, never copies.
    """

    __slots__ = (
        "_capacity",
        "_close",
        "_high",
        "_low",
        "_open",
        "_size",
        "_ts",
        "_volume",
        "instrument_id",
        "timeframe",
    )

    def __init__(
        self,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        capacity: int = 1024,
    ) -> None:
        cap = max(int(capacity), 1)
        self.instrument_id = instrument_id
        self.timeframe = timeframe
        self._ts: npt.NDArray[np.int64] = np.empty(cap, dtype=np.int64)
        self._open: npt.NDArray[np.float64] = np.empty(cap, dtype=np.float64)
        self._high: npt.NDArray[np.float64] = np.empty(cap, dtype=np.float64)
        self._low: npt.NDArray[np.float64] = np.empty(cap, dtype=np.float64)
        self._close: npt.NDArray[np.float64] = np.empty(cap, dtype=np.float64)
        self._volume: npt.NDArray[np.float64] = np.empty(cap, dtype=np.float64)
        self._size = 0
        self._capacity = cap

    # ---- construction ----------------------------------------------------- #

    @classmethod
    def _adopt(
        cls,
        instrument_id: InstrumentId,
        timeframe: Timeframe,
        columns: tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ],
    ) -> BarSeries:
        """Build a series that *references* ``columns`` without copying them.

        Bypasses ``__init__``'s allocation; this is the primitive behind
        :meth:`from_arrays`, :meth:`tail` and :meth:`slice`.
        """
        series = object.__new__(cls)
        series.instrument_id = instrument_id
        series.timeframe = timeframe
        series._ts, series._open, series._high, series._low, series._close, series._volume = columns
        series._size = columns[0].shape[0]
        series._capacity = series._size
        return series

    @classmethod
    def from_bars(cls, bars: Sequence[Bar]) -> BarSeries:
        if not bars:
            raise DataIntegrityError("Cannot build a BarSeries from zero bars.")
        first = bars[0]
        count = len(bars)
        ts = np.empty(count, dtype=np.int64)
        opens = np.empty(count, dtype=np.float64)
        highs = np.empty(count, dtype=np.float64)
        lows = np.empty(count, dtype=np.float64)
        closes = np.empty(count, dtype=np.float64)
        volumes = np.empty(count, dtype=np.float64)
        for i, bar in enumerate(bars):
            ts[i] = to_epoch_ns(bar.timestamp)
            opens[i] = bar.open
            highs[i] = bar.high
            lows[i] = bar.low
            closes[i] = bar.close
            volumes[i] = bar.volume
        return cls._adopt(
            first.instrument_id, first.timeframe, (ts, opens, highs, lows, closes, volumes)
        )

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
        ts = np.asarray(timestamps, dtype=np.int64)
        cols = tuple(np.asarray(c, dtype=np.float64) for c in (opens, highs, lows, closes, volumes))
        for name, column in zip(("open", "high", "low", "close", "volume"), cols, strict=True):
            if column.shape != ts.shape:
                raise DataIntegrityError(
                    f"Column {name!r} has length {column.shape[0]}, expected {ts.shape[0]}.",
                    instrument_id=instrument_id,
                )
        return cls._adopt(instrument_id, timeframe, (ts, *cols))  # type: ignore[arg-type]

    def append(self, bar: Bar) -> None:
        """Append one bar, growing capacity geometrically when full."""
        if self._size == self._capacity:
            self._grow()
        i = self._size
        self._ts[i] = to_epoch_ns(bar.timestamp)
        self._open[i] = bar.open
        self._high[i] = bar.high
        self._low[i] = bar.low
        self._close[i] = bar.close
        self._volume[i] = bar.volume
        self._size = i + 1

    def _grow(self) -> None:
        new_cap = max(self._capacity * 2, 1)
        self._ts = np.resize(self._ts, new_cap)
        self._open = np.resize(self._open, new_cap)
        self._high = np.resize(self._high, new_cap)
        self._low = np.resize(self._low, new_cap)
        self._close = np.resize(self._close, new_cap)
        self._volume = np.resize(self._volume, new_cap)
        self._capacity = new_cap

    # ---- column views (zero-copy) ------------------------------------------ #

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        """Epoch-nanosecond column, view of the live region only."""
        return self._ts[: self._size]

    @property
    def open(self) -> npt.NDArray[np.float64]:
        return self._open[: self._size]

    @property
    def high(self) -> npt.NDArray[np.float64]:
        return self._high[: self._size]

    @property
    def low(self) -> npt.NDArray[np.float64]:
        return self._low[: self._size]

    @property
    def close(self) -> npt.NDArray[np.float64]:
        return self._close[: self._size]

    @property
    def volume(self) -> npt.NDArray[np.float64]:
        return self._volume[: self._size]

    # ---- access ------------------------------------------------------------ #

    def tail(self, n: int) -> BarSeries:
        """View of the last ``n`` bars. No copy."""
        start = max(self._size - max(n, 0), 0)
        return self._window(start, self._size)

    def slice(self, start: UtcDatetime, end: UtcDatetime) -> BarSeries:
        """View of the bars in the closed interval ``[start, end]``."""
        ts = self.timestamps
        lo = int(np.searchsorted(ts, to_epoch_ns(start), side="left"))
        hi = int(np.searchsorted(ts, to_epoch_ns(end), side="right"))
        return self._window(lo, hi)

    def _window(self, lo: int, hi: int) -> BarSeries:
        return BarSeries._adopt(
            self.instrument_id,
            self.timeframe,
            (
                self._ts[lo:hi],
                self._open[lo:hi],
                self._high[lo:hi],
                self._low[lo:hi],
                self._close[lo:hi],
                self._volume[lo:hi],
            ),
        )

    def bar_at(self, index: int) -> Bar:
        """Materialise one row back into a :class:`Bar` (negative index allowed)."""
        i = index + self._size if index < 0 else index
        if not 0 <= i < self._size:
            raise IndexError(f"Bar index {index} out of range for series of length {self._size}.")
        return Bar(
            instrument_id=self.instrument_id,
            timestamp=from_epoch_ns(int(self._ts[i])),
            open=float(self._open[i]),
            high=float(self._high[i]),
            low=float(self._low[i]),
            close=float(self._close[i]),
            volume=float(self._volume[i]),
            timeframe=self.timeframe,
        )

    def to_frame(self) -> object:
        """Export to ``pandas.DataFrame``; reporting/debugging only."""
        import pandas as pd

        return pd.DataFrame(
            {
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            },
            index=pd.DatetimeIndex(self.timestamps.astype("datetime64[ns]"), tz="UTC", name="ts"),
        )

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"BarSeries({self.instrument_id}, {self.timeframe.value}, "
            f"n={self._size})"
        )
