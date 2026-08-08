"""Data provider plugin contract.

A provider is the only component that knows where bytes come from. It exposes
two access paths, and implementations must supply both:

* :meth:`DataProvider.stream_bars` — a lazy iterator, used by the engine for
  large runs (memory-efficient streaming NFR). Never materialises the full
  history.
* :meth:`DataProvider.load_series` — an eager columnar load, used for indicator
  warm-up, screeners and short runs where random access wins.

Providers are registered under the ``sigmaloop.data_providers`` entry-point
group (see :mod:`sigmaloop.plugins.registry`) and are therefore swappable
without touching the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

from sigmaloop.domain.account import CorporateAction
from sigmaloop.domain.bar import Bar, BarSeries, OptionChain
from sigmaloop.domain.instrument import Instrument, OptionContract
from sigmaloop.errors import ValidationError
from sigmaloop.types import (
    AssetClass,
    OptionRight,
    Price,
    Symbol,
    Timeframe,
    UtcDatetime,
)
from sigmaloop.utils.timeutil import ensure_utc

__all__ = [
    "CompositeDataProvider",
    "DataProvider",
    "DataRequest",
    "OptionChainRequest",
    "OptionsDataProvider",
    "ProviderCapabilities",
]


@dataclass(frozen=True, slots=True)
class DataRequest:
    """A bounded ask for bar data. Hashable, so it doubles as a cache key."""

    symbols: tuple[Symbol, ...]
    start: UtcDatetime
    end: UtcDatetime
    timeframe: Timeframe = Timeframe.D1
    asset_class: AssetClass = AssetClass.EQUITY
    #: Apply split/dividend adjustment at load time (vs. event-driven at run time).
    adjusted: bool = True
    include_quotes: bool = False
    #: Extra bars before ``start`` to satisfy indicator warm-up without lookahead.
    warmup_bars: int = 0

    def __post_init__(self) -> None:
        """Validate ``start < end``, non-empty symbols, tz-aware bounds.

        Also normalises: symbols are coerced to an upper-cased tuple and the
        bounds to UTC, so two requests that differ only in spelling or in the
        caller's local timezone hash equal and share one cache entry.
        """
        symbols = tuple(dict.fromkeys(Symbol(str(s).strip().upper()) for s in self.symbols))
        if not symbols or any(not s for s in symbols):
            raise ValidationError(
                "DataRequest.symbols must contain at least one non-empty ticker.",
                symbols=self.symbols,
            )
        start = ensure_utc(self.start)
        end = ensure_utc(self.end)
        if start >= end:
            raise ValidationError(
                "DataRequest.start must be strictly before end.",
                start=start.isoformat(),
                end=end.isoformat(),
            )
        if self.warmup_bars < 0:
            raise ValidationError(
                "DataRequest.warmup_bars cannot be negative.", warmup_bars=self.warmup_bars
            )
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True, slots=True)
class OptionChainRequest:
    """A bounded ask for option chain snapshots.

    Narrowing at the provider (rather than filtering a full chain in the
    strategy) is what keeps options mode tractable: an unfiltered SPY chain is
    ~5k contracts per timestamp.
    """

    underlying: Symbol
    start: UtcDatetime
    end: UtcDatetime
    timeframe: Timeframe = Timeframe.D1
    rights: tuple[OptionRight, ...] = (OptionRight.CALL, OptionRight.PUT)
    min_dte: int | None = None
    max_dte: int | None = None
    #: Keep strikes within this fraction of spot, e.g. 0.20 == +/-20%.
    strike_window_pct: float | None = None
    require_greeks: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """What a provider can actually serve — checked during config validation.

    Declaring capabilities up front turns "provider silently returns nothing"
    into a clear configuration error before the run starts.
    """

    name: str
    asset_classes: frozenset[AssetClass]
    timeframes: frozenset[Timeframe]
    supports_options: bool = False
    supports_greeks: bool = False
    supports_quotes: bool = False
    supports_corporate_actions: bool = False
    supports_streaming: bool = True
    earliest_data: UtcDatetime | None = None
    requires_credentials: bool = False
    rate_limit_per_minute: int | None = None


class DataProvider(ABC):
    """Base class for every market-data source.

    Implementations must be **stateless with respect to the simulation clock**:
    the engine, not the provider, decides what "now" is. A provider that peeks
    forward is the single easiest way to introduce lookahead, so all range
    arguments are explicit and closed.
    """

    #: Registry key; also the ``sigmaloop.data_providers`` entry-point name.
    name: ClassVar[str] = "abstract"

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    # ---- discovery --------------------------------------------------------- #

    @abstractmethod
    def resolve_instrument(self, symbol: Symbol, asset_class: AssetClass) -> Instrument:
        """Return the canonical :class:`Instrument` for a ticker.

        Raises :class:`~sigmaloop.errors.InstrumentNotFoundError` if unknown.
        """
        raise NotImplementedError

    @abstractmethod
    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        """Every symbol this provider can serve. Used to validate universes."""
        raise NotImplementedError

    def coverage(self, symbol: Symbol, timeframe: Timeframe) -> tuple[UtcDatetime, UtcDatetime] | None:
        """First and last available timestamps, if cheaply knowable."""
        return None

    # ---- bar access -------------------------------------------------------- #

    @abstractmethod
    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        """Yield bars in non-decreasing timestamp order across all symbols.

        Ordering is a hard contract — :class:`~sigmaloop.data.feed.DataFeed`
        k-way merges provider streams and relies on each being sorted. Ties
        within one timestamp may be in any symbol order.
        """
        raise NotImplementedError

    @abstractmethod
    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        """Eagerly load one symbol into columnar form."""
        raise NotImplementedError

    def load_many(self, request: DataRequest) -> dict[Symbol, BarSeries]:
        """Batch form of :meth:`load_series`; override to exploit bulk endpoints."""
        return {symbol: self.load_series(symbol, request) for symbol in request.symbols}

    # ---- corporate actions -------------------------------------------------- #

    def corporate_actions(
        self, symbol: Symbol, start: UtcDatetime, end: UtcDatetime
    ) -> Sequence[CorporateAction]:
        """Splits/dividends in range. Default: empty."""
        return ()

    # ---- lifecycle ---------------------------------------------------------- #

    def open(self) -> None:
        """Acquire handles (file descriptors, HTTP sessions, credentials)."""

    def close(self) -> None:
        """Release resources. Always called by the engine, even on failure."""

    def __enter__(self) -> Self:
        # Self, not DataProvider: `with CsvDataProvider(...) as p` must keep the
        # concrete type, or an options-mode run loses get_chain() to the checker.
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class OptionsDataProvider(DataProvider):
    """A provider that can additionally serve option chains.

    Split from :class:`DataProvider` so that equity-only sources (CSV of OHLCV,
    Yahoo) are not forced to stub out chain methods, and so config validation
    can reject an options-mode run wired to an equity-only provider.
    """

    @abstractmethod
    def get_chain(self, underlying: Symbol, as_of: UtcDatetime) -> OptionChain:
        """Full chain snapshot at one instant."""
        raise NotImplementedError

    @abstractmethod
    def stream_chains(self, request: OptionChainRequest) -> Iterator[OptionChain]:
        """Chain snapshots in ascending timestamp order."""
        raise NotImplementedError

    @abstractmethod
    def resolve_contract(
        self,
        underlying: Symbol,
        expiry: UtcDatetime,
        right: OptionRight,
        strike: Price,
    ) -> OptionContract:
        raise NotImplementedError

    def expirations(self, underlying: Symbol, as_of: UtcDatetime) -> Sequence[UtcDatetime]:
        raise NotImplementedError

    def settlement_price(self, contract: OptionContract) -> Price | None:
        """Official settlement of the underlying on the contract's expiry date.

        Used by the expiry engine to decide worthless / exercise / assignment.
        """
        raise NotImplementedError


class CompositeDataProvider(OptionsDataProvider):
    """Routes requests to child providers by asset class and symbol.

    Enables mixed runs — e.g. equities from a local CSV cache, chains from
    Polygon — behind one provider-shaped object, so the engine stays unaware.
    Resolution order is the order providers were supplied; the first whose
    capabilities match wins.
    """

    name: ClassVar[str] = "composite"

    def __init__(
        self,
        providers: Iterable[DataProvider],
        *,
        overrides: dict[AssetClass, str] | None = None,
    ) -> None:
        raise NotImplementedError

    def provider_for(self, asset_class: AssetClass, symbol: Symbol | None = None) -> DataProvider:
        """Pick the child that can serve this request, or raise ``ConfigurationError``."""
        raise NotImplementedError
