"""Polygon.io REST provider (Data requirement 1c).

The reference implementation of :class:`~sigmaloop.data.provider.OptionsDataProvider`:
it serves equities, NBBO quotes, full option chains and greeks, which is what
options mode needs.

Chains are the expensive part — an unfiltered SPY snapshot is thousands of
contracts per timestamp. The provider therefore pushes
:class:`~sigmaloop.data.provider.OptionChainRequest` filters (DTE window, strike
window, rights) into the API query rather than fetching everything and
discarding client-side.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.data.cache import DataCache
from sigmaloop.data.provider import (
    DataRequest,
    OptionChainRequest,
    OptionsDataProvider,
    ProviderCapabilities,
)
from sigmaloop.domain.bar import Bar, BarSeries, OptionChain
from sigmaloop.domain.instrument import Instrument, OptionContract
from sigmaloop.types import AssetClass, OptionRight, Price, Symbol, UtcDatetime

__all__ = ["PolygonProviderConfig", "PolygonDataProvider"]


@dataclass(frozen=True, slots=True)
class PolygonProviderConfig:
    api_key: str
    base_url: str = "https://api.polygon.io"
    #: Requests per minute; the provider self-throttles below the plan limit.
    rate_limit_per_minute: int = 100
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    request_timeout_seconds: float = 60.0
    #: Concurrent HTTP workers used when pre-fetching many symbols.
    max_concurrency: int = 8
    use_adjusted: bool = False
    include_greeks: bool = True
    #: Drop chain rows with no two-sided market; they are not tradeable.
    drop_unquoted_contracts: bool = True


class PolygonDataProvider(OptionsDataProvider):
    """Equities, quotes and option chains from Polygon."""

    name: ClassVar[str] = "polygon"

    def __init__(
        self,
        config: PolygonProviderConfig,
        cache: DataCache | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    # ---- equities ---------------------------------------------------------- #

    def resolve_instrument(self, symbol: Symbol, asset_class: AssetClass) -> Instrument:
        raise NotImplementedError

    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        """Backed by the reference-tickers endpoint; paginated and cached."""
        raise NotImplementedError

    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        raise NotImplementedError

    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        raise NotImplementedError

    # ---- options ----------------------------------------------------------- #

    def get_chain(self, underlying: Symbol, as_of: UtcDatetime) -> OptionChain:
        raise NotImplementedError

    def stream_chains(self, request: OptionChainRequest) -> Iterator[OptionChain]:
        raise NotImplementedError

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
        raise NotImplementedError

    # ---- lifecycle ---------------------------------------------------------- #

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
