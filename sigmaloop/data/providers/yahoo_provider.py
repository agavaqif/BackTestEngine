"""Yahoo Finance provider (Data requirement 1b).

Equity/ETF OHLCV only. Yahoo publishes no bid/ask and no reliable greeks, so
:attr:`ProviderCapabilities.supports_quotes` is False and the execution layer
must synthesise a spread (see
:class:`~sigmaloop.execution.pricing.SpreadModel`) for ``WORST`` pricing to
mean anything.

Every response is written through :class:`~sigmaloop.data.cache.DataCache`;
Yahoo is rate-limited and unversioned, so repeated runs must not re-fetch.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.data.cache import DataCache
from sigmaloop.data.provider import DataProvider, DataRequest, ProviderCapabilities
from sigmaloop.domain.account import CorporateAction
from sigmaloop.domain.bar import Bar, BarSeries
from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import AssetClass, Symbol, UtcDatetime

__all__ = ["YahooProviderConfig", "YahooDataProvider"]


@dataclass(frozen=True, slots=True)
class YahooProviderConfig:
    #: Yahoo's "adjusted close" back-adjusts for splits AND dividends. Prefer
    #: raw prices plus explicit corporate-action events for correct cash P&L.
    auto_adjust: bool = False
    include_dividends: bool = True
    include_splits: bool = True
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0
    request_timeout_seconds: float = 30.0
    batch_size: int = 50


class YahooDataProvider(DataProvider):
    """Fetches daily and intraday bars from Yahoo Finance."""

    name: ClassVar[str] = "yahoo"

    def __init__(
        self,
        config: YahooProviderConfig | None = None,
        cache: DataCache | None = None,
    ) -> None:
        raise NotImplementedError

    @property
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def resolve_instrument(self, symbol: Symbol, asset_class: AssetClass) -> Instrument:
        raise NotImplementedError

    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        """Yahoo has no enumeration endpoint; raises ``NotImplementedError``.

        Portfolio-mode universes must therefore be supplied explicitly or come
        from a provider that can enumerate.
        """
        raise NotImplementedError

    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        raise NotImplementedError

    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        raise NotImplementedError

    def load_many(self, request: DataRequest) -> dict[Symbol, BarSeries]:
        """Uses Yahoo's multi-ticker download to cut round trips."""
        raise NotImplementedError

    def corporate_actions(
        self, symbol: Symbol, start: UtcDatetime, end: UtcDatetime
    ) -> Sequence[CorporateAction]:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
