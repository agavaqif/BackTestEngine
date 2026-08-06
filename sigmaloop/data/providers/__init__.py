"""Concrete data provider implementations.

Each module here is optional at import time: the heavy third-party dependency
(``yfinance``, ``polygon-api-client``) is imported inside the provider, not at
module scope, so a user who only needs CSV never pays for the others.
"""

from __future__ import annotations

from sigmaloop.data.providers.csv_provider import (
    CsvColumnMap,
    CsvDataProvider,
    CsvProviderConfig,
)
from sigmaloop.data.providers.polygon_provider import (
    PolygonDataProvider,
    PolygonProviderConfig,
)
from sigmaloop.data.providers.yahoo_provider import YahooDataProvider, YahooProviderConfig

__all__ = [
    "CsvColumnMap",
    "CsvDataProvider",
    "CsvProviderConfig",
    "PolygonDataProvider",
    "PolygonProviderConfig",
    "YahooDataProvider",
    "YahooProviderConfig",
]
