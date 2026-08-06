"""CSV / Parquet file provider (Data requirement 1a).

Reads OHLCV from local files with a configurable column mapping, so users can
point the engine at whatever export they already have. Parses straight into
``numpy`` columns rather than building ``Bar`` objects per row; row objects are
materialised lazily and only for the bars the engine actually visits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from sigmaloop.data.provider import DataProvider, DataRequest, ProviderCapabilities
from sigmaloop.domain.bar import Bar, BarSeries
from sigmaloop.domain.instrument import Instrument
from sigmaloop.types import AssetClass, Symbol, Timeframe

__all__ = ["CsvColumnMap", "CsvProviderConfig", "CsvDataProvider"]


@dataclass(frozen=True, slots=True)
class CsvColumnMap:
    """Maps the engine's canonical fields onto the file's column names."""

    timestamp: str = "date"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    bid: str | None = None
    ask: str | None = None
    adjusted_close: str | None = "adj_close"
    #: Present when one file holds many tickers (long format).
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class CsvProviderConfig:
    """Where the files are and how to interpret them.

    ``path`` is either a directory of ``<SYMBOL>.csv`` files (wide layout, one
    file per ticker) or a single long-format file whose
    :attr:`CsvColumnMap.symbol` column discriminates rows.
    """

    path: Path
    columns: CsvColumnMap = field(default_factory=CsvColumnMap)
    timeframe: Timeframe = Timeframe.D1
    #: IANA zone of naive timestamps in the file; converted to UTC on load.
    source_timezone: str = "America/New_York"
    timestamp_format: str | None = None
    #: True when the file's timestamps label the bar OPEN; shifted to close.
    left_labelled: bool = False
    asset_class: AssetClass = AssetClass.EQUITY
    file_glob: str = "*.csv"
    #: Cache parsed columns as Parquet next to the source for fast re-runs.
    use_parquet_cache: bool = True


class CsvDataProvider(DataProvider):
    """Local-file provider. No network, fully deterministic, fastest to replay."""

    name: ClassVar[str] = "csv"

    def __init__(self, config: CsvProviderConfig) -> None:
        raise NotImplementedError

    @property
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def resolve_instrument(self, symbol: Symbol, asset_class: AssetClass) -> Instrument:
        raise NotImplementedError

    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        """Derived from filenames (wide layout) or a scan of the symbol column."""
        raise NotImplementedError

    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        raise NotImplementedError

    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        raise NotImplementedError

    def _validate_columns(self, symbol: Symbol) -> None:
        """Raise ``DataIntegrityError`` naming the missing/misnamed columns."""
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
