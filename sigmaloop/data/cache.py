"""Two-tier data cache: in-process LRU over an on-disk Parquet store.

Rationale: parameter sweeps and walk-forward analysis re-read the same history
dozens of times. Without a cache, wall-clock is dominated by re-parsing and
re-downloading, which directly violates the performance NFR.

Correctness rule: cache keys include everything that can change the bytes
(symbol, range, timeframe, adjustment flag, provider name and provider version).
A key collision would silently serve wrong data, so keys are conservative.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from sigmaloop.data.provider import DataRequest
from sigmaloop.domain.bar import BarSeries
from sigmaloop.types import Symbol

__all__ = ["CacheKey", "CacheStats", "DataCache", "MemoryDataCache", "ParquetDataCache", "TieredDataCache"]


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Content-addressed identity for one cached series."""

    provider: str
    symbol: Symbol
    timeframe: str
    start_ns: int
    end_ns: int
    adjusted: bool
    schema_version: int = 1

    @classmethod
    def from_request(cls, provider: str, symbol: Symbol, request: DataRequest) -> CacheKey:
        raise NotImplementedError

    def digest(self) -> str:
        """Stable short hash, used as the on-disk filename."""
        raise NotImplementedError


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_resident: int = 0

    @property
    def hit_rate(self) -> float:
        raise NotImplementedError


class DataCache(ABC):
    """Storage contract for cached bar series."""

    @abstractmethod
    def get(self, key: CacheKey) -> BarSeries | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, key: CacheKey, series: BarSeries) -> None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, key: CacheKey) -> bool:
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, key: CacheKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def stats(self) -> CacheStats:
        raise NotImplementedError


class MemoryDataCache(DataCache):
    """LRU cache bounded by approximate resident bytes, not entry count.

    Bytes rather than entries because series sizes vary by four orders of
    magnitude between a daily equity history and an intraday option chain.
    """

    __slots__ = ("_entries", "_max_bytes", "_stats")

    def __init__(self, max_bytes: int = 2 << 30) -> None:
        raise NotImplementedError

    def get(self, key: CacheKey) -> BarSeries | None:
        raise NotImplementedError

    def put(self, key: CacheKey, series: BarSeries) -> None:
        raise NotImplementedError

    def contains(self, key: CacheKey) -> bool:
        raise NotImplementedError

    def invalidate(self, key: CacheKey) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def stats(self) -> CacheStats:
        raise NotImplementedError


class ParquetDataCache(DataCache):
    """Durable columnar cache. Survives process restarts; safe to share.

    Writes are atomic (temp file + rename) so a killed run cannot leave a
    truncated file that a later run would read as valid.
    """

    __slots__ = ("_root", "_stats", "_compression")

    def __init__(self, root: Path, compression: str = "zstd") -> None:
        raise NotImplementedError

    def path_for(self, key: CacheKey) -> Path:
        raise NotImplementedError

    def get(self, key: CacheKey) -> BarSeries | None:
        raise NotImplementedError

    def put(self, key: CacheKey, series: BarSeries) -> None:
        raise NotImplementedError

    def contains(self, key: CacheKey) -> bool:
        raise NotImplementedError

    def invalidate(self, key: CacheKey) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def stats(self) -> CacheStats:
        raise NotImplementedError


class TieredDataCache(DataCache):
    """Memory in front of disk. Reads promote; writes go to both."""

    __slots__ = ("_fast", "_slow")

    def __init__(self, fast: DataCache, slow: DataCache) -> None:
        raise NotImplementedError

    def get(self, key: CacheKey) -> BarSeries | None:
        raise NotImplementedError

    def put(self, key: CacheKey, series: BarSeries) -> None:
        raise NotImplementedError

    def contains(self, key: CacheKey) -> bool:
        raise NotImplementedError

    def invalidate(self, key: CacheKey) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def stats(self) -> CacheStats:
        raise NotImplementedError
