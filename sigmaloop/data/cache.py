"""Two-tier data cache: in-process LRU over an on-disk Parquet store.

Rationale: parameter sweeps and walk-forward analysis re-read the same history
dozens of times. Without a cache, wall-clock is dominated by re-parsing and
re-downloading, which directly violates the performance NFR.

Correctness rule: cache keys include everything that can change the bytes
(symbol, range, timeframe, adjustment flag, provider name and provider version).
A key collision would silently serve wrong data, so keys are conservative.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigmaloop.data.provider import DataRequest
from sigmaloop.domain.bar import BarSeries
from sigmaloop.types import InstrumentId, Symbol, Timeframe
from sigmaloop.utils.timeutil import to_epoch_ns

__all__ = [
    "CacheKey",
    "CacheStats",
    "DataCache",
    "MemoryDataCache",
    "ParquetDataCache",
    "TieredDataCache",
]

#: Bumped whenever the on-disk column layout *or the key* changes, so an old
#: cache is ignored rather than misread. v2 added ``asset_class`` and
#: ``source_digest``; entries written under v1 are keyed too loosely to trust.
SCHEMA_VERSION = 2

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Content-addressed identity for one cached series."""

    provider: str
    symbol: Symbol
    #: A ticker is not unique across asset classes, and the class is a routing
    #: dimension: ``MergedDataFeed`` picks providers by it and
    #: ``resolve_instrument`` returns a different ``instrument_id`` for it. Left
    #: out, ``SPY`` as an ETF and ``SPY`` as an equity share one entry.
    asset_class: str
    timeframe: str
    start_ns: int
    end_ns: int
    adjusted: bool
    #: Part of the key, not an afterthought: a warm-up request covers a strictly
    #: wider window than the same request without one, so sharing an entry would
    #: silently hand indicators a series that is too short.
    warmup_bars: int = 0
    #: Identity of the bytes *behind* the request: the provider's parsing
    #: configuration plus whatever it uses to tell one revision of a source from
    #: the next. Without it the disk tier outlives the data it describes, and a
    #: re-run over an edited file — or the same file read under a different
    #: timezone — is served the previous answer. Empty only for providers whose
    #: output cannot change for a fixed request.
    source_digest: str = ""
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_request(
        cls,
        provider: str,
        symbol: Symbol,
        request: DataRequest,
        source_digest: str = "",
    ) -> CacheKey:
        return cls(
            provider=provider,
            symbol=symbol,
            asset_class=str(request.asset_class.value),
            timeframe=str(request.timeframe.value),
            start_ns=to_epoch_ns(request.start),
            end_ns=to_epoch_ns(request.end),
            adjusted=request.adjusted,
            warmup_bars=request.warmup_bars,
            source_digest=source_digest,
        )

    def digest(self) -> str:
        """Stable short hash, used as the on-disk filename.

        ``blake2b`` rather than ``hash()``: the built-in is salted per process,
        which would make the disk tier miss on every restart.
        """
        payload = "|".join(
            (
                self.provider,
                self.symbol,
                self.asset_class,
                self.timeframe,
                str(self.start_ns),
                str(self.end_ns),
                "1" if self.adjusted else "0",
                str(self.warmup_bars),
                self.source_digest,
                str(self.schema_version),
            )
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=10).hexdigest()


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_resident: int = 0

    @property
    def hit_rate(self) -> float:
        looked_up = self.hits + self.misses
        return 0.0 if looked_up == 0 else self.hits / looked_up


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


def _series_bytes(series: BarSeries) -> int:
    """Resident size of the six columns a series holds."""
    return int(
        series.timestamps.nbytes
        + series.open.nbytes
        + series.high.nbytes
        + series.low.nbytes
        + series.close.nbytes
        + series.volume.nbytes
    )


class MemoryDataCache(DataCache):
    """LRU cache bounded by approximate resident bytes, not entry count.

    Bytes rather than entries because series sizes vary by four orders of
    magnitude between a daily equity history and an intraday option chain.
    """

    __slots__ = ("_entries", "_lock", "_max_bytes", "_stats")

    def __init__(self, max_bytes: int = 2 << 30) -> None:
        self._entries: OrderedDict[CacheKey, tuple[BarSeries, int]] = OrderedDict()
        self._max_bytes = max(int(max_bytes), 0)
        self._stats = CacheStats()
        self._lock = threading.Lock()

    def get(self, key: CacheKey) -> BarSeries | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self._stats.hits += 1
            return entry[0]

    def put(self, key: CacheKey, series: BarSeries) -> None:
        size = _series_bytes(series)
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._stats.bytes_resident -= existing[1]
            # A single series larger than the whole budget is not cached at all;
            # admitting it would evict everything else to no benefit.
            if size > self._max_bytes:
                return
            self._entries[key] = (series, size)
            self._stats.bytes_resident += size
            while self._stats.bytes_resident > self._max_bytes and self._entries:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._stats.bytes_resident -= evicted_size
                self._stats.evictions += 1

    def contains(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._entries

    def invalidate(self, key: CacheKey) -> None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._stats.bytes_resident -= entry[1]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._stats.bytes_resident = 0

    @property
    def stats(self) -> CacheStats:
        return self._stats


class ParquetDataCache(DataCache):
    """Durable columnar cache. Survives process restarts; safe to share.

    Writes are atomic (temp file + rename) so a killed run cannot leave a
    truncated file that a later run would read as valid.
    """

    __slots__ = ("_compression", "_root", "_stats")

    def __init__(self, root: Path, compression: str = "zstd") -> None:
        self._root = Path(root)
        self._compression = compression
        self._stats = CacheStats()

    def path_for(self, key: CacheKey) -> Path:
        safe_symbol = _UNSAFE_FILENAME.sub("_", key.symbol) or "_"
        safe_provider = _UNSAFE_FILENAME.sub("_", key.provider) or "_"
        return self._root / safe_provider / f"{safe_symbol}-{key.digest()}.parquet"

    def get(self, key: CacheKey) -> BarSeries | None:
        import pyarrow.parquet as pq

        path = self.path_for(key)
        if not path.exists():
            self._stats.misses += 1
            return None
        try:
            table = pq.read_table(path)
        except Exception:  # noqa: BLE001 - a corrupt cache entry must never be fatal
            self._stats.misses += 1
            path.unlink(missing_ok=True)
            return None
        self._stats.hits += 1
        meta = table.schema.metadata or {}
        instrument_id = InstrumentId(meta.get(b"instrument_id", b"").decode() or f"EQ:{key.symbol}")
        timeframe = Timeframe(meta.get(b"timeframe", key.timeframe.encode()).decode())
        return BarSeries.from_arrays(
            instrument_id,
            timeframe,
            table.column("ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False),
            *(
                table.column(name).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
                for name in ("open", "high", "low", "close", "volume")
            ),
        )

    def put(self, key: CacheKey, series: BarSeries) -> None:
        import pyarrow as pa

        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "ts": pa.array(series.timestamps, type=pa.int64()),
                "open": pa.array(series.open, type=pa.float64()),
                "high": pa.array(series.high, type=pa.float64()),
                "low": pa.array(series.low, type=pa.float64()),
                "close": pa.array(series.close, type=pa.float64()),
                "volume": pa.array(series.volume, type=pa.float64()),
            },
            metadata={
                "instrument_id": str(series.instrument_id),
                "timeframe": str(series.timeframe.value),
                "symbol": str(key.symbol),
                "schema_version": str(key.schema_version),
            },
        )
        write_atomic(table, path, compression=self._compression)
        self._stats.bytes_resident += _series_bytes(series)

    def contains(self, key: CacheKey) -> bool:
        return self.path_for(key).exists()

    def invalidate(self, key: CacheKey) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def clear(self) -> None:
        if not self._root.exists():
            return
        for path in self._root.rglob("*.parquet"):
            path.unlink(missing_ok=True)
        self._stats.bytes_resident = 0

    @property
    def stats(self) -> CacheStats:
        return self._stats


def write_atomic(
    table: object, path: Path, *, compression: str = "zstd", row_group_size: int | None = None
) -> None:
    """Write ``table`` to ``path`` so readers never observe a partial file.

    The temp name carries the pid so that two processes converting the same
    source concurrently (parameter sweeps run under ``ProcessExecutor``) cannot
    clobber each other's in-progress write. ``os.replace`` is atomic within a
    filesystem, and the temp file is created in the destination directory to
    guarantee that.
    """
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident():x}.tmp")
    try:
        pq.write_table(
            table,
            tmp,
            compression=compression,
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class TieredDataCache(DataCache):
    """Memory in front of disk. Reads promote; writes go to both."""

    __slots__ = ("_fast", "_min_slow_rows", "_slow")

    def __init__(self, fast: DataCache, slow: DataCache, min_slow_rows: int = 0) -> None:
        """``min_slow_rows`` keeps trivially small series out of the disk tier.

        A portfolio load produces one series per ticker, and each is a separate
        file on disk. For a few hundred short series the writes cost more than
        recomputing them ever would — measured at 20x the load itself — so
        below the threshold the entry is memory-only.
        """
        self._fast = fast
        self._slow = slow
        self._min_slow_rows = max(min_slow_rows, 0)

    def get(self, key: CacheKey) -> BarSeries | None:
        series = self._fast.get(key)
        if series is not None:
            return series
        series = self._slow.get(key)
        if series is not None:
            self._fast.put(key, series)
        return series

    def put(self, key: CacheKey, series: BarSeries) -> None:
        self._fast.put(key, series)
        if len(series) >= self._min_slow_rows:
            self._slow.put(key, series)

    def contains(self, key: CacheKey) -> bool:
        return self._fast.contains(key) or self._slow.contains(key)

    def invalidate(self, key: CacheKey) -> None:
        self._fast.invalidate(key)
        self._slow.invalidate(key)

    def clear(self) -> None:
        self._fast.clear()
        self._slow.clear()

    @property
    def stats(self) -> CacheStats:
        """Combined counters across both tiers."""
        fast, slow = self._fast.stats, self._slow.stats
        return CacheStats(
            hits=fast.hits + slow.hits,
            # A fast-tier miss that the slow tier serves is a hit overall, so
            # only misses that reached disk count.
            misses=slow.misses,
            evictions=fast.evictions + slow.evictions,
            bytes_resident=fast.bytes_resident,
        )
