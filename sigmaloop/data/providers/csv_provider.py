"""CSV / Parquet file provider (Data requirement 1a).

Reads OHLCV **and** top-of-book quotes from local files with a configurable
column mapping, so users can point the engine at whatever export they already
have. Parses straight into ``numpy``/``arrow`` columns rather than building
``Bar`` objects per row; row objects are materialised lazily and only for the
bars the engine actually visits.

Layouts
-------
``path`` may be a single file or a directory, and a directory's files may be
sliced along either axis — one file per ticker (``MSFT.csv``), one file per
session holding the whole universe (``2023-03-28.csv``), one long-format file
holding everything, or any mixture. The provider never assumes a layout: it
reads each file's header to classify it, then records which symbols and which
timestamp range it actually contains.

Speed
-----
Three mechanisms, in the order they pay off:

1. **Parquet conversion cache.** Each source CSV is normalised once into a
   columnar Parquet file sorted by ``(ts, symbol)``. Re-runs skip CSV parsing
   entirely, which is the dominant cost in a parameter sweep.
2. **Date and symbol pruning.** A persisted index records each file's symbol
   set and timestamp range, so a request touches only overlapping files. Before
   the index exists, filename conventions (``MSFT.csv``, ``2023-03-28.csv``)
   prune candidates without opening them at all.
3. **Predicate pushdown.** Within a file, Parquet row-group statistics skip
   blocks outside the requested window, and only the needed columns are read.

Streaming stays memory-bounded: :meth:`CsvDataProvider.stream_bars` walks the
requested window in time slices sized to a row budget, so residency is
independent of how much history the files hold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeAlias

import numpy as np

from sigmaloop.data.cache import (
    CacheKey,
    DataCache,
    MemoryDataCache,
    ParquetDataCache,
    TieredDataCache,
    write_atomic,
)
from sigmaloop.data.provider import DataProvider, DataRequest, ProviderCapabilities
from sigmaloop.domain.bar import Bar, BarSeries, Quote
from sigmaloop.domain.instrument import Equity, Instrument
from sigmaloop.errors import DataIntegrityError, DataNotAvailableError, DataProviderError
from sigmaloop.types import AssetClass, Symbol, Timeframe, UtcDatetime
from sigmaloop.utils.timeutil import from_epoch_ns, to_epoch_ns

if TYPE_CHECKING:
    import numpy.typing as npt
    import pyarrow as pa

    #: ``(ts, open, high, low, close, volume)`` as parallel columns.
    _BarColumns: TypeAlias = tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]

__all__ = [
    "CsvColumnMap",
    "CsvDataProvider",
    "CsvFileKind",
    "CsvProviderConfig",
    "DuplicatePolicy",
    "EpochUnit",
    "QuotePriceBasis",
    "TimestampPolicy",
]

_LOG = logging.getLogger("sigmaloop.data.providers.csv")

#: Bumped when the normalised Parquet layout changes, invalidating old caches.
CACHE_SCHEMA_VERSION: Final = 1

#: Epoch-integer timestamps are scaled into this window. It spans less than one
#: decade, which is what makes "which power of ten is this?" have exactly one
#: answer. Data outside it must declare its unit via ``epoch_unit``.
_PLAUSIBLE_MIN_NS: Final = 631_152_000_000_000_000  # 1990-01-01T00:00:00Z
_PLAUSIBLE_MAX_NS: Final = 4_102_444_800_000_000_000  # 2100-01-01T00:00:00Z
_PLAUSIBLE_CENTER_NS: Final = (_PLAUSIBLE_MIN_NS + _PLAUSIBLE_MAX_NS) // 2
_INT64_MAX: Final = np.iinfo(np.int64).max

_NS_PER_DAY: Final = 86_400_000_000_000
_EPOCH_UTC: Final = datetime(1970, 1, 1, tzinfo=UTC)

#: Widening factors tried when back-filling ``warmup_bars`` of history. Sessions
#: are sparse relative to wall-clock (weekends, overnight), so the first guess
#: intentionally overshoots rather than paying for a second read.
_WARMUP_PADS: Final = (3.0, 12.0, 60.0, 400.0)


class CsvFileKind(StrEnum):
    """What a source file contains, decided from its header."""

    AGGREGATE = "aggregate"
    QUOTE = "quote"


class EpochUnit(StrEnum):
    """Unit of an integer timestamp column."""

    AUTO = "auto"
    SECONDS = "s"
    MILLIS = "ms"
    MICROS = "us"
    NANOS = "ns"


class TimestampPolicy(StrEnum):
    """What to do with a row whose timestamp does not scale into a real date.

    ``REPAIR`` rescales the individual value by its own power of ten and warns
    with a count; ``DROP`` discards the row and warns; ``STRICT`` raises
    :class:`~sigmaloop.errors.DataIntegrityError`.
    """

    REPAIR = "repair"
    DROP = "drop"
    STRICT = "strict"


class QuotePriceBasis(StrEnum):
    """Which side of the book becomes the OHLC of a quote-derived bar."""

    MID = "mid"
    BID = "bid"
    ASK = "ask"


class DuplicatePolicy(StrEnum):
    """What to do when two rows share a ``(symbol, timestamp)``."""

    KEEP_LAST = "keep_last"
    KEEP_FIRST = "keep_first"
    ERROR = "error"


# --------------------------------------------------------------------------- #
# Column mapping
# --------------------------------------------------------------------------- #

#: Alias tables for header auto-detection, in priority order. Names are matched
#: after lower-casing and replacing spaces with underscores.
_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    # ``sip_timestamp`` beats ``participant_timestamp``: the consolidated tape
    # time is the one every venue agrees on. ``trf_timestamp`` is deliberately
    # absent — it is zero for non-TRF prints.
    "timestamp": (
        "window_start",
        "sip_timestamp",
        "participant_timestamp",
        "timestamp",
        "datetime",
        "date_time",
        "bar_time",
        "date",
        "time",
        "dt",
        "epoch",
    ),
    "symbol": ("ticker", "symbol", "sym", "instrument", "asset", "underlying"),
    "open": ("open", "open_price", "o"),
    "high": ("high", "high_price", "h"),
    "low": ("low", "low_price", "l"),
    "close": ("close", "close_price", "c"),
    "volume": ("volume", "vol", "v"),
    "vwap": ("vwap", "vw"),
    "trade_count": ("transactions", "trade_count", "num_trades", "trades", "n"),
    "adjusted_close": ("adj_close", "adjusted_close", "adjclose"),
    "bid": ("bid_price", "bid", "bid_px", "bidprice"),
    "ask": ("ask_price", "ask", "ask_px", "askprice"),
    "bid_size": ("bid_size", "bidsize", "bid_qty"),
    "ask_size": ("ask_size", "asksize", "ask_qty"),
}

#: Polygon's compact JSON-style headers reuse ``t``/``T`` for two different
#: things, so those two are matched before case is discarded.
_CASE_SENSITIVE_ALIASES: Final[dict[str, str]] = {"T": "symbol", "t": "timestamp"}

#: Timestamp columns that label the bar's OPEN rather than its close.
_LEFT_LABELLED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"window_start", "start", "bar_start", "window_start_ns", "t"}
)


@dataclass(frozen=True, slots=True)
class CsvColumnMap:
    """Maps the engine's canonical fields onto the file's column names.

    Leave :attr:`CsvProviderConfig.columns` as ``None`` to detect all of these
    from the header. Supplying a map instead makes the mapping *strict*: every
    non-``None`` name here must exist in the file, otherwise the provider raises
    :class:`~sigmaloop.errors.DataIntegrityError` naming what is missing. That
    is deliberate — a silently ignored typo produces a backtest over zeros.
    """

    timestamp: str = "date"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    bid: str | None = None
    ask: str | None = None
    bid_size: str | None = None
    ask_size: str | None = None
    #: Back-adjusts OHLC when ``DataRequest.adjusted`` is set, if the file has it.
    adjusted_close: str | None = None
    #: Present when one file holds many tickers (long format).
    symbol: str | None = None
    vwap: str | None = None
    trade_count: str | None = None

    def named_fields(self) -> dict[str, str]:
        """The subset of fields that name a column, keyed by canonical role."""
        return {
            role: name
            for role, name in (
                ("timestamp", self.timestamp),
                ("open", self.open),
                ("high", self.high),
                ("low", self.low),
                ("close", self.close),
                ("volume", self.volume),
                ("bid", self.bid),
                ("ask", self.ask),
                ("bid_size", self.bid_size),
                ("ask_size", self.ask_size),
                ("adjusted_close", self.adjusted_close),
                ("symbol", self.symbol),
                ("vwap", self.vwap),
                ("trade_count", self.trade_count),
            )
            if name
        }


@dataclass(frozen=True, slots=True)
class CsvProviderConfig:
    """Where the files are and how to interpret them.

    ``path`` is either a single file or a directory. A directory may hold one
    file per ticker (wide), one file per session holding the whole universe, a
    single long-format file, or a mixture — each file is classified on its own
    header, so no layout flag is needed.
    """

    path: Path
    #: ``None`` auto-detects every column from the header. See :class:`CsvColumnMap`.
    columns: CsvColumnMap | None = None
    #: ``None`` infers the bar width from the modal timestamp spacing per file.
    timeframe: Timeframe | None = None
    #: IANA zone of naive timestamps in the file; converted to UTC on load.
    #: Ignored for integer epoch columns, which are UTC by definition.
    source_timezone: str = "America/New_York"
    #: How to read a naive local timestamp that occurs twice, in the hour a
    #: DST fall-back repeats. ``"earliest"`` takes the pre-transition offset for
    #: both passes, which makes them the same instant — so the second one
    #: becomes a duplicate and ``on_duplicate`` collapses it, quietly costing
    #: one hour of bars per instrument per year. ``"raise"`` refuses instead,
    #: which is the honest choice for a feed that really does cross the
    #: transition; ``"latest"`` takes the post-transition offset. The default is
    #: unchanged because most daily and session-hours data never spans the
    #: repeated hour at all.
    ambiguous_time: Literal["earliest", "latest", "raise"] = "earliest"
    timestamp_format: str | None = None
    #: ``None`` infers from the column name: ``window_start`` and friends label
    #: the bar OPEN and are shifted forward by one timeframe to the close.
    left_labelled: bool | None = None
    asset_class: AssetClass = AssetClass.EQUITY
    file_glob: str = "*.csv"
    #: Recurse into sub-directories (``AAPL/2023/03.csv`` style trees).
    recursive: bool = False
    #: Cache parsed columns as Parquet for fast re-runs.
    use_parquet_cache: bool = True
    #: Defaults to ``<data root>/.sigmaloop_cache``.
    cache_dir: Path | None = None
    #: Byte budget for the in-process series cache used by ``load_series``.
    cache_max_bytes: int = 512 << 20
    #: Series shorter than this stay memory-only. The per-file Parquet cache
    #: already makes re-deriving a short series cheap, and one small file per
    #: ticker per window costs far more to write than it ever saves.
    min_disk_cache_rows: int = 50_000
    #: Ticker for files that carry no symbol column and no filename hint.
    default_symbol: Symbol | None = None
    #: Trust ``MSFT.csv`` to hold only MSFT when pruning before the index exists.
    #: Mismatches are warned about at conversion time. Turn off for files whose
    #: names do not describe their contents.
    trust_filename_hints: bool = True
    epoch_unit: EpochUnit = EpochUnit.AUTO
    on_bad_timestamp: TimestampPolicy = TimestampPolicy.REPAIR
    on_duplicate: DuplicatePolicy = DuplicatePolicy.KEEP_LAST
    #: Discard quotes that are non-positive or crossed (ask < bid).
    drop_invalid_quotes: bool = True
    #: When a dataset holds quotes and no aggregates, build bars from the quotes
    #: rather than serving nothing. The engine consumes bars, so a quote-only
    #: feed is otherwise unusable. Turn off to make that case an explicit error.
    derive_bars_from_quotes: bool = True
    #: Which price the derived OHLC tracks. The real closing bid/ask is attached
    #: to the bar regardless, so execution still prices off the true book.
    quote_bar_price: QuotePriceBasis = QuotePriceBasis.MID
    #: Refuse to attach a quote older than this to a bar. ``None`` never expires.
    max_quote_age: timedelta | None = None
    delimiter: str = ","
    #: Target rows held in memory per streaming slice.
    stream_chunk_rows: int = 1_000_000
    #: Rows per Parquet row group — the granularity of statistics-based pruning.
    row_group_size: int = 131_072
    compression: str = "zstd"
    #: Threads used to convert cold files. Arrow's CSV reader releases the GIL.
    max_workers: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.default_symbol is not None:
            object.__setattr__(self, "default_symbol", Symbol(str(self.default_symbol).upper()))


# --------------------------------------------------------------------------- #
# Internal records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Schema:
    """A file's resolved header mapping."""

    kind: CsvFileKind
    #: Canonical role -> the file's actual column name.
    roles: Mapping[str, str]
    left_labelled: bool

    def column(self, role: str) -> str | None:
        return self.roles.get(role)


@dataclass(frozen=True, slots=True)
class _FileEntry:
    """What the index knows about one source file. Serialised into the index."""

    path: str
    size: int
    mtime_ns: int
    fingerprint: str
    kind: CsvFileKind
    symbols: tuple[Symbol, ...]
    min_ts: int
    max_ts: int
    rows: int
    timeframe: str | None
    parquet: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "fingerprint": self.fingerprint,
            "kind": self.kind.value,
            "symbols": list(self.symbols),
            "min_ts": self.min_ts,
            "max_ts": self.max_ts,
            "rows": self.rows,
            "timeframe": self.timeframe,
            "parquet": self.parquet,
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> _FileEntry:
        return cls(
            path=str(raw["path"]),
            size=int(raw["size"]),
            mtime_ns=int(raw["mtime_ns"]),
            fingerprint=str(raw["fingerprint"]),
            kind=CsvFileKind(str(raw["kind"])),
            symbols=tuple(Symbol(str(s)) for s in raw["symbols"]),
            min_ts=int(raw["min_ts"]),
            max_ts=int(raw["max_ts"]),
            rows=int(raw["rows"]),
            timeframe=None if raw.get("timeframe") is None else str(raw["timeframe"]),
            parquet=None if raw.get("parquet") is None else str(raw["parquet"]),
        )

    def overlaps(self, start_ns: int, end_ns: int) -> bool:
        return self.rows > 0 and self.min_ts <= end_ns and self.max_ts >= start_ns


@dataclass(frozen=True, slots=True)
class _NameHint:
    """What a filename suggests, before anything is read."""

    symbol: Symbol | None
    day: date | None


# --------------------------------------------------------------------------- #
# Filename heuristics
# --------------------------------------------------------------------------- #

_DATE_IN_NAME = re.compile(
    r"(?<!\d)(19|20)(\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)"
)
#: Upper-case only: ``MSFT.csv`` is a ticker by convention, ``universe.csv`` is
#: not. Guessing wrong here would prune real data, so the bar is set high.
_TICKER_STEM = re.compile(r"^[A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z0-9]{1,3})?$")


def _name_hint(path: Path) -> _NameHint:
    stem = path.stem
    day: date | None = None
    match = _DATE_IN_NAME.search(stem)
    if match is not None:
        try:
            day = date(
                int(match.group(1) + match.group(2)), int(match.group(3)), int(match.group(4))
            )
        except ValueError:
            day = None
    remainder = _DATE_IN_NAME.sub("", stem).strip(" _-.")
    symbol = Symbol(remainder) if remainder and _TICKER_STEM.match(remainder) else None
    return _NameHint(symbol=symbol, day=day)


def _day_start_ns(day: date) -> int:
    return to_epoch_ns(datetime(day.year, day.month, day.day, tzinfo=UTC))


def _symbol_codes(table: pa.Table) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Dictionary-encode the symbol column: integer codes plus distinct values.

    Grouping on ``codes == k`` costs one pass over the rows. The obvious
    alternative — an object array of Python strings compared once per ticker —
    is O(tickers x rows) and dominates the load as soon as a universe grows
    past a handful of names.
    """
    import pyarrow.compute as pc

    encoded = pc.dictionary_encode(table.column("symbol").combine_chunks())
    codes = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False)
    return np.asarray(codes, dtype=np.int64), encoded.dictionary.to_pylist()


_CONVERSION_POOL: ThreadPoolExecutor | None = None
_CONVERSION_POOL_LOCK = threading.Lock()


def _conversion_pool(max_workers: int) -> ThreadPoolExecutor:
    """The process-wide pool that converts cold CSV files.

    One pool, created once and deliberately never shut down. A fresh
    ``ThreadPoolExecutor`` per call looks harmless but is not: letting its
    threads exit leaves pyarrow's per-thread state dangling, and the *next*
    pool to run a timestamp cast segfaults. That reproduces deterministically,
    and only once a second provider runs in the same process — which is exactly
    what a parameter sweep or a multi-source run does. Interpreter shutdown
    still joins the threads, via ``concurrent.futures``' own exit hook.

    The first caller fixes the size. A later, larger ``max_workers`` is ignored
    rather than replacing a pool that other providers may be mid-way through.
    """
    global _CONVERSION_POOL
    with _CONVERSION_POOL_LOCK:
        if _CONVERSION_POOL is None:
            _CONVERSION_POOL = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="sigmaloop-csv"
            )
        return _CONVERSION_POOL


def _group_rows_by_code(
    codes: npt.NDArray[np.int64], group_count: int
) -> list[npt.NDArray[np.int64]]:
    """Row indices belonging to each symbol code.

    One stable sort for the whole table, then a slice per symbol. Testing
    ``codes == k`` once per symbol instead would touch every row once per
    ticker — at 300 names over a session of minute bars that is the difference
    between milliseconds and a third of a second.

    The sort is stable, so each group keeps the table's timestamp ordering.
    """
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(group_count + 1))
    return [order[bounds[i] : bounds[i + 1]] for i in range(group_count)]


def _clean_header(name: str) -> str:
    """Canonical form of a column name, used only for matching aliases."""
    return name.lstrip("﻿").strip().lower().replace(" ", "_")


# --------------------------------------------------------------------------- #
# Timestamp normalisation
# --------------------------------------------------------------------------- #


def _decimal_shift(value: int) -> int | None:
    """Power of ten that moves ``value`` into the plausible epoch-ns window."""
    if value <= 0:
        return None
    for k in range(-3, 11):
        scaled = value * 10**k if k >= 0 else value // 10 ** (-k)
        if _PLAUSIBLE_MIN_NS <= scaled <= _PLAUSIBLE_MAX_NS:
            return k
    return None


def _apply_shift(values: npt.NDArray[np.int64], k: int) -> npt.NDArray[np.int64]:
    if k == 0:
        return values
    if k < 0:
        return values // np.int64(10 ** (-k))
    factor = 10**k
    # Silent int64 wraparound would turn a bad row into a plausible-looking
    # date, so oversized values are clamped to 0 and caught by the range check.
    safe = values <= (_INT64_MAX // factor)
    return np.where(safe, values * np.int64(factor), np.int64(0))


class _TimestampNormaliser:
    """Turns a raw timestamp column into int64 UTC nanoseconds.

    Integer columns carry no unit, and real exports disagree about which one
    they use — seconds, millis, micros, nanos, and occasionally a single row
    whose trailing zeros were eaten by a spreadsheet. Rather than trusting a
    declared unit, the column's median is scaled into a plausible date window
    (which is narrower than one decade, so the scale is unambiguous) and any
    row that lands outside is handled per :class:`TimestampPolicy`.
    """

    __slots__ = ("_config", "_source")

    def __init__(self, config: CsvProviderConfig, source: Path) -> None:
        self._config = config
        self._source = source

    def normalise(
        self, column: pa.ChunkedArray | pa.Array
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Return ``(epoch_ns, valid_mask)``."""
        import pyarrow as pa_

        arrow_type = column.type
        if pa_.types.is_integer(arrow_type) or pa_.types.is_floating(arrow_type):
            raw = np.asarray(column.to_numpy(zero_copy_only=False))
            if raw.dtype.kind == "f":
                finite = np.isfinite(raw)
                values = np.where(finite, raw, 0.0).astype(np.int64)
                values[~finite] = 0
            else:
                values = raw.astype(np.int64, copy=False)
            return self._scale_integers(values)

        if pa_.types.is_timestamp(arrow_type) or pa_.types.is_date(arrow_type):
            return self._from_arrow_timestamps(column)

        return self._from_strings(column)

    # -- integer epochs ---------------------------------------------------- #

    def _scale_integers(
        self, values: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        unit = self._config.epoch_unit
        if unit is not EpochUnit.AUTO:
            factor = {
                EpochUnit.SECONDS: 9,
                EpochUnit.MILLIS: 6,
                EpochUnit.MICROS: 3,
                EpochUnit.NANOS: 0,
            }[unit]
            scaled = _apply_shift(values, factor)
            return scaled, values > 0

        positive = values[values > 0]
        if positive.size == 0:
            return values, np.zeros(values.shape, dtype=bool)

        column_shift = _decimal_shift(int(np.median(positive)))
        if column_shift is None:
            raise DataIntegrityError(
                "Timestamp column does not scale to any date between 1990 and 2100. "
                "Set CsvProviderConfig.epoch_unit explicitly, or point "
                "CsvColumnMap.timestamp at the right column.",
                file=str(self._source),
                median=int(np.median(positive)),
            )

        scaled = _apply_shift(values, column_shift)
        valid = (scaled >= _PLAUSIBLE_MIN_NS) & (scaled <= _PLAUSIBLE_MAX_NS)
        if bool(valid.all()):
            return scaled, valid
        return self._handle_outliers(values, scaled, valid)

    def _handle_outliers(
        self,
        raw: npt.NDArray[np.int64],
        scaled: npt.NDArray[np.int64],
        valid: npt.NDArray[np.bool_],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        bad = ~valid
        count = int(bad.sum())
        policy = self._config.on_bad_timestamp

        if policy is TimestampPolicy.STRICT:
            first = int(raw[np.argmax(bad)])
            raise DataIntegrityError(
                f"{count} timestamp(s) do not scale to a date between 1990 and 2100.",
                file=str(self._source),
                example=first,
                hint="Set on_bad_timestamp=TimestampPolicy.REPAIR to rescale them individually.",
            )
        if policy is TimestampPolicy.DROP:
            _LOG.warning("Dropping %d row(s) with unusable timestamps in %s", count, self._source)
            return scaled, valid

        # REPAIR: give each offending value its own power of ten. Grouping by
        # the needed exponent keeps this vectorised — there are only ever a
        # couple of distinct exponents in practice.
        repaired = scaled.copy()
        still_bad = bad.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.log10(np.maximum(raw.astype(np.float64), 1.0))
        wanted = np.rint(np.log10(float(_PLAUSIBLE_CENTER_NS)) - magnitude).astype(np.int64)
        for k in np.unique(wanted[bad]):
            group = bad & (wanted == k)
            candidate = _apply_shift(raw[group], int(k))
            ok = (candidate >= _PLAUSIBLE_MIN_NS) & (candidate <= _PLAUSIBLE_MAX_NS)
            indices = np.flatnonzero(group)[ok]
            repaired[indices] = candidate[ok]
            still_bad[indices] = False

        healed = count - int(still_bad.sum())
        if healed:
            _LOG.warning(
                "Rescaled %d timestamp(s) in %s whose magnitude did not match the column "
                "(likely trailing zeros lost in export). Set epoch_unit or "
                "on_bad_timestamp=STRICT to reject instead.",
                healed,
                self._source,
            )
        if still_bad.any():
            _LOG.warning(
                "Dropping %d unrecoverable timestamp(s) in %s", int(still_bad.sum()), self._source
            )
        return repaired, ~still_bad

    # -- textual and native timestamps ------------------------------------- #

    def _from_arrow_timestamps(
        self, column: pa.ChunkedArray | pa.Array
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        import pyarrow as pa_
        import pyarrow.compute as pc

        stamps = pc.cast(column, pa_.timestamp("ns")) if pa_.types.is_date(column.type) else column
        if stamps.type.tz is None:
            stamps = pc.assume_timezone(
                pc.cast(stamps, pa_.timestamp("ns")),
                self._config.source_timezone,
                ambiguous=self._config.ambiguous_time,
                nonexistent="earliest",
            )
        epoch = pc.cast(pc.cast(stamps, pa_.timestamp("ns", tz="UTC")), pa_.int64())
        values = np.asarray(epoch.to_numpy(zero_copy_only=False))
        valid = ~np.asarray(pc.is_null(epoch).to_numpy(zero_copy_only=False))
        return np.nan_to_num(values).astype(np.int64, copy=False), valid

    def _from_strings(
        self, column: pa.ChunkedArray | pa.Array
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        import pyarrow as pa_
        import pyarrow.compute as pc

        fmt = self._config.timestamp_format
        try:
            if fmt is not None:
                parsed = pc.strptime(column, format=fmt, unit="ns", error_is_null=True)
            else:
                parsed = pc.cast(column, pa_.timestamp("ns"), safe=False)
        except (pa_.ArrowInvalid, pa_.ArrowNotImplementedError) as exc:
            raise DataIntegrityError(
                "Could not parse the timestamp column as dates. Supply "
                "CsvProviderConfig.timestamp_format (strptime syntax).",
                file=str(self._source),
            ) from exc
        return self._from_arrow_timestamps(parsed)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class CsvDataProvider(DataProvider):
    """Local-file provider. No network, fully deterministic, fastest to replay."""

    name: ClassVar[str] = "csv"

    __slots__ = (
        "_cache",
        "_columns_fingerprint",
        "_config",
        "_discovered",
        "_index",
        "_index_dirty",
        "_index_loaded",
        "_lock",
        "_schemas",
    )

    def __init__(self, config: CsvProviderConfig) -> None:
        if isinstance(config, (str, Path)):  # ergonomic: CsvDataProvider("data/")
            config = CsvProviderConfig(path=Path(config))
        if not config.path.exists():
            raise DataProviderError(
                self.name, f"No such file or directory: {config.path}", path=str(config.path)
            )
        self._config = config
        self._columns_fingerprint = self._fingerprint(config)
        self._index: dict[str, _FileEntry] = {}
        self._index_dirty = False
        self._index_loaded = False
        self._lock = threading.RLock()
        self._schemas: dict[Path, _Schema] = {}
        self._discovered: tuple[Path, ...] | None = None
        self._tables: dict[str, pa.Table] = {}
        memory = MemoryDataCache(max_bytes=config.cache_max_bytes)
        self._cache: DataCache = (
            TieredDataCache(
                memory,
                ParquetDataCache(self._cache_root() / "series", compression=config.compression),
                min_slow_rows=config.min_disk_cache_rows,
            )
            if config.use_parquet_cache
            else memory
        )

    # ---- configuration surface ------------------------------------------- #

    @staticmethod
    def _fingerprint(config: CsvProviderConfig) -> str:
        """Hash of every setting that changes the normalised bytes.

        Anything omitted here would let a stale Parquet file be served for a
        different interpretation of the same CSV, which is the one cache bug
        that produces plausible-looking wrong results.
        """
        payload = json.dumps(
            {
                "v": CACHE_SCHEMA_VERSION,
                "columns": None if config.columns is None else config.columns.named_fields(),
                "timeframe": None if config.timeframe is None else config.timeframe.value,
                "tz": config.source_timezone,
                "ambiguous": config.ambiguous_time,
                "fmt": config.timestamp_format,
                "left": config.left_labelled,
                "epoch": config.epoch_unit.value,
                "bad_ts": config.on_bad_timestamp.value,
                "delim": config.delimiter,
                "default_symbol": config.default_symbol,
                "drop_invalid_quotes": config.drop_invalid_quotes,
                "quote_bars": config.derive_bars_from_quotes,
                "quote_basis": config.quote_bar_price.value,
            },
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()

    def _source_digest(self) -> str:
        """Identity of the bytes the next read will see.

        The per-file Parquet layer already folds ``_columns_fingerprint`` into
        its filenames; the series cache above it keyed only on the *request*, so
        a re-run over an edited CSV — or the same CSV read under a different
        ``source_timezone`` — was served the previous run's bars from disk. This
        closes that gap by putting the config hash and the current revision of
        every source file into the series key too.

        Deliberately covers all discovered files rather than only those a given
        symbol touches: resolving that per symbol would mean a file-selection
        pass on every cache *lookup*, and paying a few stats to over-invalidate
        is the cheaper mistake.
        """
        parts = [self._columns_fingerprint]
        for path in self._discover_files():
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - vanished between glob and stat
                parts.append(f"{path}|missing")
            else:
                parts.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
        return hashlib.blake2b("\x00".join(parts).encode("utf-8"), digest_size=10).hexdigest()

    def _cache_root(self) -> Path:
        if self._config.cache_dir is not None:
            return self._config.cache_dir
        root = self._config.path if self._config.path.is_dir() else self._config.path.parent
        return root / ".sigmaloop_cache"

    @property
    def capabilities(self) -> ProviderCapabilities:
        entries = self._index.values()
        earliest = min((e.min_ts for e in entries if e.rows), default=None)
        return ProviderCapabilities(
            name=self.name,
            asset_classes=frozenset({AssetClass.EQUITY, AssetClass.ETF, AssetClass.INDEX}),
            timeframes=frozenset(Timeframe),
            supports_options=False,
            supports_greeks=False,
            supports_quotes=True,
            supports_corporate_actions=False,
            supports_streaming=True,
            earliest_data=None if earliest is None else from_epoch_ns(earliest),
            requires_credentials=False,
            rate_limit_per_minute=None,
        )

    # ---- discovery -------------------------------------------------------- #

    def resolve_instrument(
        self, symbol: Symbol, asset_class: AssetClass = AssetClass.EQUITY
    ) -> Instrument:
        ticker = Symbol(str(symbol).strip().upper())
        return Equity(
            instrument_id=Equity.make_id(ticker),
            symbol=ticker,
            asset_class=asset_class if asset_class is not AssetClass.OPTION else AssetClass.EQUITY,
        )

    def available_symbols(self, asset_class: AssetClass = AssetClass.EQUITY) -> Sequence[Symbol]:
        """Every ticker the files contain.

        Requires a full index build on first call, because a long-format file
        cannot be enumerated from its name. Subsequent calls read the index.
        """
        self._ensure_indexed(self._discover_files())
        symbols: set[Symbol] = set()
        for entry in self._index.values():
            symbols.update(entry.symbols)
        return tuple(sorted(symbols))

    def coverage(
        self, symbol: Symbol, timeframe: Timeframe
    ) -> tuple[UtcDatetime, UtcDatetime] | None:
        ticker = Symbol(str(symbol).upper())
        self._ensure_indexed(self._discover_files())
        spans = [
            (e.min_ts, e.max_ts) for e in self._index.values() if ticker in e.symbols and e.rows
        ]
        if not spans:
            return None
        return from_epoch_ns(min(s for s, _ in spans)), from_epoch_ns(max(e for _, e in spans))

    def _discover_files(self) -> tuple[Path, ...]:
        """Every source file under ``config.path``, cache directory excluded."""
        with self._lock:
            if self._discovered is not None:
                return self._discovered
            config = self._config
            if config.path.is_file():
                found = [config.path]
            else:
                pattern = config.file_glob
                paths = (
                    config.path.rglob(pattern) if config.recursive else config.path.glob(pattern)
                )
                cache_root = self._cache_root().resolve()
                found = sorted(
                    p
                    for p in paths
                    if p.is_file()
                    and cache_root not in p.resolve().parents
                    and not p.name.startswith(".")
                )
            if not found:
                raise DataProviderError(
                    self.name,
                    f"No files matched {config.file_glob!r} under {config.path}.",
                    path=str(config.path),
                    glob=config.file_glob,
                )
            self._discovered = tuple(found)
            return self._discovered

    # ---- pruning ---------------------------------------------------------- #

    def _prune_files(
        self,
        request: DataRequest,
        *,
        kind: CsvFileKind = CsvFileKind.AGGREGATE,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> tuple[Path, ...]:
        """Candidate files for ``request`` — requirement 3, "don't read what you don't need".

        Three filters in increasing cost order: the filename, then the persisted
        index, then (inside :meth:`_load_file_table`) Parquet row-group
        statistics. Everything here is conservative: a file is only dropped when
        it *cannot* contribute.
        """
        lo = to_epoch_ns(request.start) if start_ns is None else start_ns
        hi = to_epoch_ns(request.end) if end_ns is None else end_ns
        wanted = set(request.symbols)
        keep: list[Path] = []

        for path in self._discover_files():
            entry = self._fresh_entry(path)
            if entry is not None:
                if entry.kind is kind and entry.overlaps(lo, hi) and (wanted & set(entry.symbols)):
                    keep.append(path)
                continue
            if self._config.trust_filename_hints and self._hint_excludes(path, wanted, lo, hi):
                continue
            keep.append(path)
        return tuple(keep)

    def _hint_excludes(self, path: Path, wanted: set[Symbol], lo: int, hi: int) -> bool:
        """True when the filename alone rules the file out."""
        hint = _name_hint(path)
        if hint.symbol is not None and hint.symbol not in wanted:
            return True
        if hint.day is not None:
            # A session labelled 2023-03-28 spans roughly 2023-03-28T08:00Z to
            # 2023-03-29T01:00Z once pre- and post-market are included, and some
            # venues label in local time either side of UTC midnight. Pad a day
            # each way rather than reason about the exchange's calendar here.
            day_start = _day_start_ns(hint.day)
            if day_start - _NS_PER_DAY > hi or day_start + 2 * _NS_PER_DAY < lo:
                return True
        return False

    def _fresh_entry(self, path: Path) -> _FileEntry | None:
        """Index entry for ``path`` if it still matches the file on disk."""
        self._load_index()
        entry = self._index.get(str(path))
        if entry is None or entry.fingerprint != self._columns_fingerprint:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        if entry.size != stat.st_size or entry.mtime_ns != stat.st_mtime_ns:
            return None
        if (
            entry.parquet is not None
            and not (self._cache_root() / "files" / entry.parquet).exists()
        ):
            return None
        return entry

    # ---- index persistence ------------------------------------------------ #

    def _index_path(self) -> Path:
        return self._cache_root() / f"index-v{CACHE_SCHEMA_VERSION}.json"

    def _load_index(self) -> None:
        with self._lock:
            if self._index_loaded:
                return
            self._index_loaded = True
            path = self._index_path()
            if not path.exists():
                return
            try:
                raw = json.loads(path.read_text())
                for item in raw.get("entries", []):
                    entry = _FileEntry.from_json(item)
                    self._index[entry.path] = entry
            except (OSError, ValueError, KeyError) as exc:
                _LOG.warning("Ignoring unreadable provider index %s (%s)", path, exc)
                self._index.clear()

    def _flush_index(self) -> None:
        with self._lock:
            if not self._index_dirty or not self._config.use_parquet_cache:
                self._index_dirty = False
                return
            path = self._index_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_SCHEMA_VERSION,
                "entries": [e.to_json() for e in self._index.values()],
            }
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(payload))
                os.replace(tmp, path)
            except OSError as exc:  # a read-only data directory must not be fatal
                _LOG.warning("Could not persist provider index to %s (%s)", path, exc)
                tmp.unlink(missing_ok=True)
            self._index_dirty = False

    # ---- header inspection ------------------------------------------------ #

    def _read_header(self, path: Path) -> list[str]:
        """Column names exactly as Arrow will see them.

        Arrow drops a byte-order mark but keeps surrounding whitespace, so
        ``utf-8-sig`` with no trimming reproduces its view. That matters because
        ``include_columns`` matches on the literal name; trimming here would
        make ``" Close"`` unrequestable. Cleaning happens in
        :func:`_clean_header`, which is only used for matching aliases.
        """
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.reader(handle, delimiter=self._config.delimiter), None)
        except OSError as exc:
            raise DataProviderError(
                self.name, f"Cannot read {path}: {exc}", path=str(path)
            ) from exc
        if not row or not any(c.strip() for c in row):
            raise DataIntegrityError("File is empty — no header row.", file=str(path))
        return row

    def _schema_for(self, path: Path) -> _Schema:
        with self._lock:
            cached = self._schemas.get(path)
        if cached is not None:
            return cached
        schema = self._detect_schema(path, self._read_header(path))
        with self._lock:
            self._schemas[path] = schema
        return schema

    def _detect_schema(self, path: Path, header: Sequence[str]) -> _Schema:
        # clean name -> the file's literal name, which is what Arrow needs.
        by_clean: dict[str, str] = {}
        by_trimmed: dict[str, str] = {}
        for name in header:
            by_clean.setdefault(_clean_header(name), name)
            by_trimmed.setdefault(name.strip(), name)

        declared = self._config.columns
        if declared is not None:
            roles: dict[str, str] = {}
            missing: dict[str, str] = {}
            for role, wanted in declared.named_fields().items():
                resolved = by_trimmed.get(wanted) or by_clean.get(_clean_header(wanted))
                if resolved is None:
                    missing[role] = wanted
                else:
                    roles[role] = resolved
            if missing:
                raise DataIntegrityError(
                    "CsvColumnMap names columns that the file does not have: "
                    + ", ".join(f"{role}={col!r}" for role, col in sorted(missing.items()))
                    + ".",
                    file=str(path),
                    available=[n.strip() for n in header],
                )
            kind = (
                CsvFileKind.AGGREGATE
                if {"open", "high", "low", "close"} <= roles.keys()
                else CsvFileKind.QUOTE
            )
            return _Schema(kind, roles, self._resolve_left_labelled(roles.get("timestamp")))

        roles = {}
        for role, aliases in _ALIASES.items():
            for literal, mapped_role in _CASE_SENSITIVE_ALIASES.items():
                if mapped_role == role and literal in by_trimmed:
                    roles[role] = by_trimmed[literal]
                    break
            if role in roles:
                continue
            for alias in aliases:
                if alias in by_clean:
                    roles[role] = by_clean[alias]
                    break

        has_ohlc = {"open", "high", "low", "close"} <= roles.keys()
        has_quote = {"bid", "ask"} <= roles.keys()
        if "timestamp" not in roles or not (has_ohlc or has_quote):
            raise DataIntegrityError(
                "Could not identify the columns in this file. Expected a timestamp plus "
                "either open/high/low/close or bid/ask. Pass an explicit "
                "CsvProviderConfig(columns=CsvColumnMap(...)) for non-standard headers.",
                file=str(path),
                header=[n.strip() for n in header],
                detected=sorted(roles),
            )
        if has_ohlc:
            kind = CsvFileKind.AGGREGATE
            roles = {
                k: v for k, v in roles.items() if k not in {"bid", "ask", "bid_size", "ask_size"}
            }
        else:
            kind = CsvFileKind.QUOTE
            roles = {
                k: v
                for k, v in roles.items()
                if k in {"timestamp", "symbol", "bid", "ask", "bid_size", "ask_size"}
            }
        return _Schema(kind, roles, self._resolve_left_labelled(roles.get("timestamp")))

    def _resolve_left_labelled(self, timestamp_column: str | None) -> bool:
        declared = self._config.left_labelled
        if declared is not None:
            return declared
        if timestamp_column is None:
            return False
        return timestamp_column.strip().lower() in _LEFT_LABELLED_COLUMNS

    def _validate_columns(self, symbol: Symbol) -> None:
        """Raise ``DataIntegrityError`` naming the missing/misnamed columns."""
        ticker = Symbol(str(symbol).upper())
        for path in self._discover_files():
            entry = self._fresh_entry(path)
            if entry is not None and ticker not in entry.symbols:
                continue
            self._schema_for(path)

    # ---- conversion ------------------------------------------------------- #

    def _ensure_indexed(self, paths: Iterable[Path]) -> list[_FileEntry]:
        """Convert any cold file in ``paths`` and return the resulting entries."""
        pending = [p for p in paths if self._fresh_entry(p) is None]
        if pending:
            workers = max(1, min(self._config.max_workers, len(pending)))
            if workers == 1:
                for path in pending:
                    self._convert(path)
            else:
                list(_conversion_pool(workers).map(self._convert, pending))
            self._flush_index()
        return [e for e in (self._fresh_entry(p) for p in paths) if e is not None]

    def _convert(self, path: Path) -> _FileEntry:
        """Parse one CSV into the normalised columnar form and cache it."""

        schema = self._schema_for(path)
        table = self._read_csv(path, schema)
        stat = path.stat()

        import pyarrow.compute as pc

        symbols = (
            tuple(sorted(pc.unique(table.column("symbol")).to_pylist())) if table.num_rows else ()
        )
        ts = (
            table.column("ts").to_numpy(zero_copy_only=False)
            if table.num_rows
            else np.empty(0, np.int64)
        )
        timeframe = self._infer_timeframe(table) if schema.kind is CsvFileKind.AGGREGATE else None

        self._warn_on_hint_mismatch(path, symbols, ts)

        parquet_name: str | None = None
        if self._config.use_parquet_cache:
            digest = hashlib.blake2b(str(path.resolve()).encode(), digest_size=6).hexdigest()
            parquet_name = f"{path.stem}-{self._columns_fingerprint}-{digest}.parquet"
            write_atomic(
                table,
                self._cache_root() / "files" / parquet_name,
                compression=self._config.compression,
                row_group_size=self._config.row_group_size,
            )

        entry = _FileEntry(
            path=str(path),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            fingerprint=self._columns_fingerprint,
            kind=schema.kind,
            symbols=tuple(Symbol(s) for s in symbols),
            min_ts=int(ts.min()) if ts.size else 0,
            max_ts=int(ts.max()) if ts.size else 0,
            rows=table.num_rows,
            timeframe=None if timeframe is None else timeframe.value,
            parquet=parquet_name,
        )
        with self._lock:
            self._index[entry.path] = entry
            self._index_dirty = True
            if not self._config.use_parquet_cache:
                # Without a disk cache the parsed table is the only copy; hold
                # it so a single run does not re-parse per request.
                self._schemas[path] = schema
                self._tables[str(path)] = table
        return entry

    def _warn_on_hint_mismatch(
        self, path: Path, symbols: tuple[str, ...], ts: npt.NDArray[np.int64]
    ) -> None:
        if not self._config.trust_filename_hints:
            return
        hint = _name_hint(path)
        if hint.symbol is not None and symbols and set(symbols) != {str(hint.symbol)}:
            _LOG.warning(
                "%s is named for %s but contains %s. Pruning by filename may skip data; "
                "set trust_filename_hints=False if names do not describe contents.",
                path.name,
                hint.symbol,
                ", ".join(sorted(symbols)[:5]),
            )
        if hint.day is not None and ts.size:
            hinted = _day_start_ns(hint.day)
            if int(ts.min()) < hinted - _NS_PER_DAY or int(ts.max()) > hinted + 2 * _NS_PER_DAY:
                _LOG.warning(
                    "%s is named for %s but holds data from %s to %s.",
                    path.name,
                    hint.day,
                    from_epoch_ns(int(ts.min())).date(),
                    from_epoch_ns(int(ts.max())).date(),
                )

    def _read_csv(self, path: Path, schema: _Schema) -> pa.Table:
        """CSV -> normalised Arrow table sorted by ``(ts, symbol)``."""
        import pyarrow as pa_
        import pyarrow.compute as pc
        import pyarrow.csv as pacsv

        roles = dict(schema.roles)
        wanted = list(dict.fromkeys(roles.values()))
        numeric_roles = (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "adjusted_close",
        )
        column_types = {roles[r]: pa_.float64() for r in numeric_roles if r in roles}
        if "symbol" in roles:
            column_types[roles["symbol"]] = pa_.string()

        try:
            raw = pacsv.read_csv(
                path,
                read_options=pacsv.ReadOptions(use_threads=True),
                parse_options=pacsv.ParseOptions(delimiter=self._config.delimiter),
                convert_options=pacsv.ConvertOptions(
                    include_columns=wanted, column_types=column_types, strings_can_be_null=True
                ),
            )
        except pa_.ArrowInvalid as exc:
            raise DataIntegrityError(
                f"Could not parse {path.name}: {exc}", file=str(path), columns=wanted
            ) from exc

        ts, valid = _TimestampNormaliser(self._config, path).normalise(
            raw.column(roles["timestamp"])
        )

        n = raw.num_rows
        symbol_values = self._symbol_column(raw, roles, path, n)

        columns: dict[str, Any] = {"ts": ts, "symbol": symbol_values}
        if schema.kind is CsvFileKind.AGGREGATE:
            for role in ("open", "high", "low", "close"):
                columns[role] = self._float_column(raw, roles, role, n)
            columns["volume"] = self._float_column(raw, roles, "volume", n)
            if "adjusted_close" in roles:
                columns["adj_close"] = self._float_column(raw, roles, "adjusted_close", n)
            if "vwap" in roles:
                columns["vwap"] = self._float_column(raw, roles, "vwap", n)
            valid = valid & self._ohlc_valid(columns, path)
        else:
            for role, out in (
                ("bid", "bid"),
                ("ask", "ask"),
                ("bid_size", "bid_size"),
                ("ask_size", "ask_size"),
            ):
                columns[out] = self._float_column(raw, roles, role, n)
            if self._config.drop_invalid_quotes:
                bid, ask = columns["bid"], columns["ask"]
                valid = valid & (bid > 0.0) & (ask > 0.0) & (ask >= bid)

        if schema.left_labelled and schema.kind is CsvFileKind.AGGREGATE:
            usable = valid if valid.any() else np.ones_like(valid)
            first = pc.equal(symbol_values, symbol_values[0].as_py()).to_numpy(zero_copy_only=False)
            width = self._bar_width_ns(path, ts[usable & first]) if n else 0
            if width:
                columns["ts"] = ts + np.int64(width)

        table = pa_.table(
            {k: (v if isinstance(v, pa_.Array) else pa_.array(v)) for k, v in columns.items()}
        )
        if not bool(valid.all()):
            table = table.filter(pa_.array(valid))
        return table.sort_by([("ts", "ascending"), ("symbol", "ascending")])

    def _symbol_column(
        self, raw: pa.Table, roles: Mapping[str, str], path: Path, n: int
    ) -> pa.Array:
        """Upper-cased tickers, left as an Arrow array.

        Materialising these as Python strings is the single most expensive
        thing this provider could do to a million-row file, so they stay in
        Arrow until something genuinely needs one name.
        """
        import pyarrow as pa_
        import pyarrow.compute as pc

        if "symbol" in roles:
            column = pc.utf8_upper(pc.utf8_trim_whitespace(raw.column(roles["symbol"])))
            return pc.fill_null(column, "").combine_chunks()
        hint = _name_hint(path)
        fallback = self._config.default_symbol or hint.symbol
        if fallback is None:
            raise DataIntegrityError(
                f"{path.name} has no symbol column and its name is not a ticker. "
                "Set CsvProviderConfig(default_symbol=...) or add a ticker column.",
                file=str(path),
            )
        return pa_.array(np.full(n, str(fallback).upper(), dtype=object), type=pa_.string())

    @staticmethod
    def _float_column(
        raw: pa.Table, roles: Mapping[str, str], role: str, n: int
    ) -> npt.NDArray[np.float64]:
        if role not in roles:
            return np.zeros(n, dtype=np.float64)
        values = np.asarray(
            raw.column(roles[role]).to_numpy(zero_copy_only=False), dtype=np.float64
        )
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _ohlc_valid(
        self, columns: Mapping[str, npt.NDArray[Any]], path: Path
    ) -> npt.NDArray[np.bool_]:
        """Vectorised form of ``Bar.__post_init__``, applied once at conversion."""
        o, h, l, c = (np.asarray(columns[k]) for k in ("open", "high", "low", "close"))
        ok = (l <= o) & (o <= h) & (l <= c) & (c <= h) & (l <= h) & np.isfinite(o) & np.isfinite(c)
        bad = int((~ok).sum())
        if bad:
            _LOG.warning(
                "Dropping %d row(s) in %s that violate low <= open/close <= high", bad, path
            )
        return ok

    def _bar_width_ns(self, path: Path, ts: npt.NDArray[np.int64]) -> int:
        """Nanoseconds to add when shifting a left-labelled bar to its close.

        ``ts`` must already be narrowed to a single symbol; the gap between two
        different tickers is not a bar width.
        """
        configured = self._config.timeframe
        if configured is not None:
            return (
                0
                if configured is Timeframe.TICK
                else int(configured.duration.total_seconds() * 1e9)
            )
        inferred = self._modal_spacing(ts)
        if inferred <= 0:
            # One bar per symbol per file (a daily per-session layout) leaves no
            # gap to measure, so the shift would silently not happen.
            _LOG.warning(
                "%s has left-labelled timestamps but too few bars per symbol to infer the bar "
                "width; timestamps are left as-is. Set CsvProviderConfig(timeframe=...) to fix.",
                path.name,
            )
        return inferred

    @staticmethod
    def _modal_spacing(ts: npt.NDArray[np.int64]) -> int:
        """Most common positive gap between consecutive timestamps of one symbol."""
        if ts.size < 3:
            return 0
        sample = np.sort(ts[: min(ts.size, 200_000)])
        diffs = np.diff(sample)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 0
        values, counts = np.unique(diffs, return_counts=True)
        return int(values[int(np.argmax(counts))])

    def _infer_timeframe(self, table: pa.Table) -> Timeframe | None:
        """Map the observed bar spacing onto a standard :class:`Timeframe`."""
        if self._config.timeframe is not None:
            return self._config.timeframe
        if table.num_rows < 3:
            return None
        ts = table.column("ts").to_numpy(zero_copy_only=False)
        codes, _ = _symbol_codes(table)
        spacing = self._modal_spacing(ts[codes == codes[0]])
        if spacing <= 0:
            return None
        for frame in Timeframe:
            if frame is Timeframe.TICK:
                continue
            width = int(frame.duration.total_seconds() * 1e9)
            # 1% tolerance absorbs a DST-shifted daily bar without matching H4.
            if abs(width - spacing) <= max(1, width // 100):
                return frame
        _LOG.debug("Timestamp spacing %d ns matches no standard timeframe", spacing)
        return None

    def _timeframe_for(self, entries: Sequence[_FileEntry], request: DataRequest) -> Timeframe:
        if self._config.timeframe is not None:
            return self._config.timeframe
        for entry in entries:
            if entry.timeframe:
                return Timeframe(entry.timeframe)
        return request.timeframe

    # ---- reading ---------------------------------------------------------- #

    def _load_file_table(
        self,
        entry: _FileEntry,
        symbols: Sequence[Symbol],
        start_ns: int,
        end_ns: int,
        columns: Sequence[str],
    ) -> pa.Table | None:
        """Rows of one indexed file inside the window, or ``None`` if empty.

        Row-group statistics do the coarse skipping; the residual predicate
        trims the surviving groups.
        """
        import pyarrow.parquet as pq

        filters = [
            ("ts", ">=", start_ns),
            ("ts", "<=", end_ns),
            ("symbol", "in", set(map(str, symbols))),
        ]
        # Optional columns (adj_close, vwap) may exist in some files and not
        # others; asking for a missing one is an error, so narrow per file and
        # let the concatenation fill the gap with nulls.
        present = self._file_columns(entry)
        columns = [c for c in columns if c in present] or list(columns)
        if entry.parquet is not None:
            path = self._cache_root() / "files" / entry.parquet
            try:
                table = pq.read_table(path, columns=list(columns), filters=filters)
            except (OSError, ValueError) as exc:
                _LOG.warning("Re-reading %s: cached Parquet unusable (%s)", entry.path, exc)
                self._invalidate(Path(entry.path))
                self._convert(Path(entry.path))
                refreshed = self._fresh_entry(Path(entry.path))
                if refreshed is None or refreshed.parquet is None:
                    return None
                table = pq.read_table(
                    self._cache_root() / "files" / refreshed.parquet,
                    columns=list(columns),
                    filters=filters,
                )
        else:
            cached = self._tables.get(entry.path)
            if cached is None:
                cached = self._read_csv(Path(entry.path), self._schema_for(Path(entry.path)))
                self._tables[entry.path] = cached
            table = self._filter_in_memory(
                cached.select(list(columns)) if columns else cached, symbols, start_ns, end_ns
            )
        return table if table.num_rows else None

    @staticmethod
    def _filter_in_memory(
        table: pa.Table, symbols: Sequence[Symbol], start_ns: int, end_ns: int
    ) -> pa.Table:
        import pyarrow as pa_
        import pyarrow.compute as pc

        mask = pc.and_(
            pc.greater_equal(table.column("ts"), start_ns),
            pc.less_equal(table.column("ts"), end_ns),
        )
        mask = pc.and_(
            mask, pc.is_in(table.column("symbol"), value_set=pa_.array([str(s) for s in symbols]))
        )
        return table.filter(mask)

    def _invalidate(self, path: Path) -> None:
        with self._lock:
            self._index.pop(str(path), None)
            self._schemas.pop(path, None)
            self._index_dirty = True

    def _gather(
        self,
        entries: Sequence[_FileEntry],
        symbols: Sequence[Symbol],
        start_ns: int,
        end_ns: int,
        columns: Sequence[str],
    ) -> pa.Table | None:
        """Concatenate, order and de-duplicate rows from several files."""
        import pyarrow as pa_

        tables = [
            t
            for t in (self._load_file_table(e, symbols, start_ns, end_ns, columns) for e in entries)
            if t is not None
        ]
        if not tables:
            return None
        table = (
            tables[0] if len(tables) == 1 else pa_.concat_tables(tables, promote_options="default")
        )
        if len(tables) > 1:
            table = table.sort_by([("ts", "ascending"), ("symbol", "ascending")])
        return self._deduplicate(table)

    def _deduplicate(self, table: pa.Table) -> pa.Table:
        """Collapse rows sharing ``(symbol, ts)`` — identical files, restatements."""
        import pyarrow as pa_

        n = table.num_rows
        if n < 2:
            return table
        ts = table.column("ts").to_numpy(zero_copy_only=False)
        codes, _ = _symbol_codes(table)
        boundary = (ts[1:] != ts[:-1]) | (codes[1:] != codes[:-1])
        if bool(boundary.all()):
            return table
        policy = self._config.on_duplicate
        duplicates = n - 1 - int(boundary.sum())
        if policy is DuplicatePolicy.ERROR:
            raise DataIntegrityError(
                f"{duplicates} duplicate (symbol, timestamp) row(s) across the selected files. "
                "Set on_duplicate=DuplicatePolicy.KEEP_LAST to collapse them.",
                duplicates=duplicates,
            )
        keep = np.empty(n, dtype=bool)
        if policy is DuplicatePolicy.KEEP_FIRST:
            keep[0] = True
            keep[1:] = boundary
        else:
            keep[-1] = True
            keep[:-1] = boundary
        _LOG.debug("Collapsed %d duplicate row(s)", duplicates)
        return table.filter(pa_.array(keep))

    # ---- window resolution (warm-up) -------------------------------------- #

    def _warmup_start_ns(self, request: DataRequest, entries: Sequence[_FileEntry]) -> int:
        """Earliest timestamp to read so every symbol gets ``warmup_bars`` of priming.

        Widens geometrically rather than reading from the beginning of history:
        the whole point of warm-up is a handful of extra bars, not a full load.
        """
        start_ns = to_epoch_ns(request.start)
        if request.warmup_bars <= 0 or not entries:
            return start_ns
        frame = self._timeframe_for(entries, request)
        width = (
            int(frame.duration.total_seconds() * 1e9)
            if frame is not Timeframe.TICK
            else _NS_PER_DAY
        )
        floor_ns = min(e.min_ts for e in entries)
        needed = request.warmup_bars

        for pad in _WARMUP_PADS:
            probe = max(floor_ns, start_ns - int(width * needed * pad))
            cut = self._probe_warmup_cut(request, entries, probe, start_ns, needed)
            if cut is not None:
                return cut
            if probe <= floor_ns:
                break
        return floor_ns

    def _probe_warmup_cut(
        self,
        request: DataRequest,
        entries: Sequence[_FileEntry],
        probe_ns: int,
        start_ns: int,
        needed: int,
    ) -> int | None:
        """Read only ``ts``/``symbol`` to find where the warm-up window begins."""
        candidates = [e for e in entries if e.overlaps(probe_ns, start_ns)]
        if not candidates:
            return None
        table = self._gather(candidates, request.symbols, probe_ns, start_ns - 1, ("ts", "symbol"))
        if table is None:
            return None
        ts = table.column("ts").to_numpy(zero_copy_only=False)
        codes, names = _symbol_codes(table)
        cut = start_ns
        for rows in _group_rows_by_code(codes, len(names)):
            if rows.size == 0:
                continue
            if rows.size < needed:
                return None  # widen the probe and try again
            cut = min(cut, int(ts[rows[-needed]]))
        return cut

    # ---- bar access -------------------------------------------------------- #

    def stream_bars(self, request: DataRequest) -> Iterator[Bar]:
        """Yield bars in non-decreasing timestamp order across all symbols.

        Walks the window in time slices so residency stays bounded by
        ``stream_chunk_rows`` regardless of how much history the files hold.
        """
        entries, from_quotes, read_from = self._prepare_read(request)
        if not entries:
            return iter(())
        return self._stream(request, entries, from_quotes, read_from)

    def _history_lookback_ns(self, request: DataRequest) -> int:
        """How far before ``request.start`` file selection must reach.

        Warm-up bars live in *earlier files*. In a per-session directory the
        files holding them sit outside the request window, so pruning on the
        window alone silently returns no warm-up at all and every indicator
        primes on a truncated history.

        The multiplier converts bar counts into wall-clock: sessions are sparse
        (weekends, holidays, and overnight for intraday frames), so N bars span
        rather more than N bar-widths. Anything this misses is caught by the
        unbounded retry in :meth:`_prepare_read`.
        """
        if request.warmup_bars <= 0:
            return 0
        frame = self._config.timeframe or request.timeframe
        width = self._frame_width_ns(frame) or _NS_PER_DAY
        return request.warmup_bars * width * (8 if frame.is_intraday else 2)

    def _prepare_read(self, request: DataRequest) -> tuple[list[_FileEntry], bool, int]:
        """Files to read, whether they are quotes to fold, and where to start.

        Widens once to the full history if the warm-up probe consumed every
        bar the first selection offered — that means the look-back estimate was
        too short, not that the data ran out.
        """
        entries, from_quotes = self._bar_source(request)
        if not entries:
            return [], False, 0
        read_from = self._warmup_start_ns(request, entries)
        if request.warmup_bars > 0 and read_from <= min(e.min_ts for e in entries):
            wider, from_quotes = self._bar_source(request, unbounded_history=True)
            if len(wider) > len(entries):
                entries = wider
                read_from = self._warmup_start_ns(request, entries)
        return entries, from_quotes, read_from

    def _bar_source(
        self, request: DataRequest, *, unbounded_history: bool = False
    ) -> tuple[list[_FileEntry], bool]:
        """Files that can answer a bar request, and whether they need folding.

        A dataset is one thing or the other in practice: a directory of daily or
        minute aggregates, or a directory of raw bid/ask. Aggregates win when
        both are present; otherwise quotes are folded into bars, because the
        engine consumes bars and a quote-only feed would otherwise serve none.
        """
        lookback = None if unbounded_history else self._history_lookback_ns(request)
        aggregates = self._aggregate_entries(request, lookback_ns=lookback)
        if aggregates:
            return aggregates, False
        quotes = self._quote_entries(request, lookback_ns=lookback)
        if not quotes:
            return [], False
        if not self._config.derive_bars_from_quotes:
            raise DataIntegrityError(
                "This dataset holds quotes but no OHLC aggregates, and "
                "derive_bars_from_quotes is off, so no bars can be produced. "
                "Enable it, or point the provider at aggregate files.",
                symbols=list(request.symbols),
            )
        if request.timeframe is Timeframe.TICK:
            _LOG.debug("Emitting one bar per quote for a TICK request")
        return quotes, True

    def _aggregate_entries(
        self, request: DataRequest, lookback_ns: int | None = 0
    ) -> list[_FileEntry]:
        start_ns = self._window_start_ns(request, lookback_ns)
        end_ns = to_epoch_ns(request.end)
        candidates = self._prune_files(request, kind=CsvFileKind.AGGREGATE, start_ns=start_ns)
        indexed = self._ensure_indexed(candidates)
        wanted = set(request.symbols)
        return [
            e
            for e in indexed
            if e.kind is CsvFileKind.AGGREGATE
            and e.overlaps(start_ns, end_ns)
            and (wanted & set(e.symbols))
        ]

    @staticmethod
    def _window_start_ns(request: DataRequest, lookback_ns: int | None) -> int:
        """Request start, pushed back by the warm-up look-back (``None`` == all)."""
        if lookback_ns is None:
            return 0
        return to_epoch_ns(request.start) - lookback_ns

    def _stream(
        self,
        request: DataRequest,
        entries: list[_FileEntry],
        from_quotes: bool,
        start_ns: int,
    ) -> Iterator[Bar]:
        end_ns = to_epoch_ns(request.end)
        frame = self._timeframe_for(entries, request)
        adjust = request.adjusted and not from_quotes
        columns = self._QUOTE_COLUMNS if from_quotes else self._bar_columns(entries, adjust)
        is_adjusted = adjust and "adj_close" in columns
        width_ns = self._frame_width_ns(frame)
        # A derived bar already carries its own closing quote; running the
        # as-of reader over it as well would be redundant work.
        quotes = (
            _QuoteReader(self, request, frame)
            if request.include_quotes and not from_quotes
            else None
        )
        align = width_ns if from_quotes else 0
        if from_quotes:
            # Widen back to the start of the bar that contains start_ns, so the
            # first bar is folded from all its quotes rather than a suffix.
            start_ns = max(((start_ns - 1) // width_ns) * width_ns + 1, 0) if width_ns else start_ns

        for slice_start, slice_end in self._slices(entries, start_ns, end_ns, align_ns=align):
            active = [e for e in entries if e.overlaps(slice_start, slice_end)]
            if not active:
                continue
            table = self._gather(active, request.symbols, slice_start, slice_end, columns)
            if table is None:
                continue
            if from_quotes:
                table = self._bars_from_quotes(table, width_ns)
                if table is None:
                    continue
            ts, o, h, l, c, v = self._bar_arrays(table, adjust)
            codes, names = _symbol_codes(table)
            # One id per distinct ticker in the slice, indexed by code — the
            # alternative is a dict lookup on a fresh Python string per bar.
            ids = [Equity.make_id(Symbol(name)) for name in names]
            attached: list[Quote | None] | None = None
            if quotes is not None:
                attached = quotes.attach(codes, names, ts, slice_start, slice_end)
            elif from_quotes:
                attached = self._closing_quotes(table)
            # Rows are ordered by timestamp, so in a portfolio the same instant
            # repeats once per symbol. Building the datetime once per instant
            # instead of once per row is the difference between N and N/symbols
            # object allocations on the hottest loop in the provider.
            previous_ns, moment = -1, _EPOCH_UTC
            for i in range(len(ts)):
                epoch_ns = int(ts[i])
                if epoch_ns > end_ns:
                    continue
                if epoch_ns != previous_ns:
                    previous_ns, moment = epoch_ns, from_epoch_ns(epoch_ns)
                yield Bar(
                    instrument_id=ids[codes[i]],
                    timestamp=moment,
                    open=float(o[i]),
                    high=float(h[i]),
                    low=float(l[i]),
                    close=float(c[i]),
                    volume=float(v[i]),
                    timeframe=frame,
                    quote=None if attached is None else attached[i],
                    is_adjusted=is_adjusted,
                )

    _QUOTE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "ts",
        "symbol",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
    )

    @staticmethod
    def _frame_width_ns(frame: Timeframe) -> int:
        """Bar width in nanoseconds; 0 for TICK, which is one bar per update."""
        return 0 if frame is Timeframe.TICK else int(frame.duration.total_seconds() * 1e9)

    @staticmethod
    def _closing_quotes(table: pa.Table) -> list[Quote | None]:
        """The bid/ask a folded bar closed on, in row order."""
        bid, ask, bid_size, ask_size = (
            table.column(name).to_numpy(zero_copy_only=False)
            for name in ("bid", "ask", "bid_size", "ask_size")
        )
        return [
            Quote(bid=float(b), ask=float(a), bid_size=float(bs), ask_size=float(asz))
            for b, a, bs, asz in zip(bid, ask, bid_size, ask_size, strict=True)
        ]

    def _bars_from_quotes(self, table: pa.Table, width_ns: int) -> pa.Table | None:
        """Fold top-of-book updates into OHLC bars of ``width_ns``.

        OHLC tracks :attr:`CsvProviderConfig.quote_bar_price` (mid by default).
        Volume is zero and stays zero: quotes are not trades, and inventing a
        number from quoted sizes would feed a volume-participation cap that has
        no basis in anything that traded.

        The bar also carries the LAST quote inside it, so execution still prices
        off a real bid/ask rather than a spread reconstructed from the mid.
        """
        import pyarrow as pa_

        if table.num_rows == 0:
            return None
        ts = table.column("ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        bid = table.column("bid").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        ask = table.column("ask").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        bid_size = (
            table.column("bid_size").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        )
        ask_size = (
            table.column("ask_size").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        )
        basis = self._config.quote_bar_price
        price = (
            bid
            if basis is QuotePriceBasis.BID
            else ask
            if basis is QuotePriceBasis.ASK
            else (bid + ask) * 0.5
        )

        codes, names = _symbol_codes(table)
        # Right-labelled close of the half-open interval (close - w, close].
        close_ns = ((ts - 1) // width_ns + 1) * width_ns if width_ns > 0 else ts

        out_ts: list[npt.NDArray[np.int64]] = []
        out_sym: list[str] = []
        opens: list[npt.NDArray[np.float64]] = []
        highs: list[npt.NDArray[np.float64]] = []
        lows: list[npt.NDArray[np.float64]] = []
        closes: list[npt.NDArray[np.float64]] = []
        last_bid: list[npt.NDArray[np.float64]] = []
        last_ask: list[npt.NDArray[np.float64]] = []
        last_bid_size: list[npt.NDArray[np.float64]] = []
        last_ask_size: list[npt.NDArray[np.float64]] = []
        counts: list[int] = []

        for name, rows in zip(names, _group_rows_by_code(codes, len(names)), strict=True):
            if rows.size == 0:
                continue
            buckets = close_ns[rows]
            # rows are already in ascending ts within a symbol, so a bucket
            # change marks a segment boundary and reduceat can do the rest.
            starts = np.flatnonzero(np.r_[True, buckets[1:] != buckets[:-1]])
            ends = np.r_[starts[1:], buckets.size] - 1
            segment_price = price[rows]
            out_ts.append(buckets[starts])
            opens.append(segment_price[starts])
            closes.append(segment_price[ends])
            highs.append(np.maximum.reduceat(segment_price, starts))
            lows.append(np.minimum.reduceat(segment_price, starts))
            last_bid.append(bid[rows][ends])
            last_ask.append(ask[rows][ends])
            last_bid_size.append(bid_size[rows][ends])
            last_ask_size.append(ask_size[rows][ends])
            out_sym.append(name)
            counts.append(starts.size)

        if not out_ts:
            return None
        symbols = np.repeat(np.array(out_sym, dtype=object), counts)
        folded = pa_.table(
            {
                "ts": pa_.array(np.concatenate(out_ts)),
                "symbol": pa_.array(symbols, type=pa_.string()),
                "open": pa_.array(np.concatenate(opens)),
                "high": pa_.array(np.concatenate(highs)),
                "low": pa_.array(np.concatenate(lows)),
                "close": pa_.array(np.concatenate(closes)),
                "volume": pa_.array(np.zeros(int(sum(counts)), dtype=np.float64)),
                "bid": pa_.array(np.concatenate(last_bid)),
                "ask": pa_.array(np.concatenate(last_ask)),
                "bid_size": pa_.array(np.concatenate(last_bid_size)),
                "ask_size": pa_.array(np.concatenate(last_ask_size)),
            }
        )
        return folded.sort_by([("ts", "ascending"), ("symbol", "ascending")])

    def _bar_columns(self, entries: Sequence[_FileEntry], adjusted: bool) -> tuple[str, ...]:
        base = ("ts", "symbol", "open", "high", "low", "close", "volume")
        if adjusted and self._has_adjusted_close(entries):
            return (*base, "adj_close")
        return base

    def _has_adjusted_close(self, entries: Sequence[_FileEntry]) -> bool:
        """True when *any* selected file carries an adjusted close.

        Any, not all, and not just the first: a directory can mix a Yahoo export
        that has the column with a Polygon one that does not, and which of them
        sorts first must not decide whether adjustment happens at all.
        Files without the column are simply left unadjusted.
        """
        return any("adj_close" in self._file_columns(entry) for entry in entries)

    def _file_columns(self, entry: _FileEntry) -> frozenset[str]:
        """Columns physically present in one converted file."""
        import pyarrow.parquet as pq

        if entry.parquet is None:
            cached = self._tables.get(entry.path)
            return frozenset(cached.column_names) if cached is not None else frozenset()
        path = self._cache_root() / "files" / entry.parquet
        if not path.exists():
            return frozenset()
        try:
            return frozenset(pq.read_schema(path).names)
        except OSError:
            return frozenset()

    @staticmethod
    def _bar_arrays(table: pa.Table, adjusted: bool) -> _BarColumns:
        ts = table.column("ts").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        cols = [
            table.column(name).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for name in ("open", "high", "low", "close", "volume")
        ]
        if adjusted and "adj_close" in table.column_names:
            adj = (
                table.column("adj_close")
                .to_numpy(zero_copy_only=False)
                .astype(np.float64, copy=False)
            )
            close = cols[3]
            # Rows from a file that had no adj_close arrive as null; those keep
            # their raw prices rather than being scaled by a NaN ratio.
            usable = np.isfinite(adj) & (adj > 0.0) & (close > 0.0)
            ratio = np.ones_like(close)
            np.divide(adj, close, out=ratio, where=usable)
            cols[0] = cols[0] * ratio
            cols[1] = cols[1] * ratio
            cols[2] = cols[2] * ratio
            cols[3] = np.where(usable, adj, close)
            cols[4] = cols[4] / ratio
        return ts, cols[0], cols[1], cols[2], cols[3], cols[4]

    def _slices(
        self, entries: Sequence[_FileEntry], start_ns: int, end_ns: int, align_ns: int = 0
    ) -> Iterator[tuple[int, int]]:
        """Split ``[start_ns, end_ns]`` into chunks of roughly the row budget.

        ``align_ns`` snaps every boundary to a bar edge. That matters only when
        folding quotes into bars: a slice cutting through a bar would emit two
        half-bars for the same minute instead of one.
        """
        span = end_ns - start_ns
        total = sum(e.rows for e in entries)
        budget = max(self._config.stream_chunk_rows, 1)
        if span <= 0 or total <= budget:
            yield start_ns, end_ns
            return
        covered = max(max(e.max_ts for e in entries) - min(e.min_ts for e in entries), 1)
        rows_per_ns = total / covered
        width = max(int(budget / rows_per_ns), 1) if rows_per_ns > 0 else span
        count = min(max((span // width) + 1, 1), 100_000)
        step = span // count + 1
        if align_ns > 0:
            step = max((step + align_ns - 1) // align_ns, 1) * align_ns
        cursor = start_ns
        while cursor <= end_ns:
            yield cursor, min(cursor + step - 1, end_ns)
            cursor += step

    def load_series(self, symbol: Symbol, request: DataRequest) -> BarSeries:
        ticker = Symbol(str(symbol).strip().upper())
        scoped = request if request.symbols == (ticker,) else replace(request, symbols=(ticker,))
        key = CacheKey.from_request(self.name, ticker, scoped, self._source_digest())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        series = self._load_many_uncached(scoped).get(ticker)
        if series is None:
            raise DataNotAvailableError(ticker, request.start, request.end, provider=self.name)
        self._cache.put(key, series)
        return series

    def load_many(self, request: DataRequest) -> dict[Symbol, BarSeries]:
        """One pass over the files for every symbol, rather than N passes."""
        result: dict[Symbol, BarSeries] = {}
        missing: list[Symbol] = []
        # Once per call, not once per symbol: it stats every source file.
        digest = self._source_digest()
        for symbol in request.symbols:
            scoped = replace(request, symbols=(symbol,))
            cached = self._cache.get(CacheKey.from_request(self.name, symbol, scoped, digest))
            if cached is not None:
                result[symbol] = cached
            else:
                missing.append(symbol)
        if missing:
            loaded = self._load_many_uncached(replace(request, symbols=tuple(missing)))
            for symbol, series in loaded.items():
                self._cache.put(
                    CacheKey.from_request(
                        self.name, symbol, replace(request, symbols=(symbol,)), digest
                    ),
                    series,
                )
            result.update(loaded)
        return result

    def _load_many_uncached(self, request: DataRequest) -> dict[Symbol, BarSeries]:
        entries, from_quotes, read_from = self._prepare_read(request)
        if not entries:
            return {}
        end_ns = to_epoch_ns(request.end)
        start_ns = to_epoch_ns(request.start)
        frame = self._timeframe_for(entries, request)
        adjusted = request.adjusted and not from_quotes
        columns = self._QUOTE_COLUMNS if from_quotes else self._bar_columns(entries, adjusted)
        if from_quotes:
            width = self._frame_width_ns(frame)
            read_from = max(((read_from - 1) // width) * width + 1, 0) if width else read_from
        table = self._gather(entries, request.symbols, read_from, end_ns, columns)
        if table is not None and from_quotes:
            table = self._bars_from_quotes(table, self._frame_width_ns(frame))
        if table is None:
            return {}

        ts, o, h, l, c, v = self._bar_arrays(table, adjusted)
        codes, names = _symbol_codes(table)
        out: dict[Symbol, BarSeries] = {}
        for name, rows in zip(names, _group_rows_by_code(codes, len(names)), strict=True):
            if rows.size == 0:
                continue
            own_ts = ts[rows]
            keep = rows[self._trim_warmup(own_ts, start_ns, request.warmup_bars)]
            if keep.size == 0:
                continue
            symbol = Symbol(name)
            out[symbol] = BarSeries.from_arrays(
                Equity.make_id(symbol), frame, ts[keep], o[keep], h[keep], l[keep], c[keep], v[keep]
            )
        return out

    @staticmethod
    def _trim_warmup(
        ts: npt.NDArray[np.int64], start_ns: int, warmup_bars: int
    ) -> npt.NDArray[np.int64]:
        """Indices of the in-range bars plus exactly ``warmup_bars`` before them."""
        first = int(np.searchsorted(ts, start_ns, side="left"))
        keep_from = max(first - warmup_bars, 0)
        return np.arange(keep_from, ts.size)

    # ---- quotes ------------------------------------------------------------ #

    def stream_quotes(self, request: DataRequest) -> Iterator[tuple[Symbol, UtcDatetime, Quote]]:
        """Yield raw top-of-book updates in timestamp order.

        Not part of :class:`~sigmaloop.data.provider.DataProvider` — quotes reach
        the engine attached to bars via ``DataRequest.include_quotes``. This is
        the direct path for microstructure work and for inspecting a quote file.
        """
        entries = self._quote_entries(request)
        if not entries:
            return
        start_ns, end_ns = to_epoch_ns(request.start), to_epoch_ns(request.end)
        columns = ("ts", "symbol", "bid", "ask", "bid_size", "ask_size")
        for slice_start, slice_end in self._slices(entries, start_ns, end_ns):
            active = [e for e in entries if e.overlaps(slice_start, slice_end)]
            table = (
                self._gather(active, request.symbols, slice_start, slice_end, columns)
                if active
                else None
            )
            if table is None:
                continue
            ts = table.column("ts").to_numpy(zero_copy_only=False)
            codes, names = _symbol_codes(table)
            tickers = [Symbol(n) for n in names]
            bid, ask, bid_size, ask_size = (
                table.column(name).to_numpy(zero_copy_only=False)
                for name in ("bid", "ask", "bid_size", "ask_size")
            )
            for i in range(len(ts)):
                yield (
                    tickers[codes[i]],
                    from_epoch_ns(int(ts[i])),
                    Quote(
                        bid=float(bid[i]),
                        ask=float(ask[i]),
                        bid_size=float(bid_size[i]),
                        ask_size=float(ask_size[i]),
                    ),
                )

    def _quote_entries(self, request: DataRequest, lookback_ns: int | None = 0) -> list[_FileEntry]:
        candidates = self._prune_files(
            request, kind=CsvFileKind.QUOTE, start_ns=self._window_start_ns(request, lookback_ns)
        )
        indexed = self._ensure_indexed(candidates)
        wanted = set(request.symbols)
        return [e for e in indexed if e.kind is CsvFileKind.QUOTE and (wanted & set(e.symbols))]

    # ---- lifecycle ---------------------------------------------------------- #

    def open(self) -> None:
        self._load_index()

    def close(self) -> None:
        self._flush_index()
        with self._lock:
            self._tables.clear()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CsvDataProvider(path={self._config.path!s})"


class _QuoteReader:
    """As-of joins bars to the most recent quote at or before each bar close.

    Strictly backward-looking: ``searchsorted(..., side="right") - 1`` can only
    select a quote that already existed, so attaching one cannot leak future
    information into a fill.
    """

    __slots__ = ("_carry", "_entries", "_frame_ns", "_max_age_ns", "_provider", "_request")

    def __init__(self, provider: CsvDataProvider, request: DataRequest, frame: Timeframe) -> None:
        self._provider = provider
        self._request = request
        self._entries = provider._quote_entries(request)
        self._carry: dict[str, tuple[int, Quote]] = {}
        max_age = provider._config.max_quote_age
        self._max_age_ns = None if max_age is None else int(max_age.total_seconds() * 1e9)
        self._frame_ns = (
            int(frame.duration.total_seconds() * 1e9)
            if frame is not Timeframe.TICK
            else _NS_PER_DAY
        )

    def attach(
        self,
        bar_codes: npt.NDArray[np.int64],
        bar_names: Sequence[str],
        bar_ts: npt.NDArray[np.int64],
        slice_start: int,
        slice_end: int,
    ) -> list[Quote | None]:
        """One quote (or ``None``) per bar in the slice, positionally aligned."""
        if not self._entries:
            return [None] * len(bar_ts)
        # Reach back one bar so the first bar of a slice can still see the quote
        # that preceded it; anything older arrives via the carry-over.
        table = self._provider._gather(
            [e for e in self._entries if e.overlaps(slice_start - self._frame_ns, slice_end)],
            self._request.symbols,
            slice_start - self._frame_ns,
            slice_end,
            ("ts", "symbol", "bid", "ask", "bid_size", "ask_size"),
        )
        by_symbol: dict[str, tuple[npt.NDArray[np.int64], list[Quote]]] = {}
        if table is not None:
            ts = table.column("ts").to_numpy(zero_copy_only=False)
            codes, names = _symbol_codes(table)
            bid, ask, bid_size, ask_size = (
                table.column(n).to_numpy(zero_copy_only=False)
                for n in ("bid", "ask", "bid_size", "ask_size")
            )
            for name, rows in zip(names, _group_rows_by_code(codes, len(names)), strict=True):
                by_symbol[name] = (
                    ts[rows],
                    [
                        Quote(bid=float(b), ask=float(a), bid_size=float(bs), ask_size=float(asz))
                        for b, a, bs, asz in zip(
                            bid[rows], ask[rows], bid_size[rows], ask_size[rows], strict=True
                        )
                    ],
                )

        out: list[Quote | None] = []
        for i in range(len(bar_ts)):
            out.append(self._resolve(bar_names[bar_codes[i]], int(bar_ts[i]), by_symbol))
        for name, (ts_arr, quotes) in by_symbol.items():
            if ts_arr.size:
                self._carry[name] = (int(ts_arr[-1]), quotes[-1])
        return out

    def _resolve(
        self,
        symbol: str,
        bar_ts: int,
        by_symbol: Mapping[str, tuple[npt.NDArray[np.int64], list[Quote]]],
    ) -> Quote | None:
        found = by_symbol.get(symbol)
        best: tuple[int, Quote] | None = self._carry.get(symbol)
        if found is not None:
            index = int(np.searchsorted(found[0], bar_ts, side="right")) - 1
            if index >= 0:
                best = (int(found[0][index]), found[1][index])
        if best is None:
            return None
        if self._max_age_ns is not None and bar_ts - best[0] > self._max_age_ns:
            return None
        return best[1]
