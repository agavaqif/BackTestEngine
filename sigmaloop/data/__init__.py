"""Data acquisition layer: providers, caching, feeds, universes, calendars."""

from __future__ import annotations

from sigmaloop.data.cache import (
    CacheKey,
    CacheStats,
    DataCache,
    MemoryDataCache,
    ParquetDataCache,
    TieredDataCache,
)
from sigmaloop.data.calendar import ContinuousCalendar, NyseCalendar, Session, TradingCalendar
from sigmaloop.data.feed import (
    DataFeed,
    FeedPlan,
    HistoryWindow,
    MergedDataFeed,
    PrefetchDataFeed,
    ReplayDataFeed,
)
from sigmaloop.data.provider import (
    CompositeDataProvider,
    DataProvider,
    DataRequest,
    OptionChainRequest,
    OptionsDataProvider,
    ProviderCapabilities,
)
from sigmaloop.data.universe import (
    CallableScreen,
    LiquidityScreen,
    RankedUniverse,
    Screen,
    ScreenedUniverse,
    StaticUniverse,
    Universe,
    UniverseSpec,
)

__all__ = [
    "CacheKey",
    "CacheStats",
    "CallableScreen",
    "CompositeDataProvider",
    "ContinuousCalendar",
    "DataCache",
    "DataFeed",
    "DataProvider",
    "DataRequest",
    "FeedPlan",
    "HistoryWindow",
    "LiquidityScreen",
    "MemoryDataCache",
    "MergedDataFeed",
    "NyseCalendar",
    "OptionChainRequest",
    "OptionsDataProvider",
    "ParquetDataCache",
    "PrefetchDataFeed",
    "ProviderCapabilities",
    "RankedUniverse",
    "ReplayDataFeed",
    "Screen",
    "ScreenedUniverse",
    "Session",
    "StaticUniverse",
    "TieredDataCache",
    "TradingCalendar",
    "Universe",
    "UniverseSpec",
]
