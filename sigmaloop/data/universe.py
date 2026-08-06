"""Universe definition and screening for portfolio mode.

Portfolio strategies do not name their instruments up front — they say "all
breakout stocks" or "top 10% gainers". A :class:`Universe` turns that intent
into a concrete, point-in-time symbol list on every rebalance, which the feed
then subscribes to.

Point-in-time correctness is the whole job here: a universe must answer "what
was investable *as of* this timestamp", never "what is investable today".
Ignoring that is survivorship bias, and it silently inflates every metric
downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.types import Money, Symbol, UtcDatetime

__all__ = [
    "UniverseSpec",
    "Universe",
    "StaticUniverse",
    "ScreenedUniverse",
    "RankedUniverse",
    "Screen",
    "LiquidityScreen",
    "CallableScreen",
]


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """Declarative universe config, serialisable for reproducible runs."""

    #: Explicit tickers, or a named source ("sp500", "nasdaq100", a file path).
    symbols: tuple[Symbol, ...] = ()
    source: str | None = None
    max_size: int | None = None
    #: How often membership is recomputed. ``None`` == every bar.
    rebalance_every: int | None = None
    min_price: float | None = None
    min_avg_dollar_volume: Money | None = None
    exclude: frozenset[Symbol] = frozenset()


class Screen(ABC):
    """A composable point-in-time membership filter."""

    @abstractmethod
    def passes(self, symbol: Symbol, snapshot: MarketSnapshot, context: object) -> bool:
        """True if ``symbol`` qualifies at ``snapshot.timestamp``."""
        raise NotImplementedError

    def __and__(self, other: Screen) -> Screen:
        """Conjunction, so screens compose: ``liquid & breakout``."""
        raise NotImplementedError

    def __or__(self, other: Screen) -> Screen:
        raise NotImplementedError

    def __invert__(self) -> Screen:
        raise NotImplementedError


class LiquidityScreen(Screen):
    """Drops names that are too cheap or too thin to trade realistically."""

    __slots__ = ("min_price", "min_avg_dollar_volume", "lookback")

    def __init__(
        self,
        min_price: float = 5.0,
        min_avg_dollar_volume: Money = 1_000_000.0,
        lookback: int = 20,
    ) -> None:
        raise NotImplementedError

    def passes(self, symbol: Symbol, snapshot: MarketSnapshot, context: object) -> bool:
        raise NotImplementedError


class CallableScreen(Screen):
    """Adapts a plain user function into a :class:`Screen`."""

    __slots__ = ("_fn", "name")

    def __init__(
        self,
        fn: Callable[[Symbol, MarketSnapshot, object], bool],
        name: str = "custom",
    ) -> None:
        raise NotImplementedError

    def passes(self, symbol: Symbol, snapshot: MarketSnapshot, context: object) -> bool:
        raise NotImplementedError


class Universe(ABC):
    """Resolves the tradeable symbol set at a point in time."""

    @abstractmethod
    def resolve(self, snapshot: MarketSnapshot, context: object) -> Sequence[Symbol]:
        """Members as of ``snapshot.timestamp``, most-preferred first."""
        raise NotImplementedError

    @abstractmethod
    def candidate_symbols(self, start: UtcDatetime, end: UtcDatetime) -> Sequence[Symbol]:
        """Superset of everything that could ever be a member in this window.

        The feed subscribes to this up front so data loading stays predictable
        and parallelisable, even though membership changes per bar.
        """
        raise NotImplementedError

    def should_rebalance(self, snapshot: MarketSnapshot, bar_index: int) -> bool:
        raise NotImplementedError


class StaticUniverse(Universe):
    """A fixed ticker list. The simplest and fastest option."""

    __slots__ = ("_symbols",)

    def __init__(self, symbols: Sequence[Symbol]) -> None:
        raise NotImplementedError

    def resolve(self, snapshot: MarketSnapshot, context: object) -> Sequence[Symbol]:
        raise NotImplementedError

    def candidate_symbols(self, start: UtcDatetime, end: UtcDatetime) -> Sequence[Symbol]:
        raise NotImplementedError


class ScreenedUniverse(Universe):
    """A candidate pool narrowed by an ordered chain of :class:`Screen` s."""

    __slots__ = ("_candidates", "_screens", "_spec")

    def __init__(
        self,
        candidates: Sequence[Symbol],
        screens: Sequence[Screen] = (),
        spec: UniverseSpec | None = None,
    ) -> None:
        raise NotImplementedError

    def resolve(self, snapshot: MarketSnapshot, context: object) -> Sequence[Symbol]:
        raise NotImplementedError

    def candidate_symbols(self, start: UtcDatetime, end: UtcDatetime) -> Sequence[Symbol]:
        raise NotImplementedError


class RankedUniverse(Universe):
    """Screens, then ranks by a score and keeps the top N or top X%.

    This is the "top 10% gainers" case from the requirements.
    """

    __slots__ = ("_inner", "_score_fn", "_top_n", "_top_pct", "_descending")

    def __init__(
        self,
        inner: Universe,
        score_fn: Callable[[Symbol, MarketSnapshot, object], float],
        *,
        top_n: int | None = None,
        top_pct: float | None = None,
        descending: bool = True,
    ) -> None:
        raise NotImplementedError

    def resolve(self, snapshot: MarketSnapshot, context: object) -> Sequence[Symbol]:
        raise NotImplementedError

    def candidate_symbols(self, start: UtcDatetime, end: UtcDatetime) -> Sequence[Symbol]:
        raise NotImplementedError

    def last_scores(self) -> dict[Symbol, float]:
        """Scores from the most recent :meth:`resolve`, for logging/debugging."""
        raise NotImplementedError
