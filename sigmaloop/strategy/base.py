"""Strategy base classes — one per mode, all driven by one engine.

Functional requirement 5: "There can be different Strategy bases for different
modes but a single engine should run all." That is achieved by keeping the
engine's contract narrow — :meth:`Strategy.handle_bar` — and letting each base
class adapt it into a mode-appropriate hook:

::

    Strategy                      (engine-facing contract; users rarely subclass)
    +-- SingleAssetStrategy       on_bar(ctx, bar)
    |     +-- OptionsStrategy     on_bar(ctx, bar, chain)
    +-- PortfolioStrategy         on_bar(ctx, snapshot) + select_universe(ctx)

The engine only ever calls :meth:`Strategy.handle_bar`; the subclass decides how
to unpack the snapshot. Adding a fourth mode later means adding a base class,
not touching the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from sigmaloop.data.universe import Universe
from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionChain
from sigmaloop.domain.order import Fill, Order, Rejection
from sigmaloop.domain.position import Trade
from sigmaloop.execution.expiry import ExpiryOutcome
from sigmaloop.indicators.base import IndicatorSpec
from sigmaloop.strategy.context import StrategyContext
from sigmaloop.strategy.params import ParameterSet, ParameterSpec
from sigmaloop.types import InstrumentId, StrategyMode, Symbol

__all__ = [
    "Strategy",
    "SingleAssetStrategy",
    "OptionsStrategy",
    "PortfolioStrategy",
]


class Strategy(ABC):
    """Engine-facing strategy contract.

    Lifecycle, in order::

        __init__(params)
        param_spec()            (classmethod, called before instantiation)
        declare_indicators()
        declare_instruments()
        on_start(ctx)
        for each bar:
            [on_fill / on_order_rejected / on_trade_closed / on_option_expiry]
            handle_bar(ctx)
        on_finish(ctx)

    Event callbacks fire BEFORE ``handle_bar`` for the bar in which they
    occurred, so the strategy reacts to yesterday's fills with today's data —
    the same information ordering a live system would see.

    Strategies must be **stateless across runs**: :meth:`reset` is called
    between sweep points and walk-forward folds, and any state not cleared
    there leaks between runs and corrupts comparisons.
    """

    #: Which mode this base class implements; used to validate the config.
    mode: ClassVar[StrategyMode] = StrategyMode.SINGLE_ASSET
    #: Display name for reports; defaults to the class name.
    name: ClassVar[str] = ""

    def __init__(self, params: ParameterSet | None = None) -> None:
        raise NotImplementedError

    # ---- declarations (called once, before the run) ------------------------- #

    @classmethod
    def param_spec(cls) -> ParameterSpec:
        """Declare tunable parameters. Default: no parameters."""
        raise NotImplementedError

    def declare_indicators(self) -> Sequence[IndicatorSpec]:
        """Indicators the engine should instantiate and update automatically.

        Declaring rather than constructing means the engine knows the warm-up
        requirement before loading data, and can therefore prepend exactly
        enough history without the strategy trading on a partial window.
        """
        raise NotImplementedError

    @abstractmethod
    def declare_instruments(self) -> Sequence[Symbol]:
        """Symbols required up front. Portfolio mode may return the universe's
        candidate set; options mode returns just the underlying."""
        raise NotImplementedError

    # ---- engine contract ------------------------------------------------------ #

    @abstractmethod
    def handle_bar(self, context: StrategyContext) -> None:
        """Single entry point the engine calls each bar.

        Implemented by the mode base classes; user strategies override the
        mode-specific ``on_bar`` instead.
        """
        raise NotImplementedError

    # ---- optional lifecycle hooks -------------------------------------------- #

    def on_start(self, context: StrategyContext) -> None:
        """Called once before the first tradeable bar (after warm-up)."""

    def on_finish(self, context: StrategyContext) -> None:
        """Called once after the last bar, before final liquidation."""

    def on_fill(self, context: StrategyContext, fill: Fill) -> None:
        """An order (fully or partially) executed."""

    def on_order_rejected(
        self, context: StrategyContext, order: Order, rejection: Rejection
    ) -> None:
        """An order was refused — insufficient capital, risk limit, no data."""

    def on_trade_closed(self, context: StrategyContext, trade: Trade) -> None:
        """A round trip completed."""

    def on_option_expiry(self, context: StrategyContext, outcome: ExpiryOutcome) -> None:
        """An option position expired, was exercised, or was assigned."""

    def reset(self) -> None:
        """Clear all per-run state. Must be implemented if the strategy keeps any."""

    @property
    def params(self) -> ParameterSet:
        raise NotImplementedError


class SingleAssetStrategy(Strategy):
    """Mode (a): one underlying instrument.

    Example from the requirements: *buy TQQQ at every open; sell the lot when
    the gain reaches 10%*.
    """

    mode: ClassVar[StrategyMode] = StrategyMode.SINGLE_ASSET

    #: Set by config or by overriding; the single traded symbol.
    symbol: ClassVar[Symbol | None] = None

    @property
    def instrument_id(self) -> InstrumentId:
        """The resolved id of :attr:`symbol`, available from ``on_start``."""
        raise NotImplementedError

    @abstractmethod
    def on_bar(self, context: StrategyContext, bar: Bar) -> None:
        """Called once per bar with the traded instrument's bar."""
        raise NotImplementedError

    def declare_instruments(self) -> Sequence[Symbol]:
        raise NotImplementedError

    def handle_bar(self, context: StrategyContext) -> None:
        """Unpacks the snapshot and delegates to :meth:`on_bar`.

        Skips the bar entirely if the instrument did not trade, rather than
        calling ``on_bar`` with a stale bar.
        """
        raise NotImplementedError


class OptionsStrategy(SingleAssetStrategy):
    """Mode (b): one underlying, plus its options, plus combinations.

    Examples from the requirements: *buy the SPY 0DTE 20-delta put and call*;
    *test a covered call on AMZN*.

    The chain is passed in rather than fetched by the strategy, so contract
    selection is always evaluated against the same point-in-time snapshot the
    engine will price fills from.
    """

    mode: ClassVar[StrategyMode] = StrategyMode.SINGLE_ASSET_OPTIONS

    #: Declared filters, pushed down into the provider's chain request so the
    #: engine never materialises contracts the strategy cannot use.
    min_dte: ClassVar[int | None] = None
    max_dte: ClassVar[int | None] = None
    strike_window_pct: ClassVar[float | None] = None
    require_greeks: ClassVar[bool] = True

    @abstractmethod
    def on_bar(  # type: ignore[override]
        self, context: StrategyContext, bar: Bar, chain: OptionChain
    ) -> None:
        """Called with the underlying's bar and its option chain."""
        raise NotImplementedError

    def handle_bar(self, context: StrategyContext) -> None:
        raise NotImplementedError

    def on_option_expiry(self, context: StrategyContext, outcome: ExpiryOutcome) -> None:
        """Override to roll, re-hedge, or record assignment handling."""


class PortfolioStrategy(Strategy):
    """Mode (c): evaluate a universe each bar and act on everything that fits.

    Examples from the requirements: *buy every breakout stock daily, exit on
    stop or target*; *buy ITM puts on all top-10% gainers*.

    Two extension points:

    * :meth:`select_universe` — which names are investable now.
    * :meth:`on_bar` — what to do given the snapshot restricted to them.

    ``rebalance_frequency`` throttles universe recomputation; screening
    thousands of names every bar is usually the dominant cost in this mode.
    """

    mode: ClassVar[StrategyMode] = StrategyMode.PORTFOLIO

    #: Recompute universe membership every N bars. 1 == every bar.
    rebalance_frequency: ClassVar[int] = 1
    #: Hard cap on concurrent positions, enforced by the risk layer.
    max_positions: ClassVar[int | None] = None

    @abstractmethod
    def select_universe(self, context: StrategyContext) -> Universe | Sequence[Symbol]:
        """Define the investable set — either a :class:`Universe` (re-resolved
        each rebalance) or a static symbol list."""
        raise NotImplementedError

    @abstractmethod
    def on_bar(self, context: StrategyContext, snapshot: MarketSnapshot) -> None:
        """Called once per bar with the full snapshot for the active universe."""
        raise NotImplementedError

    def declare_instruments(self) -> Sequence[Symbol]:
        raise NotImplementedError

    def handle_bar(self, context: StrategyContext) -> None:
        raise NotImplementedError

    def active_universe(self) -> Sequence[InstrumentId]:
        """Members resolved at the most recent rebalance."""
        raise NotImplementedError
