"""``RunContext`` — the engine's internal state container, and the concrete
:class:`~sigmaloop.strategy.context.StrategyContext` handed to user code.

One object holds everything scoped to a single run: clock, registry, feed,
portfolio, broker, indicators, collected intents and diagnostics. Assembling it
once and passing it around (rather than threading eight arguments through every
call) keeps the loop readable and makes the run's state trivially inspectable
in a debugger or a post-mortem.

The same object implements :class:`StrategyContext`, but strategies see it
through that narrower interface: they cannot reach the broker or the mutable
portfolio, and every data accessor routes through the clock's lookahead guard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sigmaloop.data.feed import DataFeed, HistoryWindow
from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionChain
from sigmaloop.domain.instrument import Instrument, InstrumentRegistry, OptionContract
from sigmaloop.domain.order import Order, OrderIntent
from sigmaloop.domain.position import Position
from sigmaloop.engine.clock import SimulationClock
from sigmaloop.engine.config import BacktestConfig
from sigmaloop.engine.events import EventBus
from sigmaloop.execution.broker import Broker
from sigmaloop.indicators.base import Indicator, IndicatorSet
from sigmaloop.portfolio.accounting import Portfolio, PortfolioView
from sigmaloop.portfolio.risk import RiskManager
from sigmaloop.portfolio.sizing import PositionSizer
from sigmaloop.strategy.context import StrategyContext
from sigmaloop.strategy.params import ParameterSet
from sigmaloop.types import (
    InstrumentId,
    Money,
    Price,
    Quantity,
    RunId,
    Symbol,
    UtcDatetime,
)

__all__ = ["RunDiagnostics", "RunContext"]


@dataclass(slots=True)
class RunDiagnostics:
    """Counters and messages accumulated during a run.

    Surfaced in the summary so a result always carries the caveats that apply
    to it — a Sharpe of 3.0 computed over 4,000 synthetic-spread fills is a very
    different claim from the same number on real quotes.
    """

    warnings: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    bars_processed: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    orders_expired: int = 0
    fills: int = 0
    partial_fills: int = 0
    synthetic_quote_fills: int = 0
    stale_marks: int = 0
    missing_bars: int = 0
    capital_breaches: int = 0
    #: Per-bar custom series recorded via ``ctx.record``.
    recorded: dict[str, list[float]] = field(default_factory=dict)

    def warn_once(self, message: str) -> None:
        """Deduplicated warning — a per-bar caveat must not emit 40,000 lines."""
        raise NotImplementedError


class RunContext(StrategyContext):
    """Concrete strategy context and internal run state.

    Constructed by :class:`~sigmaloop.engine.core.BacktestEngine.prepare` after
    every component has been resolved from config, so by the time a strategy
    sees it, all plugins are loaded and validated.
    """

    __slots__ = (
        "run_id",
        "config",
        "clock",
        "registry",
        "feed",
        "_portfolio",
        "broker",
        "indicators",
        "sizer",
        "risk",
        "events",
        "diagnostics",
        "strategy_params",
        "state",
        "_snapshot",
        "_pending_intents",
        "_histories",
        "_active_universe",
    )

    def __init__(
        self,
        run_id: RunId,
        config: BacktestConfig,
        clock: SimulationClock,
        registry: InstrumentRegistry,
        feed: DataFeed,
        portfolio: Portfolio,
        broker: Broker,
        indicators: IndicatorSet,
        sizer: PositionSizer,
        risk: RiskManager,
        events: EventBus,
        strategy_params: ParameterSet,
    ) -> None:
        raise NotImplementedError

    # ---- engine-internal API (not visible through StrategyContext) ---------- #

    def begin_bar(self, snapshot: MarketSnapshot) -> None:
        """Advance the clock, install the snapshot, clear pending intents."""
        raise NotImplementedError

    def drain_intents(self) -> Sequence[OrderIntent]:
        """Take and clear intents raised during this bar's ``handle_bar``."""
        raise NotImplementedError

    def set_active_universe(self, instrument_ids: Sequence[InstrumentId]) -> None:
        raise NotImplementedError

    def mutable_portfolio(self) -> Portfolio:
        """Engine-only accessor for the writable ledger."""
        raise NotImplementedError

    # ---- StrategyContext ----------------------------------------------------- #

    @property
    def now(self) -> UtcDatetime:
        raise NotImplementedError

    @property
    def bar_index(self) -> int:
        raise NotImplementedError

    @property
    def snapshot(self) -> MarketSnapshot:
        raise NotImplementedError

    @property
    def params(self) -> ParameterSet:
        raise NotImplementedError

    @property
    def portfolio(self) -> PortfolioView:
        """Read-only view. The writable ledger is engine-only, via
        :meth:`mutable_portfolio`."""
        raise NotImplementedError

    @property
    def is_warmup(self) -> bool:
        raise NotImplementedError

    @property
    def is_last_bar(self) -> bool:
        raise NotImplementedError

    def bar(self, instrument_id: InstrumentId) -> Bar | None:
        raise NotImplementedError

    def price(self, instrument_id: InstrumentId) -> Price | None:
        raise NotImplementedError

    def history(self, instrument_id: InstrumentId) -> HistoryWindow:
        raise NotImplementedError

    def instrument(self, instrument_id: InstrumentId) -> Instrument:
        raise NotImplementedError

    def resolve(self, symbol: Symbol) -> InstrumentId:
        raise NotImplementedError

    def indicator(self, alias: str, instrument_id: InstrumentId | None = None) -> Indicator[object]:
        raise NotImplementedError

    def indicator_value(self, alias: str, instrument_id: InstrumentId | None = None) -> object:
        raise NotImplementedError

    def chain(self, underlying: InstrumentId | Symbol) -> OptionChain:
        raise NotImplementedError

    def subscribe_option(self, contract: OptionContract) -> InstrumentId:
        raise NotImplementedError

    def position(self, instrument_id: InstrumentId) -> Position | None:
        raise NotImplementedError

    def positions(self) -> Sequence[Position]:
        raise NotImplementedError

    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        raise NotImplementedError

    def submit(self, intent: OrderIntent) -> OrderIntent:
        raise NotImplementedError

    def cancel(self, order: Order | str) -> bool:
        raise NotImplementedError

    def cancel_all(self, instrument_id: InstrumentId | None = None) -> int:
        raise NotImplementedError

    def buy(
        self,
        instrument_id: InstrumentId,
        *,
        quantity: Quantity | None = None,
        notional: Money | None = None,
        percent_equity: float | None = None,
        limit_price: Price | None = None,
        stop_loss: Price | None = None,
        take_profit: Price | None = None,
        tag: str = "",
    ) -> OrderIntent:
        raise NotImplementedError

    def sell(
        self,
        instrument_id: InstrumentId,
        *,
        quantity: Quantity | None = None,
        notional: Money | None = None,
        percent_equity: float | None = None,
        limit_price: Price | None = None,
        tag: str = "",
    ) -> OrderIntent:
        raise NotImplementedError

    def close(self, instrument_id: InstrumentId, *, tag: str = "") -> OrderIntent | None:
        raise NotImplementedError

    def close_all(self, *, tag: str = "") -> Sequence[OrderIntent]:
        raise NotImplementedError

    def order_target_percent(
        self, instrument_id: InstrumentId, target: float, *, tag: str = ""
    ) -> OrderIntent | None:
        raise NotImplementedError

    def rebalance(self, weights: Mapping[InstrumentId, float]) -> Sequence[OrderIntent]:
        raise NotImplementedError

    def log(self, message: str, /, **fields: object) -> None:
        raise NotImplementedError

    def record(self, name: str, value: float) -> None:
        raise NotImplementedError

    def warn(self, message: str) -> None:
        raise NotImplementedError
