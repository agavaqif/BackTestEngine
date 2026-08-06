"""``StrategyContext`` — the entire surface a strategy is allowed to touch.

Design rule: a strategy never holds a reference to the engine, the portfolio,
the broker or the feed. It gets a context, and the context exposes read-only
views plus an order-submission API. Two consequences:

* User code cannot mutate the ledger or reach forward in the data. Lookahead
  becomes structurally impossible rather than merely discouraged.
* The engine can hand a strategy a different context per run (or per fold in
  walk-forward analysis) without the strategy noticing.

Everything on the context is scoped to the current bar. Asking for data at a
timestamp beyond :attr:`StrategyContext.now` raises
:class:`~sigmaloop.errors.LookaheadViolationError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from sigmaloop.data.feed import HistoryWindow
from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionChain
from sigmaloop.domain.instrument import Instrument, OptionContract
from sigmaloop.domain.order import Order, OrderIntent
from sigmaloop.domain.position import Position
from sigmaloop.indicators.base import Indicator
from sigmaloop.portfolio.accounting import PortfolioView
from sigmaloop.strategy.params import ParameterSet
from sigmaloop.types import (
    InstrumentId,
    Money,
    Price,
    Quantity,
    Symbol,
    UtcDatetime,
)

__all__ = ["StrategyContext"]


class StrategyContext(ABC):
    """Read-only market/account view plus the order API, for one bar."""

    # ---- clock and market state -------------------------------------------- #

    @property
    @abstractmethod
    def now(self) -> UtcDatetime:
        """Timestamp of the bar currently being processed."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bar_index(self) -> int:
        """Zero-based index of this bar within the run (post warm-up)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def snapshot(self) -> MarketSnapshot:
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> ParameterSet:
        raise NotImplementedError

    @property
    @abstractmethod
    def portfolio(self) -> PortfolioView:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_warmup(self) -> bool:
        """True while indicators are still filling; orders are ignored."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_last_bar(self) -> bool:
        raise NotImplementedError

    # ---- data access -------------------------------------------------------- #

    @abstractmethod
    def bar(self, instrument_id: InstrumentId) -> Bar | None:
        raise NotImplementedError

    @abstractmethod
    def price(self, instrument_id: InstrumentId) -> Price | None:
        """Current mark price."""
        raise NotImplementedError

    @abstractmethod
    def history(self, instrument_id: InstrumentId) -> HistoryWindow:
        """Bars up to and including :attr:`now`. Never further."""
        raise NotImplementedError

    @abstractmethod
    def instrument(self, instrument_id: InstrumentId) -> Instrument:
        raise NotImplementedError

    @abstractmethod
    def resolve(self, symbol: Symbol) -> InstrumentId:
        """Symbol -> equity instrument id, subscribing to the feed if needed."""
        raise NotImplementedError

    @abstractmethod
    def indicator(self, alias: str, instrument_id: InstrumentId | None = None) -> Indicator[object]:
        raise NotImplementedError

    @abstractmethod
    def indicator_value(self, alias: str, instrument_id: InstrumentId | None = None) -> object:
        """Shorthand for ``ctx.indicator(alias).value``."""
        raise NotImplementedError

    # ---- options ------------------------------------------------------------ #

    @abstractmethod
    def chain(self, underlying: InstrumentId | Symbol) -> OptionChain:
        """Option chain at :attr:`now`.

        Raises ``OptionChainUnavailableError`` when the run's provider cannot
        serve chains, rather than returning an empty chain that would make a
        strategy look like it simply found no contracts.
        """
        raise NotImplementedError

    @abstractmethod
    def subscribe_option(self, contract: OptionContract) -> InstrumentId:
        """Ensure a contract is tracked by the feed and the registry."""
        raise NotImplementedError

    # ---- positions and orders ------------------------------------------------ #

    @abstractmethod
    def position(self, instrument_id: InstrumentId) -> Position | None:
        raise NotImplementedError

    @abstractmethod
    def positions(self) -> Sequence[Position]:
        raise NotImplementedError

    @abstractmethod
    def working_orders(self, instrument_id: InstrumentId | None = None) -> Sequence[Order]:
        raise NotImplementedError

    @abstractmethod
    def submit(self, intent: OrderIntent) -> OrderIntent:
        """Queue an intent. Sized and risk-checked by the engine after
        ``on_bar`` returns, then worked by the broker from the next bar."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, order: Order | str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def cancel_all(self, instrument_id: InstrumentId | None = None) -> int:
        raise NotImplementedError

    # ---- convenience order helpers ------------------------------------------- #
    # Thin wrappers over ``submit``; see sigmaloop.strategy.api for the full set.

    @abstractmethod
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
        """Exactly one of ``quantity`` / ``notional`` / ``percent_equity`` must
        be supplied; omitting all three uses the run's default sizer."""
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def close(self, instrument_id: InstrumentId, *, tag: str = "") -> OrderIntent | None:
        """Flatten one position. ``None`` when already flat."""
        raise NotImplementedError

    @abstractmethod
    def close_all(self, *, tag: str = "") -> Sequence[OrderIntent]:
        raise NotImplementedError

    @abstractmethod
    def order_target_percent(
        self, instrument_id: InstrumentId, target: float, *, tag: str = ""
    ) -> OrderIntent | None:
        """Rebalance one name toward a target weight; ``None`` if already there."""
        raise NotImplementedError

    @abstractmethod
    def rebalance(self, weights: Mapping[InstrumentId, float]) -> Sequence[OrderIntent]:
        """Move the whole book toward a weight vector, closing omitted names.

        The portfolio-mode primitive.
        """
        raise NotImplementedError

    # ---- diagnostics ---------------------------------------------------------- #

    @abstractmethod
    def log(self, message: str, /, **fields: object) -> None:
        """Structured, bar-stamped logging. Captured into the result."""
        raise NotImplementedError

    @abstractmethod
    def record(self, name: str, value: float) -> None:
        """Record a custom per-bar series, plotted alongside the equity curve."""
        raise NotImplementedError

    @abstractmethod
    def warn(self, message: str) -> None:
        """Surface a caveat into ``BacktestResult.warnings``."""
        raise NotImplementedError
