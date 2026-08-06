"""Portfolio ledger — cash, positions, equity and realised trades.

Satisfies the Accounting requirements:

1. Cash, positions and total equity are recomputed and recorded on **every**
   bar (:meth:`Portfolio.mark_to_market` -> :class:`EquityPoint`).
2. Orders exceeding available capital are rejected or flagged, per
   :attr:`~sigmaloop.engine.config.AccountingConfig.on_capital_breach`.
3. Long and short equity and option positions are supported, with signed
   quantities and a margin model for shorts.
4. Sizing is delegated to :mod:`sigmaloop.portfolio.sizing`, never assumed.

Invariant enforced after every mutation::

    equity == cash + sum(position.market_value for open positions)

A breach raises :class:`~sigmaloop.errors.AccountingError` immediately rather
than propagating silently into the metrics — an accounting drift found at the
end of a run is unattributable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from sigmaloop.domain.account import AccountState, CashFlow, CorporateAction, EquityPoint
from sigmaloop.domain.bar import MarketSnapshot
from sigmaloop.domain.instrument import Instrument, InstrumentRegistry
from sigmaloop.domain.order import Fill, Order, Rejection
from sigmaloop.domain.position import Position, Trade
from sigmaloop.execution.expiry import ExpiryOutcome
from sigmaloop.types import (
    InstrumentId,
    Money,
    Price,
    Quantity,
    TradeCloseReason,
    UtcDatetime,
)

__all__ = ["PortfolioView", "Portfolio", "LedgerPortfolio"]


class PortfolioView(ABC):
    """Read-only portfolio surface handed to strategies.

    Strategies receive this, never the mutable :class:`Portfolio`, so user code
    cannot corrupt the ledger — it can only observe it and emit intents.
    """

    @property
    @abstractmethod
    def cash(self) -> Money:
        raise NotImplementedError

    @property
    @abstractmethod
    def equity(self) -> Money:
        """Cash plus mark-to-market value of all open positions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def buying_power(self) -> Money:
        raise NotImplementedError

    @property
    @abstractmethod
    def positions_value(self) -> Money:
        raise NotImplementedError

    @property
    @abstractmethod
    def gross_exposure(self) -> Money:
        raise NotImplementedError

    @property
    @abstractmethod
    def net_exposure(self) -> Money:
        raise NotImplementedError

    @abstractmethod
    def position(self, instrument_id: InstrumentId) -> Position | None:
        raise NotImplementedError

    @abstractmethod
    def open_positions(self) -> Sequence[Position]:
        raise NotImplementedError

    @abstractmethod
    def has_position(self, instrument_id: InstrumentId) -> bool:
        raise NotImplementedError

    @abstractmethod
    def quantity(self, instrument_id: InstrumentId) -> Quantity:
        """Signed held quantity; 0.0 when flat."""
        raise NotImplementedError

    @abstractmethod
    def weight(self, instrument_id: InstrumentId) -> float:
        """Position market value as a fraction of equity."""
        raise NotImplementedError

    @abstractmethod
    def closed_trades(self) -> Sequence[Trade]:
        raise NotImplementedError

    @property
    @abstractmethod
    def realized_pnl(self) -> Money:
        raise NotImplementedError

    @property
    @abstractmethod
    def unrealized_pnl(self) -> Money:
        raise NotImplementedError


class Portfolio(PortfolioView):
    """Mutable ledger. Owned by the engine; the only writer of account state."""

    # ---- capital checks ------------------------------------------------------ #

    @abstractmethod
    def can_afford(self, order: Order, price: Price) -> Rejection | None:
        """Pre-trade capital check.

        Returns ``None`` when the order fits, otherwise the
        :class:`~sigmaloop.domain.order.Rejection` describing the shortfall
        (with ``required`` and ``available`` populated). The caller decides
        whether that rejects the order or merely flags it.
        """
        raise NotImplementedError

    @abstractmethod
    def reserve(self, order: Order, estimated_cost: Money) -> None:
        """Earmark cash for a working order so two orders cannot spend it twice."""
        raise NotImplementedError

    @abstractmethod
    def release(self, order: Order) -> None:
        """Release a reservation when an order fills, cancels or expires."""
        raise NotImplementedError

    # ---- mutation ------------------------------------------------------------ #

    @abstractmethod
    def apply_fill(self, fill: Fill, at: UtcDatetime) -> Trade | None:
        """Apply a fill; returns the closed :class:`Trade` if one completed."""
        raise NotImplementedError

    @abstractmethod
    def apply_expiry(self, outcome: ExpiryOutcome, at: UtcDatetime) -> Sequence[Trade]:
        """Settle an option expiry/assignment, including any share delivery."""
        raise NotImplementedError

    @abstractmethod
    def apply_corporate_action(self, action: CorporateAction, at: UtcDatetime) -> None:
        """Adjust positions and cost basis for a split or dividend."""
        raise NotImplementedError

    @abstractmethod
    def apply_cash_flow(self, flow: CashFlow) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_to_market(self, snapshot: MarketSnapshot) -> EquityPoint:
        """Revalue every position and emit this bar's equity point.

        Positions with no bar in the snapshot keep their previous mark; the
        count of stale marks is reported so a run built on gappy data is
        visible rather than silently wrong.
        """
        raise NotImplementedError

    @abstractmethod
    def liquidate_all(
        self, snapshot: MarketSnapshot, reason: TradeCloseReason
    ) -> Sequence[Trade]:
        """Flatten everything at the final bar so the trade log has no open legs."""
        raise NotImplementedError

    # ---- state --------------------------------------------------------------- #

    @property
    @abstractmethod
    def account(self) -> AccountState:
        raise NotImplementedError

    @abstractmethod
    def validate_invariants(self) -> None:
        """Assert the equity identity; raises ``AccountingError`` on breach."""
        raise NotImplementedError


class LedgerPortfolio(Portfolio):
    """Reference implementation.

    Positions live in a plain dict keyed by :class:`InstrumentId`; closed trades
    accumulate in a list. Both are append-mostly, which keeps the per-bar cost
    proportional to the number of *touched* instruments rather than the number
    held — important in portfolio mode, where the book can hold thousands of
    names but only a handful trade on any given bar.
    """

    __slots__ = (
        "_account",
        "_positions",
        "_closed_trades",
        "_registry",
        "_reservations",
        "_cash_flows",
        "_config",
        "_last_equity",
        "_high_water_mark",
        "_trade_seq",
    )

    def __init__(
        self,
        initial_cash: Money,
        registry: InstrumentRegistry,
        config: object | None = None,
    ) -> None:
        raise NotImplementedError

    # PortfolioView -------------------------------------------------------- #

    @property
    def cash(self) -> Money:
        raise NotImplementedError

    @property
    def equity(self) -> Money:
        raise NotImplementedError

    @property
    def buying_power(self) -> Money:
        raise NotImplementedError

    @property
    def positions_value(self) -> Money:
        raise NotImplementedError

    @property
    def gross_exposure(self) -> Money:
        raise NotImplementedError

    @property
    def net_exposure(self) -> Money:
        raise NotImplementedError

    def position(self, instrument_id: InstrumentId) -> Position | None:
        raise NotImplementedError

    def open_positions(self) -> Sequence[Position]:
        raise NotImplementedError

    def has_position(self, instrument_id: InstrumentId) -> bool:
        raise NotImplementedError

    def quantity(self, instrument_id: InstrumentId) -> Quantity:
        raise NotImplementedError

    def weight(self, instrument_id: InstrumentId) -> float:
        raise NotImplementedError

    def closed_trades(self) -> Sequence[Trade]:
        raise NotImplementedError

    @property
    def realized_pnl(self) -> Money:
        raise NotImplementedError

    @property
    def unrealized_pnl(self) -> Money:
        raise NotImplementedError

    # Portfolio ------------------------------------------------------------- #

    def can_afford(self, order: Order, price: Price) -> Rejection | None:
        raise NotImplementedError

    def reserve(self, order: Order, estimated_cost: Money) -> None:
        raise NotImplementedError

    def release(self, order: Order) -> None:
        raise NotImplementedError

    def apply_fill(self, fill: Fill, at: UtcDatetime) -> Trade | None:
        raise NotImplementedError

    def apply_expiry(self, outcome: ExpiryOutcome, at: UtcDatetime) -> Sequence[Trade]:
        raise NotImplementedError

    def apply_corporate_action(self, action: CorporateAction, at: UtcDatetime) -> None:
        raise NotImplementedError

    def apply_cash_flow(self, flow: CashFlow) -> None:
        raise NotImplementedError

    def mark_to_market(self, snapshot: MarketSnapshot) -> EquityPoint:
        raise NotImplementedError

    def liquidate_all(
        self, snapshot: MarketSnapshot, reason: TradeCloseReason
    ) -> Sequence[Trade]:
        raise NotImplementedError

    @property
    def account(self) -> AccountState:
        raise NotImplementedError

    def validate_invariants(self) -> None:
        raise NotImplementedError

    # internals -------------------------------------------------------------- #

    def _get_or_create_position(self, instrument: Instrument) -> Position:
        raise NotImplementedError

    def _close_trade(
        self,
        position: Position,
        exit_price: Price,
        exit_quantity: Quantity,
        at: UtcDatetime,
        reason: TradeCloseReason,
    ) -> Trade:
        """Build the trade-log row, choosing ``Trade`` or ``OptionTrade`` by
        asset class so options rows carry strike/expiry/greeks context."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[Position]:
        raise NotImplementedError
