"""Open positions, tax lots and closed round-trip trades."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import Fill
from sigmaloop.types import (
    AssetClass,
    FillId,
    InstrumentId,
    Money,
    OptionRight,
    Percent,
    PositionSide,
    Price,
    Quantity,
    Symbol,
    TradeCloseReason,
    TradeId,
    UtcDatetime,
)

__all__ = ["Lot", "Position", "Trade", "OptionTrade"]


@dataclass(slots=True)
class Lot:
    """One acquisition tranche within a position.

    Lots let the accounting layer compute realised P&L under an explicit
    matching policy (FIFO by default, configurable via
    ``AccountingConfig.lot_matching``) instead of only average cost, which
    matters for partial exits and for options roll analysis.
    """

    quantity: Quantity
    price: Price
    opened_at: UtcDatetime
    fill_id: FillId
    commission: Money = 0.0


@dataclass(slots=True)
class Position:
    """Live holding in one instrument. Mutated in place by the portfolio.

    :attr:`quantity` is signed: positive long, negative short, zero means the
    position object is retained only until end-of-bar cleanup.
    """

    instrument: Instrument
    quantity: Quantity = 0.0
    avg_price: Price = 0.0
    realized_pnl: Money = 0.0
    commission_paid: Money = 0.0
    fees_paid: Money = 0.0
    borrow_cost_paid: Money = 0.0
    dividends_received: Money = 0.0
    mark_price: Price = 0.0
    last_update: UtcDatetime | None = None
    opened_at: UtcDatetime | None = None
    lots: deque[Lot] = field(default_factory=deque)
    #: Running extremes since entry, used for MAE/MFE on the closed trade.
    max_favorable_price: Price = 0.0
    max_adverse_price: Price = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def instrument_id(self) -> InstrumentId:
        raise NotImplementedError

    @property
    def side(self) -> PositionSide:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    @property
    def is_short(self) -> bool:
        raise NotImplementedError

    @property
    def cost_basis(self) -> Money:
        """``|quantity| * avg_price * multiplier``."""
        raise NotImplementedError

    @property
    def market_value(self) -> Money:
        """Signed mark-to-market value: ``quantity * mark_price * multiplier``."""
        raise NotImplementedError

    @property
    def unrealized_pnl(self) -> Money:
        raise NotImplementedError

    @property
    def unrealized_pnl_pct(self) -> Percent:
        raise NotImplementedError

    @property
    def total_pnl(self) -> Money:
        """Realised + unrealised, net of commissions, fees and borrow."""
        raise NotImplementedError

    @property
    def exposure(self) -> Money:
        """Absolute notional at risk — always positive, shorts included."""
        raise NotImplementedError

    def apply_fill(self, fill: Fill) -> Money:
        """Fold a fill in and return the realised P&L it produced.

        Handles the four cases: open, increase, reduce (realises), and
        flip-through-zero (realises the whole old side, opens the new one).
        """
        raise NotImplementedError

    def mark(self, price: Price, at: UtcDatetime) -> None:
        """Update the mark price and the MAE/MFE extremes."""
        raise NotImplementedError

    def accrue_borrow(self, at: UtcDatetime, bar_fraction_of_year: float) -> Money:
        """Charge short-borrow cost for one bar; returns the amount charged."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip — the row type of the trade log.

    Produced by the portfolio when a position's exposure returns to (or crosses)
    zero. Entry figures are the lot-weighted averages of the opening fills.
    """

    trade_id: TradeId
    instrument_id: InstrumentId
    symbol: Symbol
    asset_class: AssetClass
    direction: PositionSide
    quantity: Quantity
    entry_time: UtcDatetime
    entry_price: Price
    exit_time: UtcDatetime
    exit_price: Price
    gross_pnl: Money
    commission: Money
    fees: Money
    net_pnl: Money
    return_pct: Percent
    close_reason: TradeCloseReason
    #: Max adverse / favourable excursion in cash while the trade was open.
    mae: Money = 0.0
    mfe: Money = 0.0
    bars_held: int = 0
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def holding_period(self) -> timedelta:
        raise NotImplementedError

    @property
    def is_winner(self) -> bool:
        raise NotImplementedError

    @property
    def r_multiple(self) -> float | None:
        """Net P&L divided by initial risk, when a stop distance was recorded."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OptionTrade(Trade):
    """Round trip on an option contract — the options trade log row.

    Adds contract identity and the entry/exit market context an options
    post-mortem needs (was it delta, theta or the underlying that paid?).
    """

    underlying_symbol: Symbol = Symbol("")
    right: OptionRight = OptionRight.CALL
    strike: Price = 0.0
    expiry: date | None = None
    multiplier: float = 100.0
    dte_at_entry: int = 0
    dte_at_exit: int = 0
    delta_at_entry: float | None = None
    iv_at_entry: float | None = None
    iv_at_exit: float | None = None
    underlying_price_at_entry: Price | None = None
    underlying_price_at_exit: Price | None = None
    was_assigned: bool = False
    was_exercised: bool = False
    expired_worthless: bool = False

    @property
    def premium_collected(self) -> Money:
        """Positive for short-premium trades, negative for long."""
        raise NotImplementedError

    @property
    def underlying_return_pct(self) -> Percent | None:
        raise NotImplementedError
