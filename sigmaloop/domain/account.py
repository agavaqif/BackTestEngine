"""Account-level state: per-bar equity points and corporate actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sigmaloop.types import (
    CorporateActionType,
    Currency,
    InstrumentId,
    Money,
    Percent,
    Price,
    Symbol,
    UtcDatetime,
)

__all__ = ["EquityPoint", "AccountState", "CorporateAction", "CashFlow"]


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One row of the equity curve — emitted on every bar (Accounting req. #1).

    Recorded AFTER fills, expiries and mark-to-market for the bar, so
    :attr:`equity` is the true end-of-bar account value.

    Stored by :class:`~sigmaloop.results.curves.EquityCurve` in columnar form;
    this row type exists for the streaming/callback path and for tests.
    """

    timestamp: UtcDatetime
    cash: Money
    positions_value: Money
    equity: Money
    #: Sum of absolute position notionals.
    gross_exposure: Money = 0.0
    #: Long notional minus short notional.
    net_exposure: Money = 0.0
    margin_used: Money = 0.0
    buying_power: Money = 0.0
    open_positions: int = 0
    realized_pnl_cum: Money = 0.0
    unrealized_pnl: Money = 0.0
    #: Fraction below the running high-water mark; <= 0.
    drawdown: Percent = 0.0
    high_water_mark: Money = 0.0

    @property
    def leverage(self) -> float:
        """``gross_exposure / equity``."""
        raise NotImplementedError


@dataclass(slots=True)
class AccountState:
    """Mutable cash-and-margin state owned by the portfolio.

    Separated from :class:`~sigmaloop.portfolio.accounting.Portfolio` so the
    margin/buying-power policy can be reasoned about (and unit-tested) apart
    from position bookkeeping.
    """

    initial_cash: Money
    cash: Money
    currency: Currency = Currency.USD
    margin_used: Money = 0.0
    #: Cash reserved against working (unfilled) orders.
    reserved_cash: Money = 0.0
    realized_pnl: Money = 0.0
    total_commission: Money = 0.0
    total_fees: Money = 0.0
    total_borrow_cost: Money = 0.0
    total_dividends: Money = 0.0
    high_water_mark: Money = 0.0

    @property
    def available_cash(self) -> Money:
        """``cash - reserved_cash`` — what a new order may consume."""
        raise NotImplementedError

    def buying_power(self, positions_value: Money, leverage: float) -> Money:
        """Cash plus margin capacity under the configured margin model."""
        raise NotImplementedError

    def can_afford(self, cost: Money, positions_value: Money, leverage: float) -> bool:
        raise NotImplementedError

    def credit(self, amount: Money, reason: str = "") -> None:
        raise NotImplementedError

    def debit(self, amount: Money, reason: str = "") -> None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CashFlow:
    """A non-trade cash movement (dividend, borrow charge, interest, fee).

    Kept distinct from :class:`~sigmaloop.domain.order.Fill` so P&L attribution
    can separate market returns from carry.
    """

    timestamp: UtcDatetime
    amount: Money
    reason: str
    instrument_id: InstrumentId | None = None


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A split, dividend or delisting applied before trading on its ex-date.

    Applied by the engine at the top of the bar so that positions, open orders
    and indicator history are all adjusted consistently.
    """

    action_type: CorporateActionType
    instrument_id: InstrumentId
    symbol: Symbol
    ex_date: date
    #: For SPLIT: shares out per share in (2.0 == 2-for-1). Else 1.0.
    ratio: float = 1.0
    #: For dividends: cash per share.
    amount: Price = 0.0
    new_symbol: Symbol | None = None
    payable_date: date | None = None
