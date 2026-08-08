"""Account-level state: per-bar equity points and corporate actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from sigmaloop.errors import ValidationError
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
from sigmaloop.utils.timeutil import ensure_utc

__all__ = ["AccountState", "CashFlow", "CorporateAction", "EquityPoint"]

#: Cash amounts differing by less than this are equal. Guards ``can_afford``
#: against refusing an order that costs exactly the buying power available, when
#: the two sides were reached by different float64 routes.
_CASH_TOLERANCE: Money = 1e-9


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

    def __post_init__(self) -> None:
        """Normalise the bar instant to tz-aware UTC.

        This row is the x-axis of the equity curve; a naive stamp among aware
        ones makes the curve unsortable and every annualised metric wrong by an
        offset that never shows up as an error.
        """
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def leverage(self) -> float:
        """``gross_exposure / equity``.

        A wiped-out account holding nothing is 0x, not undefined; one still
        holding exposure is infinitely levered, and reporting that as ``inf``
        keeps the blow-up visible instead of rounding it away.
        """
        if self.equity == 0.0:
            return 0.0 if self.gross_exposure == 0.0 else math.inf
        return self.gross_exposure / self.equity


@dataclass(slots=True)
class AccountState:
    """Mutable cash-and-margin state owned by the portfolio.

    Separated from :class:`~sigmaloop.portfolio.accounting.Portfolio` so the
    margin/buying-power policy can be reasoned about (and unit-tested) apart
    from position bookkeeping.

    :meth:`credit` and :meth:`debit` move :attr:`cash` and nothing else. The
    attribution counters below — ``realized_pnl``, the ``total_*`` buckets and
    ``high_water_mark`` — are written by the portfolio, which is the only layer
    that knows *what* a cash movement was. Classifying it here would mean
    inferring the category from the free-text ``reason``, and a ledger that
    guesses its own P&L attribution is worse than one that declines to.
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
        return self.cash - self.reserved_cash

    def buying_power(self, positions_value: Money, leverage: float) -> Money:
        """Cash plus margin capacity under the configured margin model.

        ``(cash + positions_value) * leverage - margin_used - reserved_cash``,
        floored at zero.

        The margin model enters entirely through ``margin_used``, which is why
        one expression serves all of them: a CASH account books each long at full
        notional, so ``margin_used`` cancels ``positions_value`` and the result
        collapses to spendable cash; Reg-T books half at ``leverage=2.0`` and the
        same expression yields the familiar 2x. Reserved cash comes off the top
        so two orders raised on the same bar cannot both spend the same dollar.

        Because that identity leans on ``margin_used`` being maintained, an
        unlevered account is additionally capped at :attr:`available_cash`. A
        caller that forgets to book margin then merely gets a conservative
        number, instead of being invited to spend the value of positions it is
        still holding.
        """
        equity = self.cash + positions_value
        capacity = equity * leverage - self.margin_used - self.reserved_cash
        if leverage <= 1.0:
            capacity = min(capacity, self.available_cash)
        return max(capacity, 0.0)

    def can_afford(self, cost: Money, positions_value: Money, leverage: float) -> bool:
        if cost <= 0.0:
            return True
        return cost <= self.buying_power(positions_value, leverage) + _CASH_TOLERANCE

    def credit(self, amount: Money, reason: str = "") -> None:
        self._require_non_negative(amount, "credit", "debit", reason)
        self.cash += amount

    def debit(self, amount: Money, reason: str = "") -> None:
        """Remove cash unconditionally.

        Deliberately does not refuse to go negative. Whether an over-capital
        order is stopped or merely flagged is ``AccountingConfig.on_capital_breach``'s
        call, made pre-trade by :meth:`can_afford` and the risk checks; balking
        here would leave the ledger disagreeing with a fill that already happened,
        and a silently unbooked debit is worse than a visible overdraft.
        """
        self._require_non_negative(amount, "debit", "credit", reason)
        self.cash -= amount

    @staticmethod
    def _require_non_negative(amount: Money, action: str, inverse: str, reason: str) -> None:
        """Keep direction in the method name, never in the sign of the argument.

        Non-finite amounts are refused for the same reason: one ``inf`` from a
        bad price makes ``cash`` infinite, and the next movement the other way
        turns it into ``nan``, from which no later arithmetic recovers.
        """
        if not math.isfinite(amount) or amount < 0.0:
            raise ValidationError(
                f"AccountState.{action} takes a non-negative, finite amount; "
                f"use {inverse}() to move cash the other way.",
                amount=amount,
                reason=reason or None,
            )


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


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
