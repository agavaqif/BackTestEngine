"""Execution models — when and whether a working order fills.

This is the lookahead firewall (Execution requirement 2). A signal computed
from bar *t*'s close cannot transact at bar *t*'s open, because that price was
already in the past when the signal existed. The default
:class:`NextBarOpenExecutionModel` enforces that by construction: an order
submitted during bar *t* becomes eligible only at bar *t+1*.

The model answers two questions per working order per bar:

1. *Is it eligible yet?* — :meth:`ExecutionModel.is_eligible`
2. *Does the bar's price action trigger it, and at what price?* —
   :meth:`ExecutionModel.try_fill`

Limit/stop triggering uses the bar's high/low, which is the standard OHLC
approximation. Its known weakness — intrabar path is unknown, so a bar that
touches both a stop and a target is ambiguous — is resolved pessimistically:
the stop is assumed to trigger first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from sigmaloop.domain.bar import Bar, OptionQuote
from sigmaloop.domain.instrument import Instrument
from sigmaloop.domain.order import Order
from sigmaloop.types import (
    ExecutionTiming,
    OrderSide,
    OrderType,
    Price,
    Quantity,
    TimeInForce,
    UtcDatetime,
)

__all__ = [
    "FillDecision",
    "ExecutionContext",
    "ExecutionModel",
    "NextBarOpenExecutionModel",
    "NextBarCloseExecutionModel",
    "SameBarCloseExecutionModel",
]

#: Prices differing by less than this are the same price, matching
#: :mod:`sigmaloop.domain.order`. A limit exactly equal to the bar's low must
#: trigger; float64 error on the way in must not decide otherwise.
_PRICE_TOLERANCE: Price = 1e-9

#: Order types that transact at whatever the bar offers, with no price test.
_MARKET_TYPES = frozenset({OrderType.MARKET, OrderType.MARKET_ON_OPEN, OrderType.MARKET_ON_CLOSE})

_NO_FILL_REASONS = {
    OrderType.LIMIT: "limit not reached on this bar",
    OrderType.STOP: "stop not triggered on this bar",
    OrderType.STOP_LIMIT: "stop not triggered on this bar",
}


@dataclass(frozen=True, slots=True)
class FillDecision:
    """Outcome of evaluating one order against one bar."""

    should_fill: bool
    quantity: Quantity = 0.0
    #: Pre-slippage price; the broker applies slippage and commission after.
    reference_price: Price = 0.0
    is_partial: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Everything needed to evaluate one order at one timestamp."""

    order: Order
    instrument: Instrument
    timestamp: UtcDatetime
    bar: Bar | None
    option_quote: OptionQuote | None = None
    is_session_close: bool = False


def _no_fill(reason: str) -> FillDecision:
    return FillDecision(should_fill=False, reason=reason)


def _triggered_price(
    order: Order, bar: Bar, anchor: Price, *, allow_gap_improvement: bool
) -> Price | None:
    """Price at which ``bar`` fills ``order``, or ``None`` if it does not.

    ``anchor`` is the point in the bar the model transacts at — the open for
    next-bar-open, the close for the close-anchored models. The level rules are
    the ones in DESIGN §6.1:

    ===========  ==================  ======================
    order        triggers when       fills at
    ===========  ==================  ======================
    buy limit    ``low <= limit``    ``min(anchor, limit)``
    sell limit   ``high >= limit``   ``max(anchor, limit)``
    buy stop     ``high >= stop``    ``max(anchor, stop)``
    sell stop    ``low <= stop``     ``min(anchor, stop)``
    ===========  ==================  ======================

    Gaps are where backtests manufacture money. A buy stop whose bar opens
    *through* the trigger fills at that open, not at the trigger, because the
    trigger price was never available. ``allow_gap_improvement`` reinstates the
    optimistic convention for comparison against published results that use it;
    it is off by default and the engine warns when it is on.
    """
    is_buy = order.side is OrderSide.BUY
    order_type = order.order_type

    if order_type in _MARKET_TYPES:
        return anchor

    if order_type is OrderType.LIMIT:
        limit = order.limit_price
        if limit is None:  # pragma: no cover - Order.__post_init__ forbids it
            return None
        if is_buy:
            return min(anchor, limit) if bar.low <= limit + _PRICE_TOLERANCE else None
        return max(anchor, limit) if bar.high >= limit - _PRICE_TOLERANCE else None

    stop = order.stop_price
    if stop is None:  # pragma: no cover - Order.__post_init__ forbids it
        return None
    if is_buy:
        if bar.high < stop - _PRICE_TOLERANCE:
            return None
        triggered = stop if allow_gap_improvement else max(anchor, stop)
    else:
        if bar.low > stop + _PRICE_TOLERANCE:
            return None
        triggered = stop if allow_gap_improvement else min(anchor, stop)

    if order_type is OrderType.STOP:
        return triggered
    # STOP_LIMIT: the trigger turns it into a limit order, and a triggered price
    # through the limit is not fillable. The bar may well have traded back
    # inside the limit afterwards, but the intrabar path is unknown and assuming
    # it did is precisely the free money the gap rule above exists to refuse.
    return triggered if order.accepts_price(triggered) else None


class ExecutionModel(ABC):
    """Decides eligibility and fill outcome for working orders."""

    name: ClassVar[str] = "abstract"

    @property
    @abstractmethod
    def timing(self) -> ExecutionTiming:
        raise NotImplementedError

    @property
    def introduces_lookahead(self) -> bool:
        """True for models that let bar *t* signals fill at bar *t* prices.

        The engine emits a prominent warning into
        :attr:`~sigmaloop.results.result.BacktestResult.warnings` when this is
        True, so a lookahead-tainted result can never be mistaken for a clean
        one.
        """
        return self.timing is ExecutionTiming.SAME_BAR_CLOSE

    @abstractmethod
    def is_eligible(self, context: ExecutionContext) -> bool:
        """True if the order may be considered for filling at this bar."""
        raise NotImplementedError

    @abstractmethod
    def try_fill(self, context: ExecutionContext) -> FillDecision:
        """Evaluate the order against the bar."""
        raise NotImplementedError

    def should_expire(self, context: ExecutionContext) -> bool:
        """Apply time-in-force: DAY expires at session close, GTD at its date.

        Measured from :attr:`Order.activated_at`, not from submission. Under the
        default next-bar-open timing a DAY order raised on bar *t*'s close is an
        instruction for session *t+1*; expiring it against its submission date
        would kill it before the session it was written for ever opened.

        ``IOC`` and ``FOK`` are not handled here — they hinge on how much of the
        order the bar actually filled, which only the broker knows.
        """
        order = context.order
        if not order.is_open:
            return False
        if order.expires_at is not None and context.timestamp >= order.expires_at:
            return True
        if order.time_in_force is not TimeInForce.DAY:
            return False
        activated = order.activated_at
        if activated is None:
            # Never eligible for a bar, so it has not had its session yet.
            return False
        # The UTC date stands in for the session id: a US equity session closes
        # at 20:00/21:00 UTC on its own date, so the two agree. A calendar-aware
        # session key can replace this when intraday non-US venues arrive.
        return context.timestamp.date() > activated.date()


class NextBarOpenExecutionModel(ExecutionModel):
    """DEFAULT. Market orders fill at the next bar's open.

    Limit and stop orders become eligible at the next bar and are then tested
    against each subsequent bar's range until filled, cancelled or expired:

    * Buy limit  fills if ``low <= limit``, at ``min(open, limit)``.
    * Sell limit fills if ``high >= limit``, at ``max(open, limit)``.
    * Buy stop   triggers if ``high >= stop``, at ``max(open, stop)``.
    * Sell stop  triggers if ``low <= stop``, at ``min(open, stop)``.

    Gap handling matters: when a bar opens through the level, the fill uses the
    open, not the level. Assuming the limit price on a gap is the most common
    way backtests manufacture free money.
    """

    name: ClassVar[str] = "next_bar_open"

    def __init__(self, allow_gap_improvement: bool = False) -> None:
        """``allow_gap_improvement`` credits a gapped stop with its trigger price.

        Off by default, which is the table above. Turning it on reproduces the
        common convention where a stop that gapped through is still filled at
        the level — money the market never offered — and exists only so a run
        can be compared against published results that assume it.
        """
        self._allow_gap_improvement = allow_gap_improvement

    @property
    def allow_gap_improvement(self) -> bool:
        return self._allow_gap_improvement

    @property
    def timing(self) -> ExecutionTiming:
        return ExecutionTiming.NEXT_BAR_OPEN

    def is_eligible(self, context: ExecutionContext) -> bool:
        """Strictly after the submitting bar — the firewall, in one comparison."""
        return context.order.is_open and context.timestamp > context.order.submitted_at

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        bar = context.bar
        order = context.order
        if bar is None:
            return _no_fill("no bar for this instrument at this step")
        if order.order_type is OrderType.MARKET_ON_CLOSE and not context.is_session_close:
            return _no_fill("market-on-close waits for the session close")
        anchor = bar.close if order.order_type is OrderType.MARKET_ON_CLOSE else bar.open
        price = _triggered_price(
            order, bar, anchor, allow_gap_improvement=self._allow_gap_improvement
        )
        if price is None:
            return _no_fill(_NO_FILL_REASONS.get(order.order_type, "not marketable"))
        return FillDecision(
            should_fill=True, quantity=order.remaining_quantity, reference_price=price
        )


class NextBarCloseExecutionModel(ExecutionModel):
    """Fills at the next bar's close — models a VWAP/TWAP-style working order."""

    name: ClassVar[str] = "next_bar_close"

    @property
    def timing(self) -> ExecutionTiming:
        return ExecutionTiming.NEXT_BAR_CLOSE

    def is_eligible(self, context: ExecutionContext) -> bool:
        return context.order.is_open and context.timestamp > context.order.submitted_at

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        return _close_anchored_fill(context)


class SameBarCloseExecutionModel(ExecutionModel):
    """Fills at the close of the bar that produced the signal.

    LOOKAHEAD-PRONE and offered only for comparison against published results
    that use this convention. :attr:`introduces_lookahead` is True.
    """

    name: ClassVar[str] = "same_bar_close"

    @property
    def timing(self) -> ExecutionTiming:
        return ExecutionTiming.SAME_BAR_CLOSE

    def is_eligible(self, context: ExecutionContext) -> bool:
        """Includes the submitting bar — which is exactly the lookahead."""
        return context.order.is_open and context.timestamp >= context.order.submitted_at

    def try_fill(self, context: ExecutionContext) -> FillDecision:
        return _close_anchored_fill(context)


def _close_anchored_fill(context: ExecutionContext) -> FillDecision:
    """Shared body of the two close-anchored models.

    A working order that finishes at the close still has to honour its limit, so
    a triggered buy limit prints at ``min(close, limit)`` rather than at the
    close outright — the same rule as the open-anchored model, with the anchor
    moved.
    """
    bar = context.bar
    order = context.order
    if bar is None:
        return _no_fill("no bar for this instrument at this step")
    price = _triggered_price(order, bar, bar.close, allow_gap_improvement=False)
    if price is None:
        return _no_fill(_NO_FILL_REASONS.get(order.order_type, "not marketable"))
    return FillDecision(should_fill=True, quantity=order.remaining_quantity, reference_price=price)
