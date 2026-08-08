"""Order lifecycle objects: intents, orders, fills and rejections.

Two-stage design
----------------
Strategies emit an :class:`OrderIntent` — *what* they want, expressed in
strategy terms ("go long 2% of equity"). The engine then runs it through the
:class:`~sigmaloop.portfolio.sizing.PositionSizer` and
:class:`~sigmaloop.portfolio.risk.RiskManager` to produce a concrete
:class:`Order` with a resolved quantity.

This split is what makes the sizing requirement (fixed qty / notional / % of
equity / custom) a first-class, swappable concern instead of something baked
into every strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sigmaloop.errors import ExecutionError, ValidationError
from sigmaloop.types import (
    FillId,
    FillLiquidity,
    InstrumentId,
    IntentId,
    Money,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    Price,
    Quantity,
    RejectReason,
    SizingMode,
    TimeInForce,
    UtcDatetime,
)
from sigmaloop.utils.timeutil import ensure_utc

__all__ = [
    "BracketSpec",
    "Fill",
    "Order",
    "OrderIntent",
    "Rejection",
    "SizingRequest",
]

#: Quantities differing by less than this are the same quantity. Summing fills
#: in float64 leaves dust in the 1e-13 range on realistic sizes, which would
#: otherwise leave a fully worked order stuck in ``PARTIALLY_FILLED`` forever.
_QUANTITY_TOLERANCE: Quantity = 1e-9

#: Prices differing by less than this are the same price. Absolute, because
#: float64 error on any tradeable price is many orders of magnitude smaller.
_PRICE_TOLERANCE: Price = 1e-9

_NEEDS_LIMIT_PRICE: frozenset[OrderType] = frozenset({OrderType.LIMIT, OrderType.STOP_LIMIT})
_NEEDS_STOP_PRICE: frozenset[OrderType] = frozenset({OrderType.STOP, OrderType.STOP_LIMIT})


def _validate_price_fields(
    order_type: OrderType,
    limit_price: Price | None,
    stop_price: Price | None,
    time_in_force: TimeInForce,
    expires_at: UtcDatetime | None,
    raised_at: UtcDatetime,
    /,
    **context: Any,
) -> None:
    """Shared check for the intent and the order it resolves into.

    Both ends must agree: validating only the intent would let a sizer or a
    plugin hand the broker a ``LIMIT`` order with no limit, which the execution
    model would then work as an unbounded market order — the strategy's price
    ceiling silently gone.
    """
    if order_type in _NEEDS_LIMIT_PRICE and limit_price is None:
        raise ValidationError(f"{order_type.value!r} orders require a limit_price.", **context)
    # A stop_price on a MARKET order is deliberately allowed: RISK_PERCENT sizing
    # measures the risk budget against it, so the level is meaningful even when
    # it is not a trigger.
    if order_type in _NEEDS_STOP_PRICE and stop_price is None:
        raise ValidationError(f"{order_type.value!r} orders require a stop_price.", **context)
    # A limit_price has no such second reading. Accepting a stray one would arm
    # Order.accepts_price against an order that is not a limit order at all, so
    # a mistyped MARKET would refuse every fill the market offered it.
    if limit_price is not None and order_type not in _NEEDS_LIMIT_PRICE:
        raise ValidationError(
            f"{order_type.value!r} orders take no limit_price; use "
            f"OrderType.LIMIT or OrderType.STOP_LIMIT to cap the fill price.",
            **context,
            limit_price=limit_price,
        )
    for name, value in (("limit_price", limit_price), ("stop_price", stop_price)):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValidationError(
                f"{name} must be a positive, finite price.", **context, **{name: value}
            )
    if time_in_force is TimeInForce.GTD and expires_at is None:
        raise ValidationError(
            "TimeInForce.GTD requires expires_at; without it the order would "
            "never expire and would behave as GTC.",
            **context,
        )
    if expires_at is not None and ensure_utc(expires_at) <= ensure_utc(raised_at):
        raise ValidationError(
            "expires_at is at or before the moment the order was raised, so it "
            "would expire on the bar that created it and could never fill.",
            **context,
            expires_at=expires_at,
            raised_at=raised_at,
        )


@dataclass(frozen=True, slots=True)
class SizingRequest:
    """Declarative position size, resolved later by a ``PositionSizer``.

    Exactly one of the value fields is meaningful, selected by :attr:`mode`:

    ===========================  =======================================
    mode                         meaning of ``value``
    ===========================  =======================================
    ``FIXED_QUANTITY``           shares / contracts
    ``FIXED_NOTIONAL``           cash notional in base currency
    ``PERCENT_EQUITY``           fraction of total equity (0.02 == 2%)
    ``RISK_PERCENT``             fraction of equity risked to ``stop_price``
    ``TARGET_WEIGHT``            desired post-trade portfolio weight
    ``CUSTOM``                   ignored; ``sizer_name`` decides
    ===========================  =======================================
    """

    mode: SizingMode
    value: float
    sizer_name: str | None = None
    max_quantity: Quantity | None = None
    allow_fractional: bool = False


@dataclass(frozen=True, slots=True)
class BracketSpec:
    """Protective child orders attached to an entry.

    Levels may be absolute prices or offsets from the realised entry fill; the
    broker resolves offsets once the parent fills. Stops are evaluated against
    each subsequent bar's high/low by the execution model.
    """

    stop_loss_price: Price | None = None
    stop_loss_pct: float | None = None
    take_profit_price: Price | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None


@dataclass(slots=True)
class OrderIntent:
    """A strategy's unsized trading request for one instrument.

    Created via the :class:`~sigmaloop.strategy.api.StrategyAPI` helpers
    (``ctx.buy(...)``, ``ctx.close(...)``) rather than instantiated directly.
    """

    intent_id: IntentId
    instrument_id: InstrumentId
    side: OrderSide
    sizing: SizingRequest
    created_at: UtcDatetime
    order_type: OrderType = OrderType.MARKET
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    expires_at: UtcDatetime | None = None
    bracket: BracketSpec | None = None
    reduce_only: bool = False
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate price fields against ``order_type`` and normalise times to UTC."""
        _validate_price_fields(
            self.order_type,
            self.limit_price,
            self.stop_price,
            self.time_in_force,
            self.expires_at,
            self.created_at,
            intent_id=self.intent_id,
            instrument_id=self.instrument_id,
        )
        self.created_at = ensure_utc(self.created_at)
        if self.expires_at is not None:
            self.expires_at = ensure_utc(self.expires_at)


@dataclass(frozen=True, slots=True)
class Rejection:
    """Immutable record of why an order was refused."""

    reason: RejectReason
    message: str
    timestamp: UtcDatetime
    #: Populated for capital breaches so the summary can quantify the shortfall.
    required: Money | None = None
    available: Money | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))


@dataclass(slots=True)
class Order:
    """A sized, broker-visible order. Mutable: status advances as it is worked.

    Lifecycle
    ---------
    ``PENDING_NEW`` -> ``ACCEPTED`` -> (``PARTIALLY_FILLED``) -> ``FILLED``
    with ``REJECTED`` / ``CANCELLED`` / ``EXPIRED`` as terminal alternatives.

    ``submitted_at`` is when the strategy raised it; ``activated_at`` is the bar
    at which the execution model made it eligible to fill. Under the default
    next-bar-open model those differ by exactly one bar, and the gap is what
    guarantees no lookahead.
    """

    order_id: OrderId
    instrument_id: InstrumentId
    side: OrderSide
    quantity: Quantity
    order_type: OrderType
    submitted_at: UtcDatetime
    intent_id: IntentId | None = None
    parent_order_id: OrderId | None = None
    limit_price: Price | None = None
    stop_price: Price | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    expires_at: UtcDatetime | None = None
    status: OrderStatus = OrderStatus.PENDING_NEW
    activated_at: UtcDatetime | None = None
    filled_quantity: Quantity = 0.0
    avg_fill_price: Price = 0.0
    commission_paid: Money = 0.0
    rejection: Rejection | None = None
    reduce_only: bool = False
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Re-run the intent's price checks and normalise times to UTC.

        The order is what the broker actually works, and it can be built without
        an intent (bracket children, end-of-run liquidation), so it cannot
        inherit :class:`OrderIntent`'s validation by assuming one ran.
        """
        _validate_price_fields(
            self.order_type,
            self.limit_price,
            self.stop_price,
            self.time_in_force,
            self.expires_at,
            self.submitted_at,
            order_id=self.order_id,
            instrument_id=self.instrument_id,
        )
        self.submitted_at = ensure_utc(self.submitted_at)
        if self.activated_at is not None:
            self.activated_at = ensure_utc(self.activated_at)
        if self.expires_at is not None:
            self.expires_at = ensure_utc(self.expires_at)

    @property
    def remaining_quantity(self) -> Quantity:
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def signed_quantity(self) -> Quantity:
        """``+quantity`` for BUY, ``-quantity`` for SELL."""
        return self.side.sign * self.quantity

    def accepts_price(self, price: Price) -> bool:
        """Whether ``price`` honours the limit: at it or better, never through it.

        Buy at or below the limit, sell at or above it; orders without one accept
        anything. This is the marketability test the execution model applies
        before it prices a bar, and :meth:`apply_fill` re-applies it as the
        backstop — a limit ignored on either side of that seam is a fill the
        market never offered.

        A *stop* price is not checked: a triggered stop is a market order, and
        filling it worse than the trigger is slippage, which is real.
        """
        if self.limit_price is None:
            return True
        if self.side is OrderSide.BUY:
            return price <= self.limit_price + _PRICE_TOLERANCE
        return price >= self.limit_price - _PRICE_TOLERANCE

    def apply_fill(self, fill: Fill) -> None:
        """Fold a fill in: advance status, update VWAP and filled quantity.

        ``commission_paid`` accumulates commission only, as its name says;
        exchange fees stay on the :class:`Fill` and are aggregated onto the
        position and the closed :class:`~sigmaloop.domain.position.Trade`, which
        report the two separately.
        """
        self._require_workable("fill")
        if fill.order_id != self.order_id:
            raise ExecutionError(
                "Fill belongs to a different order.",
                order_id=self.order_id,
                fill_order_id=fill.order_id,
                fill_id=fill.fill_id,
            )
        if fill.instrument_id != self.instrument_id:
            raise ExecutionError(
                "Fill is for a different instrument than the order it claims; "
                "booking it would price one symbol's execution off another's tape.",
                order_id=self.order_id,
                instrument_id=self.instrument_id,
                fill_instrument_id=fill.instrument_id,
                fill_id=fill.fill_id,
            )
        if fill.side is not self.side:
            raise ExecutionError(
                "Fill side contradicts the order side.",
                order_id=self.order_id,
                order_side=self.side.value,
                fill_side=fill.side.value,
            )
        if fill.quantity <= 0.0:
            raise ExecutionError(
                "Fill quantity must be positive.",
                order_id=self.order_id,
                fill_quantity=fill.quantity,
            )
        if fill.quantity > self.remaining_quantity + _QUANTITY_TOLERANCE:
            raise ExecutionError(
                "Fill exceeds the order's remaining quantity; the broker would be "
                "handing the strategy size it never asked for.",
                order_id=self.order_id,
                fill_quantity=fill.quantity,
                remaining=self.remaining_quantity,
            )
        if not self.accepts_price(fill.price):
            raise ExecutionError(
                "Fill price is through the order's limit. A limit order may fill "
                "at its limit or better, never worse — otherwise the run books "
                "prices the simulated market never offered.",
                order_id=self.order_id,
                side=self.side.value,
                limit_price=self.limit_price,
                fill_price=fill.price,
            )

        filled = self.filled_quantity + fill.quantity
        # Volume-weighted, so a partially filled order always reports the price
        # it actually achieved rather than the price of its last slice.
        self.avg_fill_price = (
            self.avg_fill_price * self.filled_quantity + fill.price * fill.quantity
        ) / filled
        self.commission_paid += fill.commission
        if filled >= self.quantity - _QUANTITY_TOLERANCE:
            # Snap to the order's own size. Three 0.1 fills against 0.3 sum to
            # 0.30000000000000004, and a fill report that claims more shares
            # than were ordered is a reconciliation break downstream.
            self.filled_quantity = self.quantity
            self.status = OrderStatus.FILLED
        else:
            self.filled_quantity = filled
            self.status = OrderStatus.PARTIALLY_FILLED

    def reject(self, rejection: Rejection) -> None:
        self._require_workable("reject")
        self.status = OrderStatus.REJECTED
        self.rejection = rejection

    def cancel(self, at: UtcDatetime, message: str = "") -> None:
        self._require_workable("cancel")
        self.status = OrderStatus.CANCELLED
        # There is no dedicated field: `rejection` records a refusal, and a
        # cancel is not one. Metadata keeps the audit trail without widening a
        # row the engine allocates once per order.
        self.metadata["cancelled_at"] = ensure_utc(at)
        if message:
            self.metadata["cancel_message"] = message

    def _require_workable(self, action: str) -> None:
        """Guard every mutation: terminal states are final by definition."""
        if self.status.is_terminal:
            raise ExecutionError(
                f"Cannot {action} an order in the terminal state {self.status.value!r}.",
                order_id=self.order_id,
                status=self.status.value,
            )


@dataclass(frozen=True, slots=True)
class Fill:
    """An immutable execution event — the only thing that moves cash.

    ``price`` is the per-unit execution price AFTER slippage and price selection
    but BEFORE commission. Cash impact is
    ``-side.sign * price * quantity * multiplier - commission - fees``.
    """

    fill_id: FillId
    order_id: OrderId
    instrument_id: InstrumentId
    timestamp: UtcDatetime
    side: OrderSide
    quantity: Quantity
    price: Price
    commission: Money = 0.0
    fees: Money = 0.0
    slippage_per_unit: Price = 0.0
    reference_price: Price = 0.0
    liquidity: FillLiquidity = FillLiquidity.TAKER
    is_partial: bool = False

    def __post_init__(self) -> None:
        """Normalise the execution instant to tz-aware UTC.

        The fill is what stamps ``Position.opened_at`` and both ends of the
        closed :class:`~sigmaloop.domain.position.Trade`, so a naive timestamp
        landing here would put a whole timezone of error into holding periods and
        into every time-bucketed metric downstream.
        """
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))

    @property
    def gross_value(self) -> Money:
        """``price * quantity * multiplier`` — multiplier applied by the caller.

        The :class:`Fill` does not know its instrument's contract size, and
        guessing ``1.0`` here would understate every option by 100x.
        """
        return self.price * self.quantity

    @property
    def total_cost(self) -> Money:
        """``commission + fees``."""
        return self.commission + self.fees

    @property
    def slippage_cost(self) -> Money:
        """Total cash given up to slippage on this fill.

        ``slippage_per_unit`` is signed adverse-positive (paid up when buying,
        given up when selling), so a model that hands back price improvement
        reports a negative cost rather than an inflated positive one.
        """
        return self.slippage_per_unit * self.quantity
