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

from dataclasses import dataclass, field
from typing import Any

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

__all__ = [
    "SizingRequest",
    "BracketSpec",
    "OrderIntent",
    "Order",
    "Fill",
    "Rejection",
]


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
        """Validate price fields against ``order_type`` (e.g. LIMIT needs a limit)."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Rejection:
    """Immutable record of why an order was refused."""

    reason: RejectReason
    message: str
    timestamp: UtcDatetime
    #: Populated for capital breaches so the summary can quantify the shortfall.
    required: Money | None = None
    available: Money | None = None


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

    @property
    def remaining_quantity(self) -> Quantity:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError

    @property
    def is_filled(self) -> bool:
        raise NotImplementedError

    @property
    def signed_quantity(self) -> Quantity:
        """``+quantity`` for BUY, ``-quantity`` for SELL."""
        raise NotImplementedError

    def apply_fill(self, fill: Fill) -> None:
        """Fold a fill in: advance status, update VWAP and filled quantity."""
        raise NotImplementedError

    def reject(self, rejection: Rejection) -> None:
        raise NotImplementedError

    def cancel(self, at: UtcDatetime, message: str = "") -> None:
        raise NotImplementedError


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

    @property
    def gross_value(self) -> Money:
        """``price * quantity * multiplier`` — multiplier applied by the caller."""
        raise NotImplementedError

    @property
    def total_cost(self) -> Money:
        """``commission + fees``."""
        raise NotImplementedError

    @property
    def slippage_cost(self) -> Money:
        """Total cash given up to slippage on this fill."""
        raise NotImplementedError
