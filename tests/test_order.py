"""Order lifecycle: intent validation, VWAP fills, terminal-state guards."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from sigmaloop.domain.order import Fill, Order, OrderIntent, Rejection, SizingRequest
from sigmaloop.errors import ExecutionError, ValidationError
from sigmaloop.types import (
    FillId,
    InstrumentId,
    IntentId,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    RejectReason,
    SizingMode,
    TimeInForce,
)

MSFT = InstrumentId("EQ:MSFT")
STAMP = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
OID = OrderId("O-1")


def intent(**overrides: object) -> OrderIntent:
    kwargs: dict[str, object] = {
        "intent_id": IntentId("I-1"),
        "instrument_id": MSFT,
        "side": OrderSide.BUY,
        "sizing": SizingRequest(mode=SizingMode.FIXED_QUANTITY, value=10.0),
        "created_at": STAMP,
    }
    kwargs.update(overrides)
    return OrderIntent(**kwargs)  # type: ignore[arg-type]


def order(**overrides: object) -> Order:
    kwargs: dict[str, object] = {
        "order_id": OID,
        "instrument_id": MSFT,
        "side": OrderSide.BUY,
        "quantity": 100.0,
        "order_type": OrderType.MARKET,
        "submitted_at": STAMP,
    }
    kwargs.update(overrides)
    return Order(**kwargs)  # type: ignore[arg-type]


def fill(quantity: float, price: float, **overrides: object) -> Fill:
    kwargs: dict[str, object] = {
        "fill_id": FillId("F-1"),
        "order_id": OID,
        "instrument_id": MSFT,
        "timestamp": STAMP,
        "side": OrderSide.BUY,
        "quantity": quantity,
        "price": price,
    }
    kwargs.update(overrides)
    return Fill(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Sides and statuses
# --------------------------------------------------------------------------- #


def test_side_sign_and_opposite() -> None:
    assert OrderSide.BUY.sign == 1
    assert OrderSide.SELL.sign == -1
    assert OrderSide.BUY.opposite is OrderSide.SELL
    assert OrderSide.SELL.opposite is OrderSide.BUY


def test_terminal_statuses_are_exactly_the_four_documented_ones() -> None:
    terminal = {s for s in OrderStatus if s.is_terminal}

    assert terminal == {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
    assert all(s.is_open is not s.is_terminal for s in OrderStatus)
    assert OrderStatus.PENDING_NEW.is_open, "raised and unresolved is still open"


# --------------------------------------------------------------------------- #
# Intent validation
# --------------------------------------------------------------------------- #


def test_market_intent_needs_no_prices() -> None:
    assert intent().order_type is OrderType.MARKET


@pytest.mark.parametrize(
    ("order_type", "kwargs", "missing"),
    [
        (OrderType.LIMIT, {}, "limit_price"),
        (OrderType.STOP, {}, "stop_price"),
        (OrderType.STOP_LIMIT, {"limit_price": 100.0}, "stop_price"),
        (OrderType.STOP_LIMIT, {"stop_price": 100.0}, "limit_price"),
    ],
)
def test_price_fields_are_required_by_order_type(
    order_type: OrderType, kwargs: dict[str, float], missing: str
) -> None:
    with pytest.raises(ValidationError, match=missing):
        intent(order_type=order_type, **kwargs)


def test_a_market_order_may_still_carry_a_stop_price() -> None:
    """RISK_PERCENT sizing measures its budget against the stop, so the level is
    meaningful even when it is not a trigger."""
    raised = intent(
        sizing=SizingRequest(mode=SizingMode.RISK_PERCENT, value=0.01),
        stop_price=95.0,
    )
    assert raised.stop_price == pytest.approx(95.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_prices_are_rejected(bad: float) -> None:
    with pytest.raises(ValidationError, match="positive, finite price"):
        intent(order_type=OrderType.LIMIT, limit_price=bad)


def test_gtd_without_an_expiry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="GTD requires expires_at"):
        intent(time_in_force=TimeInForce.GTD)

    dated = intent(time_in_force=TimeInForce.GTD, expires_at=STAMP + timedelta(days=1))
    assert dated.expires_at is not None


# --------------------------------------------------------------------------- #
# Order arithmetic
# --------------------------------------------------------------------------- #


def test_signed_quantity_carries_direction() -> None:
    assert order(side=OrderSide.BUY).signed_quantity == pytest.approx(100.0)
    assert order(side=OrderSide.SELL).signed_quantity == pytest.approx(-100.0)


def test_a_new_order_is_open_unfilled_and_owes_its_full_size() -> None:
    fresh = order()

    assert fresh.status is OrderStatus.PENDING_NEW
    assert fresh.is_open and not fresh.is_filled
    assert fresh.remaining_quantity == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Filling
# --------------------------------------------------------------------------- #


def test_partial_fill_advances_status_and_leaves_a_remainder() -> None:
    working = order()
    working.apply_fill(fill(40.0, 10.0, commission=1.0))

    assert working.status is OrderStatus.PARTIALLY_FILLED
    assert working.is_open and not working.is_filled
    assert working.filled_quantity == pytest.approx(40.0)
    assert working.remaining_quantity == pytest.approx(60.0)
    assert working.avg_fill_price == pytest.approx(10.0)
    assert working.commission_paid == pytest.approx(1.0)


def test_avg_fill_price_is_volume_weighted_not_last() -> None:
    """A 90/10 split at 10 and 20 averages to 11, not to the last slice's 20."""
    working = order()
    working.apply_fill(fill(90.0, 10.0, commission=1.0))
    working.apply_fill(fill(10.0, 20.0, fill_id=FillId("F-2"), commission=0.5))

    assert working.status is OrderStatus.FILLED
    assert working.is_filled and not working.is_open
    assert working.avg_fill_price == pytest.approx(11.0)
    assert working.remaining_quantity == pytest.approx(0.0)
    assert working.commission_paid == pytest.approx(1.5)


def test_float_dust_does_not_strand_an_order_in_partially_filled() -> None:
    working = order(quantity=0.3)
    for _ in range(3):
        working.apply_fill(fill(0.1, 10.0))

    assert working.status is OrderStatus.FILLED


def test_overfilling_is_refused() -> None:
    working = order(quantity=10.0)
    with pytest.raises(ExecutionError, match="exceeds the order's remaining quantity"):
        working.apply_fill(fill(10.5, 10.0))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"order_id": OrderId("O-2")}, "different order"),
        ({"instrument_id": InstrumentId("EQ:AAPL")}, "different instrument"),
        ({"side": OrderSide.SELL}, "contradicts the order side"),
    ],
)
def test_mismatched_fills_are_refused(overrides: dict[str, object], match: str) -> None:
    working = order()
    with pytest.raises(ExecutionError, match=match):
        working.apply_fill(fill(1.0, 10.0, **overrides))


def test_zero_quantity_fill_is_refused() -> None:
    working = order()
    with pytest.raises(ExecutionError, match="quantity must be positive"):
        working.apply_fill(fill(0.0, 10.0))


# --------------------------------------------------------------------------- #
# Terminal states
# --------------------------------------------------------------------------- #


def test_rejection_is_recorded_with_its_reason() -> None:
    working = order()
    refusal = Rejection(
        reason=RejectReason.INSUFFICIENT_CAPITAL,
        message="needs 1000, has 400",
        timestamp=STAMP,
        required=1000.0,
        available=400.0,
    )
    working.reject(refusal)

    assert working.status is OrderStatus.REJECTED
    assert working.rejection is refusal
    assert not working.is_open


def test_cancel_records_when_and_why_without_faking_a_rejection() -> None:
    working = order()
    working.cancel(STAMP, "end of day")

    assert working.status is OrderStatus.CANCELLED
    assert working.rejection is None, "a cancel is not a refusal"
    assert working.metadata["cancelled_at"] == STAMP
    assert working.metadata["cancel_message"] == "end of day"


def test_a_partially_filled_order_can_still_be_cancelled() -> None:
    working = order()
    working.apply_fill(fill(25.0, 10.0))
    working.cancel(STAMP)

    assert working.status is OrderStatus.CANCELLED
    assert working.filled_quantity == pytest.approx(25.0), "the fill stands"


@pytest.mark.parametrize("action", ["fill", "reject", "cancel"])
def test_terminal_orders_refuse_every_further_mutation(action: str) -> None:
    working = order(quantity=10.0)
    working.apply_fill(fill(10.0, 10.0))
    assert working.status is OrderStatus.FILLED

    with pytest.raises(ExecutionError, match="terminal state"):
        if action == "fill":
            working.apply_fill(fill(1.0, 10.0, fill_id=FillId("F-2")))
        elif action == "reject":
            working.reject(Rejection(RejectReason.MARKET_CLOSED, "late", STAMP))
        else:
            working.cancel(STAMP)


# --------------------------------------------------------------------------- #
# The limit price binds
# --------------------------------------------------------------------------- #


def test_an_order_re_runs_the_intent_price_checks() -> None:
    """A sizer or plugin can build an Order without an intent; a LIMIT that lost
    its limit on the way would be worked as an unbounded market order."""
    with pytest.raises(ValidationError, match="limit_price"):
        order(order_type=OrderType.LIMIT)

    with pytest.raises(ValidationError, match="stop_price"):
        order(order_type=OrderType.STOP_LIMIT, limit_price=100.0)


def test_an_order_without_a_limit_accepts_any_price() -> None:
    assert order().accepts_price(1e9)


@pytest.mark.parametrize(
    ("side", "price", "accepted"),
    [
        (OrderSide.BUY, 99.0, True),  # better
        (OrderSide.BUY, 100.0, True),  # at the limit
        (OrderSide.BUY, 100.01, False),  # through it
        (OrderSide.SELL, 101.0, True),
        (OrderSide.SELL, 100.0, True),
        (OrderSide.SELL, 99.99, False),
    ],
)
def test_a_limit_is_a_ceiling_for_buys_and_a_floor_for_sells(
    side: OrderSide, price: float, accepted: bool
) -> None:
    limited = order(side=side, order_type=OrderType.LIMIT, limit_price=100.0)
    assert limited.accepts_price(price) is accepted


def test_a_fill_through_the_limit_is_refused() -> None:
    limited = order(order_type=OrderType.LIMIT, limit_price=100.0)

    with pytest.raises(ExecutionError, match="through the order's limit"):
        limited.apply_fill(fill(100.0, 100.5))

    assert limited.filled_quantity == pytest.approx(0.0)
    assert limited.status is OrderStatus.PENDING_NEW, "the refusal left no trace"


def test_a_limit_order_still_takes_price_improvement() -> None:
    limited = order(order_type=OrderType.LIMIT, limit_price=100.0)
    limited.apply_fill(fill(100.0, 98.5))

    assert limited.is_filled
    assert limited.avg_fill_price == pytest.approx(98.5)


def test_the_limit_binds_slice_by_slice_not_only_on_average() -> None:
    """A good first slice must not buy room for a bad second one."""
    limited = order(order_type=OrderType.LIMIT, limit_price=100.0)
    limited.apply_fill(fill(50.0, 90.0))

    with pytest.raises(ExecutionError, match="through the order's limit"):
        limited.apply_fill(fill(50.0, 105.0, fill_id=FillId("F-2")))


def test_a_triggered_stop_may_fill_worse_than_its_trigger() -> None:
    """That gap is slippage, and slippage is real; only a limit caps the price."""
    stopped = order(side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=95.0)
    stopped.apply_fill(fill(100.0, 92.0, side=OrderSide.SELL))

    assert stopped.is_filled
    assert stopped.avg_fill_price == pytest.approx(92.0)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


LOCAL_CLOSE = datetime(2023, 3, 28, 16, 0, tzinfo=ZoneInfo("America/New_York"))
NAIVE = datetime(2023, 3, 28, 20, 0)  # noqa: DTZ001


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (lambda ts: intent(created_at=ts), "created_at"),
        # Back-dated so the expiry is strictly after it: an order that expires
        # the instant it is raised is refused, and LOCAL_CLOSE is STAMP.
        (
            lambda ts: intent(
                created_at=ts - timedelta(days=1),
                time_in_force=TimeInForce.GTD,
                expires_at=ts,
            ),
            "expires_at",
        ),
        (lambda ts: order(submitted_at=ts), "submitted_at"),
        (lambda ts: order(activated_at=ts), "activated_at"),
        (lambda ts: fill(1.0, 10.0, timestamp=ts), "timestamp"),
        (lambda ts: Rejection(RejectReason.MARKET_CLOSED, "shut", ts), "timestamp"),
    ],
)
def test_every_timestamp_is_normalised_to_utc(factory: object, field_name: str) -> None:
    built = factory(LOCAL_CLOSE)  # type: ignore[operator]

    assert getattr(built, field_name) == datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
    assert getattr(built, field_name).tzinfo is UTC


@pytest.mark.parametrize(
    "factory",
    [
        lambda ts: intent(created_at=ts),
        lambda ts: intent(time_in_force=TimeInForce.GTD, expires_at=ts),
        lambda ts: order(submitted_at=ts),
        lambda ts: order(activated_at=ts),
        lambda ts: fill(1.0, 10.0, timestamp=ts),
        lambda ts: Rejection(RejectReason.MARKET_CLOSED, "shut", ts),
    ],
)
def test_naive_timestamps_are_rejected_not_assumed(factory: object) -> None:
    with pytest.raises(ValidationError, match="Naive datetime"):
        factory(NAIVE)  # type: ignore[operator]


def test_cancelling_at_a_naive_instant_is_rejected() -> None:
    working = order()
    with pytest.raises(ValidationError, match="Naive datetime"):
        working.cancel(NAIVE)


def test_cancelling_at_a_local_instant_records_utc() -> None:
    working = order()
    working.cancel(LOCAL_CLOSE)

    assert working.metadata["cancelled_at"] == datetime(2023, 3, 28, 20, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fill economics
# --------------------------------------------------------------------------- #


def test_gross_value_leaves_the_multiplier_to_the_caller() -> None:
    executed = fill(2.0, 3.25)
    assert executed.gross_value == pytest.approx(6.5)


def test_total_cost_sums_commission_and_fees() -> None:
    executed = fill(10.0, 100.0, commission=1.0, fees=0.25)
    assert executed.total_cost == pytest.approx(1.25)


def test_slippage_cost_scales_with_size_and_keeps_its_sign() -> None:
    adverse = fill(100.0, 10.02, slippage_per_unit=0.02, reference_price=10.0)
    assert adverse.slippage_cost == pytest.approx(2.0)

    improved = fill(100.0, 9.99, slippage_per_unit=-0.01, reference_price=10.0)
    assert improved.slippage_cost == pytest.approx(-1.0), "improvement is not a cost"


# --------------------------------------------------------------------------- #
# Price fields that do not belong, and expiries that cannot happen
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "order_type",
    [OrderType.MARKET, OrderType.STOP, OrderType.MARKET_ON_OPEN, OrderType.MARKET_ON_CLOSE],
)
def test_a_limit_price_on_a_non_limit_order_is_refused(order_type: OrderType) -> None:
    """accepts_price() enforces any limit it finds, so a stray one on a MARKET
    order would make it refuse every fill the market offered."""
    extra: dict[str, object] = {"stop_price": 90.0} if order_type is OrderType.STOP else {}
    with pytest.raises(ValidationError, match="take no limit_price"):
        intent(order_type=order_type, limit_price=95.0, **extra)


def test_the_same_refusal_applies_to_an_order_built_without_an_intent() -> None:
    with pytest.raises(ValidationError, match="take no limit_price"):
        order(order_type=OrderType.MARKET, limit_price=95.0)


@pytest.mark.parametrize("offset", [timedelta(0), timedelta(days=-1)])
def test_an_expiry_at_or_before_the_raising_instant_is_refused(offset: timedelta) -> None:
    """It would expire on the bar that created it and could never fill."""
    with pytest.raises(ValidationError, match="at or before"):
        intent(time_in_force=TimeInForce.GTD, expires_at=STAMP + offset)


def test_filled_quantity_snaps_to_the_order_size_rather_than_drifting_past_it() -> None:
    """Three 0.1 fills sum to 0.30000000000000004; a report claiming more shares
    than were ordered is a reconciliation break downstream."""
    working = order(quantity=0.3)
    for _ in range(3):
        working.apply_fill(fill(0.1, 10.0))

    assert working.status is OrderStatus.FILLED
    assert working.filled_quantity == 0.3
    assert working.remaining_quantity == pytest.approx(0.0)
