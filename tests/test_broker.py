"""The simulated broker: intake, matching, costs, brackets and OCO."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from sigmaloop.domain.bar import Bar, Greeks, MarketSnapshot, OptionChain, OptionQuote, Quote
from sigmaloop.domain.instrument import Equity, InstrumentRegistry, OptionContract
from sigmaloop.domain.order import BracketSpec, Order
from sigmaloop.errors import ExecutionError
from sigmaloop.execution.broker import BRACKET_METADATA_KEY, SimulatedBroker
from sigmaloop.execution.commission import (
    CommissionModel,
    PerShareCommissionModel,
    PerTradeCommissionModel,
    ZeroCommissionModel,
)
from sigmaloop.execution.models import NextBarOpenExecutionModel
from sigmaloop.execution.pricing import FixedBpsSpreadModel, QuoteFillPriceModel
from sigmaloop.execution.slippage import (
    FixedBpsSlippageModel,
    NoSlippageModel,
    SlippageModel,
    SpreadFractionSlippageModel,
    VolumeShareSlippageModel,
)
from sigmaloop.types import (
    InstrumentId,
    OptionRight,
    OrderId,
    OrderSide,
    OrderStatus,
    OrderType,
    PriceSelection,
    RejectReason,
    Symbol,
    TimeInForce,
)

DAY1 = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
DAY3 = DAY1 + timedelta(days=2)
DAY4 = DAY1 + timedelta(days=3)

MSFT = Equity(instrument_id=InstrumentId("EQ:MSFT"), symbol=Symbol("MSFT"))
NO_BORROW = Equity(instrument_id=InstrumentId("EQ:HTB"), symbol=Symbol("HTB"), is_shortable=False)
DELISTED = Equity(
    instrument_id=InstrumentId("EQ:OLD"), symbol=Symbol("OLD"), delisted_on=date(2020, 1, 1)
)
EXPIRY = date(2023, 4, 21)
CALL = OptionContract(
    instrument_id=OptionContract.make_id(Symbol("MSFT"), EXPIRY, OptionRight.CALL, 280.0),
    symbol=Symbol("MSFT230421C00280000"),
    underlying_id=MSFT.instrument_id,
    underlying_symbol=Symbol("MSFT"),
    right=OptionRight.CALL,
    strike=280.0,
    expiry=EXPIRY,
)


def make_broker(
    *,
    slippage: SlippageModel | None = None,
    commission: CommissionModel | None = None,
    spread_bps: float = 2.0,
    selection: PriceSelection = PriceSelection.WORST,
    **flags: bool,
) -> SimulatedBroker:
    registry = InstrumentRegistry()
    for instrument in (MSFT, NO_BORROW, DELISTED, CALL):
        registry.register(instrument)
    return SimulatedBroker(
        execution_model=NextBarOpenExecutionModel(),
        price_model=QuoteFillPriceModel(FixedBpsSpreadModel(spread_bps=spread_bps)),
        slippage_model=slippage or NoSlippageModel(),
        commission_model=commission or ZeroCommissionModel(),
        registry=registry,
        price_selection=selection,
        **flags,
    )


def order(**overrides: object) -> Order:
    kwargs: dict[str, object] = {
        "order_id": OrderId("O-1"),
        "instrument_id": MSFT.instrument_id,
        "side": OrderSide.BUY,
        "quantity": 100.0,
        "order_type": OrderType.MARKET,
        "submitted_at": DAY1,
        "time_in_force": TimeInForce.GTC,
    }
    kwargs.update(overrides)
    return Order(**kwargs)  # type: ignore[arg-type]


def bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    at: datetime = DAY2,
    volume: float = 10_000.0,
    instrument_id: InstrumentId = MSFT.instrument_id,
) -> Bar:
    return Bar(
        instrument_id=instrument_id,
        timestamp=at,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def snapshot(*bars: Bar, at: datetime | None = None, **overrides: object) -> MarketSnapshot:
    kwargs: dict[str, object] = {
        "timestamp": at or (bars[0].timestamp if bars else DAY2),
        "bars": {market.instrument_id: market for market in bars},
    }
    kwargs.update(overrides)
    return MarketSnapshot(**kwargs)  # type: ignore[arg-type]


def option_snapshot(
    bid: float, ask: float, *, at: datetime = DAY2, volume: float = 50.0
) -> MarketSnapshot:
    quote = OptionQuote(
        instrument_id=CALL.instrument_id,
        contract=CALL,
        timestamp=at,
        quote=Quote(bid=bid, ask=ask),
        volume=volume,
        greeks=Greeks(delta=0.45),
        underlying_price=280.0,
    )
    chain = OptionChain(
        underlying_id=MSFT.instrument_id,
        underlying_symbol=Symbol("MSFT"),
        timestamp=at,
        underlying_price=280.0,
        quotes=(quote,),
    )
    return MarketSnapshot(timestamp=at, bars={}, chains={MSFT.instrument_id: chain})


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #


def test_a_submitted_order_joins_the_working_book() -> None:
    broker = make_broker()
    working = order()
    broker.submit(working)
    assert working.status is OrderStatus.ACCEPTED
    assert broker.working_orders() == (working,)
    assert broker.working_orders(MSFT.instrument_id) == (working,)
    assert broker.get_order(working.order_id) is working


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"instrument_id": InstrumentId("EQ:NOPE")}, RejectReason.INSTRUMENT_NOT_TRADEABLE),
        ({"instrument_id": DELISTED.instrument_id}, RejectReason.INSTRUMENT_EXPIRED),
        ({"quantity": 0.0}, RejectReason.ZERO_OR_NEGATIVE_QUANTITY),
        ({"quantity": 0.4}, RejectReason.BELOW_MIN_LOT),
        (
            {"instrument_id": NO_BORROW.instrument_id, "side": OrderSide.SELL},
            RejectReason.NOT_SHORTABLE,
        ),
    ],
)
def test_structurally_impossible_orders_are_refused(
    overrides: dict[str, object], reason: RejectReason
) -> None:
    broker = make_broker()
    refused = order(**overrides)
    broker.submit(refused)
    assert refused.status is OrderStatus.REJECTED
    assert refused.rejection is not None and refused.rejection.reason is reason
    assert broker.working_orders() == ()


def test_a_covering_sell_is_allowed_on_a_hard_to_borrow_name() -> None:
    broker = make_broker()
    covering = order(instrument_id=NO_BORROW.instrument_id, side=OrderSide.SELL, reduce_only=True)
    broker.submit(covering)
    assert covering.status is OrderStatus.ACCEPTED


def test_rejections_surface_on_the_next_bar() -> None:
    broker = make_broker()
    broker.submit(order(quantity=0.0))
    result = broker.process_bar(snapshot(bar(100, 101, 99, 100)))
    assert [order_id for order_id, _ in result.rejections] == [OrderId("O-1")]
    assert not result.is_empty
    # Drained, not repeated.
    assert broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY3))).rejections == []


def test_a_duplicate_order_id_is_a_programming_error() -> None:
    broker = make_broker()
    broker.submit(order())
    with pytest.raises(ExecutionError, match="Duplicate order id"):
        broker.submit(order())


def test_resubmitting_a_working_order_is_refused() -> None:
    broker = make_broker()
    working = order()
    broker.submit(working)
    with pytest.raises(ExecutionError, match="PENDING_NEW"):
        broker.submit(working)


# --------------------------------------------------------------------------- #
# Matching and the firewall
# --------------------------------------------------------------------------- #


def test_an_order_cannot_fill_on_the_bar_that_raised_it() -> None:
    broker = make_broker()
    broker.submit(order())
    assert broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY1))).fills == []
    assert broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY2))).fills != []


def test_a_market_buy_pays_the_ask_at_the_next_open() -> None:
    """The open, not the close — and the bar is chosen so the two differ.

    A bar whose open equals its close cannot tell the two apart, which is exactly
    how a price model that quietly re-anchored the fill on the close would slip
    through. Here the bar runs 100 -> 150, so anchoring on the close would print
    a 50% better entry than the market offered when the signal existed.
    """
    broker = make_broker()
    working = order()
    broker.submit(working)
    (fill,) = broker.process_bar(snapshot(bar(100.0, 155.0, 99.0, 150.0))).fills
    # Open 100.00 plus the half-spread the 2bp model puts on the bar's mark.
    assert fill.reference_price == 100.0
    assert fill.price == pytest.approx(100.01)
    assert fill.quantity == 100.0
    assert working.status is OrderStatus.FILLED
    assert broker.working_orders() == ()


def test_a_market_sell_receives_the_bid() -> None:
    broker = make_broker()
    broker.submit(order(side=OrderSide.SELL))
    (fill,) = broker.process_bar(snapshot(bar(100.0, 105.0, 99.0, 100.0))).fills
    assert fill.price == 99.99


def test_slippage_is_adverse_on_both_sides_and_snapped_against_the_trader() -> None:
    buyer = make_broker(slippage=FixedBpsSlippageModel(bps=1.3))
    buyer.submit(order())
    (bought,) = buyer.process_bar(snapshot(bar(100.0, 105.0, 99.0, 100.0))).fills
    # 100.00 open + 0.01 spread + 0.013 slippage = 100.023, rounded up a tick.
    assert bought.price == 100.03
    assert bought.slippage_per_unit == pytest.approx(0.013)

    seller = make_broker(slippage=FixedBpsSlippageModel(bps=1.3))
    seller.submit(order(side=OrderSide.SELL))
    (sold,) = seller.process_bar(snapshot(bar(100.0, 105.0, 99.0, 100.0))).fills
    assert sold.price == 99.97


def test_the_spread_is_charged_where_the_trade_happened_not_at_the_close() -> None:
    """A bar that opened at 100 and closed at 140 has two very different
    spreads; an opening fill crossed the one at 100."""
    broker = make_broker(spread_bps=100.0)
    broker.submit(order())
    (fill,) = broker.process_bar(snapshot(bar(100.0, 140.0, 100.0, 140.0))).fills
    # 100bp of 100 is a 1.00 spread, so half a point over the open.
    assert fill.price == pytest.approx(100.50)


def test_a_stop_at_the_high_cannot_print_beyond_what_the_bar_offered() -> None:
    """Spread and slippage stack on a level already at the bar's extreme; left
    alone they book a purchase at a price nobody was offering."""
    broker = make_broker(slippage=FixedBpsSlippageModel(bps=50.0))
    stop = order(order_type=OrderType.STOP, stop_price=100.0)
    broker.submit(stop)
    # Gaps open to its own high: the trigger fills at 104, the top of the range.
    (fill,) = broker.process_bar(snapshot(bar(104.0, 104.0, 100.0, 101.0))).fills
    assert fill.reference_price == 104.0
    # 50bp of slippage would have printed 104.53; the offer was 104.01.
    assert fill.price == pytest.approx(104.01)


def test_a_sell_stop_at_the_low_is_bounded_by_the_bid() -> None:
    broker = make_broker(slippage=FixedBpsSlippageModel(bps=50.0))
    stop = order(side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=100.0)
    broker.submit(stop)
    (fill,) = broker.process_bar(snapshot(bar(90.0, 96.0, 90.0, 95.0))).fills
    assert fill.reference_price == 90.0
    assert fill.price == pytest.approx(89.99)


def test_slippage_still_bites_inside_the_range() -> None:
    """The bound is a backstop on impossible prints, not a cap on cost."""
    broker = make_broker(slippage=FixedBpsSlippageModel(bps=50.0))
    broker.submit(order())
    (fill,) = broker.process_bar(snapshot(bar(100.0, 120.0, 99.0, 110.0))).fills
    # Open 100 + 0.01 spread + 0.50 slippage, nowhere near the 120 high.
    assert fill.price == pytest.approx(100.51)


def test_option_slippage_survives_the_market_bound() -> None:
    """A chain quote is a book, not a traded range; bounding to its mid would
    delete the spread-fraction cost that options actually pay."""
    broker = make_broker(slippage=SpreadFractionSlippageModel(fraction=0.5))
    broker.submit(order(instrument_id=CALL.instrument_id, quantity=5.0))
    (fill,) = broker.process_bar(option_snapshot(bid=2.00, ask=2.20)).fills
    # The ask, plus half the 0.20 spread in impact.
    assert fill.price == pytest.approx(2.30)


def test_commission_and_fees_ride_on_the_fill() -> None:
    broker = make_broker(commission=PerShareCommissionModel(rate=0.005, minimum=1.0))
    working = order()
    broker.submit(working)
    (fill,) = broker.process_bar(snapshot(bar(100.0, 105.0, 99.0, 100.0))).fills
    assert fill.commission == pytest.approx(1.0)
    assert working.commission_paid == pytest.approx(1.0)


def test_a_limit_order_never_prints_through_its_limit() -> None:
    """Spread and tick rounding push the price past the cap; the cap wins."""
    broker = make_broker()
    limit = order(order_type=OrderType.LIMIT, limit_price=100.0)
    broker.submit(limit)
    (fill,) = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0))).fills
    assert fill.price == 100.0
    assert limit.status is OrderStatus.FILLED


def test_a_resting_limit_waits_for_its_price() -> None:
    broker = make_broker()
    broker.submit(order(order_type=OrderType.LIMIT, limit_price=95.0))
    assert broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY2))).fills == []
    assert broker.process_bar(snapshot(bar(97, 98, 94, 95, at=DAY3))).fills != []


def test_an_instrument_that_did_not_trade_leaves_the_order_working() -> None:
    broker = make_broker()
    working = order()
    broker.submit(working)
    result = broker.process_bar(snapshot(at=DAY2))
    assert result.is_empty
    assert broker.working_orders() == (working,)


def test_fills_are_counted_as_synthetic_when_the_feed_had_no_book() -> None:
    broker = make_broker()
    broker.submit(order())
    broker.process_bar(snapshot(bar(100.0, 105.0, 99.0, 100.0)))
    assert broker.synthetic_quote_fills == 1


def test_a_real_book_is_not_counted_as_synthetic() -> None:
    broker = make_broker()
    broker.submit(order())
    quoted = Bar(
        instrument_id=MSFT.instrument_id,
        timestamp=DAY2,
        open=100.0,
        high=105.0,
        low=99.0,
        close=100.0,
        volume=10_000.0,
        quote=Quote(bid=99.5, ask=100.5),
    )
    (fill,) = broker.process_bar(snapshot(quoted)).fills
    assert broker.synthetic_quote_fills == 0
    assert fill.price == 100.5  # open 100 + the observed half-spread of 0.50


# --------------------------------------------------------------------------- #
# Liquidity, partial fills and one-shot time in force
# --------------------------------------------------------------------------- #


def test_a_participation_cap_truncates_the_fill() -> None:
    broker = make_broker(slippage=VolumeShareSlippageModel(max_volume_share=0.025))
    working = order(quantity=1_000.0)
    broker.submit(working)
    (fill,) = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0))).fills
    assert fill.quantity == 250.0
    assert fill.is_partial
    assert working.status is OrderStatus.PARTIALLY_FILLED
    assert broker.partial_fills == 1
    # Carried by default: the rest is still working.
    assert working.remaining_quantity == 750.0


def test_a_carried_remainder_finishes_on_a_deeper_bar() -> None:
    broker = make_broker(slippage=VolumeShareSlippageModel(max_volume_share=0.025))
    working = order(quantity=1_000.0)
    broker.submit(working)
    (first,) = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0))).fills
    (second,) = broker.process_bar(
        snapshot(bar(102.0, 103.0, 101.0, 102.0, at=DAY3, volume=1_000_000.0))
    ).fills
    assert second.quantity == 750.0
    assert not second.is_partial
    assert working.status is OrderStatus.FILLED
    # Volume-weighted across both prints, so the order reports what it achieved.
    assert working.avg_fill_price == pytest.approx(0.25 * first.price + 0.75 * second.price)
    # The deeper bar absorbed the rest with far less impact than the thin one.
    assert second.slippage_per_unit < first.slippage_per_unit


def test_the_remainder_can_be_cancelled_instead_of_carried() -> None:
    broker = make_broker(
        slippage=VolumeShareSlippageModel(max_volume_share=0.025),
        carry_unfilled_remainder=False,
    )
    working = order(quantity=1_000.0)
    broker.submit(working)
    result = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    assert result.cancellations == [working.order_id]
    assert working.status is OrderStatus.CANCELLED
    assert broker.working_orders() == ()


def test_partial_fills_can_be_switched_off_entirely() -> None:
    broker = make_broker(
        slippage=VolumeShareSlippageModel(max_volume_share=0.025), allow_partial_fills=False
    )
    working = order(quantity=1_000.0)
    broker.submit(working)
    assert broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0))).fills == []
    assert working.status is OrderStatus.ACCEPTED


def test_fill_or_kill_refuses_a_truncated_fill() -> None:
    broker = make_broker(slippage=VolumeShareSlippageModel(max_volume_share=0.025))
    fok = order(quantity=1_000.0, time_in_force=TimeInForce.FOK)
    broker.submit(fok)
    result = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    assert result.fills == []
    assert result.expirations == [fok.order_id]
    assert fok.status is OrderStatus.EXPIRED


def test_immediate_or_cancel_gets_one_look() -> None:
    broker = make_broker(slippage=VolumeShareSlippageModel())
    ioc = order(quantity=1_000.0, time_in_force=TimeInForce.IOC)
    broker.submit(ioc)
    # A bar that printed nothing can absorb nothing, and IOC does not wait.
    result = broker.process_bar(snapshot(bar(100.0, 100.0, 100.0, 100.0, volume=0.0)))
    assert result.expirations == [ioc.order_id]


def test_a_day_order_expires_after_its_session() -> None:
    broker = make_broker()
    day = order(order_type=OrderType.LIMIT, limit_price=1.0, time_in_force=TimeInForce.DAY)
    broker.submit(day)
    assert broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY2))).expirations == []
    result = broker.process_bar(snapshot(bar(100, 101, 99, 100, at=DAY3)))
    assert result.expirations == [day.order_id]
    assert day.status is OrderStatus.EXPIRED
    assert broker.working_orders() == ()


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_cancel_removes_the_order_from_the_book() -> None:
    broker = make_broker()
    working = order()
    broker.submit(working)
    assert broker.cancel(working.order_id, DAY2, "changed my mind")
    assert working.status is OrderStatus.CANCELLED
    assert broker.working_orders() == ()
    # Cancelling twice is a no-op, not an error.
    assert not broker.cancel(working.order_id, DAY2)


def test_cancel_all_can_be_narrowed_to_one_instrument() -> None:
    broker = make_broker()
    broker.submit(order(order_id=OrderId("O-1")))
    broker.submit(order(order_id=OrderId("O-2")))
    broker.submit(order(order_id=OrderId("O-3"), instrument_id=NO_BORROW.instrument_id))
    assert broker.cancel_all(DAY2, MSFT.instrument_id) == 2
    assert len(broker.working_orders()) == 1
    assert broker.cancel_all(DAY2) == 1


# --------------------------------------------------------------------------- #
# Brackets
# --------------------------------------------------------------------------- #


def bracketed(spec: BracketSpec, **overrides: object) -> Order:
    return order(metadata={BRACKET_METADATA_KEY: spec}, **overrides)


def test_bracket_children_appear_only_after_the_parent_fills() -> None:
    broker = make_broker()
    parent = bracketed(BracketSpec(stop_loss_pct=0.10, take_profit_pct=0.20))
    broker.submit(parent)
    assert len(broker.working_orders()) == 1

    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    children = broker.working_orders()
    assert {child.order_id for child in children} == {OrderId("O-1-SL"), OrderId("O-1-TP")}
    stop, target = children[0], children[1]
    assert stop.order_type is OrderType.STOP and stop.side is OrderSide.SELL
    assert stop.stop_price == pytest.approx(90.01)  # 10% under the 100.01 entry
    assert target.limit_price == pytest.approx(120.01)
    assert all(child.reduce_only and child.time_in_force is TimeInForce.GTC for child in children)


def test_a_stop_cannot_be_hit_by_the_bar_that_opened_the_position() -> None:
    broker = make_broker()
    broker.submit(bracketed(BracketSpec(stop_loss_pct=0.10)))
    # The entry bar collapses far through the stop, but the child was not yet
    # working — the position cannot be stopped out by its own entry print.
    (fill,) = broker.process_bar(snapshot(bar(100.0, 101.0, 50.0, 60.0))).fills
    # And the entry itself still prints at the open. The bar closes 40% lower,
    # so a fill anchored anywhere but the open would be visible here.
    assert fill.reference_price == 100.0


def test_filling_one_bracket_leg_cancels_its_sibling() -> None:
    broker = make_broker()
    broker.submit(bracketed(BracketSpec(stop_loss_pct=0.10, take_profit_pct=0.20)))
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    result = broker.process_bar(snapshot(bar(115.0, 125.0, 114.0, 124.0, at=DAY3)))
    (fill,) = result.fills
    assert fill.order_id == OrderId("O-1-TP")
    assert result.cancellations == [OrderId("O-1-SL")]
    assert broker.working_orders() == ()


def test_a_bar_that_touches_both_legs_stops_out() -> None:
    """The intrabar path is unknown, so the pessimistic leg is assumed first."""
    broker = make_broker()
    broker.submit(bracketed(BracketSpec(stop_loss_pct=0.10, take_profit_pct=0.20)))
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    result = broker.process_bar(snapshot(bar(100.0, 125.0, 85.0, 120.0, at=DAY3)))
    (fill,) = result.fills
    assert fill.order_id == OrderId("O-1-SL")
    assert result.cancellations == [OrderId("O-1-TP")]


def test_a_partly_filled_entry_is_protected_for_what_it_owns() -> None:
    """A thin market that fills half the entry still leaves a real position."""
    broker = make_broker(slippage=VolumeShareSlippageModel(max_volume_share=0.025))
    parent = bracketed(BracketSpec(stop_loss_pct=0.10), quantity=1_000.0)
    broker.submit(parent)
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    assert parent.filled_quantity == 250.0
    (stop,) = [child for child in broker.working_orders() if child.order_id == "O-1-SL"]
    assert stop.quantity == 250.0

    # The rest fills on a deeper bar, and the stop grows with the position.
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0, at=DAY3, volume=1_000_000.0)))
    assert parent.status is OrderStatus.FILLED
    assert stop.quantity == 1_000.0


def test_a_dropped_remainder_still_leaves_the_stop_behind() -> None:
    broker = make_broker(
        slippage=VolumeShareSlippageModel(max_volume_share=0.025),
        carry_unfilled_remainder=False,
    )
    parent = bracketed(BracketSpec(stop_loss_pct=0.10), quantity=1_000.0)
    broker.submit(parent)
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    assert parent.status is OrderStatus.CANCELLED
    (stop,) = broker.working_orders()
    assert stop.order_id == "O-1-SL"
    assert stop.quantity == 250.0


def test_absolute_bracket_levels_are_used_as_given() -> None:
    broker = make_broker()
    broker.submit(bracketed(BracketSpec(stop_loss_price=95.0, take_profit_price=110.0)))
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    stop, target = broker.working_orders()
    assert stop.stop_price == 95.0
    assert target.limit_price == 110.0


def test_a_trailing_stop_ratchets_after_the_bar_it_watched() -> None:
    broker = make_broker()
    broker.submit(bracketed(BracketSpec(trailing_stop_pct=0.10)))
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    (stop,) = broker.working_orders()
    assert stop.stop_price == pytest.approx(90.01)

    # A bar that runs to 120 lifts the stop to 108 — but only once the bar it
    # printed on has already been tested, so its own low cannot be caught by it.
    broker.process_bar(snapshot(bar(101.0, 120.0, 100.0, 119.0, at=DAY3)))
    assert stop.stop_price == pytest.approx(108.0)

    result = broker.process_bar(snapshot(bar(118.0, 119.0, 105.0, 106.0, at=DAY4)))
    (fill,) = result.fills
    assert fill.order_id == OrderId("O-1-SL")
    assert fill.reference_price == pytest.approx(108.0)  # min(open, stop)


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #


def test_a_run_with_no_book_and_no_spread_model_counts_its_free_crossings() -> None:
    """WORST asks for a side of the book. With neither quotes nor a spread model
    there is no side to pay for, so the fill crosses for nothing — the one case
    the synthetic spread exists to prevent. It is counted rather than hidden, so
    the run can say how much of its result was priced as if trading were free.
    """
    registry = InstrumentRegistry()
    registry.register(MSFT)
    broker = SimulatedBroker(
        execution_model=NextBarOpenExecutionModel(),
        price_model=QuoteFillPriceModel(spread_model=None),
        slippage_model=NoSlippageModel(),
        commission_model=ZeroCommissionModel(),
        registry=registry,
        price_selection=PriceSelection.WORST,
    )
    broker.submit(order())
    (fill,) = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0))).fills

    assert fill.price == 100.0, "no book, so nothing was charged for crossing it"
    assert broker.uncosted_spread_fills == 1
    assert broker.synthetic_quote_fills == 0, "nothing was synthesised either"


def test_a_priced_book_is_not_counted_as_a_free_crossing() -> None:
    broker = make_broker()  # carries a FixedBpsSpreadModel
    broker.submit(order())
    broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0)))
    assert broker.uncosted_spread_fills == 0
    assert broker.synthetic_quote_fills == 1


def test_an_order_split_by_a_liquidity_cap_pays_one_flat_fee() -> None:
    """The broker reports what has already filled, so per-order schedules bill
    the order once however many bars it takes to work."""
    broker = make_broker(
        slippage=VolumeShareSlippageModel(coefficient=0.0, max_volume_share=0.025),
        commission=PerTradeCommissionModel(fee=5.0),
    )
    broker.submit(order(quantity=1_000.0))
    charged = []
    for index, volume in enumerate((10_000.0, 10_000.0, 10_000.0, 1_000_000.0)):
        at = DAY2 + timedelta(days=index)
        result = broker.process_bar(snapshot(bar(100.0, 101.0, 99.0, 100.0, at=at, volume=volume)))
        charged += [fill.commission for fill in result.fills]

    assert len(charged) == 4, "the cap should have split the order four ways"
    assert charged[0] == pytest.approx(5.0)
    assert sum(charged) == pytest.approx(5.0)


def test_an_option_order_fills_off_the_chain_quote() -> None:
    broker = make_broker()
    working = order(instrument_id=CALL.instrument_id, quantity=5.0)
    broker.submit(working)
    (fill,) = broker.process_bar(option_snapshot(bid=2.00, ask=2.10)).fills
    assert fill.price == 2.10  # the ask, since the order buys
    assert fill.quantity == 5.0
    # The chain published a real book, so nothing was synthesised.
    assert broker.synthetic_quote_fills == 0


def test_an_option_without_a_quote_leaves_the_order_working() -> None:
    broker = make_broker()
    working = order(instrument_id=CALL.instrument_id, quantity=5.0)
    broker.submit(working)
    assert broker.process_bar(snapshot(at=DAY2)).is_empty
    assert broker.working_orders() == (working,)
