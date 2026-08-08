"""Execution models: the lookahead firewall, the trigger table and gaps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sigmaloop.domain.bar import Bar
from sigmaloop.domain.instrument import Equity
from sigmaloop.domain.order import Order
from sigmaloop.execution.models import (
    ExecutionContext,
    ExecutionModel,
    NextBarCloseExecutionModel,
    NextBarOpenExecutionModel,
    SameBarCloseExecutionModel,
)
from sigmaloop.types import (
    ExecutionTiming,
    InstrumentId,
    OrderId,
    OrderSide,
    OrderType,
    Symbol,
    TimeInForce,
)

MSFT = Equity(instrument_id=InstrumentId("EQ:MSFT"), symbol=Symbol("MSFT"))
DAY1 = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
DAY3 = DAY1 + timedelta(days=2)


def order(**overrides: object) -> Order:
    kwargs: dict[str, object] = {
        "order_id": OrderId("O-1"),
        "instrument_id": MSFT.instrument_id,
        "side": OrderSide.BUY,
        "quantity": 100.0,
        "order_type": OrderType.MARKET,
        "submitted_at": DAY1,
    }
    kwargs.update(overrides)
    return Order(**kwargs)  # type: ignore[arg-type]


def bar(open_: float, high: float, low: float, close: float, at: datetime = DAY2) -> Bar:
    return Bar(
        instrument_id=MSFT.instrument_id,
        timestamp=at,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10_000.0,
    )


def context(target: Order, market: Bar | None, **overrides: object) -> ExecutionContext:
    kwargs: dict[str, object] = {
        "order": target,
        "instrument": MSFT,
        "timestamp": market.timestamp if market is not None else DAY2,
        "bar": market,
        "is_session_close": True,
    }
    kwargs.update(overrides)
    return ExecutionContext(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The firewall
# --------------------------------------------------------------------------- #


def test_next_bar_open_refuses_the_submitting_bar() -> None:
    model = NextBarOpenExecutionModel()
    working = order()
    assert not model.is_eligible(context(working, bar(100, 101, 99, 100, at=DAY1)))
    assert model.is_eligible(context(working, bar(100, 101, 99, 100, at=DAY2)))


def test_same_bar_close_fills_on_the_signal_bar_and_says_so() -> None:
    model = SameBarCloseExecutionModel()
    assert model.is_eligible(context(order(), bar(100, 101, 99, 100, at=DAY1)))
    assert model.introduces_lookahead
    assert model.timing is ExecutionTiming.SAME_BAR_CLOSE


def test_the_default_model_is_lookahead_free() -> None:
    model = NextBarOpenExecutionModel()
    assert not model.introduces_lookahead
    assert model.timing is ExecutionTiming.NEXT_BAR_OPEN
    assert not NextBarCloseExecutionModel().introduces_lookahead


# --------------------------------------------------------------------------- #
# Market orders
# --------------------------------------------------------------------------- #


def test_market_fills_at_the_next_open() -> None:
    decision = NextBarOpenExecutionModel().try_fill(context(order(), bar(102, 105, 101, 104)))
    assert decision.should_fill
    assert decision.reference_price == 102.0
    assert decision.quantity == 100.0


def test_market_on_close_waits_for_the_session_close() -> None:
    model = NextBarOpenExecutionModel()
    moc = order(order_type=OrderType.MARKET_ON_CLOSE)
    intraday = context(moc, bar(102, 105, 101, 104), is_session_close=False)
    assert not model.try_fill(intraday).should_fill
    assert model.try_fill(context(moc, bar(102, 105, 101, 104))).reference_price == 104.0


def test_close_anchored_models_fill_at_the_close() -> None:
    decision = NextBarCloseExecutionModel().try_fill(context(order(), bar(102, 105, 101, 104)))
    assert decision.reference_price == 104.0


def test_no_bar_means_no_fill() -> None:
    decision = NextBarOpenExecutionModel().try_fill(context(order(), None))
    assert not decision.should_fill
    assert "no bar" in decision.reason


# --------------------------------------------------------------------------- #
# The trigger table (DESIGN §6.1)
# --------------------------------------------------------------------------- #


def test_buy_limit_needs_the_low_to_reach_it() -> None:
    model = NextBarOpenExecutionModel()
    limit = order(order_type=OrderType.LIMIT, limit_price=99.0)
    assert not model.try_fill(context(limit, bar(100, 101, 99.5, 100))).should_fill
    filled = model.try_fill(context(limit, bar(100, 101, 98.0, 99.5)))
    assert filled.should_fill
    assert filled.reference_price == 99.0  # min(open, limit)


def test_sell_limit_needs_the_high_to_reach_it() -> None:
    model = NextBarOpenExecutionModel()
    limit = order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=101.0)
    assert not model.try_fill(context(limit, bar(100, 100.5, 99, 100))).should_fill
    filled = model.try_fill(context(limit, bar(100, 102.0, 99, 101.5)))
    assert filled.reference_price == 101.0


def test_a_limit_that_gaps_in_your_favour_prints_at_the_open() -> None:
    """The open is a real trade; the limit would understate the improvement."""
    limit = order(order_type=OrderType.LIMIT, limit_price=99.0)
    decision = NextBarOpenExecutionModel().try_fill(context(limit, bar(97.0, 98.0, 96.0, 97.5)))
    assert decision.reference_price == 97.0


def test_a_stop_that_gaps_through_pays_the_open_not_the_trigger() -> None:
    """The single most common way a backtest manufactures free money."""
    stop = order(order_type=OrderType.STOP, stop_price=100.0)
    decision = NextBarOpenExecutionModel().try_fill(context(stop, bar(103.0, 104, 102.5, 103.5)))
    assert decision.reference_price == 103.0


def test_gap_improvement_reinstates_the_optimistic_convention() -> None:
    stop = order(order_type=OrderType.STOP, stop_price=100.0)
    model = NextBarOpenExecutionModel(allow_gap_improvement=True)
    assert model.try_fill(context(stop, bar(103.0, 104, 102.5, 103.5))).reference_price == 100.0


def test_sell_stop_triggers_on_the_low() -> None:
    model = NextBarOpenExecutionModel()
    stop = order(side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=95.0)
    assert not model.try_fill(context(stop, bar(100, 101, 96, 98))).should_fill
    filled = model.try_fill(context(stop, bar(100, 101, 94, 94.5)))
    assert filled.reference_price == 95.0  # min(open, stop)


def test_stop_limit_does_not_fill_through_its_limit() -> None:
    model = NextBarOpenExecutionModel()
    stop_limit = order(order_type=OrderType.STOP_LIMIT, stop_price=100.0, limit_price=100.5)
    # Triggered, but the bar opened at 103 — above the limit, so no fill.
    assert not model.try_fill(context(stop_limit, bar(103, 104, 102, 103))).should_fill
    # Triggered inside the limit: fills at the trigger.
    assert model.try_fill(context(stop_limit, bar(99, 101, 98, 100.2))).reference_price == 100.0


def test_a_limit_exactly_at_the_bar_low_triggers() -> None:
    limit = order(order_type=OrderType.LIMIT, limit_price=99.0)
    assert (
        NextBarOpenExecutionModel().try_fill(context(limit, bar(100, 101, 99.0, 100))).should_fill
    )


# --------------------------------------------------------------------------- #
# Time in force
# --------------------------------------------------------------------------- #


def test_a_day_order_survives_the_session_it_was_written_for() -> None:
    """Raised at Monday's close, it is an instruction for Tuesday — expiring it
    against its submission date would kill it before that session opened."""
    model = NextBarOpenExecutionModel()
    day = order(time_in_force=TimeInForce.DAY)
    assert not model.should_expire(context(day, bar(100, 101, 99, 100, at=DAY2)))

    day.activated_at = DAY2
    assert not model.should_expire(context(day, bar(100, 101, 99, 100, at=DAY2)))
    assert model.should_expire(context(day, bar(100, 101, 99, 100, at=DAY3)))


def test_gtc_never_lapses() -> None:
    model = NextBarOpenExecutionModel()
    gtc = order(time_in_force=TimeInForce.GTC)
    gtc.activated_at = DAY2
    far_future = bar(100, 101, 99, 100, at=DAY1 + timedelta(days=400))
    assert not model.should_expire(context(gtc, far_future))


def test_gtd_lapses_at_its_date() -> None:
    model = NextBarOpenExecutionModel()
    gtd = order(time_in_force=TimeInForce.GTD, expires_at=DAY3)
    gtd.activated_at = DAY2
    assert not model.should_expire(context(gtd, bar(100, 101, 99, 100, at=DAY2)))
    assert model.should_expire(context(gtd, bar(100, 101, 99, 100, at=DAY3)))


def test_a_terminal_order_never_expires_again() -> None:
    model = NextBarOpenExecutionModel()
    cancelled = order(time_in_force=TimeInForce.DAY)
    cancelled.activated_at = DAY2
    cancelled.cancel(DAY2)
    assert not model.should_expire(context(cancelled, bar(100, 101, 99, 100, at=DAY3)))


@pytest.mark.parametrize(
    "model",
    [NextBarOpenExecutionModel(), NextBarCloseExecutionModel(), SameBarCloseExecutionModel()],
)
def test_every_model_reports_a_timing(model: ExecutionModel) -> None:
    assert isinstance(model.timing, ExecutionTiming)
