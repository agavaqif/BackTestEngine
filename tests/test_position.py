"""Positions and round trips: lot matching, flips, excursions, borrow, trade rows."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from sigmaloop.domain.instrument import Equity, OptionContract
from sigmaloop.domain.order import Fill
from sigmaloop.domain.position import INITIAL_RISK_KEY, OptionTrade, Position, Trade
from sigmaloop.errors import ValidationError
from sigmaloop.types import (
    AssetClass,
    FillId,
    InstrumentId,
    OptionRight,
    OrderId,
    OrderSide,
    PositionSide,
    Symbol,
    TradeCloseReason,
    TradeId,
)

STAMP = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
LATER = STAMP + timedelta(days=3)

MSFT = Equity(
    instrument_id=InstrumentId("EQ:MSFT"),
    symbol=Symbol("MSFT"),
    asset_class=AssetClass.EQUITY,
)
HARD_TO_BORROW = Equity(
    instrument_id=InstrumentId("EQ:GME"),
    symbol=Symbol("GME"),
    asset_class=AssetClass.EQUITY,
    borrow_rate_annual=0.25,
)
MSFT_CALL = OptionContract(
    instrument_id=InstrumentId("OPT:MSFT:20230421:C:00280000"),
    symbol=Symbol("MSFT230421C00280000"),
    underlying_id=MSFT.instrument_id,
    underlying_symbol=Symbol("MSFT"),
    right=OptionRight.CALL,
    strike=280.0,
    expiry=date(2023, 4, 21),
)

_SEQUENCE = iter(range(1, 10_000))


def fill(side: OrderSide, quantity: float, price: float, **overrides: object) -> Fill:
    n = next(_SEQUENCE)
    kwargs: dict[str, object] = {
        "fill_id": FillId(f"F-{n}"),
        "order_id": OrderId(f"O-{n}"),
        "instrument_id": MSFT.instrument_id,
        "timestamp": STAMP,
        "side": side,
        "quantity": quantity,
        "price": price,
    }
    kwargs.update(overrides)
    return Fill(**kwargs)  # type: ignore[arg-type]


def buy(quantity: float, price: float, **overrides: object) -> Fill:
    return fill(OrderSide.BUY, quantity, price, **overrides)


def sell(quantity: float, price: float, **overrides: object) -> Fill:
    return fill(OrderSide.SELL, quantity, price, **overrides)


def trade(**overrides: object) -> Trade:
    kwargs: dict[str, object] = {
        "trade_id": TradeId("T-1"),
        "instrument_id": MSFT.instrument_id,
        "symbol": Symbol("MSFT"),
        "asset_class": AssetClass.EQUITY,
        "direction": PositionSide.LONG,
        "quantity": 100.0,
        "entry_time": STAMP,
        "entry_price": 100.0,
        "exit_time": LATER,
        "exit_price": 110.0,
        "gross_pnl": 1_000.0,
        "commission": 2.0,
        "fees": 0.5,
        "net_pnl": 997.5,
        "return_pct": 0.09975,
        "close_reason": TradeCloseReason.SIGNAL,
    }
    kwargs.update(overrides)
    return Trade(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Side and identity
# --------------------------------------------------------------------------- #


def test_a_fresh_position_is_flat() -> None:
    flat = Position(instrument=MSFT)

    assert flat.instrument_id == MSFT.instrument_id
    assert flat.side is PositionSide.FLAT
    assert not flat.is_open and not flat.is_short
    assert flat.exposure == pytest.approx(0.0)


def test_side_follows_the_sign_of_quantity() -> None:
    long = Position(instrument=MSFT)
    long.apply_fill(buy(10.0, 100.0))
    short = Position(instrument=MSFT)
    short.apply_fill(sell(10.0, 100.0))

    assert long.side is PositionSide.LONG and not long.is_short
    assert short.side is PositionSide.SHORT and short.is_short
    assert short.quantity == pytest.approx(-10.0)


# --------------------------------------------------------------------------- #
# Opening and increasing
# --------------------------------------------------------------------------- #


def test_opening_records_a_lot_and_marks_at_the_fill() -> None:
    """Without marking at the fill, exposure reads zero until the next bar and
    every risk check in between sees no risk."""
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0, commission=1.0, fees=0.25))

    assert position.quantity == pytest.approx(100.0)
    assert position.avg_price == pytest.approx(50.0)
    assert position.mark_price == pytest.approx(50.0)
    assert position.opened_at == STAMP
    assert position.market_value == pytest.approx(5_000.0)
    assert position.exposure == pytest.approx(5_000.0)
    assert position.commission_paid == pytest.approx(1.0)
    assert position.fees_paid == pytest.approx(0.25)
    assert len(position.lots) == 1


def test_increasing_averages_the_entry_price() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 10.0))
    realized = position.apply_fill(buy(300.0, 20.0))

    assert realized == pytest.approx(0.0), "adding size realises nothing"
    assert position.quantity == pytest.approx(400.0)
    assert position.avg_price == pytest.approx(17.5)
    assert len(position.lots) == 2


def test_a_negative_fill_quantity_is_refused() -> None:
    position = Position(instrument=MSFT)
    with pytest.raises(ValidationError, match="positive fill quantity"):
        position.apply_fill(buy(-10.0, 50.0))


def test_another_instruments_fill_is_refused() -> None:
    """Booking it would balance the ledger with both rows wrong."""
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0))

    with pytest.raises(ValidationError, match="different instrument"):
        position.apply_fill(sell(100.0, 3.0, instrument_id=MSFT_CALL.instrument_id))

    assert position.quantity == pytest.approx(100.0)
    assert position.realized_pnl == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Realised P&L and lot matching
# --------------------------------------------------------------------------- #


def test_closing_a_long_realises_the_gain() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0))
    realized = position.apply_fill(sell(100.0, 60.0))

    assert realized == pytest.approx(1_000.0)
    assert position.realized_pnl == pytest.approx(1_000.0)
    assert not position.is_open


def test_closing_a_short_realises_the_fall() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(sell(100.0, 60.0))
    realized = position.apply_fill(buy(100.0, 50.0))

    assert realized == pytest.approx(1_000.0)
    assert not position.is_open


def test_partial_exit_matches_the_oldest_lot_first() -> None:
    """FIFO: selling 100 out of 10@100 + 20@100 realises the 10s, and what is
    left is priced at 20, not at the 15 average."""
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 10.0))
    position.apply_fill(buy(100.0, 20.0))
    assert position.avg_price == pytest.approx(15.0)

    realized = position.apply_fill(sell(100.0, 30.0))

    assert realized == pytest.approx(2_000.0), "(30 - 10) * 100, the oldest lot"
    assert position.quantity == pytest.approx(100.0)
    assert position.avg_price == pytest.approx(20.0)
    assert len(position.lots) == 1


def test_an_exit_spanning_two_lots_realises_both() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 10.0))
    position.apply_fill(buy(100.0, 20.0))

    realized = position.apply_fill(sell(150.0, 30.0))

    # 100 @ 10 -> 2000, then 50 @ 20 -> 500.
    assert realized == pytest.approx(2_500.0)
    assert position.quantity == pytest.approx(50.0)
    assert position.avg_price == pytest.approx(20.0)


def test_the_option_multiplier_scales_realised_pnl() -> None:
    position = Position(instrument=MSFT_CALL)
    position.apply_fill(buy(2.0, 3.00, instrument_id=MSFT_CALL.instrument_id))
    realized = position.apply_fill(sell(2.0, 4.50, instrument_id=MSFT_CALL.instrument_id))

    assert realized == pytest.approx(300.0), "1.50 x 2 contracts x 100"


def test_flip_realises_the_whole_old_side_and_reopens_at_the_fill() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0))

    realized = position.apply_fill(sell(150.0, 60.0))

    assert realized == pytest.approx(1_000.0), "only the 100 long units realise"
    assert position.quantity == pytest.approx(-50.0)
    assert position.side is PositionSide.SHORT
    assert position.avg_price == pytest.approx(60.0), "the new side entered here"
    assert len(position.lots) == 1
    assert position.opened_at == STAMP


def test_going_flat_clears_the_entry_but_keeps_the_running_totals() -> None:
    """Otherwise a re-entry on the same instrument inherits the previous round
    trip's entry price and excursions."""
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0, commission=1.0))
    position.apply_fill(sell(100.0, 60.0, commission=1.0))

    assert position.avg_price == pytest.approx(0.0)
    assert position.opened_at is None
    assert not position.lots
    assert position.realized_pnl == pytest.approx(1_000.0), "cumulative, kept"
    assert position.commission_paid == pytest.approx(2.0), "cumulative, kept"

    position.apply_fill(buy(10.0, 70.0))
    assert position.avg_price == pytest.approx(70.0)
    assert position.realized_pnl == pytest.approx(1_000.0)


def test_a_position_seeded_without_lots_still_realises_against_average_cost() -> None:
    seeded = Position(instrument=MSFT, quantity=100.0, avg_price=50.0, mark_price=50.0)

    realized = seeded.apply_fill(sell(100.0, 55.0))

    assert realized == pytest.approx(500.0)
    assert not seeded.is_open


# --------------------------------------------------------------------------- #
# Marking
# --------------------------------------------------------------------------- #


def test_marking_moves_value_and_unrealised_pnl() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0))
    position.mark(55.0, LATER)

    assert position.mark_price == pytest.approx(55.0)
    assert position.last_update == LATER
    assert position.market_value == pytest.approx(5_500.0)
    assert position.unrealized_pnl == pytest.approx(500.0)
    assert position.unrealized_pnl_pct == pytest.approx(0.10)


def test_a_short_gains_when_the_price_falls() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(sell(100.0, 50.0))
    position.mark(45.0, LATER)

    assert position.market_value == pytest.approx(-4_500.0), "signed"
    assert position.exposure == pytest.approx(4_500.0), "absolute"
    assert position.unrealized_pnl == pytest.approx(500.0)


def test_excursion_extremes_invert_with_the_side() -> None:
    long = Position(instrument=MSFT)
    long.apply_fill(buy(10.0, 50.0))
    short = Position(instrument=MSFT)
    short.apply_fill(sell(10.0, 50.0))
    for price in (55.0, 45.0, 52.0):
        long.mark(price, LATER)
        short.mark(price, LATER)

    assert (long.max_favorable_price, long.max_adverse_price) == (55.0, 45.0)
    assert (short.max_favorable_price, short.max_adverse_price) == (45.0, 55.0)


def test_unrealised_pct_is_zero_rather_than_undefined_without_a_basis() -> None:
    assert Position(instrument=MSFT).unrealized_pnl_pct == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Carry
# --------------------------------------------------------------------------- #


def test_borrow_is_charged_on_shorts_only() -> None:
    long = Position(instrument=HARD_TO_BORROW)
    long.apply_fill(buy(100.0, 40.0, instrument_id=HARD_TO_BORROW.instrument_id))

    assert long.accrue_borrow(LATER, 1.0 / 252.0) == pytest.approx(0.0)
    assert long.borrow_cost_paid == pytest.approx(0.0)


def test_borrow_accrues_on_current_notional() -> None:
    position = Position(instrument=HARD_TO_BORROW)
    position.apply_fill(sell(100.0, 40.0, instrument_id=HARD_TO_BORROW.instrument_id))
    position.mark(50.0, LATER)  # squeezed: 5_000 of exposure, not 4_000

    charged = position.accrue_borrow(LATER, 1.0 / 252.0)

    assert charged == pytest.approx(5_000.0 * 0.25 / 252.0)
    assert position.borrow_cost_paid == pytest.approx(charged)


def test_a_freely_borrowable_short_costs_nothing() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(sell(100.0, 50.0))

    assert position.accrue_borrow(LATER, 1.0 / 252.0) == pytest.approx(0.0)


def test_total_pnl_nets_costs_and_adds_carry() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0, commission=1.0, fees=0.25))
    position.mark(55.0, LATER)
    position.dividends_received = 30.0
    position.borrow_cost_paid = 2.0

    assert position.total_pnl == pytest.approx(500.0 + 30.0 - 1.0 - 0.25 - 2.0)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


LOCAL_CLOSE = datetime(2023, 3, 28, 16, 0, tzinfo=ZoneInfo("America/New_York"))
NAIVE = datetime(2023, 3, 28, 20, 0)  # noqa: DTZ001


def test_a_fill_stamps_the_position_in_utc() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 50.0, timestamp=LOCAL_CLOSE))

    assert position.opened_at == STAMP
    assert position.last_update == STAMP


@pytest.mark.parametrize(
    "act",
    [
        lambda p, ts: p.mark(55.0, ts),
        lambda p, ts: p.accrue_borrow(ts, 1.0 / 252.0),
    ],
)
def test_naive_instants_are_rejected_by_the_mutating_methods(act: object) -> None:
    position = Position(instrument=HARD_TO_BORROW)
    position.apply_fill(sell(100.0, 40.0, instrument_id=HARD_TO_BORROW.instrument_id))

    with pytest.raises(ValidationError, match="Naive datetime"):
        act(position, NAIVE)  # type: ignore[operator]

    act(position, LOCAL_CLOSE)  # type: ignore[operator]
    assert position.last_update == STAMP


@pytest.mark.parametrize("field_name", ["entry_time", "exit_time"])
def test_trade_ends_are_normalised_to_utc(field_name: str) -> None:
    row = trade(**{field_name: LOCAL_CLOSE})

    assert getattr(row, field_name) == STAMP
    with pytest.raises(ValidationError, match="Naive datetime"):
        trade(**{field_name: NAIVE})


def test_a_holding_period_never_mixes_naive_and_aware_ends() -> None:
    row = trade(entry_time=LOCAL_CLOSE, exit_time=LATER)

    assert row.holding_period == timedelta(days=3)


# --------------------------------------------------------------------------- #
# Trade rows
# --------------------------------------------------------------------------- #


def test_holding_period_and_winner_flag() -> None:
    won = trade()

    assert won.holding_period == timedelta(days=3)
    assert won.is_winner


def test_a_gross_win_eaten_by_costs_is_not_a_winner() -> None:
    assert not trade(gross_pnl=5.0, commission=4.0, fees=2.0, net_pnl=-1.0).is_winner


def test_r_multiple_needs_a_recorded_risk() -> None:
    assert trade().r_multiple is None, "no stop, no R"
    assert trade(metadata={INITIAL_RISK_KEY: 500.0}).r_multiple == pytest.approx(1.995)
    assert trade(metadata={INITIAL_RISK_KEY: -500.0}).r_multiple == pytest.approx(1.995)


@pytest.mark.parametrize("recorded", [0.0, "500", None, True])
def test_unusable_recorded_risk_yields_no_r_multiple(recorded: object) -> None:
    assert trade(metadata={INITIAL_RISK_KEY: recorded}).r_multiple is None


# --------------------------------------------------------------------------- #
# Option trade rows
# --------------------------------------------------------------------------- #


def option_trade(**overrides: object) -> OptionTrade:
    kwargs: dict[str, object] = {
        "trade_id": TradeId("T-2"),
        "instrument_id": MSFT_CALL.instrument_id,
        "symbol": MSFT_CALL.symbol,
        "asset_class": AssetClass.OPTION,
        "direction": PositionSide.SHORT,
        "quantity": 2.0,
        "entry_time": STAMP,
        "entry_price": 3.0,
        "exit_time": LATER,
        "exit_price": 1.0,
        "gross_pnl": 400.0,
        "commission": 2.0,
        "fees": 0.2,
        "net_pnl": 397.8,
        "return_pct": 0.663,
        "close_reason": TradeCloseReason.SIGNAL,
        "underlying_symbol": Symbol("MSFT"),
        "right": OptionRight.CALL,
        "strike": 280.0,
        "expiry": date(2023, 4, 21),
    }
    kwargs.update(overrides)
    return OptionTrade(**kwargs)  # type: ignore[arg-type]


def test_premium_is_collected_when_short_and_paid_when_long() -> None:
    assert option_trade().premium_collected == pytest.approx(600.0)
    assert option_trade(direction=PositionSide.LONG).premium_collected == pytest.approx(-600.0)


def test_underlying_return_needs_both_ends() -> None:
    both = option_trade(underlying_price_at_entry=275.0, underlying_price_at_exit=264.0)
    assert both.underlying_return_pct == pytest.approx(-0.04)

    assert option_trade().underlying_return_pct is None, "unknown, not flat"
    assert option_trade(underlying_price_at_entry=275.0).underlying_return_pct is None


def test_an_option_trade_is_still_a_trade() -> None:
    """The options log inherits every column of the equity log (Outputs req. 2)."""
    expired = option_trade(close_reason=TradeCloseReason.OPTION_EXPIRY_WORTHLESS)

    assert isinstance(expired, Trade)
    assert expired.holding_period == timedelta(days=3)
    assert expired.is_winner
    assert expired.multiplier == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Short-side lot arithmetic
# --------------------------------------------------------------------------- #


def test_partially_covering_a_short_matches_the_oldest_lot_first() -> None:
    """The short side keeps signed lots, so FIFO consumption has to preserve the
    sign as it partially draws a lot down."""
    position = Position(instrument=MSFT)
    position.apply_fill(sell(100.0, 60.0))
    position.apply_fill(sell(100.0, 80.0))
    assert position.avg_price == pytest.approx(70.0)

    realized = position.apply_fill(buy(150.0, 50.0))

    # 100 short @ 60 -> 1000, then 50 short @ 80 -> 1500.
    assert realized == pytest.approx(2_500.0)
    assert position.quantity == pytest.approx(-50.0)
    assert position.avg_price == pytest.approx(80.0), "the 60s were consumed first"
    assert [lot.quantity for lot in position.lots] == [pytest.approx(-50.0)]


def test_flipping_from_short_to_long_reopens_on_the_other_side() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(sell(100.0, 60.0))

    realized = position.apply_fill(buy(150.0, 50.0))

    assert realized == pytest.approx(1_000.0), "only the 100 short units realise"
    assert position.quantity == pytest.approx(50.0)
    assert position.side is PositionSide.LONG
    assert position.avg_price == pytest.approx(50.0)
    assert [lot.quantity for lot in position.lots] == [pytest.approx(50.0)]


def test_adding_after_a_partial_exit_averages_against_what_survived() -> None:
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 10.0))
    position.apply_fill(buy(100.0, 20.0))
    position.apply_fill(sell(100.0, 30.0))  # consumes the 10s, leaving 100 @ 20

    position.apply_fill(buy(100.0, 30.0))

    assert position.quantity == pytest.approx(200.0)
    assert position.avg_price == pytest.approx(25.0), "20 and 30, not the original 15"


# --------------------------------------------------------------------------- #
# Excursions
# --------------------------------------------------------------------------- #


def test_a_fill_counts_towards_the_excursions_like_any_other_print() -> None:
    """Scaling out into a spike is a price the position genuinely traded through.
    Ignoring it leaves MFE reading the last quiet mark."""
    position = Position(instrument=MSFT)
    position.apply_fill(buy(100.0, 10.0))
    position.mark(11.0, LATER)

    position.apply_fill(sell(50.0, 20.0))

    assert position.max_favorable_price == pytest.approx(20.0)


def test_a_worthless_mark_is_recorded_rather_than_read_as_unset() -> None:
    """0.0 is a real price for an option that expired worthless; a zero sentinel
    would throw away the very extreme the trade turned on."""
    seeded = Position(instrument=MSFT, quantity=100.0, avg_price=50.0)
    seeded.mark(0.0, STAMP)
    seeded.mark(5.0, LATER)

    assert seeded.max_adverse_price == pytest.approx(0.0)
    assert seeded.max_favorable_price == pytest.approx(5.0)


def test_marking_a_flat_position_moves_the_price_but_tracks_no_excursion() -> None:
    position = Position(instrument=MSFT)
    position.mark(42.0, LATER)

    assert position.mark_price == pytest.approx(42.0)
    assert position.last_update == LATER
    assert position.max_favorable_price == pytest.approx(0.0)
    assert position.opened_at is None


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_a_fill_for_another_instrument_is_refused() -> None:
    """Booking it here would realise one symbol's P&L against another's cost
    basis, and the ledger would still balance while both rows were wrong."""
    position = Position(instrument=MSFT)
    with pytest.raises(ValidationError, match="different instrument"):
        position.apply_fill(buy(10.0, 50.0, instrument_id=InstrumentId("EQ:AAPL")))


@pytest.mark.parametrize("fraction", [0.0, -1.0 / 252.0])
def test_borrow_is_not_charged_for_a_non_positive_bar_fraction(fraction: float) -> None:
    position = Position(instrument=HARD_TO_BORROW)
    position.apply_fill(sell(100.0, 40.0, instrument_id=HARD_TO_BORROW.instrument_id))

    assert position.accrue_borrow(LATER, fraction) == pytest.approx(0.0)
    assert position.borrow_cost_paid == pytest.approx(0.0)
