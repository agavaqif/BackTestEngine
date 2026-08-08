"""Option expiry: worthless, exercise, assignment, and the cash they move."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from sigmaloop.domain.bar import Bar, MarketSnapshot
from sigmaloop.domain.instrument import Equity, OptionContract
from sigmaloop.domain.position import Position
from sigmaloop.errors import DataNotAvailableError, ValidationError
from sigmaloop.execution.expiry import ExpiryPolicy, StandardExpiryEngine
from sigmaloop.types import (
    InstrumentId,
    OptionRight,
    OptionStyle,
    OrderSide,
    SettlementType,
    Symbol,
    TradeCloseReason,
)

EXPIRY = date(2023, 4, 21)
EXPIRY_CLOSE = datetime(2023, 4, 21, 20, 0, tzinfo=UTC)
DAY_BEFORE = EXPIRY_CLOSE - timedelta(days=1)

MSFT = Equity(instrument_id=InstrumentId("EQ:MSFT"), symbol=Symbol("MSFT"))


def contract(
    right: OptionRight = OptionRight.CALL,
    strike: float = 280.0,
    *,
    settlement: SettlementType = SettlementType.PHYSICAL,
    style: OptionStyle = OptionStyle.AMERICAN,
    expiry: date = EXPIRY,
) -> OptionContract:
    return OptionContract(
        instrument_id=OptionContract.make_id(Symbol("MSFT"), expiry, right, strike),
        symbol=Symbol("MSFT_OPT"),
        underlying_id=MSFT.instrument_id,
        underlying_symbol=Symbol("MSFT"),
        right=right,
        strike=strike,
        expiry=expiry,
        settlement=settlement,
        style=style,
    )


def position(instrument: OptionContract, quantity: float, avg_price: float = 5.0) -> Position:
    return Position(instrument=instrument, quantity=quantity, avg_price=avg_price)


def snapshot(
    underlying: float | None = 300.0,
    *,
    at: datetime = EXPIRY_CLOSE,
    is_session_close: bool = True,
) -> MarketSnapshot:
    bars = {}
    if underlying is not None:
        bars[MSFT.instrument_id] = Bar(
            instrument_id=MSFT.instrument_id,
            timestamp=at,
            open=underlying,
            high=underlying,
            low=underlying,
            close=underlying,
            volume=1_000.0,
        )
    return MarketSnapshot(timestamp=at, bars=bars, is_session_close=is_session_close)


def engine(**policy: object) -> StandardExpiryEngine:
    return StandardExpiryEngine(ExpiryPolicy(**policy), seed=7)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_only_expiring_contracts_are_selected() -> None:
    expiring = position(contract(), 1.0)
    later = position(contract(expiry=date(2023, 6, 16)), 1.0)
    equity = Position(instrument=MSFT, quantity=100.0)
    picked = engine().expiring_positions([expiring, later, equity], snapshot())
    assert picked == (expiring,)


def test_settlement_waits_for_the_session_close() -> None:
    held = position(contract(), 1.0)
    intraday = snapshot(at=EXPIRY_CLOSE - timedelta(hours=2), is_session_close=False)
    assert engine().expiring_positions([held], intraday) == ()


def test_a_flat_position_is_not_expiring() -> None:
    assert engine().expiring_positions([position(contract(), 0.0)], snapshot()) == ()


def test_the_policy_can_flatten_before_expiry() -> None:
    held = position(contract(), 1.0)
    resolver = engine(close_before_expiry_bars=2)
    two_days_out = snapshot(at=EXPIRY_CLOSE - timedelta(days=2))
    week_out = snapshot(at=EXPIRY_CLOSE - timedelta(days=7))
    assert resolver.positions_to_close_early([held], two_days_out) == (held,)
    assert resolver.positions_to_close_early([held], week_out) == ()
    # Off by default: nothing is closed early.
    assert engine().positions_to_close_early([held], two_days_out) == ()


# --------------------------------------------------------------------------- #
# Worthless expiry
# --------------------------------------------------------------------------- #


def test_an_out_of_the_money_long_expires_worthless() -> None:
    outcome = engine().resolve(position(contract(), 1.0), snapshot(underlying=250.0))
    assert outcome.reason is TradeCloseReason.OPTION_EXPIRY_WORTHLESS
    assert outcome.cash_impact == 0.0
    assert outcome.underlying_quantity_delta == 0.0
    (fill,) = outcome.fills
    assert (fill.side, fill.price, fill.quantity) == (OrderSide.SELL, 0.0, 1.0)


def test_barely_in_the_money_is_left_unexercised() -> None:
    """Below the auto-exercise threshold, nobody exercises for half a cent."""
    outcome = engine(auto_exercise_threshold=0.01).resolve(
        position(contract(), 1.0), snapshot(underlying=280.005)
    )
    assert outcome.reason is TradeCloseReason.OPTION_EXPIRY_WORTHLESS


def test_a_short_the_holder_does_not_exercise_keeps_the_premium() -> None:
    outcome = engine(expiry_assignment_probability=0.0).resolve(
        position(contract(), -1.0), snapshot(underlying=300.0)
    )
    assert outcome.reason is TradeCloseReason.OPTION_EXPIRY_WORTHLESS
    assert outcome.underlying_quantity_delta == 0.0


# --------------------------------------------------------------------------- #
# Physical settlement
# --------------------------------------------------------------------------- #


def test_an_exercised_long_call_buys_shares_at_the_strike() -> None:
    outcome = engine().resolve(position(contract(), 1.0), snapshot(underlying=300.0))
    assert outcome.reason is TradeCloseReason.OPTION_EXERCISE
    assert outcome.settlement_price == pytest.approx(20.0)
    assert outcome.underlying_quantity_delta == 100.0
    assert outcome.cash_impact == pytest.approx(-28_000.0)
    option_leg, share_leg = outcome.fills
    # The contract itself expires at nothing; the value arrives as shares, and
    # booking both would count the same $20 twice.
    assert (option_leg.price, option_leg.side) == (0.0, OrderSide.SELL)
    assert (share_leg.side, share_leg.quantity, share_leg.price) == (OrderSide.BUY, 100.0, 280.0)
    assert share_leg.instrument_id == MSFT.instrument_id


def test_an_exercised_long_put_sells_shares_at_the_strike() -> None:
    outcome = engine().resolve(position(contract(OptionRight.PUT), 1.0), snapshot(underlying=250.0))
    assert outcome.underlying_quantity_delta == -100.0
    assert outcome.cash_impact == pytest.approx(28_000.0)
    assert outcome.fills[1].side is OrderSide.SELL


def test_an_assigned_short_call_delivers_the_shares() -> None:
    outcome = engine().resolve(position(contract(), -2.0), snapshot(underlying=300.0))
    assert outcome.reason is TradeCloseReason.OPTION_ASSIGNMENT
    assert outcome.underlying_quantity_delta == -200.0
    assert outcome.cash_impact == pytest.approx(56_000.0)
    option_leg, share_leg = outcome.fills
    assert option_leg.side is OrderSide.BUY  # closing the short contract
    assert share_leg.side is OrderSide.SELL


def test_an_assigned_short_put_takes_delivery() -> None:
    outcome = engine().resolve(
        position(contract(OptionRight.PUT), -1.0), snapshot(underlying=250.0)
    )
    assert outcome.underlying_quantity_delta == 100.0
    assert outcome.cash_impact == pytest.approx(-28_000.0)


def test_exercise_fees_are_charged_per_contract() -> None:
    outcome = engine(exercise_fee_per_contract=0.50).resolve(
        position(contract(), 4.0), snapshot(underlying=300.0)
    )
    assert outcome.fees == pytest.approx(2.0)
    assert outcome.cash_impact == pytest.approx(-4 * 100 * 280.0 - 2.0)


# --------------------------------------------------------------------------- #
# Cash settlement
# --------------------------------------------------------------------------- #


def test_a_cash_settled_contract_pays_its_intrinsic_value() -> None:
    cash_contract = contract(settlement=SettlementType.CASH)
    outcome = engine().resolve(position(cash_contract, 1.0), snapshot(underlying=300.0))
    assert outcome.underlying_quantity_delta == 0.0
    assert outcome.cash_impact == pytest.approx(2_000.0)
    (fill,) = outcome.fills
    assert fill.price == pytest.approx(20.0)


def test_physical_settlement_can_be_converted_to_cash() -> None:
    """For runs with no equity data loaded, shares cannot be delivered."""
    outcome = engine(allow_physical_settlement=False).resolve(
        position(contract(), 1.0), snapshot(underlying=300.0)
    )
    assert outcome.underlying_quantity_delta == 0.0
    assert outcome.cash_impact == pytest.approx(2_000.0)


def test_a_cash_settled_short_pays_out() -> None:
    cash_contract = contract(settlement=SettlementType.CASH)
    outcome = engine().resolve(position(cash_contract, -1.0), snapshot(underlying=300.0))
    assert outcome.cash_impact == pytest.approx(-2_000.0)
    assert outcome.fills[0].side is OrderSide.BUY


# --------------------------------------------------------------------------- #
# Early assignment
# --------------------------------------------------------------------------- #


def test_early_assignment_is_off_by_default() -> None:
    held = position(contract(), -1.0)
    assert engine().check_early_assignment(held, snapshot(underlying=300.0)) is None


def test_a_certain_early_assignment_resolves_the_short() -> None:
    outcome = engine(early_assignment_probability=1.0).check_early_assignment(
        position(contract(), -1.0), snapshot(underlying=300.0, at=DAY_BEFORE)
    )
    assert outcome is not None
    assert outcome.reason is TradeCloseReason.OPTION_ASSIGNMENT


def test_only_short_in_the_money_american_contracts_are_assignable() -> None:
    resolver = engine(early_assignment_probability=1.0)
    market = snapshot(underlying=300.0, at=DAY_BEFORE)
    assert resolver.check_early_assignment(position(contract(), 1.0), market) is None
    otm = snapshot(underlying=250.0, at=DAY_BEFORE)
    assert resolver.check_early_assignment(position(contract(), -1.0), otm) is None
    european = contract(style=OptionStyle.EUROPEAN)
    assert resolver.check_early_assignment(position(european, -1.0), market) is None


def test_assignment_draws_are_reproducible_for_one_seed() -> None:
    """Sweep points are only comparable if the randomness is not."""
    market = snapshot(underlying=300.0, at=DAY_BEFORE)

    def draws(seed: int) -> list[bool]:
        resolver = StandardExpiryEngine(ExpiryPolicy(early_assignment_probability=0.5), seed=seed)
        return [
            resolver.check_early_assignment(position(contract(), -1.0), market) is not None
            for _ in range(20)
        ]

    assert draws(7) == draws(7)
    assert draws(7) != draws(8)


# --------------------------------------------------------------------------- #
# Settlement prices
# --------------------------------------------------------------------------- #


def test_the_last_seen_underlying_price_is_carried_into_the_expiry_bar() -> None:
    resolver = engine()
    held = position(contract(), 1.0)
    # Seen the day before, absent on the expiry bar itself.
    resolver.expiring_positions([held], snapshot(underlying=300.0, at=DAY_BEFORE))
    outcome = resolver.resolve(held, snapshot(underlying=None))
    assert outcome.underlying_price == 300.0
    assert outcome.reason is TradeCloseReason.OPTION_EXERCISE


def test_an_underlying_that_was_never_priced_is_an_error_not_a_guess() -> None:
    with pytest.raises(DataNotAvailableError, match="never been priced"):
        engine().resolve(position(contract(), 1.0), snapshot(underlying=None))


def test_only_options_expire() -> None:
    with pytest.raises(ValidationError, match="not an option"):
        engine().resolve(Position(instrument=MSFT, quantity=100.0), snapshot())


def test_a_flat_position_cannot_be_resolved() -> None:
    with pytest.raises(ValidationError, match="flat position"):
        engine().resolve(position(contract(), 0.0), snapshot())
