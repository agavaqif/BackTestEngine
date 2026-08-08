"""Instruments: identity invariants and the tick/lot grids fills are snapped to."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigmaloop.domain.instrument import Equity, InstrumentRegistry, OptionContract
from sigmaloop.errors import InstrumentNotFoundError, ValidationError
from sigmaloop.types import AssetClass, InstrumentId, OptionRight, SettlementType, Symbol


def equity(**overrides: object) -> Equity:
    kwargs: dict[str, object] = {
        "instrument_id": InstrumentId("EQ:MSFT"),
        "symbol": Symbol("MSFT"),
    }
    kwargs.update(overrides)
    return Equity(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Construction invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"symbol": Symbol("")}, "non-empty ticker"),
        ({"symbol": Symbol("   ")}, "non-empty ticker"),
        ({"instrument_id": InstrumentId("")}, "instrument_id must be non-empty"),
        ({"multiplier": 0.0}, "multiplier must be positive"),
        ({"tick_size": 0.0}, "tick_size must be positive"),
        ({"tick_size": -0.01}, "tick_size must be positive"),
        ({"lot_size": 0.0}, "lot_size must be positive"),
    ],
)
def test_an_unusable_instrument_is_refused_at_construction(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        equity(**overrides)


def test_ids_are_built_canonically_from_the_ticker() -> None:
    assert Equity.make_id(Symbol("  msft ")) == "EQ:MSFT"


def test_an_equity_notional_is_just_price_times_quantity() -> None:
    assert equity().notional(10.0, 3.0) == pytest.approx(30.0)


def test_a_delisted_equity_is_expired_on_and_after_its_last_day() -> None:
    delisted = equity(delisted_on=date(2023, 6, 15))

    assert not delisted.is_expired(datetime(2023, 6, 14, tzinfo=UTC))
    assert delisted.is_expired(datetime(2023, 6, 15, tzinfo=UTC))
    assert delisted.is_expired(datetime(2024, 1, 1, tzinfo=UTC))


def option(**overrides: object) -> OptionContract:
    kwargs: dict[str, object] = {
        "instrument_id": InstrumentId("OPT:MSFT:20230421:C:00280000"),
        "symbol": Symbol("MSFT230421C00280000"),
        "underlying_id": InstrumentId("EQ:MSFT"),
        "underlying_symbol": Symbol("MSFT"),
        "right": OptionRight.CALL,
        "strike": 280.0,
        "expiry": date(2023, 4, 21),
    }
    kwargs.update(overrides)
    return OptionContract(**kwargs)  # type: ignore[arg-type]


def test_an_option_carries_the_contract_multiplier() -> None:
    contract = option()

    assert contract.asset_class is AssetClass.OPTION
    assert contract.multiplier == pytest.approx(100.0)
    assert contract.notional(2.50, 3.0) == pytest.approx(750.0)


# --------------------------------------------------------------------------- #
# Option contracts
# --------------------------------------------------------------------------- #


def test_option_ids_follow_the_occ_layout() -> None:
    built = OptionContract.make_id(Symbol("spy"), date(2025, 1, 17), OptionRight.CALL, 500.0)
    assert built == "OPT:SPY:20250117:C:00500000"
    put = OptionContract.make_id(Symbol("SPY"), date(2025, 1, 17), OptionRight.PUT, 7.5)
    assert put == "OPT:SPY:20250117:P:00007500"


def test_a_contract_needs_a_positive_strike() -> None:
    """moneyness() divides by it, and a zero-strike exercise is free shares."""
    with pytest.raises(ValidationError, match="strike must be a positive"):
        option(strike=0.0)


def test_an_occ_symbol_round_trips_through_make_id() -> None:
    """``from_occ`` is the inverse of ``make_id``: what a feed names a contract
    and what the registry keys it by have to agree, or one run holds two objects
    for one contract."""
    parsed = OptionContract.from_occ("SPY   250117C00500000")

    assert parsed.underlying_symbol == Symbol("SPY")
    assert parsed.underlying_id == "EQ:SPY"
    assert parsed.expiry == date(2025, 1, 17)
    assert parsed.right is OptionRight.CALL
    assert parsed.strike == pytest.approx(500.0)
    assert parsed.instrument_id == OptionContract.make_id(
        Symbol("SPY"), date(2025, 1, 17), OptionRight.CALL, 500.0
    )


def test_an_occ_root_is_read_from_the_fixed_width_tail() -> None:
    """The last 15 characters are fixed; the root is whatever precedes them. A
    feed that stripped the padding still parses, and so does a 4-letter root."""
    padded = OptionContract.from_occ("SPY   250117P00012500")
    stripped = OptionContract.from_occ("SPY250117P00012500")
    four_letter = OptionContract.from_occ("MSFT230421P00012500")

    assert padded.instrument_id == stripped.instrument_id
    assert padded.strike == pytest.approx(12.5), "strike is in thousandths"
    assert padded.right is OptionRight.PUT
    assert four_letter.underlying_symbol == Symbol("MSFT")


def test_occ_overrides_supply_what_the_symbol_cannot_carry() -> None:
    """Style, settlement and an adjusted multiplier are not in the 21 characters."""
    cash_settled = OptionContract.from_occ(
        "SPX   250117C05000000", settlement=SettlementType.CASH, multiplier=10.0
    )

    assert cash_settled.settlement is SettlementType.CASH
    assert cash_settled.multiplier == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("symbol", "why"),
    [
        ("250117C00500000", "no root before the tail"),
        ("SPY   250117X00500000", "right is neither C nor P"),
        ("SPY   251317C00500000", "month 13"),
        ("SPY   2501", "shorter than the fixed tail"),
        ("SPY   250117C0050000A", "non-numeric strike"),
    ],
)
def test_a_malformed_occ_symbol_is_refused(symbol: str, why: str) -> None:
    """Silently mis-parsing one would book trades against the wrong contract."""
    with pytest.raises(ValidationError):
        OptionContract.from_occ(symbol)


def test_a_contract_trades_through_its_expiry_date() -> None:
    contract = option()

    assert not contract.is_expired(datetime(2023, 4, 21, 13, 30, tzinfo=UTC))
    assert contract.is_expired(datetime(2023, 4, 22, tzinfo=UTC))


def test_days_to_expiry_goes_negative_after_the_fact() -> None:
    contract = option()

    assert contract.days_to_expiry(datetime(2023, 4, 21, 20, tzinfo=UTC)) == 0
    assert contract.days_to_expiry(datetime(2023, 4, 14, tzinfo=UTC)) == 7
    assert contract.days_to_expiry(datetime(2023, 4, 28, tzinfo=UTC)) == -7


@pytest.mark.parametrize(
    ("right", "underlying", "intrinsic", "itm"),
    [
        (OptionRight.CALL, 300.0, 20.0, True),
        (OptionRight.CALL, 250.0, 0.0, False),
        (OptionRight.PUT, 250.0, 30.0, True),
        (OptionRight.PUT, 300.0, 0.0, False),
        (OptionRight.CALL, 280.0, 0.0, False),
    ],
)
def test_intrinsic_value_is_zero_out_of_the_money(
    right: OptionRight, underlying: float, intrinsic: float, itm: bool
) -> None:
    contract = option(right=right)

    assert contract.intrinsic_value(underlying) == pytest.approx(intrinsic)
    assert contract.is_itm(underlying) is itm


def test_moneyness_reads_above_one_in_the_money_for_either_right() -> None:
    call = option()
    put = option(right=OptionRight.PUT)

    assert call.moneyness(280.0) == pytest.approx(1.0)
    assert call.moneyness(308.0) == pytest.approx(1.1)
    assert put.moneyness(280.0) == pytest.approx(1.0)
    assert put.moneyness(254.5454545) == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_the_registry_interns_one_object_per_instrument() -> None:
    registry = InstrumentRegistry()
    first = registry.register(equity())
    second = registry.register(equity())

    assert first is second
    assert len(registry) == 1
    assert registry.get(InstrumentId("EQ:MSFT")) is first
    assert InstrumentId("EQ:MSFT") in registry


def test_two_instruments_cannot_claim_one_id() -> None:
    """Every position and fill keys off the id; rebinding it would reprice
    holdings the strategy already opened."""
    registry = InstrumentRegistry()
    registry.register(equity())
    with pytest.raises(ValidationError, match="same instrument_id"):
        registry.register(equity(symbol=Symbol("MSFT"), tick_size=0.05))


def test_an_unknown_id_is_named_in_the_error() -> None:
    with pytest.raises(InstrumentNotFoundError, match="EQ:NOPE"):
        InstrumentRegistry().get(InstrumentId("EQ:NOPE"))
    assert InstrumentRegistry().try_get(InstrumentId("EQ:NOPE")) is None


def test_a_ticker_resolves_to_the_equity_and_its_options() -> None:
    registry = InstrumentRegistry()
    share = registry.register(equity())
    call = registry.register(option())

    assert registry.by_symbol(Symbol("MSFT")) == (share,)
    assert registry.options_on(InstrumentId("EQ:MSFT")) == (call,)
    assert registry.options_on(InstrumentId("EQ:AAPL")) == ()


# --------------------------------------------------------------------------- #
# Tick grid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("price", "expected"),
    [(0.125, 0.13), (0.375, 0.38), (0.625, 0.63), (0.875, 0.88)],
)
def test_half_ticks_always_round_the_same_way(price: float, expected: float) -> None:
    """Python's round() is banker's rounding: it would send 0.125 down and 0.375
    up off the same grid, so a series landing on half-ticks would drift in a
    direction no reader could predict."""
    assert equity().round_price(price) == pytest.approx(expected)


def test_the_tick_grid_is_symmetric_about_zero() -> None:
    grid = equity()

    assert grid.round_price(-0.125) == pytest.approx(-0.13)
    assert grid.round_price(-10.004) == pytest.approx(-10.0)


def test_snapping_absorbs_the_representation_error_in_the_tick_size() -> None:
    """0.01 is not exactly representable, so ticks * tick_size drifts into 1e-17
    and would leak into every reported price."""
    assert equity().round_price(10.004999) == 10.0
    assert equity().round_price(19.99) == 19.99


def test_a_coarser_tick_still_lands_on_the_grid() -> None:
    quarters = equity(tick_size=0.25)

    assert quarters.round_price(10.30) == pytest.approx(10.25)
    assert quarters.round_price(10.125) == pytest.approx(10.25), "half-tick, away from zero"


# --------------------------------------------------------------------------- #
# Lot grid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("quantity", "expected"), [(2.9999999999, 3.0), (3.0, 3.0), (2.5, 2.0), (0.9, 0.0)]
)
def test_lots_floor_toward_zero_so_a_sizer_cannot_overspend(
    quantity: float, expected: float
) -> None:
    assert equity().round_quantity(quantity) == pytest.approx(expected)


def test_a_short_is_floored_by_magnitude_not_toward_minus_infinity() -> None:
    assert equity().round_quantity(-2.7) == pytest.approx(-2.0)


def test_rounding_a_short_away_to_nothing_does_not_produce_negative_zero() -> None:
    """-0.0 compares equal to 0.0, so no assertion catches it, and it prints as
    "-0" in every report that reaches a human."""
    flattened = equity().round_quantity(-0.4)

    assert flattened == 0.0
    assert str(flattened) == "0.0"


def test_the_lot_epsilon_rescues_a_clean_division_that_float_spoiled() -> None:
    """0.3 / 0.1 == 2.9999999999999996 would otherwise floor three lots to two."""
    assert equity().round_quantity(0.3 / 0.1) == pytest.approx(3.0)


def test_fractional_lots_are_honoured_when_configured() -> None:
    fractional = equity(lot_size=0.001)

    assert fractional.round_quantity(1.9999) == pytest.approx(1.999)
    assert fractional.round_quantity(0.0005) == pytest.approx(0.0)
