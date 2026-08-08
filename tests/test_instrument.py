"""Instruments: identity invariants and the tick/lot grids fills are snapped to."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigmaloop.domain.instrument import Equity, OptionContract
from sigmaloop.errors import ValidationError
from sigmaloop.types import AssetClass, InstrumentId, OptionRight, Symbol


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


def test_an_option_carries_the_contract_multiplier() -> None:
    contract = OptionContract(
        instrument_id=InstrumentId("OPT:MSFT:20230421:C:00280000"),
        symbol=Symbol("MSFT230421C00280000"),
        underlying_id=InstrumentId("EQ:MSFT"),
        underlying_symbol=Symbol("MSFT"),
        right=OptionRight.CALL,
        strike=280.0,
        expiry=date(2023, 4, 21),
    )

    assert contract.asset_class is AssetClass.OPTION
    assert contract.multiplier == pytest.approx(100.0)


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
