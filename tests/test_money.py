"""Money helpers: the two rounding boundaries, and the divisions metrics make."""

from __future__ import annotations

import math

import pytest

from sigmaloop.errors import ValidationError
from sigmaloop.utils.money import (
    is_close,
    notional,
    pct_change,
    round_money,
    round_to_lot,
    round_to_tick,
    safe_divide,
)


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(0.125, 0.13), (0.375, 0.38), (0.625, 0.63), (-0.125, -0.13), (0.0, 0.0), (10.0, 10.0)],
)
def test_money_rounds_half_away_from_zero(amount: float, expected: float) -> None:
    """Halves that are exact in binary, so the test measures the rule and not
    float representation: banker's rounding would send 0.125 down to 0.12 but
    0.375 up to 0.38, off one grid, in a direction no reader could predict."""
    assert round_money(amount) == pytest.approx(expected)


def test_money_rounding_leaves_a_broken_number_visible() -> None:
    assert math.isnan(round_money(math.nan))


@pytest.mark.parametrize(
    ("price", "mode", "expected"),
    [
        (100.024, "nearest", 100.02),
        (100.025, "nearest", 100.03),
        (100.021, "up", 100.03),
        (100.029, "down", 100.02),
        (-100.021, "up", -100.02),
        (-100.021, "down", -100.03),
    ],
)
def test_tick_rounding_is_directional_along_the_number_line(
    price: float, mode: str, expected: float
) -> None:
    assert round_to_tick(price, 0.01, mode) == pytest.approx(expected)


def test_a_price_already_on_the_grid_is_left_alone() -> None:
    """100.03 / 0.01 is 10002.999999999998; a bare ceil would add a tick."""
    assert round_to_tick(100.03, 0.01, "up") == 100.03
    assert round_to_tick(100.03, 0.01, "down") == 100.03


def test_tick_rounding_does_not_produce_negative_zero() -> None:
    assert str(round_to_tick(-0.001, 0.01, "nearest")) == "0.0"


def test_an_unusable_tick_or_mode_is_refused() -> None:
    with pytest.raises(ValidationError, match="tick_size"):
        round_to_tick(1.0, 0.0)
    with pytest.raises(ValidationError, match="rounding mode"):
        round_to_tick(1.0, 0.01, "sideways")


def test_lots_floor_so_a_sizer_cannot_overspend() -> None:
    assert round_to_lot(2.7, 1.0) == 2.0
    assert round_to_lot(-2.7, 1.0) == -2.0
    assert round_to_lot(0.3 / 0.1, 1.0) == pytest.approx(3.0)


def test_fractional_shares_pass_through_untouched() -> None:
    assert round_to_lot(2.7, 1.0, allow_fractional=True) == 2.7


def test_is_close_tolerates_accounting_dust() -> None:
    assert is_close(1e6, 1e6 + 1e-7)
    assert not is_close(1.0, 1.01)


def test_safe_divide_declines_rather_than_inventing_infinity() -> None:
    assert safe_divide(1.0, 0.0) is None
    assert safe_divide(1.0, 0.0, default=0.0) == 0.0
    assert safe_divide(1.0, math.inf) is None
    assert safe_divide(10.0, 4.0) == pytest.approx(2.5)


def test_pct_change_keeps_growth_from_nothing_visible() -> None:
    assert pct_change(100.0, 110.0) == pytest.approx(0.10)
    assert pct_change(0.0, 0.0) == 0.0
    assert pct_change(0.0, 5.0) == math.inf
    assert pct_change(0.0, -5.0) == -math.inf


def test_notional_is_the_one_place_the_multiplier_is_applied() -> None:
    assert notional(2.50, 3.0) == pytest.approx(7.5)
    assert notional(2.50, 3.0, 100.0) == pytest.approx(750.0)
