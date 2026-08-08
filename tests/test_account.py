"""Account state: cash movements, buying power under a margin model, equity rows."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from sigmaloop.domain.account import AccountState, CashFlow, EquityPoint
from sigmaloop.errors import ValidationError
from sigmaloop.types import Currency

STAMP = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)


def account(**overrides: object) -> AccountState:
    kwargs: dict[str, object] = {"initial_cash": 100_000.0, "cash": 100_000.0}
    kwargs.update(overrides)
    return AccountState(**kwargs)  # type: ignore[arg-type]


def equity_point(**overrides: object) -> EquityPoint:
    kwargs: dict[str, object] = {
        "timestamp": STAMP,
        "cash": 40_000.0,
        "positions_value": 60_000.0,
        "equity": 100_000.0,
    }
    kwargs.update(overrides)
    return EquityPoint(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Cash
# --------------------------------------------------------------------------- #


def test_defaults_are_a_clean_usd_account() -> None:
    fresh = account()

    assert fresh.currency is Currency.USD
    assert fresh.available_cash == pytest.approx(100_000.0)


def test_reserved_cash_is_not_available() -> None:
    """Two orders raised on the same bar must not both spend the same dollar."""
    committed = account(reserved_cash=25_000.0)

    assert committed.cash == pytest.approx(100_000.0)
    assert committed.available_cash == pytest.approx(75_000.0)


def test_credit_and_debit_move_cash() -> None:
    ledger = account()
    ledger.debit(30_000.0, reason="entry")
    ledger.credit(500.0, reason="dividend")

    assert ledger.cash == pytest.approx(70_500.0)


def test_debit_may_overdraw_rather_than_silently_drop_a_booked_fill() -> None:
    """The breach policy is decided pre-trade; refusing here would leave the
    ledger disagreeing with a fill that already happened."""
    ledger = account(cash=100.0)
    ledger.debit(250.0, reason="settled fill")

    assert ledger.cash == pytest.approx(-150.0)


@pytest.mark.parametrize(("action", "inverse"), [("credit", "debit"), ("debit", "credit")])
def test_direction_lives_in_the_method_name_not_the_sign(action: str, inverse: str) -> None:
    ledger = account()
    with pytest.raises(ValidationError, match=f"use {inverse}"):
        getattr(ledger, action)(-1.0, "backwards")


def test_a_nan_amount_is_refused_before_it_poisons_the_ledger() -> None:
    ledger = account()
    with pytest.raises(ValidationError):
        ledger.debit(float("nan"))
    assert ledger.cash == pytest.approx(100_000.0)


# --------------------------------------------------------------------------- #
# Buying power
# --------------------------------------------------------------------------- #


def test_a_cash_account_can_spend_only_its_cash() -> None:
    """Under MarginModel.CASH each long is booked at full notional, so
    ``margin_used`` cancels ``positions_value`` and only cash is left."""
    ledger = account(cash=40_000.0, margin_used=60_000.0)

    assert ledger.buying_power(positions_value=60_000.0, leverage=1.0) == pytest.approx(40_000.0)


def test_reg_t_doubles_equity_and_nets_off_margin_already_used() -> None:
    ledger = account(cash=40_000.0, margin_used=30_000.0)

    # (40_000 + 60_000) * 2 - 30_000
    assert ledger.buying_power(positions_value=60_000.0, leverage=2.0) == pytest.approx(170_000.0)


def test_reserved_cash_comes_off_buying_power_too() -> None:
    ledger = account(cash=100_000.0, reserved_cash=25_000.0)

    assert ledger.buying_power(positions_value=0.0, leverage=1.0) == pytest.approx(75_000.0)


def test_buying_power_floors_at_zero() -> None:
    blown = account(cash=-5_000.0, margin_used=10_000.0)

    assert blown.buying_power(positions_value=1_000.0, leverage=1.0) == pytest.approx(0.0)


def test_can_afford_admits_a_cost_exactly_equal_to_buying_power() -> None:
    ledger = account(cash=10_000.0)

    assert ledger.can_afford(10_000.0, positions_value=0.0, leverage=1.0)
    assert not ledger.can_afford(10_000.01, positions_value=0.0, leverage=1.0)


def test_a_zero_cost_order_is_always_affordable() -> None:
    broke = account(cash=0.0)

    assert broke.can_afford(0.0, positions_value=0.0, leverage=1.0)
    assert not broke.can_afford(1.0, positions_value=0.0, leverage=1.0)


# --------------------------------------------------------------------------- #
# Equity curve rows
# --------------------------------------------------------------------------- #


def test_leverage_is_gross_exposure_over_equity() -> None:
    point = equity_point(gross_exposure=150_000.0)

    assert point.leverage == pytest.approx(1.5)


def test_an_unlevered_row_reports_zero_not_a_division_error() -> None:
    assert equity_point(gross_exposure=0.0).leverage == pytest.approx(0.0)


def test_a_wiped_out_account_holding_nothing_is_flat_not_undefined() -> None:
    assert equity_point(cash=0.0, positions_value=0.0, equity=0.0).leverage == pytest.approx(0.0)


def test_a_wiped_out_account_still_holding_exposure_reports_infinite_leverage() -> None:
    """Rounding a blow-up down to 0x would hide it in the equity curve."""
    ruined = equity_point(cash=0.0, positions_value=0.0, equity=0.0, gross_exposure=25_000.0)

    assert math.isinf(ruined.leverage)


def test_defaults_leave_the_optional_columns_empty() -> None:
    point = equity_point()

    assert point.margin_used == pytest.approx(0.0)
    assert point.open_positions == 0
    assert point.drawdown == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


LOCAL_CLOSE = datetime(2023, 3, 28, 16, 0, tzinfo=ZoneInfo("America/New_York"))
NAIVE = datetime(2023, 3, 28, 20, 0)  # noqa: DTZ001


def test_the_equity_curve_x_axis_is_utc() -> None:
    """One naive stamp among aware ones makes the curve unsortable."""
    assert equity_point(timestamp=LOCAL_CLOSE).timestamp == STAMP

    with pytest.raises(ValidationError, match="Naive datetime"):
        equity_point(timestamp=NAIVE)


def test_cash_flows_are_stamped_in_utc() -> None:
    assert CashFlow(timestamp=LOCAL_CLOSE, amount=12.0, reason="dividend").timestamp == STAMP

    with pytest.raises(ValidationError, match="Naive datetime"):
        CashFlow(timestamp=NAIVE, amount=12.0, reason="dividend")


# --------------------------------------------------------------------------- #
# Non-finite guards and margin bookkeeping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("action", ["credit", "debit"])
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_amounts_never_reach_the_ledger(action: str, bad: float) -> None:
    """One inf makes cash infinite; the next movement the other way makes it nan,
    and no later arithmetic recovers from that."""
    ledger = account()
    with pytest.raises(ValidationError, match="non-negative, finite"):
        getattr(ledger, action)(bad, "bad feed")

    assert ledger.cash == pytest.approx(100_000.0)


def test_an_unlevered_account_cannot_spend_the_value_of_what_it_holds() -> None:
    """The single buying-power expression leans on ``margin_used`` being booked.
    A caller that forgets must get a conservative number, not an invitation to
    spend positions it is still holding."""
    unbooked = account(cash=0.0, margin_used=0.0)

    assert unbooked.buying_power(positions_value=100_000.0, leverage=1.0) == pytest.approx(0.0)


def test_the_margin_cap_does_not_clip_a_levered_account() -> None:
    levered = account(cash=40_000.0, margin_used=30_000.0)

    assert levered.buying_power(positions_value=60_000.0, leverage=2.0) == pytest.approx(170_000.0)
