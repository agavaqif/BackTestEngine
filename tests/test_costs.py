"""Slippage and commission: adverse impact, participation caps, fee schedules."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigmaloop.domain.bar import Bar, Quote
from sigmaloop.domain.instrument import Equity, OptionContract
from sigmaloop.errors import ValidationError
from sigmaloop.execution.commission import (
    CommissionContext,
    CommissionModel,
    CompositeCommissionModel,
    PercentValueCommissionModel,
    PerContractCommissionModel,
    PerShareCommissionModel,
    PerTradeCommissionModel,
    RegulatoryFeeModel,
    TieredCommissionModel,
    ZeroCommissionModel,
)
from sigmaloop.execution.slippage import (
    FixedBpsSlippageModel,
    NoSlippageModel,
    SlippageContext,
    SpreadFractionSlippageModel,
    TickSlippageModel,
    VolumeShareSlippageModel,
)
from sigmaloop.types import InstrumentId, OptionRight, OrderSide, Symbol

STAMP = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
MSFT = Equity(instrument_id=InstrumentId("EQ:MSFT"), symbol=Symbol("MSFT"))
CALL = OptionContract(
    instrument_id=OptionContract.make_id(
        Symbol("MSFT"), date(2023, 4, 21), OptionRight.CALL, 280.0
    ),
    symbol=Symbol("MSFT230421C00280000"),
    underlying_id=MSFT.instrument_id,
    underlying_symbol=Symbol("MSFT"),
    right=OptionRight.CALL,
    strike=280.0,
    expiry=date(2023, 4, 21),
)


def bar(volume: float = 10_000.0, quote: Quote | None = None) -> Bar:
    return Bar(
        instrument_id=MSFT.instrument_id,
        timestamp=STAMP,
        open=100.0,
        high=100.0,
        low=100.0,
        close=100.0,
        volume=volume,
        quote=quote,
    )


def slippage_context(**overrides: object) -> SlippageContext:
    kwargs: dict[str, object] = {
        "instrument": MSFT,
        "side": OrderSide.BUY,
        "quantity": 100.0,
        "reference_price": 100.0,
        "bar": bar(),
    }
    kwargs.update(overrides)
    return SlippageContext(**kwargs)  # type: ignore[arg-type]


def commission_context(**overrides: object) -> CommissionContext:
    kwargs: dict[str, object] = {
        "instrument": MSFT,
        "side": OrderSide.BUY,
        "quantity": 100.0,
        "price": 100.0,
    }
    kwargs.update(overrides)
    return CommissionContext(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Slippage
# --------------------------------------------------------------------------- #


def test_no_slippage_is_free_and_uncapped() -> None:
    model = NoSlippageModel()
    assert model.slippage_per_unit(slippage_context()) == 0.0
    assert model.fillable_quantity(slippage_context()) == 100.0


def test_fixed_bps_scales_with_the_reference_price() -> None:
    model = FixedBpsSlippageModel(bps=5.0)
    assert model.slippage_per_unit(slippage_context()) == pytest.approx(0.05)
    assert model.slippage_per_unit(slippage_context(reference_price=200.0)) == pytest.approx(0.10)


def test_tick_slippage_ignores_the_price_level() -> None:
    model = TickSlippageModel(ticks=2.0)
    assert model.slippage_per_unit(slippage_context(reference_price=3.0)) == pytest.approx(0.02)


def test_volume_share_caps_participation() -> None:
    model = VolumeShareSlippageModel(max_volume_share=0.025)
    # 2.5% of a 10,000-share bar is 250 shares, whatever the order asked for.
    assert model.fillable_quantity(slippage_context(quantity=1_000.0)) == 250.0
    assert model.fillable_quantity(slippage_context(quantity=100.0)) == 100.0


def test_volume_share_impact_grows_with_the_square_root_of_participation() -> None:
    model = VolumeShareSlippageModel(coefficient=0.1, max_volume_share=1.0)
    small = model.slippage_per_unit(slippage_context(quantity=100.0))
    large = model.slippage_per_unit(slippage_context(quantity=400.0))
    assert large == pytest.approx(small * 2.0)


def test_a_bar_that_printed_nothing_absorbs_nothing() -> None:
    model = VolumeShareSlippageModel()
    context = slippage_context(bar=bar(volume=0.0))
    assert model.fillable_quantity(context) == 0.0
    assert model.slippage_per_unit(context) > 0.0


def test_spread_fraction_takes_its_cut_of_the_book() -> None:
    model = SpreadFractionSlippageModel(fraction=0.5)
    wide = slippage_context(bar=bar(quote=Quote(bid=99.0, ask=101.0)))
    assert model.slippage_per_unit(wide) == pytest.approx(1.0)


def test_spread_fraction_charges_nothing_without_a_book() -> None:
    """Rather than invent a spread the pricing layer has already accounted for."""
    assert SpreadFractionSlippageModel().slippage_per_unit(slippage_context()) == 0.0


def test_negative_slippage_is_refused() -> None:
    with pytest.raises(ValidationError):
        FixedBpsSlippageModel(bps=-1.0)


# --------------------------------------------------------------------------- #
# Commission
# --------------------------------------------------------------------------- #


def test_zero_commission_costs_nothing() -> None:
    model = ZeroCommissionModel()
    assert model.total(commission_context()) == 0.0


def test_per_share_applies_the_floor_then_the_cap() -> None:
    model = PerShareCommissionModel(rate=0.005, minimum=1.0, maximum_pct_of_value=0.01)
    # 100 shares at 0.005 is 0.50, lifted to the 1.00 floor.
    assert model.commission(commission_context()) == pytest.approx(1.0)
    # 10 shares of a $2 stock: the floor says 1.00, the 1%-of-value cap says 0.20.
    tiny = commission_context(quantity=10.0, price=2.0)
    assert model.commission(tiny) == pytest.approx(0.20)


def test_per_share_without_a_cap_keeps_the_floor() -> None:
    model = PerShareCommissionModel(rate=0.005, minimum=1.0, maximum_pct_of_value=None)
    assert model.commission(commission_context(quantity=10.0, price=2.0)) == pytest.approx(1.0)


def test_per_trade_is_flat() -> None:
    model = PerTradeCommissionModel(fee=0.65)
    assert model.commission(commission_context(quantity=1.0)) == 0.65
    assert model.commission(commission_context(quantity=10_000.0)) == 0.65


def test_percent_value_prices_the_notional() -> None:
    model = PercentValueCommissionModel(rate=0.001)
    assert model.commission(commission_context()) == pytest.approx(10.0)


def test_per_contract_uses_the_multiplier_free_contract_count() -> None:
    model = PerContractCommissionModel(per_contract=0.65, per_order=1.0)
    context = commission_context(instrument=CALL, quantity=10.0, price=2.0)
    assert model.commission(context) == pytest.approx(7.5)


def test_per_contract_waives_a_cheap_close() -> None:
    model = PerContractCommissionModel(waive_close_below_price=0.05)
    cheap = commission_context(instrument=CALL, quantity=10.0, price=0.02, is_closing=True)
    assert model.commission(cheap) == 0.0
    # Opening at the same price is still billed — the waiver is a closing perk.
    assert model.commission(commission_context(instrument=CALL, quantity=10.0, price=0.02)) > 0.0


def test_regulatory_fees_are_sell_side_for_equities() -> None:
    model = RegulatoryFeeModel()
    buy = commission_context(side=OrderSide.BUY)
    sell = commission_context(side=OrderSide.SELL)
    assert model.commission(sell) == 0.0
    assert model.fees(buy) == 0.0
    # SEC on $10,000 of notional plus TAF on 100 shares.
    assert model.fees(sell) == pytest.approx(0.0000278 * 10_000 + 0.000166 * 100)


def test_finra_taf_is_capped() -> None:
    model = RegulatoryFeeModel(sec_fee_rate=0.0, finra_taf_per_share=0.000166, finra_taf_cap=8.30)
    huge = commission_context(side=OrderSide.SELL, quantity=10_000_000.0)
    assert model.fees(huge) == pytest.approx(8.30)


def test_the_options_regulatory_fee_is_charged_both_ways() -> None:
    model = RegulatoryFeeModel(options_orf_per_contract=0.0388, sec_fee_rate=0.0)
    for side in (OrderSide.BUY, OrderSide.SELL):
        context = commission_context(instrument=CALL, side=side, quantity=10.0, price=2.0)
        assert model.fees(context) == pytest.approx(0.388)


def test_tiered_rates_step_down_with_monthly_volume() -> None:
    model = TieredCommissionModel(tiers=[(300_000.0, 0.0035), (0.0, 0.005)])
    starting = commission_context(quantity=1_000.0)
    heavy = commission_context(quantity=1_000.0, period_volume=500_000.0)
    assert model.commission(starting) == pytest.approx(5.0)
    assert model.commission(heavy) == pytest.approx(3.5)


def test_a_tiered_schedule_needs_at_least_one_band() -> None:
    with pytest.raises(ValidationError):
        TieredCommissionModel(tiers=[])


def test_composite_sums_commission_and_fees_separately() -> None:
    model = CompositeCommissionModel([PerShareCommissionModel(), RegulatoryFeeModel()])
    sell = commission_context(side=OrderSide.SELL)
    assert model.commission(sell) == pytest.approx(PerShareCommissionModel().commission(sell))
    assert model.fees(sell) == pytest.approx(RegulatoryFeeModel().fees(sell))
    assert model.total(sell) == pytest.approx(model.commission(sell) + model.fees(sell))


def test_composite_needs_a_child() -> None:
    with pytest.raises(ValidationError, match="ZeroCommissionModel"):
        CompositeCommissionModel([])


# --------------------------------------------------------------------------- #
# Per-order schedules across partial fills
# --------------------------------------------------------------------------- #


def charge_in_slices(
    model: CommissionModel, total: float, slices: int, **overrides: object
) -> float:
    """What ``total`` units cost when a liquidity cap splits them ``slices`` ways."""
    step = total / slices
    return sum(
        model.total(commission_context(quantity=step, filled_before=step * index, **overrides))
        for index in range(slices)
    )


@pytest.mark.parametrize(
    ("model", "total", "overrides"),
    [
        (PerTradeCommissionModel(fee=5.0), 1_000.0, {}),
        # 10 shares of a $2 name: the $1 floor binds, then the 1% cap claws it back.
        (
            PerShareCommissionModel(rate=0.005, minimum=1.0, maximum_pct_of_value=0.01),
            10.0,
            {"price": 2.0},
        ),
        (PercentValueCommissionModel(rate=0.001, minimum=25.0), 100.0, {}),
        (PerContractCommissionModel(per_contract=0.65, per_order=1.0), 10.0, {"instrument": CALL}),
        # Far past the $8.30 TAF cap, which is a ceiling on the order, not the fill.
        (RegulatoryFeeModel(), 1_000_000.0, {"side": OrderSide.SELL}),
        (TieredCommissionModel(tiers=[(0.0, 0.005)], minimum=10.0), 100.0, {}),
    ],
    ids=["per_trade", "per_share", "percent_value", "per_contract", "regulatory", "tiered"],
)
def test_a_split_order_costs_what_a_single_print_costs(
    model: CommissionModel, total: float, overrides: dict[str, object]
) -> None:
    """Floors, caps and flat fees belong to the order, not to the print that
    happened to complete it.

    A thin market that fills one order over four bars must not bill the minimum
    four times, nor collect a capped fee once per slice. Every schedule here has
    a per-order component that binds at the size given.
    """
    whole = charge_in_slices(model, total, 1, **overrides)
    assert whole > 0.0, "the per-order component must actually bind at this size"
    for slices in (2, 4, 10):
        assert charge_in_slices(model, total, slices, **overrides) == pytest.approx(whole)


def test_a_flat_fee_lands_on_the_first_print() -> None:
    """Not spread across fills, and not deferred to the last one — a strategy
    reading its first fill should already see what the order cost to place."""
    model = PerTradeCommissionModel(fee=5.0)
    assert model.commission(commission_context(quantity=250.0)) == pytest.approx(5.0)
    assert model.commission(commission_context(quantity=250.0, filled_before=250.0)) == 0.0


def test_the_first_fill_is_the_one_with_nothing_behind_it() -> None:
    assert commission_context().is_first_fill
    assert commission_context(filled_before=1e-12).is_first_fill, "float dust is not a fill"
    assert not commission_context(filled_before=1.0).is_first_fill
