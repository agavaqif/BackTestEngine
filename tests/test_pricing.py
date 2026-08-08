"""Fill pricing: synthetic spreads, quote sides and the spread adjustment."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigmaloop.domain.bar import Bar, Quote
from sigmaloop.domain.instrument import Equity, OptionContract
from sigmaloop.errors import ExecutionError, ValidationError
from sigmaloop.execution.pricing import (
    FixedBpsSpreadModel,
    OhlcFillPriceModel,
    PricingContext,
    QuoteFillPriceModel,
    TickSpreadModel,
    VolatilitySpreadModel,
)
from sigmaloop.types import InstrumentId, OptionRight, OrderSide, PriceSelection, Symbol

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


def bar(
    close: float = 100.0, *, low: float | None = None, high: float | None = None, **kw: object
) -> Bar:
    return Bar(
        instrument_id=MSFT.instrument_id,
        timestamp=STAMP,
        open=close,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        volume=1_000.0,
        **kw,  # type: ignore[arg-type]
    )


def context(**overrides: object) -> PricingContext:
    kwargs: dict[str, object] = {
        "instrument": MSFT,
        "side": OrderSide.BUY,
        "selection": PriceSelection.WORST,
        "bar": bar(),
    }
    kwargs.update(overrides)
    return PricingContext(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Spread models
# --------------------------------------------------------------------------- #


def test_fixed_bps_spread_straddles_the_close() -> None:
    quote = FixedBpsSpreadModel(spread_bps=2.0).quote_for(bar(100.0), MSFT)
    # 2bp of 100 is 0.02 wide, so a penny either side of the mark.
    assert (quote.bid, quote.ask) == (99.99, 100.01)
    assert quote.is_synthetic


def test_fixed_bps_spread_uses_the_option_rate_for_contracts() -> None:
    model = FixedBpsSpreadModel(spread_bps=2.0, option_spread_bps=100.0)
    equity_half = model.half_spread(2.0, MSFT)
    option_half = model.half_spread(2.0, CALL)
    assert option_half == pytest.approx(0.01)
    assert option_half == pytest.approx(equity_half * 50)


def test_tick_spread_is_independent_of_price_level() -> None:
    quote = TickSpreadModel(ticks=2.0).quote_for(bar(3.00), MSFT)
    assert (quote.bid, quote.ask) == (2.99, 3.01)


def test_volatility_spread_widens_with_the_bar_range() -> None:
    model = VolatilitySpreadModel(base_bps=0.0, range_multiplier=0.1)
    calm = model.quote_for(bar(100.0, low=99.9, high=100.1), MSFT)
    stressed = model.quote_for(bar(100.0, low=95.0, high=105.0), MSFT)
    assert stressed.spread > calm.spread
    # 10% of a 10-point range, half either side.
    assert stressed.ask - 100.0 == pytest.approx(0.5)


def test_a_spread_can_be_centred_where_the_fill_transacts() -> None:
    """The bar's close is the wrong basis for a fill at its open."""
    wide = bar(140.0, low=100.0, high=140.0)
    model = FixedBpsSpreadModel(spread_bps=100.0)

    at_close = model.quote_for(wide, MSFT)
    at_open = model.quote_for(wide, MSFT, at=100.0)
    assert at_close.ask == pytest.approx(140.70)
    assert at_open.ask == pytest.approx(100.50)


def test_the_volatility_model_keeps_its_range_term_when_recentred() -> None:
    model = VolatilitySpreadModel(base_bps=0.0, range_multiplier=0.1)
    quote = model.quote_for(bar(140.0, low=100.0, high=140.0), MSFT, at=100.0)
    # Centred on the open, still widened by 10% of the 40-point range.
    assert quote.mid == pytest.approx(100.0)
    assert quote.ask == pytest.approx(102.0)


def test_the_price_model_centres_the_book_on_the_execution_reference() -> None:
    model = QuoteFillPriceModel(FixedBpsSpreadModel(spread_bps=100.0))
    wide = bar(140.0, low=100.0, high=140.0)
    priced = context(bar=wide, reference_price=100.0)
    quote = model.quote_used(priced)

    assert quote is not None and quote.ask == pytest.approx(100.50)
    assert model.spread_adjustment(priced) == pytest.approx(0.50)


def test_synthetic_bid_never_goes_negative() -> None:
    quote = FixedBpsSpreadModel(spread_bps=100_000.0).quote_for(bar(0.05), MSFT)
    assert quote.bid == 0.0


def test_negative_spread_is_refused() -> None:
    with pytest.raises(ValidationError):
        FixedBpsSpreadModel(spread_bps=-1.0)


# --------------------------------------------------------------------------- #
# Quote pricing
# --------------------------------------------------------------------------- #


def test_worst_pays_the_ask_and_hits_the_bid() -> None:
    model = QuoteFillPriceModel()
    quoted = bar(100.0, quote=Quote(bid=99.9, ask=100.1))
    assert model.resolve(context(bar=quoted, side=OrderSide.BUY)) == 100.1
    assert model.resolve(context(bar=quoted, side=OrderSide.SELL)) == 99.9


def test_best_and_mid_are_the_optimistic_and_neutral_sides() -> None:
    model = QuoteFillPriceModel()
    quoted = bar(100.0, quote=Quote(bid=99.9, ask=100.1))
    best = model.resolve(context(bar=quoted, selection=PriceSelection.BEST))
    mid = model.resolve(context(bar=quoted, selection=PriceSelection.MID))
    assert (best, mid) == (99.9, 100.0)


def test_last_ignores_the_book_entirely() -> None:
    model = QuoteFillPriceModel()
    quoted = bar(100.0, quote=Quote(bid=90.0, ask=110.0))
    assert model.resolve(context(bar=quoted, selection=PriceSelection.LAST)) == 100.0


def test_a_quoteless_feed_gets_the_synthetic_spread() -> None:
    model = QuoteFillPriceModel(FixedBpsSpreadModel(spread_bps=2.0))
    assert model.resolve(context()) == 100.01
    quote = model.quote_used(context())
    assert quote is not None and quote.is_synthetic


def test_without_a_spread_model_worst_collapses_to_the_close() -> None:
    """The exact failure the synthetic spread exists to prevent, made explicit."""
    assert QuoteFillPriceModel().resolve(context()) == 100.0


def test_require_real_quotes_refuses_a_synthesised_book() -> None:
    model = QuoteFillPriceModel(FixedBpsSpreadModel(), require_real_quotes=True)
    with pytest.raises(ExecutionError, match="require_real_quotes"):
        model.resolve(context())


def test_require_real_quotes_accepts_an_observed_book() -> None:
    model = QuoteFillPriceModel(FixedBpsSpreadModel(), require_real_quotes=True)
    quoted = bar(100.0, quote=Quote(bid=99.9, ask=100.1))
    assert model.resolve(context(bar=quoted)) == 100.1


def test_a_supplied_synthetic_quote_is_reused_rather_than_rebuilt() -> None:
    model = QuoteFillPriceModel(FixedBpsSpreadModel(spread_bps=2.0))
    supplied = Quote(bid=50.0, ask=150.0, is_synthetic=True)
    assert model.resolve(context(synthetic_quote=supplied)) == 150.0


# --------------------------------------------------------------------------- #
# Spread adjustment — what the broker adds to the execution model's price
# --------------------------------------------------------------------------- #


def test_spread_adjustment_is_signed_by_side() -> None:
    model = QuoteFillPriceModel()
    quoted = bar(100.0, quote=Quote(bid=99.9, ask=100.1))
    buy = model.spread_adjustment(context(bar=quoted, side=OrderSide.BUY))
    sell = model.spread_adjustment(context(bar=quoted, side=OrderSide.SELL))
    assert buy == pytest.approx(0.1)
    assert sell == pytest.approx(-0.1)


@pytest.mark.parametrize("selection", [PriceSelection.MID, PriceSelection.LAST])
def test_neutral_selections_cost_nothing(selection: PriceSelection) -> None:
    model = QuoteFillPriceModel()
    quoted = bar(100.0, quote=Quote(bid=99.0, ask=101.0))
    assert model.spread_adjustment(context(bar=quoted, selection=selection)) == 0.0


# --------------------------------------------------------------------------- #
# OHLC pricing
# --------------------------------------------------------------------------- #


def test_ohlc_prices_from_its_field() -> None:
    model = OhlcFillPriceModel("high")
    priced = bar(100.0, low=99.0, high=101.0)
    assert model.resolve(context(bar=priced, selection=PriceSelection.LAST)) == 101.0


def test_ohlc_refuses_a_selection_it_cannot_honour() -> None:
    model = OhlcFillPriceModel()
    assert not model.supports(PriceSelection.WORST)
    with pytest.raises(ExecutionError, match="knows nothing about the spread"):
        model.resolve(context(selection=PriceSelection.WORST))


def test_ohlc_charges_no_spread() -> None:
    assert OhlcFillPriceModel().spread_adjustment(context()) == 0.0


def test_unknown_ohlc_field_is_refused_at_construction() -> None:
    with pytest.raises(ValidationError, match="Unknown OHLC field"):
        OhlcFillPriceModel("median")
