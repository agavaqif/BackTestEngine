"""MarketSnapshot: the closed world the engine sees at one timestamp."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sigmaloop.domain.bar import Bar, MarketSnapshot, OptionChain, OptionQuote, Quote
from sigmaloop.domain.instrument import OptionContract
from sigmaloop.errors import DataNotAvailableError, ValidationError
from sigmaloop.types import InstrumentId, OptionRight, PriceSelection, Symbol, Timeframe

MSFT = InstrumentId("EQ:MSFT")
AAPL = InstrumentId("EQ:AAPL")
STAMP = datetime(2023, 3, 28, 20, 0, tzinfo=UTC)


def bar(instrument_id: InstrumentId, close: float, quote: Quote | None = None) -> Bar:
    return Bar(
        instrument_id=instrument_id,
        timestamp=STAMP,
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=1_000.0,
        timeframe=Timeframe.D1,
        quote=quote,
    )


def snapshot(*bars: Bar, **kwargs: object) -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=STAMP,
        bars={b.instrument_id: b for b in bars},
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #


def test_bar_lookup_and_size() -> None:
    snap = snapshot(bar(MSFT, 100.0), bar(AAPL, 200.0))

    assert len(snap) == 2
    assert sorted(snap.instruments()) == [AAPL, MSFT]
    found = snap.bar(MSFT)
    assert found is not None
    assert found.close == pytest.approx(100.0)


def test_absent_instrument_is_absent_not_stale() -> None:
    """A closed world: no bar means it did not trade, not that nothing changed."""
    snap = snapshot(bar(MSFT, 100.0))

    assert snap.bar(AAPL) is None
    assert snap.price(AAPL) is None
    assert AAPL not in snap.instruments()


def test_require_bar_names_the_instrument_and_the_instant() -> None:
    snap = snapshot(bar(MSFT, 100.0))

    assert snap.require_bar(MSFT).close == pytest.approx(100.0)
    with pytest.raises(DataNotAvailableError) as raised:
        snap.require_bar(AAPL)
    assert raised.value.context["timestamp"] == STAMP.isoformat()
    assert "NO_MARKET_DATA" in str(raised.value)


# --------------------------------------------------------------------------- #
# Marking
# --------------------------------------------------------------------------- #


def test_price_falls_back_to_close_without_a_book() -> None:
    snap = snapshot(bar(MSFT, 100.0))
    assert snap.price(MSFT) == pytest.approx(100.0)


def test_price_marks_at_the_close_even_when_a_book_is_attached() -> None:
    """A Quote carries no timestamp, so the snapshot cannot tell a live book from
    one carried forward for hours. The close is contemporaneous with the bar by
    construction, so marking never silently freezes against a stale spread."""
    skewed = bar(MSFT, 100.0, quote=Quote(bid=99.0, ask=100.0))
    assert snapshot(skewed).price(MSFT) == pytest.approx(100.0), "close, not the 99.5 mid"


def test_attaching_a_book_does_not_move_the_mark() -> None:
    """``include_quotes`` attaches information; it must not change the equity curve."""
    unquoted = snapshot(bar(MSFT, 100.0)).price(MSFT)
    quoted = snapshot(bar(MSFT, 100.0, quote=Quote(bid=90.0, ask=110.0))).price(MSFT)
    assert unquoted == quoted


def test_execution_still_prices_off_the_real_book() -> None:
    """Marking ignores the quote; filling must not. That split is the whole point."""
    quoted = bar(MSFT, 100.0, quote=Quote(bid=99.0, ask=101.0))
    assert quoted.price_for(PriceSelection.WORST, is_buy=True) == pytest.approx(101.0)
    assert quoted.price_for(PriceSelection.WORST, is_buy=False) == pytest.approx(99.0)
    assert quoted.price_for(PriceSelection.MID, is_buy=True) == pytest.approx(100.0)


def test_price_resolves_an_option_from_its_chain() -> None:
    contract = OptionContract(
        instrument_id=InstrumentId("OPT:MSFT:20230421:C:00280000"),
        symbol=Symbol("MSFT230421C00280000"),
        underlying_id=MSFT,
        underlying_symbol=Symbol("MSFT"),
        right=OptionRight.CALL,
        strike=280.0,
        expiry=date(2023, 4, 21),
    )
    option = OptionQuote(
        instrument_id=contract.instrument_id,
        contract=contract,
        timestamp=STAMP,
        quote=Quote(bid=2.0, ask=2.4),
    )
    chain = OptionChain(
        underlying_id=MSFT,
        underlying_symbol=Symbol("MSFT"),
        timestamp=STAMP,
        underlying_price=275.0,
        quotes=(option,),
    )
    snap = snapshot(bar(MSFT, 275.0), chains={MSFT: chain})

    assert snap.chain(MSFT) is chain
    assert snap.chain(AAPL) is None
    assert snap.price(contract.instrument_id) == pytest.approx(2.2)


# --------------------------------------------------------------------------- #
# Option quotes and chains
# --------------------------------------------------------------------------- #


def option_quote(**overrides: object) -> OptionQuote:
    contract = OptionContract(
        instrument_id=InstrumentId("OPT:MSFT:20230421:C:00280000"),
        symbol=Symbol("MSFT230421C00280000"),
        underlying_id=MSFT,
        underlying_symbol=Symbol("MSFT"),
        right=OptionRight.CALL,
        strike=280.0,
        expiry=date(2023, 4, 21),
    )
    kwargs: dict[str, object] = {
        "instrument_id": contract.instrument_id,
        "contract": contract,
        "timestamp": STAMP,
        "quote": Quote(bid=2.0, ask=2.4),
    }
    kwargs.update(overrides)
    return OptionQuote(**kwargs)  # type: ignore[arg-type]


def test_an_option_quote_prices_from_its_own_book() -> None:
    option = option_quote()

    assert option.mid == pytest.approx(2.2)
    assert option.price_for(PriceSelection.WORST, is_buy=True) == pytest.approx(2.4)
    assert option.price_for(PriceSelection.WORST, is_buy=False) == pytest.approx(2.0)
    assert option.price_for(PriceSelection.BEST, is_buy=True) == pytest.approx(2.0)


def test_last_falls_back_to_the_mid_when_the_contract_never_printed() -> None:
    """An untraded contract has no last, and a stale one is not a price anyone
    could have transacted at."""
    assert option_quote().price_for(PriceSelection.LAST, is_buy=True) == pytest.approx(2.2)
    traded = option_quote(last=2.1)
    assert traded.price_for(PriceSelection.LAST, is_buy=True) == pytest.approx(2.1)


def test_delta_without_greeks_is_an_error_not_a_zero() -> None:
    from sigmaloop.domain.bar import Greeks
    from sigmaloop.errors import DataError

    with pytest.raises(DataError, match="no greeks"):
        _ = option_quote().delta
    assert option_quote(greeks=Greeks(delta=0.42)).delta == pytest.approx(0.42)


def test_dte_is_measured_from_the_quote_timestamp() -> None:
    assert option_quote().days_to_expiry == 24
    assert not option_quote().is_zero_dte
    assert option_quote(timestamp=datetime(2023, 4, 21, 14, tzinfo=UTC)).is_zero_dte


def test_a_chain_looks_its_contracts_up_by_id() -> None:
    option = option_quote()
    chain = OptionChain(
        underlying_id=MSFT,
        underlying_symbol=Symbol("MSFT"),
        timestamp=STAMP,
        underlying_price=275.0,
        quotes=(option,),
    )

    assert len(chain) == 1
    assert list(chain) == [option]
    assert chain.get(option.instrument_id) is option
    assert chain.get(InstrumentId("OPT:MSFT:20230421:P:00280000")) is None


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #


def test_timestamp_is_normalised_to_utc() -> None:
    from zoneinfo import ZoneInfo

    local = datetime(2023, 3, 28, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    snap = MarketSnapshot(timestamp=local, bars={})

    assert snap.timestamp == datetime(2023, 3, 28, 20, 0, tzinfo=UTC)
    assert snap.timestamp.tzinfo is UTC


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Naive datetime"):
        MarketSnapshot(timestamp=datetime(2023, 3, 28, 20, 0), bars={})  # noqa: DTZ001


def test_session_flags_default_to_open_and_close() -> None:
    snap = snapshot(bar(MSFT, 100.0))
    assert snap.is_session_open is True
    assert snap.is_session_close is True
